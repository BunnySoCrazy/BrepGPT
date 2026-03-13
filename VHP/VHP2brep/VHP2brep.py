"""
Graph-to-STEP Converter

Converts DGL half-edge graphs (.bin) into STEP CAD models via two surface strategies:

  Standard mode (default):
    BRepFill_Filling with Voronoi interior point constraints.

  UV mode (--use-uv):
    RBF-interpolated B-spline surface fitted from UV-XYZ sample pairs,

Pipeline:
  1. Load the DGL graph from a .bin file
  2. Build the half-edge data structure (vertices, edges, twins)
  3. Establish next/prev connectivity via the Hungarian algorithm
  4. Extract face loops from the half-edge structure
  5. Create B-Rep faces
  6. Sew faces and export as STEP + STL

Graph conventions:
  graph.ndata['x']                [N, 3]   vertex coordinates
  graph.edata['x']                [E, curve_samples, normal_samples, 3]  VHP samples
  graph.edata['next_half_edge']   [E, normal_samples, 3]
  graph.edata['edge_inner_outer'] [E, 1]   inner/outer loop flag
  graph.edata['uv']               [E, curve_samples, normal_samples, 2]  UV coords (UV mode)
"""

import os
import random
import signal
import multiprocessing as mp

import numpy as np
import torch
import dgl
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import RBFInterpolator

from OCC.Core.gp import gp_Pnt, gp_Trsf, gp_Vec
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline, GeomAPI_PointsToBSplineSurface
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeWire, BRepBuilderAPI_Transform,
)
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.TopoDS import topods
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Core.ShapeFix import ShapeFix_Wire

from halfedge_brep_reconstructor import (
    HalfEdge, Vertex,
    extract_loops, is_all_simple_loops,
    create_faces_from_loops,
    match_inner_wires_to_faces, create_faces_with_inner_loops,
    add_pcurves_to_edges, fix_wires, fix_face,
    sew_faces, export_to_step, export_to_stl,
)


# ---------------------------------------------------------------------------
# Graph loading and de-normalization
# ---------------------------------------------------------------------------

def load_graph(bin_file):
    """Load a DGL graph from a .bin file."""
    graph = dgl.load_graphs(bin_file)[0][0]

    src, dst = graph.edges()
    edge_set = set()
    self_loops = 0
    multi_edges = 0

    for i in range(len(src)):
        s, d = src[i].item(), dst[i].item()
        if s == d:
            self_loops += 1
        edge = (s, d)
        if edge in edge_set:
            multi_edges += 1
        edge_set.add(edge)

    return graph


def inverse_process_graph(graph, num_curve_samples=8, num_normal_samples=4):
    """Restore absolute coordinates from the normalized graph representation.

    Edge features are stored as offsets; this converts them back to
    world-space positions by undoing the normalization applied during encoding.
    """
    g = graph
    src, dst = g.edges()
    g.edata['ori_edge_data'] = g.edata["x"].clone()

    for eid in range(g.number_of_edges()):
        edge_data = g.edata["x"][eid].clone()
        next_edge_data = g.edata["next_half_edge"][eid].clone()
        start_pt = g.ndata["x"][src[eid]]
        end_pt = g.ndata["x"][dst[eid]]

        edge_length = 1
        edge_data = edge_data * edge_length

        t = torch.linspace(0, 1, num_curve_samples + 2, device=edge_data.device)[1:-1]
        interp = (
            start_pt.unsqueeze(0).unsqueeze(0) * (1 - t).unsqueeze(-1).unsqueeze(0)
            + end_pt.unsqueeze(0).unsqueeze(0) * t.unsqueeze(-1).unsqueeze(0)
        )

        edge_data[:, 0, :] += interp.squeeze(0)
        for i in range(1, num_normal_samples):
            edge_data[:, i, :] += edge_data[:, i - 1, :]

        next_edge_data[0, :] += end_pt
        for i in range(1, num_normal_samples):
            next_edge_data[i, :] += next_edge_data[i - 1, :]

        g.edata["x"][eid] = edge_data
        g.edata["next_half_edge"][eid] = next_edge_data

    return g


def create_bspline_curve(graph, src_id, dst_id, edge_points, he, twin_he):
    """Fit a B-spline through edge sample points and assign to both half-edges.

    Normalizes to unit scale before fitting, then transforms back.
    """
    curve_points = edge_points[:, 0, :]

    all_points = torch.cat([
        graph.ndata['x'][src_id].unsqueeze(0),
        curve_points,
        graph.ndata['x'][dst_id].unsqueeze(0),
    ], dim=0)

    pts_np = all_points.numpy()
    center = pts_np.mean(axis=0)
    scale = max(np.abs(pts_np - center).max(), 1e-12)
    pts_norm = (pts_np - center) / scale

    occ_points = TColgp_Array1OfPnt(1, len(pts_norm))
    for i, pt in enumerate(pts_norm):
        occ_points.SetValue(i + 1, gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])))

    curve_builder = GeomAPI_PointsToBSpline(occ_points, 0, 8, GeomAbs_C2, 5e-3)
    if not curve_builder.IsDone():
        return

    curve = curve_builder.Curve()
    edge_norm = BRepBuilderAPI_MakeEdge(curve).Edge()

    trsf = gp_Trsf()
    trsf.SetScaleFactor(float(scale))
    trsf_t = gp_Trsf()
    trsf_t.SetTranslation(gp_Vec(float(center[0]), float(center[1]), float(center[2])))

    trsf_back = gp_Trsf()
    trsf_back.Multiply(trsf_t)
    trsf_back.Multiply(trsf)

    edge = BRepBuilderAPI_Transform(edge_norm, trsf_back, True).Shape()
    he.edge = edge
    twin_he.edge = edge.Reversed()


# ---------------------------------------------------------------------------
# Shared half-edge infrastructure
# ---------------------------------------------------------------------------

def build_halfedge_structure(graph):
    """Initialize empty half-edge containers and create Vertex objects."""
    vertices = {}   # (x, y, z) -> Vertex
    halfedges = []
    edge_map = {}   # ((x1,y1,z1), (x2,y2,z2)) -> HalfEdge

    for nid in range(graph.number_of_nodes()):
        coord = tuple(graph.ndata['x'][nid].tolist())
        vertices[coord] = Vertex(coord)

    return vertices, halfedges, edge_map


def build_edge_connections(vertices):
    """Establish next/prev links between half-edges at each vertex.

    Uses the Hungarian algorithm to optimally match incoming to outgoing
    half-edges based on curve-sample similarity. Twin pairs are excluded.
    """
    for vertex in vertices:
        outgoing = vertex.out_halfedges
        incoming = vertex.in_halfedges

        if len(outgoing) <= 1 or len(incoming) <= 1:
            continue

        assert len(incoming) == len(outgoing), (
            "Incoming and outgoing half-edge counts must be equal"
        )

        n = len(incoming)
        cost_matrix = np.full((n, n), float('inf'))

        for i, inc_he in enumerate(incoming):
            if inc_he is None or not hasattr(inc_he, 'next_curve_samples'):
                continue
            for j, out_he in enumerate(outgoing):
                if out_he is None or not hasattr(out_he, 'curve_samples'):
                    continue
                if inc_he.twin is not None and out_he == inc_he.twin:
                    continue
                if (isinstance(inc_he.next_curve_samples, torch.Tensor)
                        and isinstance(out_he.curve_samples, torch.Tensor)):
                    num_ns = inc_he.next_curve_samples.shape[0]
                    next_pts = inc_he.next_curve_samples
                    out_pts = out_he.curve_samples[:num_ns, 0, :]
                    cost_matrix[i][j] = torch.norm(next_pts - out_pts, dim=1).sum().item()

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i][j] != float('inf'):
                incoming[i].next = outgoing[j]
                outgoing[j].prev = incoming[i]


# ---------------------------------------------------------------------------
# Standard mode: half-edge creation (no UV)
# ---------------------------------------------------------------------------

def create_halfedges(graph, vertices, halfedges, edge_map):
    """Create HalfEdge objects, pair twins, and fit B-spline curves."""
    src, dst = graph.edges()

    for eid in range(graph.number_of_edges()):
        start_coord = tuple(graph.ndata['x'][src[eid]].tolist())
        end_coord = tuple(graph.ndata['x'][dst[eid]].tolist())

        he = HalfEdge()
        he.vertex = vertices[start_coord]
        he.curve_samples = graph.edata['x'][eid]
        he.next_curve_samples = graph.edata['next_half_edge'][eid]
        he.edge_inner_outer = graph.edata['edge_inner_outer'][eid]

        vertices[start_coord].out_halfedges.append(he)
        vertices[end_coord].in_halfedges.append(he)

        reverse_key = (end_coord, start_coord)
        if reverse_key in edge_map:
            twin_he = edge_map[reverse_key]

            fwd_curve = he.curve_samples[:, 0, :]
            rev_curve = torch.flip(twin_he.curve_samples[:, 0, :], dims=[0])
            averaged = (fwd_curve + rev_curve) / 2

            he.curve_samples[:, 0, :] = averaged
            twin_he.curve_samples[:, 0, :] = torch.flip(averaged, dims=[0])

            he.twin = twin_he
            twin_he.twin = he

            create_bspline_curve(graph, src[eid], dst[eid],
                                 graph.edata['x'][eid], he, twin_he)

        edge_map[(start_coord, end_coord)] = he
        halfedges.append(he)

    return halfedges


# ---------------------------------------------------------------------------
# UV mode: half-edge creation and surface fitting
# ---------------------------------------------------------------------------

def create_halfedges_with_uv(graph, vertices, halfedges, edge_map):
    """Create HalfEdge objects with UV samples attached."""
    src, dst = graph.edges()
    has_uv = "uv" in graph.edata

    for eid in range(graph.number_of_edges()):
        start_coord = tuple(graph.ndata["x"][src[eid]].tolist())
        end_coord = tuple(graph.ndata["x"][dst[eid]].tolist())

        he = HalfEdge()
        he.vertex = vertices[start_coord]
        he.curve_samples = graph.edata["x"][eid]
        he.next_curve_samples = graph.edata["next_half_edge"][eid]
        he.edge_inner_outer = graph.edata["edge_inner_outer"][eid]
        he.uv_samples = graph.edata["uv"][eid] if has_uv else None

        vertices[start_coord].out_halfedges.append(he)
        vertices[end_coord].in_halfedges.append(he)

        reverse_key = (end_coord, start_coord)
        if reverse_key in edge_map:
            twin_he = edge_map[reverse_key]

            fwd = he.curve_samples[:, 0, :]
            rev = torch.flip(twin_he.curve_samples[:, 0, :], dims=[0])
            avg = (fwd + rev) / 2
            he.curve_samples[:, 0, :] = avg
            twin_he.curve_samples[:, 0, :] = torch.flip(avg, dims=[0])

            he.twin = twin_he
            twin_he.twin = he

            create_bspline_curve(
                graph, src[eid], dst[eid], graph.edata["x"][eid], he, twin_he
            )

        edge_map[(start_coord, end_coord)] = he
        halfedges.append(he)

    return halfedges


def _dedup_uv_by_radius(uv_points, xyz_points, eps):
    """Merge UV points closer than eps (grid quantization)."""
    keys = np.round(uv_points / eps).astype(np.int64)
    bucket = {}
    for i, k in enumerate(map(tuple, keys)):
        bucket.setdefault(k, []).append(i)

    uv_new, xyz_new = [], []
    for inds in bucket.values():
        uv_new.append(uv_points[inds].mean(axis=0))
        xyz_new.append(xyz_points[inds].mean(axis=0))
    return np.asarray(uv_new), np.asarray(xyz_new)


def _fit_bspline_surface(uv_points, xyz_points, grid_size=16):
    """Fit a B-spline surface from scattered UV-XYZ pairs via RBF interpolation.

    Normalizes XYZ to unit scale before fitting, then transforms back.
    Returns (surface, grid_xyz, u_lin, v_lin) or (None, None, None, None) on failure.
    """
    from OCC.Core.Approx import Approx_IsoParametric

    xyz_center = xyz_points.mean(axis=0)
    xyz_scale = max(np.abs(xyz_points - xyz_center).max(), 1e-12)

    u_min, v_min = uv_points.min(axis=0)
    u_max, v_max = uv_points.max(axis=0)

    u_lin = np.linspace(u_min, u_max, grid_size)
    v_lin = np.linspace(v_min, v_max, grid_size)
    uu, vv = np.meshgrid(u_lin, v_lin, indexing="ij")

    uv_dedup, xyz_dedup = _dedup_uv_by_radius(uv_points, xyz_points, eps=1e-9)
    xyz_dedup_norm = (xyz_dedup - xyz_center) / xyz_scale

    query = np.stack([uu.ravel(), vv.ravel()], axis=-1)
    grid_xyz_norm = np.zeros((grid_size, grid_size, 3))

    for d in range(3):
        rbf = RBFInterpolator(
            uv_dedup, xyz_dedup_norm[:, d:d + 1],
            kernel="thin_plate_spline", smoothing=0.0,
        )
        grid_xyz_norm[:, :, d] = rbf(query).reshape(grid_size, grid_size)

    grid_xyz = grid_xyz_norm * xyz_scale + xyz_center

    occ_pts = TColgp_Array2OfPnt(1, grid_size, 1, grid_size)
    for i in range(grid_size):
        for j in range(grid_size):
            occ_pts.SetValue(
                i + 1, j + 1,
                gp_Pnt(float(grid_xyz_norm[i, j, 0]),
                       float(grid_xyz_norm[i, j, 1]),
                       float(grid_xyz_norm[i, j, 2])),
            )

    try:
        fitter = GeomAPI_PointsToBSplineSurface()
        fitter.Init(occ_pts, Approx_IsoParametric, 2, 2, GeomAbs_C2, 1e-3, False)
        if fitter.IsDone():
            surf_norm = fitter.Surface()

            trsf = gp_Trsf()
            trsf.SetScaleFactor(float(xyz_scale))
            trsf_t = gp_Trsf()
            trsf_t.SetTranslation(gp_Vec(
                float(xyz_center[0]), float(xyz_center[1]), float(xyz_center[2]),
            ))
            trsf_back = gp_Trsf()
            trsf_back.Multiply(trsf_t)
            trsf_back.Multiply(trsf)

            surf_norm.Transform(trsf_back)
            return surf_norm, grid_xyz, u_lin, v_lin
    except Exception as e:
        print(f"  Surface fitting exception: {e}")

    return None, None, None, None


def _visualize_loop_samples(loop_idx, he_data, u_lin, v_lin, grid_xyz, debug_dir):
    """Save a PNG showing UV scatter, XYZ scatter, and interpolated surface grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n_edges = len(he_data)
    cmap = cm.get_cmap("tab20", max(n_edges, 1))
    colors = [cmap(i) for i in range(n_edges)]

    has_grid = grid_xyz is not None and u_lin is not None and v_lin is not None
    ncols = 3 if has_grid else 2
    fig = plt.figure(figsize=(6 * ncols, 5))

    ax_uv = fig.add_subplot(1, ncols, 1)
    ax_uv.set_title(f"Loop {loop_idx} - UV samples")
    ax_uv.set_xlabel("U")
    ax_uv.set_ylabel("V")
    ax_uv.set_aspect("equal", adjustable="datalim")

    for i, (uv, _) in enumerate(he_data):
        c = colors[i]
        cs, ns, _ = uv.shape
        for s in range(cs):
            ax_uv.plot(uv[s, :, 0], uv[s, :, 1], "-", color=c, alpha=0.25, linewidth=0.8)
        ax_uv.scatter(uv[:, 0, 0], uv[:, 0, 1], color=c, s=35, zorder=4, label=f"e{i}")
        if ns > 1:
            pts = uv[:, 1:, :].reshape(-1, 2)
            ax_uv.scatter(pts[:, 0], pts[:, 1], color=c, s=8, alpha=0.35, zorder=3)

    if has_grid:
        uu, vv = np.meshgrid(u_lin, v_lin, indexing="ij")
        ax_uv.plot(uu.ravel(), vv.ravel(), "k+", markersize=3, alpha=0.4, label="grid")
    ax_uv.legend(fontsize=6, loc="upper right", ncol=2)

    ax_xyz = fig.add_subplot(1, ncols, 2, projection="3d")
    ax_xyz.set_title(f"Loop {loop_idx} - XYZ samples")
    for i, (_, xyz) in enumerate(he_data):
        c = colors[i]
        ax_xyz.scatter(xyz[:, 0, 0], xyz[:, 0, 1], xyz[:, 0, 2],
                       color=c, s=20, depthshade=False, zorder=4)
        if xyz.shape[1] > 1:
            pts = xyz[:, 1:, :].reshape(-1, 3)
            ax_xyz.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                           color=c, s=4, alpha=0.3, depthshade=False)
    ax_xyz.set_xlabel("X")
    ax_xyz.set_ylabel("Y")
    ax_xyz.set_zlabel("Z")

    if has_grid:
        ax_surf = fig.add_subplot(1, ncols, 3, projection="3d")
        ax_surf.set_title(f"Loop {loop_idx} - surface grid")
        gx, gy, gz = grid_xyz[:, :, 0], grid_xyz[:, :, 1], grid_xyz[:, :, 2]
        ax_surf.plot_surface(gx, gy, gz, alpha=0.7, cmap="viridis",
                             linewidth=0.3, edgecolor="k")
        ax_surf.scatter(gx.ravel(), gy.ravel(), gz.ravel(), color="k", s=8, zorder=5)
        ax_surf.set_xlabel("X")
        ax_surf.set_ylabel("Y")
        ax_surf.set_zlabel("Z")

    plt.tight_layout()
    out = os.path.join(debug_dir, f"loop{loop_idx:03d}.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _edge_inner_outer_value(flag):
    if isinstance(flag, torch.Tensor):
        return int(flag.item())
    return int(flag) if flag else 0


def _build_face_uv(loop, grid_size=16, debug_dir=None, loop_idx=0):
    """Build a TopoDS_Face using UV-derived B-spline surface + original 3D edges.

    OCC auto-computes pcurves via ShapeFix_Edge.FixAddPCurve.
    Returns (face, inner_vote) or (None, 0).
    """
    uv_list, xyz_list = [], []
    he_data = []
    inner_vote = 0

    for he in loop:
        inner_vote += _edge_inner_outer_value(he.edge_inner_outer)

        if he.curve_samples is None or he.uv_samples is None:
            continue

        xyz = he.curve_samples.numpy() if isinstance(he.curve_samples, torch.Tensor) \
            else np.asarray(he.curve_samples)
        uv = he.uv_samples.numpy() if isinstance(he.uv_samples, torch.Tensor) \
            else np.asarray(he.uv_samples)

        he_data.append((uv, xyz))
        uv_list.append(uv.reshape(-1, 2))
        xyz_list.append(xyz.reshape(-1, 3))

    if not uv_list:
        print("  No UV data - cannot fit surface")
        return None, 0

    uv_all = np.concatenate(uv_list)
    xyz_all = np.concatenate(xyz_list)

    if len(uv_all) < 4:
        print("  Too few UV-XYZ pairs")
        return None, 0

    uv_min = uv_all.min(axis=0)
    uv_max = uv_all.max(axis=0)
    uv_range = uv_max - uv_min
    uv_range[uv_range < 1e-12] = 1.0
    uv_all_norm = (uv_all - uv_min) / uv_range

    surface, grid_xyz, u_lin, v_lin = _fit_bspline_surface(
        uv_all_norm, xyz_all, grid_size=grid_size,
    )

    if debug_dir is not None:
        he_data_norm = [(( uv - uv_min) / uv_range, xyz) for uv, xyz in he_data]
        _visualize_loop_samples(loop_idx, he_data_norm, u_lin, v_lin, grid_xyz, debug_dir)

    if surface is None:
        print("  Surface fitting failed")
        return None, 0

    wire_builder = BRepBuilderAPI_MakeWire()
    for he in loop:
        if he.edge is None:
            continue
        orig_edge = he.edge
        if orig_edge.Orientation() == TopAbs_REVERSED:
            wire_builder.Add(topods.Edge(orig_edge.Reversed()))
        else:
            wire_builder.Add(orig_edge)

    if not wire_builder.IsDone():
        print("  Wire construction failed")
        return None, 0

    wire = wire_builder.Wire()
    wf = ShapeFix_Wire()
    wf.Load(wire)
    wf.Perform()
    wire = wf.Wire()

    try:
        face_builder = BRepBuilderAPI_MakeFace(surface, wire)
        if not face_builder.IsDone():
            print(f"  MakeFace failed (error: {face_builder.Error()})")
            try:
                fb = BRepBuilderAPI_MakeFace(wire)
                if fb.IsDone():
                    return fb.Face(), inner_vote
            except Exception:
                pass
            return None, 0
        face = face_builder.Face()
    except Exception as e:
        print(f"  MakeFace exception: {e}")
        return None, 0

    fix_wires(face)
    add_pcurves_to_edges(face)
    fix_wires(face)
    face = fix_face(face)

    if not BRepCheck_Analyzer(face, False).IsValid():
        print(f"  WARNING: face {loop_idx} invalid after construction, returning anyway")

    return face, inner_vote


def create_faces_from_loops_uv(loops, debug_dir=None):
    """Build faces from all loops using UV surface + pcurve approach.

    Returns (outer_faces, inner_loops, []).
    """
    outer_faces = []
    inner_loops = []

    for idx, loop in enumerate(loops):
        # print(f"  Loop {idx}: {len(loop)} half-edges")
        face, inner_vote = _build_face_uv(loop, debug_dir=debug_dir, loop_idx=idx)

        if face is not None:
            if inner_vote < len(loop) // 2:
                outer_faces.append(face)
            else:
                inner_loops.append(loop)
        else:
            print(f"  -> Failed to create face for loop {idx}")

    print(f"Created {len(outer_faces)} outer faces, {len(inner_loops)} inner loops")
    return outer_faces, inner_loops, []


# ---------------------------------------------------------------------------
# Unified pipeline
# ---------------------------------------------------------------------------

def process_and_export_model(bin_file, output_dir, use_uv=False, debug=False):
    """Full pipeline: load graph -> build half-edges -> extract faces -> export.

    Args:
        bin_file:   Path to a .bin DGL graph file.
        output_dir: Directory to write STEP and STL outputs.
        use_uv:     If True, use UV-based surface fitting instead of BRepFill_Filling.
        debug:      If True and use_uv is True, write per-loop debug PNGs.
    """
    stem = os.path.basename(bin_file).replace(".bin", "")
    output_file = os.path.join(output_dir, stem + ".step")

    # 1. Load graph
    graph = load_graph(bin_file)

    if use_uv and "uv" not in graph.edata:
        print(f"ERROR: {bin_file} has no UV data. Re-run VHP_sampling with record_uv=True.")
        return

    # 2. De-normalize (currently identity; kept for future use)
    # graph = inverse_process_graph(graph)

    # 3. Build half-edge structure
    vertices, halfedges, edge_map = build_halfedge_structure(graph)

    if use_uv:
        halfedges = create_halfedges_with_uv(graph, vertices, halfedges, edge_map)
    else:
        halfedges = create_halfedges(graph, vertices, halfedges, edge_map)

    # 4. Build next/prev connectivity
    build_edge_connections(list(vertices.values()))

    # 5. Extract loops
    loops = extract_loops(halfedges)
    if loops is None:
        print("No loops found - skipping.")
        return

    if not is_all_simple_loops(loops):
        print(f"Complex loops detected - skipping: {stem}")
        return

    # 6. Create faces
    if use_uv:
        debug_dir = None
        if debug:
            debug_dir = os.path.join(output_dir, stem + "_debug")
            os.makedirs(debug_dir, exist_ok=True)
        outer_faces, inner_loops, _ = create_faces_from_loops_uv(loops, debug_dir=debug_dir)
    else:
        outer_faces, inner_loops, _ = create_faces_from_loops(loops)

    if not outer_faces:
        print("No faces created - skipping.")
        return

    # 7. Match inner loops to outer faces and rebuild with holes
    matched_pairs, _ = match_inner_wires_to_faces(
        outer_faces, inner_loops, distance_threshold=1e-2,
    )
    new_faces = create_faces_with_inner_loops(outer_faces, matched_pairs)

    # 8. Export STEP
    export_to_step(new_faces, output_file)

    # 9. Sew, validate, and export STL
    sew_tol = 0.01 if use_uv else 0.0001
    sewn = sew_faces(new_faces, sew_tol)

    checker = BRepCheck_Analyzer(sewn)
    if checker.IsValid():
        print("Sewn shape is valid.")
    else:
        print("WARNING: sewn shape is invalid.")

    stl_file = output_file.rsplit(".", 1)[0] + ".stl"
    export_to_stl(sewn, stl_file)
    print(f"Export complete: {stl_file}")


# ---------------------------------------------------------------------------
# Batch / parallel processing
# ---------------------------------------------------------------------------

def _process_file_with_timeout(args, timeout_seconds=120):
    """Process a single file with a SIGALRM-based timeout."""
    bin_file, output_subdir, use_uv, debug = args

    def _timeout_handler(_signum, _frame):
        raise TimeoutError(f"Timed out processing {bin_file}")

    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)

        process_and_export_model(bin_file, output_subdir, use_uv=use_uv, debug=debug)

        signal.alarm(0)
        return bin_file, True, None
    except TimeoutError as e:
        return bin_file, False, str(e)
    except Exception as e:
        return bin_file, False, str(e)
    finally:
        signal.alarm(0)


def process_directory_parallel(input_dir, output_dir, num_processes=8, use_uv=False, debug=False):
    """Process all .bin files in parallel with per-file timeouts."""
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for root, _dirs, files in os.walk(input_dir):
        for file in files:
            if not file.endswith('.bin'):
                continue
            bin_file = os.path.join(root, file)
            rel_path = os.path.relpath(root, input_dir)
            output_subdir = os.path.join(output_dir, rel_path)
            os.makedirs(output_subdir, exist_ok=True)
            tasks.append((bin_file, output_subdir, use_uv, debug))

    random.shuffle(tasks)

    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(_process_file_with_timeout, tasks)

    success  = sum(1 for _, ok, _   in results if ok)
    timeouts = sum(1 for _, ok, msg in results if not ok and msg and "Timed out" in msg)
    errors   = sum(1 for _, ok, msg in results if not ok and (msg is None or "Timed out" not in msg))

    for bin_file, ok, msg in results:
        if not ok:
            label = "Skipped" if "Timed out" in (msg or "") else "Error"
            print(f"{label} {bin_file}: {msg}")

    print(f"Done: {success} succeeded, {timeouts} timed out, {errors} errors")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Graph-to-STEP Converter")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to a single .bin file or a directory of .bin files")
    parser.add_argument("--output", "-o", type=str, default="output_step",
                        help="Output directory (default: output_step)")
    parser.add_argument("--parallel", "-p", type=int, default=0,
                        help="Number of parallel processes (0 = sequential, default: 0)")
    parser.add_argument("--use-uv", action="store_true",
                        help="Use UV-based surface fitting instead of BRepFill_Filling")
    parser.add_argument("--no-debug", action="store_true",
                        help="Disable per-loop debug PNG exports (UV mode only)")
    args = parser.parse_args()

    use_uv = args.use_uv
    debug  = not args.no_debug

    if os.path.isfile(args.input) and args.input.endswith(".bin"):
        os.makedirs(args.output, exist_ok=True)
        process_and_export_model(args.input, args.output, use_uv=use_uv, debug=debug)

    elif os.path.isdir(args.input):
        if args.parallel > 0:
            process_directory_parallel(
                args.input, args.output,
                num_processes=args.parallel, use_uv=use_uv, debug=debug,
            )
        else:
            os.makedirs(args.output, exist_ok=True)
            for root, _dirs, files in os.walk(args.input):
                for f in files:
                    if not f.endswith(".bin"):
                        continue
                    bin_path = os.path.join(root, f)
                    rel = os.path.relpath(root, args.input)
                    out_sub = os.path.join(args.output, rel)
                    os.makedirs(out_sub, exist_ok=True)
                    try:
                        process_and_export_model(bin_path, out_sub, use_uv=use_uv, debug=debug)
                    except Exception as e:
                        print(f"Error processing {bin_path}: {e}")
    else:
        print(f"Error: '{args.input}' is not a valid .bin file or directory.")
