"""
VHP sampling for BrepGPT.
"""

import os
import argparse
import multiprocessing as mp
from functools import partial

import numpy as np
import torch
import dgl
from tqdm import tqdm

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepClass import BRepClass_FaceClassifier
from OCC.Core.BRepTools import BRepTools_WireExplorer, breptools
from OCC.Core.gp import gp_Pnt2d, gp_Vec2d
from OCC.Core.TopAbs import (
    TopAbs_FACE,
    TopAbs_IN,
    TopAbs_ON,
    TopAbs_REVERSED,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopoDS import topods
from occwl.compound import Compound

from brep_utils import get_faces, split_face_by_inner_wires, sew_faces_to_solid


def normalize_vector(vec):
    """Normalize a 2D vector. Returns the original if magnitude is near zero."""
    magnitude = (vec.X() ** 2 + vec.Y() ** 2) ** 0.5
    if magnitude > 1e-6:
        return gp_Vec2d(vec.X() / magnitude, vec.Y() / magnitude)
    return vec


def calculate_distance_to_curve(point, curve, first, last, samples=50):
    """Minimum distance from a 2D point to a parametric curve."""
    min_distance = float("inf")
    for param in np.linspace(first, last, samples):
        curve_point = gp_Pnt2d()
        curve.D0(param, curve_point)
        min_distance = min(min_distance, point.Distance(curve_point))
    return min_distance


def is_in_voronoi_region(current_curve, other_curves, test_point, current_first, current_last):
    """Check whether test_point is closer to current_curve than to any other curve."""
    dist_to_current = calculate_distance_to_curve(
        test_point, current_curve, current_first, current_last
    )
    for curve, first, last in other_curves:
        if calculate_distance_to_curve(test_point, curve, first, last) < dist_to_current:
            return False
    return True


def find_max_voronoi_distance(
    curve, other_curves, pnt, perp_vec, face, current_first, current_last, max_distance
):
    """Binary search along perp_vec to find max distance inside face and Voronoi region."""
    left = 0.0
    right = max_distance
    tolerance = max_distance / 100.0

    while right - left > tolerance:
        mid = (left + right) / 2.0
        uv = pnt.Translated(perp_vec.Multiplied(mid))
        pnt2d = gp_Pnt2d(uv.X(), uv.Y())

        classifier = BRepClass_FaceClassifier(face, pnt2d, 1e-7)
        state = classifier.State()
        in_face = (state == TopAbs_IN) or (state == TopAbs_ON)

        in_voronoi = is_in_voronoi_region(
            curve, other_curves, pnt2d, current_first, current_last
        )

        if in_face and in_voronoi:
            left = mid
        else:
            right = mid

    return left


def is_point_inside_face(point2d, face):
    classifier = BRepClass_FaceClassifier()
    classifier.Perform(face, point2d, 1e-7)
    return classifier.State() == TopAbs_IN


def _get_normal_at_param(curve, param, edge_orientation, face_orientation):
    """Compute inward-pointing normal of a curve at a given parameter."""
    pnt = gp_Pnt2d()
    vec = gp_Vec2d()
    curve.D1(param, pnt, vec)

    perp_vec = gp_Vec2d(-vec.Y(), vec.X())
    if edge_orientation == TopAbs_REVERSED:
        perp_vec.Reverse()
    if face_orientation == TopAbs_REVERSED:
        perp_vec.Reverse()

    return normalize_vector(perp_vec)


def _average_normals(n1, n2, weight1=0.99):
    """Weighted normalized average of two 2D vectors."""
    return normalize_vector(
        gp_Vec2d(
            (n1.X() * weight1 + n2.X()) / 2.0,
            (n1.Y() * weight1 + n2.Y()) / 2.0,
        )
    )

def calculate_face_sampling_info(face, edge_samples):
    """
    Pre-compute per-sample perpendicular direction and Voronoi neighbor curves
    for every edge on every wire of the face.
    """
    umin, umax, vmin, vmax = breptools.UVBounds(face)
    face_orientation = face.Orientation()

    all_sampling_info = []

    wire_exp = TopExp_Explorer(face, TopAbs_WIRE)
    while wire_exp.More():
        wire = topods.Wire(wire_exp.Current())

        wire_edges_info = []
        edge_exp = BRepTools_WireExplorer(wire)
        while edge_exp.More():
            edge = edge_exp.Current()
            curve, first, last = BRep_Tool.CurveOnSurface(edge, face)
            wire_edges_info.append((edge, curve, first, last))
            edge_exp.Next()

        num_edges = len(wire_edges_info)

        for i, (edge, curve, first, last) in enumerate(wire_edges_info):
            edge_orientation = edge.Orientation()
            params = np.linspace(first, last, edge_samples)

            prev_edge_info = wire_edges_info[i - 1]
            next_edge_info = wire_edges_info[(i + 1) % num_edges]
            if edge_orientation == TopAbs_REVERSED:
                prev_edge_info, next_edge_info = next_edge_info, prev_edge_info

            for j, param in enumerate(params):
                perp_vec = _get_normal_at_param(curve, param, edge_orientation, face_orientation)

                other_curves = []
                for e, c, f, l in wire_edges_info:
                    if e == edge:
                        continue
                    if j == 0 and e == prev_edge_info[0]:
                        continue
                    if j == len(params) - 1 and e == next_edge_info[0]:
                        continue
                    other_curves.append((c, f, l))

                all_sampling_info.append(
                    {
                        "edge": edge,
                        "perp_vec": perp_vec,
                        "curve": curve,
                        "first": first,
                        "last": last,
                        "other_curves": other_curves,
                    }
                )

        wire_exp.Next()

    return all_sampling_info

def sample_face_voronoi_g(shape, edge_samples=8, normal_samples=16, wire_info_list=None, record_uv=False):
    """
    Sample the B-Rep shape and build a DGL directed graph.

    Node features: 3D vertex coordinates
    Edge features: sampled surface points along each edge + Voronoi normals

    Args:
        record_uv: If True, record UV coordinates for each sampled point
    """

    vertex_map = {}
    vertex_coords = {}
    vertex_idx = 0

    vertex_explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    while vertex_explorer.More():
        vertex = topods.Vertex(vertex_explorer.Current())
        point = BRep_Tool.Pnt(vertex)
        coords = (round(point.X(), 6), round(point.Y(), 6), round(point.Z(), 6))
        if coords not in vertex_map:
            vertex_map[coords] = vertex_idx
            vertex_coords[vertex_idx] = [point.X(), point.Y(), point.Z()]
            vertex_idx += 1
        vertex_explorer.Next()

    edges = []
    edge_samples_dict = {}
    edge_inner_outer_dict = {}
    edge_next = {}
    edge_uv_dict = {} if record_uv else None  # Store UV coordinates if requested

    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)

    while face_explorer.More():
        face = topods.Face(face_explorer.Current())
        surface = BRep_Tool.Surface(face)

        umin, umax, vmin, vmax = breptools.UVBounds(face)
        uv_diagonal = ((umax - umin) ** 2 + (vmax - vmin) ** 2) ** 0.5
        adaptive_step_size = uv_diagonal / normal_samples / 2

        all_sampling_info = calculate_face_sampling_info(face, edge_samples)

        sample_id = 0

        wire_exp = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_exp.More():
            wire = topods.Wire(wire_exp.Current())

            current_wire_vertices = _collect_wire_vertex_coords(wire)
            wire_type = _match_wire_type(current_wire_vertices, wire_info_list)

            all_edge_curves = []
            edge_exp_collect = BRepTools_WireExplorer(wire)
            while edge_exp_collect.More():
                edge = edge_exp_collect.Current()
                curve, first, last = BRep_Tool.CurveOnSurface(edge, face)
                all_edge_curves.append((curve, first, last))
                edge_exp_collect.Next()

            wire_edges = []
            edge_exp = BRepTools_WireExplorer(wire)

            while edge_exp.More():
                edge = edge_exp.Current()
                edge_orientation = edge.Orientation()

                v1 = topods.Vertex(topexp.FirstVertex(edge))
                v2 = topods.Vertex(topexp.LastVertex(edge))
                p1, p2 = BRep_Tool.Pnt(v1), BRep_Tool.Pnt(v2)
                coords1 = (round(p1.X(), 6), round(p1.Y(), 6), round(p1.Z(), 6))
                coords2 = (round(p2.X(), 6), round(p2.Y(), 6), round(p2.Z(), 6))
                src_idx = vertex_map[coords1]
                dst_idx = vertex_map[coords2]

                edge_3d_length = p1.Distance(p2)
                if edge_3d_length < 1e-6:
                    # print('Skip degenerate edge (near-zero 3D length)')
                    sample_id += edge_samples
                    edge_exp.Next()
                    continue

                curve, first, last = BRep_Tool.CurveOnSurface(edge, face)
                params = np.linspace(first, last, edge_samples)

                edge_points = []
                edge_uv_points = [] if record_uv else None  # Store UV for this edge
                for param in params:
                    pnt = gp_Pnt2d()
                    vec = gp_Vec2d()
                    curve.D1(param, pnt, vec)

                    perp_vec = all_sampling_info[sample_id]["perp_vec"]
                    other_curves = all_sampling_info[sample_id]["other_curves"]

                    max_distance = find_max_voronoi_distance(
                        curve, other_curves, pnt, perp_vec, face,
                        first, last, adaptive_step_size * normal_samples,
                    )
                    sample_id += 1

                    # actual_step = max_distance / (normal_samples + 1)
                    actual_step = max_distance / normal_samples

                    normal_points = []
                    normal_uv_points = [] if record_uv else None  # Store UV for this sample
                    for i in range(normal_samples + 1):
                        distance = i * actual_step
                        uv = pnt.Translated(perp_vec.Multiplied(distance))
                        try:
                            pnt_3d = surface.Value(uv.X(), uv.Y())
                            normal_points.append([pnt_3d.X(), pnt_3d.Y(), pnt_3d.Z()])
                            if record_uv:
                                normal_uv_points.append([uv.X(), uv.Y()])
                        except Exception as e:
                            print(e)
                            continue

                    edge_points.append(normal_points)
                    if record_uv:
                        edge_uv_points.append(normal_uv_points)

                if edge_orientation == TopAbs_REVERSED:
                    src_idx, dst_idx = dst_idx, src_idx
                    edge_points = edge_points[::-1]
                    if record_uv:
                        edge_uv_points = edge_uv_points[::-1]

                if (src_idx, dst_idx) in edges and coords1 != coords2:
                    print("need reverse", face.Orientation(), wire.Orientation(), edge_orientation)
                    return None

                edges.append((src_idx, dst_idx))
                edge_samples_dict[(src_idx, dst_idx)] = edge_points
                edge_inner_outer_dict[(src_idx, dst_idx)] = (wire_type == "inner")
                if record_uv:
                    edge_uv_dict[(src_idx, dst_idx)] = edge_uv_points

                wire_edges.append((src_idx, dst_idx))

                edge_exp.Next()

            for i in range(len(wire_edges)):
                edge_next[wire_edges[i]] = wire_edges[(i + 1) % len(wire_edges)]

            wire_exp.Next()
        face_explorer.Next()

    return _build_dgl_graph(edges, vertex_coords, edge_samples_dict, edge_inner_outer_dict, edge_next, edge_uv_dict)


def _build_dgl_graph(edges, vertex_coords, edge_samples_dict, edge_inner_outer_dict, edge_next, edge_uv_dict=None):
    """Convert collected data into a DGL graph with tensor features."""
    g = dgl.graph(edges)

    pos_array = np.array([vertex_coords[i] for i in range(len(vertex_coords))])
    points_array = np.array([edge_samples_dict[e] for e in edges])
    inner_outer_array = np.array([edge_inner_outer_dict[e] for e in edges])

    next_indices = []
    next_half_edge_points = []

    for i, edge in enumerate(edges):
        if edge in edge_next:
            next_edge = edge_next[edge]
            next_idx = edges.index(next_edge)
            next_indices.append(next_idx)

            next_edge_points = edge_samples_dict[next_edge]
            half_idx = 4
            next_half_points = np.array(next_edge_points)[:half_idx, 0]
            next_half_edge_points.append(next_half_points)
        else:
            print(f"Warning: no next edge for {edge}")
            next_indices.append(i)
            next_half_edge_points.append(
                np.zeros((0, points_array[i].shape[1], points_array[i].shape[2]))
            )

    g.ndata["x"] = torch.from_numpy(pos_array).float()
    g.edata["x"] = torch.from_numpy(points_array).float()
    g.edata["next_idx"] = torch.tensor(next_indices, dtype=torch.long)
    g.edata["next_half_edge"] = torch.from_numpy(
        np.array(next_half_edge_points)
    ).float()
    g.edata["edge_inner_outer"] = torch.from_numpy(inner_outer_array).bool()

    # Add UV coordinates if available
    if edge_uv_dict is not None:
        uv_array = np.array([edge_uv_dict[e] for e in edges])
        g.edata["uv"] = torch.from_numpy(uv_array).float()

    return g


def _collect_wire_vertex_coords(wire):
    """Return ordered list of vertex coordinate tuples for a wire."""
    vertices = []
    edge_exp = BRepTools_WireExplorer(wire)
    while edge_exp.More():
        edge = edge_exp.Current()
        v = topods.Vertex(topexp.FirstVertex(edge))
        p = BRep_Tool.Pnt(v)
        vertices.append((round(p.X(), 6), round(p.Y(), 6), round(p.Z(), 6)))
        edge_exp.Next()
    return vertices


def _match_wire_type(wire_vertices, wire_info_list):
    """Match wire vertices against wire_info_list to determine 'inner'/'outer' type."""
    for wire_info in wire_info_list:
        if len(wire_info["vertices"]) != len(wire_vertices):
            continue
        if all(
            any(np.allclose(v1, v2) for v2 in wire_info["vertices"])
            for v1 in wire_vertices
        ):
            return wire_info["type"]
    return None


def create_wire_info_list(faces_info):
    """Build wire_info_list from a list of face dicts with 'face' and 'type' keys."""
    wire_info_list = []
    for face_info in faces_info:
        face = face_info["face"]
        face_type = face_info["type"]
        wire_exp = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_exp.More():
            wire = topods.Wire(wire_exp.Current())
            vertices = _collect_wire_vertex_coords(wire)
            wire_info_list.append({"vertices": vertices, "type": face_type})
            wire_exp.Next()
    return wire_info_list


def find_step_files(input_dir, max_files=None):
    """Recursively find all .step/.stp files under input_dir."""
    step_files = []
    for root, _dirs, files in tqdm(os.walk(input_dir), desc="Finding STEP files"):
        for file in files:
            if file.lower().endswith((".step", ".stp")):
                step_files.append(os.path.join(root, file))
                if max_files and len(step_files) >= max_files:
                    return step_files
    return step_files


# ---------------------------------------------------------------------------
# Single-file processing
# ---------------------------------------------------------------------------

def process_step_file(step_file_path, input_dir, output_dir, edge_samples, normal_samples, max_vertices):
    """Load a STEP file, sample it, and save the resulting DGL graph.

    Returns a status string: 'ok' on success, or a short skip reason.
    """
    shape = Compound.load_step_with_attributes(str(step_file_path))[0]

    try:
        vertex_count = shape.num_vertices()
    except Exception:
        return "skip:vertex_count_error"
    if vertex_count > max_vertices:
        return f"skip:too_many_vertices({vertex_count})"

    try:
        faces = get_faces(shape.topods_shape())
    except Exception:
        return "skip:get_faces_error"

    new_faces = []
    for face in faces:
        split_faces = split_face_by_inner_wires(face)
        if split_faces is None:
            return "skip:split_face_error"
        new_faces.extend(split_faces)

    wire_info_list = create_wire_info_list(new_faces)

    solid = sew_faces_to_solid(new_faces)
    try:
        is_valid = BRepCheck_Analyzer(solid).IsValid()
    except Exception as e:
        return f"skip:brep_check_error"

    if not is_valid:
        return "skip:brep_invalid"

    g = sample_face_voronoi_g(
        solid, edge_samples=edge_samples, normal_samples=normal_samples,
        wire_info_list=wire_info_list,
        record_uv=True
    )
    if g is None:
        return "skip:graph_none"

    rel_path = os.path.relpath(step_file_path, input_dir)
    output_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".bin")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dgl.save_graphs(output_path, g)
    return "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Voronoi-based face sampling for B-Rep CAD models"
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Input STEP directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output .bin directory")
    parser.add_argument("--num-workers", type=int, default=12, help="Number of parallel workers")
    parser.add_argument("--max-files", type=int, default=None, help="Max number of STEP files to process")
    parser.add_argument("--max-vertices", type=int, default=256, help="Skip shapes with more vertices than this")
    parser.add_argument("--edge-samples", type=int, default=8, help="Samples per edge")
    parser.add_argument("--normal-samples", type=int, default=3, help="Samples along Voronoi normal direction")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    step_files = find_step_files(args.input_dir, max_files=args.max_files)
    total = len(step_files)
    print(f"Found {total} STEP file(s)")
    print(
        f"Config: workers={args.num_workers}, edge_samples={args.edge_samples}, "
        f"normal_samples={args.normal_samples}, max_vertices={args.max_vertices}"
    )

    process_func = partial(
        process_step_file,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        edge_samples=args.edge_samples,
        normal_samples=args.normal_samples,
        max_vertices=args.max_vertices,
    )

    from collections import Counter
    counters = Counter()

    with mp.Pool(args.num_workers) as pool:
        for status in tqdm(
            pool.imap_unordered(process_func, step_files),
            total=total,
            desc="Processing",
            unit="file",
        ):
            counters[status] += 1

    ok = counters.pop("ok", 0)
    skipped = sum(counters.values())
    print(f"\nDone. Succeeded: {ok}/{total}, Skipped: {skipped}/{total}")
    if counters:
        for reason, count in sorted(counters.items()):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()

