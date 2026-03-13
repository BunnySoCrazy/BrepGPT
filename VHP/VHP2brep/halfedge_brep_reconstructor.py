"""
Half-Edge BRep Processing

Core data structures and OCC-based geometry utilities shared by all
VHP-to-BRep conversion pipelines.
"""

import math
import random
from collections import defaultdict

import numpy as np
import torch
import networkx as nx

from OCC.Core.gp import gp_Vec, gp_Pnt, gp_Trsf
from OCC.Core.BRep import BRep_Tool, BRep_Builder
from OCC.Core.BRepTools import breptools
from OCC.Core.TopAbs import TopAbs_REVERSED, TopAbs_EDGE, TopAbs_WIRE, TopAbs_SHAPE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import (
    topods, topods_Face, topods_Edge, topods_Wire,
    TopoDS_Compound, TopoDS_Shape,
)
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_C0, GeomAbs_C2
from OCC.Core.GeomConvert import geomconvert_CurveToBSplineCurve
from OCC.Core.GeomFill import GeomFill_BSplineCurves, GeomFill_StretchStyle
from OCC.Core.Precision import precision_Confusion
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_Transform,
)
from OCC.Core.BRepFill import BRepFill_Filling
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepBndLib import brepbndlib, brepbndlib_Add
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GProp import GProp_GProps
from OCC.Core.ShapeFix import ShapeFix_Wire, ShapeFix_Face, ShapeFix_Edge
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.GCPnts import GCPnts_QuasiUniformAbscissa
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class HalfEdge:
    """One directed half of an edge in the half-edge data structure."""

    def __init__(self):
        self.vertex = None              # Start vertex
        self.face = None                # Associated face
        self.next = None                # Next half-edge in the loop
        self.prev = None                # Previous half-edge in the loop
        self.twin = None                # Opposite-direction twin half-edge
        self.edge = None                # Underlying TopoDS_Edge
        self.curve_samples = None       # [num_curve_samples, num_normal_samples, 3]
        self.next_curve_samples = None  # [num_normal_samples, 3]
        self.edge_inner_outer = None    # Scalar flag: True = inner, False = outer


class Vertex:
    """A vertex shared by one or more half-edges."""

    def __init__(self, point, shape=None):
        self.point = point              # (x, y, z) coordinate tuple
        self.in_halfedges = []          # Half-edges whose destination is this vertex
        self.out_halfedges = []         # Half-edges originating from this vertex
        self.normal = None              # Vertex normal
        self.shape = None               # Optional TopoDS_Vertex


class Face:
    """A face bounded by one or more loops of half-edges."""

    def __init__(self):
        self.halfedge = None            # One half-edge on this face
        self.loops = []                 # List of boundary loops
        self.shape = None               # Optional TopoDS_Face


# ---------------------------------------------------------------------------
# Loop utilities
# ---------------------------------------------------------------------------

def extract_loops(halfedges):
    """Walk next pointers to extract all closed loops.

    Returns None if fewer than 3 loops are found.
    """
    visited = set()
    loops = []

    for start_he in halfedges:
        if start_he in visited or not start_he.next:
            continue

        loop = []
        current = start_he
        is_valid = True

        while True:
            if current in visited:
                is_valid = False
                break
            visited.add(current)
            loop.append(current)
            current = current.next
            if not current:
                is_valid = False
                break
            if current == start_he:
                break

        if is_valid and loop:
            loops.append(loop)

    if len(loops) <= 2:
        return None
    return loops


def is_all_simple_loops(loops):
    """Return True if every loop is a simple cycle (single cycle basis)."""
    if not loops:
        return True
    for loop in loops:
        G = nx.Graph()
        for he in loop:
            G.add_edge(he.vertex, he.next.vertex)
        if len(list(nx.cycle_basis(G.to_undirected()))) > 1:
            return False
    return True


# ---------------------------------------------------------------------------
# Face building helpers
# ---------------------------------------------------------------------------

def make_coons(edges):
    """Create a Coons-patch face from 2, 3, or 4 BSpline edges."""
    curves = []
    for edge in edges:
        curve, first, last = BRep_Tool.Curve(edge)
        curves.append(geomconvert_CurveToBSplineCurve(curve))

    if len(curves) == 4:
        srf = GeomFill_BSplineCurves(curves[0], curves[1], curves[2], curves[3], GeomFill_StretchStyle)
    elif len(curves) == 3:
        srf = GeomFill_BSplineCurves(curves[0], curves[1], curves[2], GeomFill_StretchStyle)
    elif len(curves) == 2:
        srf = GeomFill_BSplineCurves(curves[0], curves[1], GeomFill_StretchStyle)
    else:
        raise ValueError("Expected 2, 3, or 4 curves")

    face_builder = BRepBuilderAPI_MakeFace(srf.Surface(), precision_Confusion())
    if face_builder.IsDone():
        return face_builder.Face()
    raise RuntimeError("Failed to create Coons face")


def _face_is_sane(fill_face, coons_face, wire, bbox_ratio_max=1.5, area_ratio_max=3.0, verbose=False):
    """Check whether fill_face is geometrically reasonable vs the Coons face / wire bbox.

    Returns True if the face looks valid, False if degenerate.
    """
    # Bounding-box inflation check (vs wire)
    wire_box = Bnd_Box()
    brepbndlib.AddOptimal(wire, wire_box)
    xmin0, ymin0, zmin0, xmax0, ymax0, zmax0 = wire_box.Get()
    wire_diag = ((xmax0 - xmin0)**2 + (ymax0 - ymin0)**2 + (zmax0 - zmin0)**2) ** 0.5

    face_box = Bnd_Box()
    brepbndlib.AddOptimal(fill_face, face_box)
    xmin1, ymin1, zmin1, xmax1, ymax1, zmax1 = face_box.Get()
    face_diag = ((xmax1 - xmin1)**2 + (ymax1 - ymin1)**2 + (zmax1 - zmin1)**2) ** 0.5

    if wire_diag > 1e-12:
        diag_ratio = face_diag / wire_diag
        if diag_ratio > bbox_ratio_max:
            if verbose:
                print(f"  REJECT fill: bbox diagonal ratio {diag_ratio:.2f} > {bbox_ratio_max}")
            return False

        wire_center = [(xmin0+xmax0)/2, (ymin0+ymax0)/2, (zmin0+zmax0)/2]
        face_center = [(xmin1+xmax1)/2, (ymin1+ymax1)/2, (zmin1+zmax1)/2]
        drift = sum((a-b)**2 for a, b in zip(wire_center, face_center)) ** 0.5
        if drift > wire_diag * 0.5:
            if verbose:
                print(f"  REJECT fill: center drift {drift:.4f} > 0.5 * wire_diag {wire_diag:.4f}")
            return False

    # Area ratio check (vs Coons)
    if coons_face is not None:
        props_coons = GProp_GProps()
        brepgprop.SurfaceProperties(coons_face, props_coons)
        area_coons = props_coons.Mass()

        props_fill = GProp_GProps()
        brepgprop.SurfaceProperties(fill_face, props_fill)
        area_fill = props_fill.Mass()

        if area_coons > 1e-12:
            a_ratio = area_fill / area_coons
            if a_ratio > area_ratio_max:
                if verbose:
                    print(f"  REJECT fill: area ratio {a_ratio:.2f} > {area_ratio_max}")
                return False

    return True

def _fit_bspline_edge(points_3d):
    """
    Fit a BSpline curve through a list of (x,y,z) tuples, return a TopoDS_Edge.
    Uses interpolation (not approximation) so the curve passes through all points.
    """
    from OCC.Core.TColgp import TColgp_Array1OfPnt
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge

    n = len(points_3d)
    arr = TColgp_Array1OfPnt(1, n)
    for i, (x, y, z) in enumerate(points_3d):
        arr.SetValue(i + 1, gp_Pnt(x, y, z))

    fitter = GeomAPI_PointsToBSpline(arr, 0, 8, GeomAbs_C2, 1e-3)
    if not fitter.IsDone():
        return None

    edge = BRepBuilderAPI_MakeEdge(fitter.Curve()).Edge()
    return edge

def build_face_with_retry(
    fixed_wire,
    edges,
    filling_param_sets=None,
    verbose=False,
    edge_target_segments=80,
    face_idx=0,
):
    """
    Build a face from a wire with interior edge constraints.

    Fallback chain:
      1) BRepFill_Filling with boundary + interior edge constraints
      2) Coons patch (if 2-4 edges)
      3) BRepFill_Filling with boundary edges only (no interior constraints)
      4) Planar face fallback

    For each edge's pts_tensor [num_curve_samples, num_normal_samples, 3],
    we pick slices near the middle curve sample and fit BSpline edge constraints.
    """
    if filling_param_sets is None:
        filling_param_sets = [
            (3, 3, 2, False, 5e-9, 5e-9, 0.1, 0.1, 9, 9),
            (3, 3, 2, False, 1e-8, 1e-7, 0.1, 0.1, 9, 9),
        ]

    if not hasattr(build_face_with_retry, "_stats"):
        build_face_with_retry._stats = {
            "total_calls": 0,
            "per_param": {},
            "coons_only": 0,
            "unconstrained_fill": 0,
            "fallback_planar": 0,
            "failed": 0,
            "param_labels": [str(p) for p in filling_param_sets],
        }

    stats = build_face_with_retry._stats
    for i in range(len(filling_param_sets)):
        if i not in stats["per_param"]:
            stats["per_param"][i] = {
                "build_ok": 0, "sane": 0, "insane": 0, "exception": 0,
            }
            if i >= len(stats["param_labels"]):
                stats["param_labels"].append(str(filling_param_sets[i]))
    stats["total_calls"] += 1

    # -- Extract wire edges and interior curve points (original scale) --
    wire_edges = []
    all_interior_curves = []
    inner_vote = 0

    exp = TopExp_Explorer(fixed_wire, TopAbs_EDGE)
    edge_idx = 0
    while exp.More():
        wire_edges.append(topods_Edge(exp.Current()))
        pts_tensor = edges[edge_idx]["points"]
        inner_vote += edges[edge_idx]["edge_inner_outer"]

        if isinstance(pts_tensor, torch.Tensor) and pts_tensor.dim() == 3:
            mid = pts_tensor.shape[0] // 2
            for offset in (0, -1, 1):
                row = pts_tensor[mid + offset, 1:-1, :]
                pts = [(float(p[0]), float(p[1]), float(p[2])) for p in row]
                if len(pts) >= 2:
                    all_interior_curves.append(pts)

        edge_idx += 1
        exp.Next()

    # -- Normalize to unit bounding box --
    bbox = Bnd_Box()
    brepbndlib_Add(fixed_wire, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    diag = math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
    scale = max(diag, 1e-12)

    trsf_fwd = gp_Trsf()
    trsf_fwd.SetTranslation(gp_Vec(-cx, -cy, -cz))
    trsf_scl = gp_Trsf()
    trsf_scl.SetScaleFactor(1.0 / scale)
    trsf_norm = gp_Trsf()
    trsf_norm.Multiply(trsf_scl)
    trsf_norm.Multiply(trsf_fwd)

    trsf_inv_scl = gp_Trsf()
    trsf_inv_scl.SetScaleFactor(scale)
    trsf_inv_tr = gp_Trsf()
    trsf_inv_tr.SetTranslation(gp_Vec(cx, cy, cz))
    trsf_inv = gp_Trsf()
    trsf_inv.Multiply(trsf_inv_tr)
    trsf_inv.Multiply(trsf_inv_scl)

    norm_wire = topods_Wire(
        BRepBuilderAPI_Transform(fixed_wire, trsf_norm, True).Shape()
    )

    norm_wire_edges = []
    exp2 = TopExp_Explorer(norm_wire, TopAbs_EDGE)
    while exp2.More():
        norm_wire_edges.append(topods_Edge(exp2.Current()))
        exp2.Next()

    # -- Normalize interior curves and fit BSpline edges --
    norm_constraint_edges = []
    for curve_pts in all_interior_curves:
        norm_pts = [
            ((x - cx) / scale, (y - cy) / scale, (z - cz) / scale)
            for x, y, z in curve_pts
        ]
        edge = _fit_bspline_edge(norm_pts)
        if edge is not None:
            norm_constraint_edges.append(edge)
    
    # Representative points for interface compatibility
    points_list_all = [
        curve_pts[len(curve_pts) // 2] for curve_pts in all_interior_curves
    ]

    # -- Precompute Coons face (available for 2-4 edges) --
    norm_coons_face = None
    if 2 <= len(norm_wire_edges) <= 4:
        try:
            norm_coons_face = make_coons(norm_wire_edges)
        except Exception:
            pass

    # -- Helper: run filling and validate --
    def _try_filling(params_list, constraint_edges):
        for idx, params in enumerate(params_list):
            try:
                fill = BRepFill_Filling(*params)
                for e in norm_wire_edges:
                    fill.Add(e, GeomAbs_C0)
                for ce in constraint_edges:
                    fill.Add(ce, GeomAbs_C0, False)
                fill.Build()
            except Exception as exc:
                if verbose:
                    print(f"[fill] param[{idx}] exception: {type(exc).__name__}: {exc}")
                continue

            if not fill.IsDone():
                continue

            candidate = fill.Face()
            try:
                if _face_is_sane(candidate, norm_coons_face, norm_wire, verbose=verbose):
                    return candidate
            except TypeError:
                pass

        return None

    # -- Helper: inverse-transform a normalized face --
    def _to_original(norm_face):
        return topods_Face(
            BRepBuilderAPI_Transform(norm_face, trsf_inv, True).Shape()
        )

    # -- Fallback chain --

    # 1) Filling with interior edge constraints
    result = _try_filling(filling_param_sets, norm_constraint_edges)
    if result is not None:
        return _to_original(result), inner_vote, points_list_all

    # 2) Coons patch
    if norm_coons_face is not None:
        stats["coons_only"] += 1
        print(f"[coons] Using Coons patch for face {face_idx}")
        return _to_original(norm_coons_face), inner_vote, points_list_all

    # 3) Filling without interior constraints (boundary only)
    result = _try_filling(filling_param_sets, [])
    if result is not None:
        stats["unconstrained_fill"] += 1
        return _to_original(result), inner_vote, points_list_all

    # 4) Planar face fallback
    try:
        fallback = BRepBuilderAPI_MakeFace(fixed_wire)
        if fallback.IsDone():
            stats["fallback_planar"] += 1
            return fallback.Face(), inner_vote, points_list_all
    except Exception:
        pass

    stats["failed"] += 1
    return None, 0, []


# ---------------------------------------------------------------------------
# Face assembly from loops
# ---------------------------------------------------------------------------

def create_faces_from_loops(loops):
    """Build TopoDS_Face objects from loops of half-edges.

    Returns (outer_faces, inner_loops, all_points_list).
    """
    builder = BRep_Builder()
    outer_faces = []
    inner_loops = []
    failed_wires = []
    all_points_list = []
    idx = 0
    for loop in loops:
        wire_builder = BRepBuilderAPI_MakeWire()
        edges = []

        for he in loop:
            if not he.edge:
                continue
            if he.edge.Orientation() == TopAbs_REVERSED:
                wire_builder.Add(he.edge.Reversed())
                edges.append({
                    'edge': he.edge.Reversed(),
                    'points': he.curve_samples,
                    'start_point': he.vertex,
                    'end_point': he.next.vertex,
                    'edge_inner_outer': he.edge_inner_outer,
                })
            else:
                wire_builder.Add(he.edge)
                edges.append({
                    'edge': he.edge,
                    'points': he.curve_samples,
                    'start_point': he.vertex,
                    'end_point': he.next.vertex,
                    'edge_inner_outer': he.edge_inner_outer,
                })

        if not wire_builder.IsDone():
            continue

        wire = wire_builder.Wire()
        wire_fixer = ShapeFix_Wire()
        wire_fixer.Load(wire)
        wire_fixer.Perform()
        fixed_wire = wire_fixer.Wire()

        face, inner_vote, points_list = build_face_with_retry(fixed_wire, edges, verbose=True, face_idx=idx)
        idx+=1
        for edge_data in edges:
            inner_vote += edge_data['edge_inner_outer']
            points_list.append(edge_data['points'])

        if face is not None:
            if inner_vote < len(loop) // 2:
                outer_faces.append(face)
            else:
                inner_loops.append(loop)
            all_points_list.extend(points_list)
        else:
            failed_wires.append(fixed_wire)

    print(f"Created {len(outer_faces)} outer faces and {len(inner_loops)} inner loops")

    if failed_wires:
        print(f"{len(failed_wires)} wire(s) failed to produce a face")
        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)
        for w in failed_wires:
            builder.Add(compound, w)

        writer = STEPControl_Writer()
        writer.Transfer(compound, STEPControl_AsIs)
        status = writer.Write("output/failed_wires.step")
        if status == IFSelect_RetDone:
            print("Failed wires exported to output/failed_wires.step")

    return outer_faces, inner_loops, all_points_list


# ---------------------------------------------------------------------------
# Inner-loop matching
# ---------------------------------------------------------------------------

def match_inner_wires_to_faces(outer_faces, inner_loops, distance_threshold=1.0):
    """Match each inner loop to the closest outer face by average point distance."""
    matched_pairs = []
    unmatched = []

    for loop in inner_loops:
        all_points = []
        for he in loop:
            if hasattr(he, 'curve_samples') and he.curve_samples is not None:
                all_points.append(he.curve_samples[:, 0, :])

        if not all_points:
            unmatched.append(loop)
            continue

        all_points = torch.cat(all_points, dim=0)
        min_avg_dist = float('inf')
        best_idx = -1

        for face_idx, face in enumerate(outer_faces):
            total_dist = 0.0
            for pt in all_points:
                vtx = BRepBuilderAPI_MakeVertex(
                    gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
                ).Vertex()
                calc = BRepExtrema_DistShapeShape(vtx, face)
                if calc.IsDone():
                    total_dist += calc.Value()

            avg_dist = total_dist / len(all_points)
            if avg_dist < min_avg_dist:
                min_avg_dist = avg_dist
                best_idx = face_idx

        if min_avg_dist <= distance_threshold:
            matched_pairs.append((best_idx, loop))
        else:
            unmatched.append(loop)

    print(f"Matched {len(matched_pairs)} inner loop(s), {len(unmatched)} unmatched")
    return matched_pairs, unmatched


def create_wire_from_loop(loop):
    """Create a TopoDS_Wire from a list of half-edges, respecting orientation."""
    wire_builder = BRepBuilderAPI_MakeWire()
    for he in loop:
        if he.edge:
            if he.edge.Orientation() == TopAbs_REVERSED:
                wire_builder.Add(he.edge.Reversed())
            else:
                wire_builder.Add(he.edge)

    if wire_builder.IsDone():
        return wire_builder.Wire()
    print("WARNING: Failed to create wire from inner loop")
    return None


def create_faces_with_inner_loops(outer_faces, matched_pairs):
    """Rebuild outer faces with their matched inner loops (holes)."""
    face_to_loops = defaultdict(list)
    for face_idx, inner_loop in matched_pairs:
        face_to_loops[face_idx].append(inner_loop)

    new_faces = []
    for idx, face in enumerate(outer_faces):
        if idx not in face_to_loops:
            new_faces.append(face)
            continue

        surface = BRep_Tool.Surface(face)
        outer_wire = breptools.OuterWire(face)
        if outer_wire.IsNull():
            print(f"WARNING: Face {idx} has invalid outer wire")
            new_faces.append(face)
            continue

        face_maker = BRepBuilderAPI_MakeFace(surface, outer_wire)

        for inner_loop in face_to_loops[idx]:
            inner_wire = create_wire_from_loop(inner_loop)
            if not inner_wire:
                continue
            wire_fixer = ShapeFix_Wire()
            wire_fixer.Load(inner_wire)
            wire_fixer.Perform()
            inner_wire = wire_fixer.Wire()
            if not inner_wire.Closed():
                print("WARNING: Inner wire is not closed, skipping")
                continue
            face_maker.Add(inner_wire)

        if not face_maker.IsDone():
            print(f"WARNING: Failed to add inner loops to face {idx}")
            new_faces.append(face)
            continue

        new_face = face_maker.Shape()
        fix_wires(new_face)
        add_pcurves_to_edges(new_face)
        fix_wires(new_face)
        new_face = fix_face(new_face)

        if BRepCheck_Analyzer(new_face, False).IsValid():
            new_faces.append(new_face)
            print(f"Successfully created face {idx} with inner loops")
        else:
            new_faces.append(face)
            print(f"WARNING: Face {idx} invalid after adding inner loops; using original")
            continue

        exp = TopExp_Explorer(new_face, TopAbs_WIRE)
        wire_count = 0
        while exp.More():
            wire_count += 1
            exp.Next()
        print(f"  Wire count: {wire_count}")

    return new_faces


# ---------------------------------------------------------------------------
# Shape fix utilities
# ---------------------------------------------------------------------------

def add_pcurves_to_edges(face):
    """Add parametric curves to all edges of a face."""
    edge_fixer = ShapeFix_Edge()
    for wire in TopologyExplorer(face).wires():
        for edge in WireExplorer(wire).ordered_edges():
            edge_fixer.FixAddPCurve(edge, face, False, 0.01)


def fix_wires(face, debug=False):
    """Validate and repair all wires of a face."""
    for wire in TopologyExplorer(face).wires():
        if debug:
            checker = ShapeAnalysis_Wire(wire, face, 0.01)
            print(f"  CheckOrder3d   : {checker.CheckOrder()}")
            print(f"  CheckGaps3d    : {checker.CheckGaps3d()}")
            print(f"  CheckClosed    : {checker.CheckClosed()}")
            print(f"  CheckConnected : {checker.CheckConnected()}")
        fixer = ShapeFix_Wire(wire, face, 0.0001)
        assert fixer.IsReady()
        fixer.Perform()


def fix_face(face):
    """Apply ShapeFix_Face to repair and orient a face."""
    fixer = ShapeFix_Face(face)
    fixer.SetPrecision(0.01)
    fixer.SetMaxTolerance(0.1)
    fixer.Perform()
    fixer.FixOrientation()
    return fixer.Face()


# ---------------------------------------------------------------------------
# Export utilities
# ---------------------------------------------------------------------------

def sew_faces(faces, tolerance=1e-6):
    """Sew a list of faces into a single shape."""
    sewing = BRepBuilderAPI_Sewing(tolerance)
    for face in faces:
        sewing.Add(face)
    sewing.Perform()
    return sewing.SewedShape()


def export_to_step(faces, output_file):
    """Sew faces and write the result as a STEP file."""
    sewn_shape = sew_faces(faces)

    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, sewn_shape)

    writer = STEPControl_Writer()
    writer.Transfer(compound, STEPControl_AsIs)
    status = writer.Write(output_file)

    if status == IFSelect_RetDone:
        print(f"STEP file exported to: {output_file}")
        return True
    print("STEP export failed")
    return False


def export_to_stl(shape, filename):
    """Export a shape to STL format."""
    try:
        if not isinstance(shape, TopoDS_Shape):
            shape = topods.Shape(shape)

        mesh = BRepMesh_IncrementalMesh(shape, 0.001, False, 0.5, True)
        # mesh = BRepMesh_IncrementalMesh(shape, 0.001, False, 0.1, True)
        mesh.Perform()

        writer = StlAPI_Writer()
        writer.SetASCIIMode(True)
        return writer.Write(shape, filename)
    except Exception as e:
        print(f"STL export failed: {e}")
        return False
