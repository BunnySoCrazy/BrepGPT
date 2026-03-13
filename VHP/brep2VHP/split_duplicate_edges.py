import os
import time
import shutil
import argparse
import multiprocessing

from OCC.Core.TopoDS import  topods_Face, topods_Wire
from OCC.Core.TopAbs import (
    TopAbs_FACE, TopAbs_WIRE, TopAbs_REVERSED,
)
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
from OCC.Core.BRepTools import BRepTools_WireExplorer, BRepTools_ReShape
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.ShapeFix import ShapeFix_Shape
from OCC.Core.TopoDS import topods
from OCC.Core.gp import gp_Pnt

from occwl.solid import Solid
from brep_utils import read_step, write_step


def find_duplicate_edges(shape):
    """Find directed half-edges that appear more than once."""
    half_edges = {}
    duplicates = []

    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    face_idx = 0

    while face_exp.More():
        face = topods_Face(face_exp.Current())
        wire_exp = TopExp_Explorer(face, TopAbs_WIRE)
        wire_idx = 0

        while wire_exp.More():
            wire = topods_Wire(wire_exp.Current())
            edge_exp = BRepTools_WireExplorer(wire)

            while edge_exp.More():
                edge = edge_exp.Current()

                v1 = topods.Vertex(topexp.FirstVertex(edge))
                v2 = topods.Vertex(topexp.LastVertex(edge))

                p1 = BRep_Tool.Pnt(v1)
                p2 = BRep_Tool.Pnt(v2)

                c1 = (round(p1.X(), 12), round(p1.Y(), 12), round(p1.Z(), 12))
                c2 = (round(p2.X(), 12), round(p2.Y(), 12), round(p2.Z(), 12))

                if edge.Orientation() == TopAbs_REVERSED:
                    c1, c2 = c2, c1

                directed_edge = (c1, c2)

                if directed_edge in half_edges and c1 != c2:
                    duplicates.append({
                        'edge': edge,
                        'first_occurrence': half_edges[directed_edge],
                        'current_occurrence': [face_idx, wire_idx],
                    })
                else:
                    half_edges[directed_edge] = [face_idx, wire_idx]

                edge_exp.Next()

            wire_idx += 1
            wire_exp.Next()

        face_idx += 1
        face_exp.Next()

    return half_edges, duplicates


def split_edge_into_wire(edge):
    """Split a single edge at its midpoint, returning a wire of two sub-edges."""
    if edge.IsNull():
        print('Edge is null')
        return None

    curve, first, last = BRep_Tool.Curve(edge)
    orientation = edge.Orientation()
    mid = (first + last) / 2.0

    if abs(last - first) < 1e-7:
        print(f'Curve parameter range too small: {first}, {last}')
        return None

    p1, pm, p2 = gp_Pnt(), gp_Pnt(), gp_Pnt()
    curve.D0(first, p1)
    curve.D0(mid, pm)
    curve.D0(last, p2)

    if pm.Distance(p1) < 1e-7 or pm.Distance(p2) < 1e-7:
        print(f'Midpoint coincides with endpoint: {first}, {last}')
        return None

    e1 = BRepBuilderAPI_MakeEdge(curve, first, mid).Edge()
    e2 = BRepBuilderAPI_MakeEdge(curve, mid, last).Edge()
    e1.Orientation(orientation)
    e2.Orientation(orientation)

    wire_maker = BRepBuilderAPI_MakeWire()
    wire_maker.Add(e1)
    wire_maker.Add(e2)

    if not wire_maker.IsDone():
        print("Wire creation failed")
        return None

    return wire_maker.Wire()


def split_duplicate_edges(shape, duplicates):
    """Replace each duplicate edge with a wire of two sub-edges."""
    reshaper = BRepTools_ReShape()
    for dup in duplicates:
        wire = split_edge_into_wire(dup['edge'])
        if wire is not None:
            reshaper.Replace(dup['edge'], wire)
    return reshaper.Apply(shape)


def fix_shape_if_invalid(shape):
    """Attempt to fix an invalid shape. Returns (shape, is_valid)."""
    analyzer = BRepCheck_Analyzer(shape)
    if analyzer.IsValid():
        return shape, True

    fixer = ShapeFix_Shape(shape)
    fixer.Perform()
    fixed = fixer.Shape()

    if BRepCheck_Analyzer(fixed).IsValid():
        print("Shape repair succeeded,", Solid(fixed).num_vertices(), "vertices")
        return fixed, True
    else:
        print(Solid(shape).num_vertices(), "vertices")
        print("Shape repair failed")
        return shape, False


def process_single_file(input_file, output_file):
    try:
        shape = read_step(input_file)
        if not BRepCheck_Analyzer(shape).IsValid():
            return

        fixer = ShapeFix_Shape(shape)
        fixer.Perform()
        shape = fixer.Shape()

        half_edges, duplicates = find_duplicate_edges(shape)

        if len(duplicates) == 0:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            shutil.copy2(input_file, output_file)
            return

        modified = split_duplicate_edges(shape, duplicates)

        try:
            modified, is_valid = fix_shape_if_invalid(modified)
            if not is_valid:
                return
        except Exception:
            # print(f"{input_file} validation error")
            pass

        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        write_step(modified, output_file)

    except Exception as e:
        print(f"{input_file} - error: {e}")


def collect_step_files(directory):
    """Recursively collect all .step/.stp files under a directory."""
    files = []
    for root, _, filenames in os.walk(directory):
        for f in filenames:
            if f.lower().endswith(('.step', '.stp')):
                files.append(os.path.join(root, f))
    return files


def process_file_list(file_pairs, num_processes, timeout, label=""):
    """Process a list of (input, output) pairs with multiprocessing and timeout."""
    if not file_pairs:
        return 0

    pool = multiprocessing.Pool(processes=num_processes)
    results = [
        pool.apply_async(process_single_file, (inp, out))
        for inp, out in file_pairs
    ]

    start = time.time()
    completed = 0
    for r in results:
        remaining = max(0, timeout - (time.time() - start))
        try:
            r.get(timeout=remaining)
            completed += 1
        except multiprocessing.TimeoutError:
            print(f"{label} timed out after {timeout}s, processed {completed}/{len(file_pairs)}")
            break
        except Exception as e:
            print(f"Error: {e}")

    pool.terminate()
    pool.join()
    return completed


def main():
    parser = argparse.ArgumentParser(description='Split duplicate half-edges in STEP files')
    parser.add_argument('--input_dir', type=str, help='Input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--single_file', type=str, help='Process a single file instead of a directory')
    parser.add_argument('--processes', type=int, default=os.cpu_count(), help='Number of worker processes')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout in seconds per subdirectory batch')
    args = parser.parse_args()

    # Single-file mode
    if args.single_file:
        if not os.path.exists(args.single_file):
            print(f"File not found: {args.single_file}")
            return
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, os.path.basename(args.single_file))
        process_single_file(args.single_file, output_file)
        return

    # Batch mode
    if not args.input_dir or not os.path.exists(args.input_dir):
        print(f"Input directory not found: {args.input_dir}")
        return

    print("Processing STEP files by subdirectory...")

    subdirs = [
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ]

    total = 0

    for subdir in subdirs:
        subdir_path = os.path.join(args.input_dir, subdir)
        print(f"Subdirectory: {subdir}")

        pairs = []
        for fpath in collect_step_files(subdir_path):
            rel = os.path.relpath(fpath, args.input_dir)
            pairs.append((fpath, os.path.join(args.output_dir, rel)))

        print(f"  Found {len(pairs)} STEP file(s)")
        done = process_file_list(pairs, args.processes, args.timeout, label=subdir)
        total += done
        print(f"  Completed {done}/{len(pairs)}")

    # Root-level files
    root_pairs = []
    for f in os.listdir(args.input_dir):
        fpath = os.path.join(args.input_dir, f)
        if os.path.isfile(fpath) and f.lower().endswith(('.step', '.stp')):
            root_pairs.append((fpath, os.path.join(args.output_dir, f)))

    if root_pairs:
        print(f"Processing {len(root_pairs)} root-level STEP file(s)")
        done = process_file_list(root_pairs, args.processes, 300, label="root")
        total += done

    print(f"Done. Total processed: {total}")


if __name__ == "__main__":
    main()

