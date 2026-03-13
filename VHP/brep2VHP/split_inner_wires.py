"""Split faces by inner wires and reassemble into solids."""

import os
import argparse
import multiprocessing as mp

from tqdm import tqdm
from OCC.Core.TopoDS import TopoDS_Shell, topods_Face, topods_Wire
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
from OCC.Core.ShapeFix import ShapeFix_Shell
from brep_utils import read_step, write_step, extract_faces, count_vertices, split_face_by_inner_wires, sew_faces_to_solid


def process_step_file(input_file, output_dir, max_vertices=256):
    try:
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)

        shape = read_step(input_file)

        if count_vertices(shape) > max_vertices:
            return

        faces = extract_faces(shape)

        split_faces = []
        for face in faces:
            split_faces.extend(split_face_by_inner_wires(face))

        solid = sew_faces_to_solid(split_faces)
        write_step(solid, output_file)

    except Exception as e:
        print(f"Error ({input_file}): {e}")


def process_wrapper(args):
    input_file, output_subdir, max_vertices = args
    try:
        os.makedirs(output_subdir, exist_ok=True)
        process_step_file(input_file, output_subdir, max_vertices)
    except Exception as e:
        print(f"Error ({input_file}): {e}")


def gather_tasks(input_dir, output_dir):
    tasks = []
    pbar = tqdm(desc="Collecting files", unit="file")
    try:
        for root, _, files in os.walk(input_dir):
            for f in files:
                if f.lower().endswith(".step"):
                    input_file = os.path.join(root, f)
                    rel = os.path.relpath(root, input_dir)
                    output_subdir = os.path.join(output_dir, rel)
                    tasks.append((input_file, output_subdir))
                    pbar.update(1)
    finally:
        pbar.close()

    print(f"Collected {len(tasks)} file(s) to process.")
    return tasks


def parse_args():
    parser = argparse.ArgumentParser(description="Split faces by inner wires and reassemble into solids.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--max-vertices", type=int, default=256, help="Skip shapes with more vertices than this")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = gather_tasks(args.input_dir, args.output_dir)
    pool_args = [(f, d, args.max_vertices) for f, d in tasks]

    with mp.Pool(args.num_workers) as pool:
        list(tqdm(
            pool.imap_unordered(process_wrapper, pool_args),
            total=len(pool_args),
            desc="Processing",
        ))


if __name__ == "__main__":
    main()

