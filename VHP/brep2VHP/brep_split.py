"""Batch process STEP files: scale to unit box and split closed faces/edges."""

import os
import signal
import argparse
import multiprocessing as mp
from pathlib import Path
from contextlib import contextmanager

from tqdm import tqdm
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from occwl.compound import Compound


@contextmanager
def alarm_timeout(seconds):
    def _handler(signum, frame):
        raise TimeoutError()

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def write_step(shape, filepath):
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    return writer.Write(filepath) == IFSelect_RetDone


def gather_step_files(root_dir, limit=None):
    files = [
        str(Path(root) / f)
        for root, _, filenames in os.walk(root_dir)
        for f in filenames
        if f.lower().endswith(".step")
    ]
    return files[:limit] if limit is not None else files


def process_single(args):
    src_path, input_dir, output_dir, timeout_sec = args

    try:
        with alarm_timeout(timeout_sec):
            rel = Path(src_path).relative_to(input_dir)
            dst_path = Path(output_dir) / rel

            if dst_path.exists():
                return f"Skipped (exists): {dst_path}"

            dst_path.parent.mkdir(parents=True, exist_ok=True)

            solid = Compound.load_step_with_attributes(str(src_path))[0]
            solid = solid.scale_to_unit_box()
            solid = solid.split_all_closed_faces(max_tol=0.01, precision=0.01, num_splits=2)
            solid = solid.split_all_closed_edges(max_tol=0.01, precision=0.01, num_splits=2)

            write_step(solid.topods_shape(), str(dst_path))
            return f"OK: {src_path}"

    except TimeoutError:
        return f"Timeout: {src_path}"
    except Exception as e:
        return f"Error ({src_path}): {e}"


def parse_args():
    parser = argparse.ArgumentParser(description="Batch process STEP files.")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory containing STEP files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for processed files")
    parser.add_argument("--num-workers", type=int, default=48)
    parser.add_argument("--timeout", type=int, default=10, help="Per-file timeout in seconds")
    parser.add_argument("--max-files", type=int, default=None, help="Max number of files to process")
    return parser.parse_args()


def main():
    args = parse_args()

    step_files = gather_step_files(args.input_dir, limit=args.max_files)
    print(f"Found {len(step_files)} STEP file(s) to process.")

    pool_args = [
        (f, args.input_dir, args.output_dir, args.timeout)
        for f in step_files
    ]

    with mp.Pool(args.num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_single, pool_args),
            total=len(step_files),
            desc="Processing",
        ):
            pass


if __name__ == "__main__":
    main()

