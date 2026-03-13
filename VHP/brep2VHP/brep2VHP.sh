#!/bin/bash

ROOT=../data
echo "Splitting BREP files..."

python brep_split.py \
    --input-dir "${ROOT}/brep" \
    --output-dir "${ROOT}/split" \
    --num-workers 12 \
    --timeout 10 \
    --max-files 10

echo "Splitting inner wires..."

python split_inner_wires.py \
    --input-dir "${ROOT}/split" \
    --output-dir "${ROOT}/break" \
    --num-workers 12 \
    --max-vertices 256

echo "Splitting duplicate edges..."

python split_duplicate_edges.py \
    --input_dir "${ROOT}/split" \
    --output_dir "${ROOT}/simple" \
    --processes 48 \
    --timeout 300

echo "Sampling VHP..."

python VHP_sampling.py \
    --input-dir "${ROOT}/simple" \
    --output-dir "${ROOT}/VHP" \
    --num-workers 12 \
    --max-files 10 \
    --max-vertices 512 \
    --edge-samples 12 \
    --normal-samples 5