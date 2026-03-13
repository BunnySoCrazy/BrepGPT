#!/bin/bash

ROOT=../data

python VHP2brep.py \
    -i "${ROOT}/VHP" \
    -o "${ROOT}/recon_brep" \
    -p 8 \
    --no-debug 
    # --use-uv
