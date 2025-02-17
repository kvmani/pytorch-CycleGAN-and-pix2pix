#!/bin/bash
#
# parallel_subcrops.sh
# --------------------
# This script finds all images in a hardcoded input folder and uses 16 parallel
# Python processes (via xargs) to run create_subcrops.py on each file, saving
# results to a hardcoded output directory.

# ---------------- EDIT THESE TWO LINES ----------------
INPUT_DIR="/home/lus04/kvmani/ml_works/kaushal_2025/outputs/data8.0/extract_only_tiffs/B/"
OUTPUT_DIR="/home/lus04/kvmani/ml_works/kaushal_2025/outputs/data9.0/tiff/B/"
# ------------------------------------------------------

# Ensure the output directory exists
# mkdir -p "$OUTPUT_DIR"

# Find images by extension (adjust as needed). Then call create_subcrops.py
# 16 at a time in parallel, passing each image path and the single output dir.
find "$INPUT_DIR" -type f \( \
    -iname '*.tif'  -o \
    -iname '*.tiff' -o \
    -iname '*.jpg'  -o \
    -iname '*.jpeg' -o \
    -iname '*.png'  \
\) | xargs -P 16 -I {} python3 cropImage.py \
    --input_file "{}" \
    --output_dir "$OUTPUT_DIR"
