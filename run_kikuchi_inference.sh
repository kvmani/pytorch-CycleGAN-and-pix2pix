#!/bin/bash

# Script: run_kikuchi_inference.sh
#
# Description: This script is used to test the CycleGAN model with PyTorch and clean up the results by keeping only necessary images.
#              It sets up the environment, activates the conda environment, runs the inference, and then processes the outputs.
#              Embedded Python code is used for image resizing and grayscale conversion.
#
# Usage: run_kikuchi_inference.sh --input_folder <input_data_folder> --results_dir <output_results_folder> --model_name <model_name> [--target_size <size>]
#
#

# Default value for CUDA_VISIBLE_DEVICES and TARGET_SIZE
CUDA_VISIBLE_DEVICES=0,1,2,3
TARGET_SIZE=460  # Default size

# Function to log messages with date and time
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to display usage
usage() {
  log "Usage: $0 --input_folder <input_data_folder> --results_dir <output_results_folder> --model_name <model_name> [--target_size <size>]"
  exit 1
}

# Check for arguments
if [[ $# -lt 6 ]]; then
  usage
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --input_folder)
      INPUT_FOLDER="$2"
      shift
      shift
      ;;
    --results_dir)
      RESULTS_DIR="$2"
      shift
      shift
      ;;
    --model_name)
      MODEL_NAME="$2"
      shift
      shift
      ;;
    --target_size)
      TARGET_SIZE="$2"
      shift
      shift
      ;;
    *)
      usage
      ;;
  esac
done

# Check if input folder exists
if [[ ! -d "$INPUT_FOLDER" ]]; then
  log "Error: Input folder does not exist."
  exit 1
fi

# Create necessary subfolders (testA and testB)
log "Setting up testA and testB directories for CycleGAN inference..."
mkdir -p "$INPUT_FOLDER/testA"
mkdir -p "$INPUT_FOLDER/testB"

log "Copying all images to testA folder..."
cp "$INPUT_FOLDER"/*.jpg "$INPUT_FOLDER/testA/"

log "Copying a random image to testB folder to fake the presence of domain B..."
cp "$(find "$INPUT_FOLDER/testA" -type f | head -n 1)" "$INPUT_FOLDER/testB/"

# Log system details
log "Starting script on system: $(uname -a)"
log "OS: $(lsb_release -d | awk -F'\t' '{print $2}')"
log "Number of CPU cores: $(nproc)"
log "Total RAM: $(free -h | grep Mem | awk '{print $2}')"
log "GPU Details:"
nvidia-smi

# Activate conda environment
log "Activating conda environment"
source $HOME/anaconda3/bin/activate
conda activate ml_env

# Start the CycleGAN testing
log "Starting CycleGAN testing"
python test.py --results_dir "$RESULTS_DIR" --dataroot "$INPUT_FOLDER" --name "$MODEL_NAME" --model cycle_gan --input_nc 1 --output_nc 1 --load_size 512 --crop_size 512 --gpu_ids -1 --num_test 50000

log "CycleGAN testing completed."

# Clean up and rename files in the results folder
OUTPUT_PATH="$RESULTS_DIR/$MODEL_NAME/test_latest/images"
log "Cleaning up output folder: $OUTPUT_PATH"

for image_file in "$OUTPUT_PATH"/*_fake_B.png; do
  # Get the original image name by removing the _fake_B.png suffix
  original_image=$(basename "$image_file" "_fake_B.png")
  original_image="${original_image}.png"
  
  # Move and rename the necessary images
  mv "$OUTPUT_PATH/${original_image%.png}_fake_B.png" "$OUTPUT_PATH/$original_image"
  

  # Python block for resizing and converting to grayscale
  python3 - <<END
from PIL import Image
import sys

# Arguments passed to the python script
image_path = "$OUTPUT_PATH/$original_image"
target_size = ${TARGET_SIZE}

# Open the image, convert it to grayscale, resize it, and save it
img = Image.open(image_path).convert('L')  # 'L' mode is for grayscale
img = img.resize((target_size, target_size))
img.save(image_path)

END

done

# Remove any remaining unnecessary files
log "Removing unnecessary images..."
rm "$OUTPUT_PATH"/*_fake_A.png
rm "$OUTPUT_PATH"/*_rec_A.png
rm "$OUTPUT_PATH"/*_rec_B.png
rm "$OUTPUT_PATH"/*_real_B.png
rm "$OUTPUT_PATH"/*_real_A.png

# End time and total execution time
end_time=$(date +%s)
execution_time=$((end_time - start_time))
log "Total execution time: $(date -d@$execution_time -u +%H:%M:%S)"
