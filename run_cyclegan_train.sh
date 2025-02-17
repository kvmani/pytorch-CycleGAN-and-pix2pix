#!/bin/bash

# Script: run_cyclegan_train.sh
#
# Description: This script is used to train cyclegan model with PyTorch.
#              It sets up the environment, activates the conda environment,
#              and starts the training process.
#
# Usage: run_cyclegan_train.sh [--cuda_id <CUDA_VISIBLE_DEVICES>]
#
# Options:
#   --cuda_id <CUDA_VISIBLE_DEVICES>: Specify GPU device(s) to use for training.
#                                     If not provided, defaults to 0,1,2,3.
#
# Example:
#   To train with CUDA_VISIBLE_DEVICES set to GPU 1:
#   $ ./train_realsrgan.sh --cuda_id 1
#
#   To train with multiple GPUs (e.g., GPU 0 and GPU 1):
#   $ ./train_realsrgan.sh --cuda_id 0,1
#
#   To train with the default GPUs (0,1,2,3):
#   $ ./train_realsrgan.sh
#

# Default value for CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0,1,2,3

# Function to display usage
usage() {
  echo "Usage: $0 [--cuda_id <CUDA_VISIBLE_DEVICES>]"
  exit 1
}

# Parse optional arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --cuda_id)
      if [[ -n "$2" && "$2" =~ ^[0-4](,[0-4])*$ ]]; then
        # Extract the first CUDA_VISIBLE_DEVICES value if it's a list
        FIRST_CUDA_VISIBLE_DEVICE=$(echo "$2" | cut -d',' -f1)
        CUDA_VISIBLE_DEVICES=$2
        shift 2
      elif [[ -n "$2" && "$2" =~ ^[0-4]$ ]]; then
        FIRST_CUDA_VISIBLE_DEVICE=$2
        CUDA_VISIBLE_DEVICES=$2
        shift 2
      else
        echo "Error: --cuda_id must be a single integer or a comma-separated list of integers from 0 to 4."
        usage
      fi
      ;;
    *)
      echo "Unknown parameter passed: $1"
      usage
      ;;
  esac
done

# If FIRST_CUDA_VISIBLE_DEVICE is not set, use the whole CUDA_VISIBLE_DEVICES
if [ -z "$FIRST_CUDA_VISIBLE_DEVICE" ]; then
  FIRST_CUDA_VISIBLE_DEVICE=$CUDA_VISIBLE_DEVICES
fi

# Calculate the number of visible GPUs
IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
n_GPUs="${#GPU_ARRAY[@]}"

# Calculate the master port
MASTER_PORT=$((4322 + FIRST_CUDA_VISIBLE_DEVICE))

echo "Super resolution - Start $MASTER_PORT"
nvidia-smi

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

killall -9 python3

which python3
which pip3

echo "Above is before activation of env"

source $HOME/anaconda3/bin/activate
conda env list
conda activate ml_env

echo "Below is after activation of env"
which python3
which pip3

#torchrun --nproc_per_node=$n_GPUs --master_port=$MASTER_PORT ../realesrgan/train.py -opt ../options/train_realesrgan_x4plus.yml --debug
#CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train.py --dataroot ./datasets/horse2zebra --name horse2zebra_cyclegan --model cycle_gan --pool_size 50 --no_dropout 

echo "Super resolution - End"

