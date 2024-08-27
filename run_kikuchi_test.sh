#!/bin/bash

# Script: run_cyclegan_test.sh
#
# Description: This script is used to test CycleGAN kikuchi model with PyTorch.
#              It sets up the environment, activates the conda environment,
#              and starts the training process with logging.
#
# Usage: run_cyclegan_train.sh 
#
#
#

# Default value for CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0,1,2,3

# Function to log messages with date and time
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to display usage
usage() {
  log "Usage: $0 [--cuda_id <CUDA_VISIBLE_DEVICES>]"
  exit 1
}

# Start time
start_time=$(date +%s)

# Log system details
log "Starting script on system: $(uname -a)"
log "OS: $(lsb_release -d | awk -F'\t' '{print $2}')"
log "Number of CPU cores: $(nproc)"
log "Total RAM: $(free -h | grep Mem | awk '{print $2}')"
log "GPU Details:"
nvidia-smi



log "Killing any existing Python3 processes"
killall -9 python3

log "Python3 path before activation of environment: $(which python3)"
log "Pip3 path before activation of environment: $(which pip3)"

log "Activating conda environment"
source $HOME/anaconda3/bin/activate
conda env list
conda activate ml_env

log "Python3 path after activation of environment: $(which python3)"
log "Pip3 path after activation of environment: $(which pip3)"

log "Starting CycleGAN training"
python test.py --dataroot ./datasets/SimulatedData --name sim_kikuchi_no_preprocess_lr00002_460X460 --model cycle_gan --input_nc 1 --output_nc 1

log "Super resolution - End"

# End time and total execution time
end_time=$(date +%s)
execution_time=$((end_time - start_time))
log "Total execution time: $(date -d@$execution_time -u +%H:%M:%S)"
