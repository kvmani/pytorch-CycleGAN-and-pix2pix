#!/bin/bash

# Script: run_cyclegan_train.sh
# Description: This script trains CycleGAN model and then tests it for each job specified in a list.
#              It includes a dryrun flag that, when set to true, prints the commands instead of executing them.
# Usage: run_cyclegan_train.sh [--cuda_id <CUDA_VISIBLE_DEVICES>] [--dryrun true|false]

# Default value for CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0,1,2,3

# Default value for dryrun (set to false)
dryrun=false

# Function to log messages with date and time
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to display usage
usage() {
  log "Usage: $0 [--cuda_id <CUDA_VISIBLE_DEVICES>] [--dryrun true|false]"
  exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cuda_id) CUDA_VISIBLE_DEVICES="$2"; shift ;;
        --dryrun) dryrun="$2"; shift ;;
        *) log "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

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

# Define the job list as an array of strings
JobList=(
  
    "--dataroot /home/lus04/kvmani/ml_works/kaushal_2025/outputs/data10.0/tiff_for_ML/AB/ --name ebsd_data_10.0_pix2pix_batch_32_A2B --gan_mode vanilla --n_epochs_decay 500 --batch_size 32 --save_epoch_freq 25 --gpu_ids $CUDA_VISIBLE_DEVICES --direction AtoB"
)
# Function to filter only --dataroot and --name arguments along with their values
filter_args() {
  local filtered_args=""
  local args=("$@")
  for ((i=0; i<${#args[@]}; i++)); do
    if [[ "${args[i]}" == "--dataroot" || "${args[i]}" == "--name" ]]; then
      filtered_args+="${args[i]} ${args[i+1]} "
    fi
  done
  echo "$filtered_args"
}

# Get the number of jobs
total_jobs=${#JobList[@]}

# Loop through each job in the JobList
for ((job_index=0; job_index<total_jobs; job_index++)); do
  job_args="${JobList[job_index]}"
  job_number=$((job_index + 1))

  log "Starting job $job_number/$total_jobs with arguments: ${job_args}"

  # Record job start time
  job_start_time=$(date +%s)

  # Split job_args into an array
  job_args_array=($job_args)

  # Construct the training command
  train_cmd="python train.py ${job_args} --model pix2pix --pool_size 100 --no_dropout --display_id -1 --batch_size 32 --n_epochs_decay 500 --lr 0.0002 --save_epoch_freq 25 --input_nc 3 --output_nc 3 --preprocess none"  #$CUDA_VISIBLE_DEVICES
  
  # Filter only --dataroot and --name for the testing command
  filtered_args=$(filter_args "${job_args_array[@]}")
  test_cmd="python test.py ${filtered_args} --model pix2pix --input_nc 3 --output_nc 3"
  
  # Check if dryrun is true
  if [ "$dryrun" == "true" ]; then
    log "Dryrun enabled. The following command would have been executed for training:"
    echo "$train_cmd"
    log "Dryrun enabled. The following command would have been executed for testing:"
    echo "$test_cmd"
  else
    # Execute the training command
    log "Starting training for job $job_number/$total_jobs"
    $train_cmd
    log "Training for job $job_number/$total_jobs complete"

    # Execute the testing command
    log "Starting testing for job $job_number/$total_jobs"
    $test_cmd
    log "Testing for job $job_number/$total_jobs complete"
  fi

  # Record job end time and calculate execution time
  job_end_time=$(date +%s)
  job_execution_time=$((job_end_time - job_start_time))
  log "Job $job_number/$total_jobs completed in $(date -d@$job_execution_time -u +%H:%M:%S)"
done

log "All jobs completed"

# End time and total execution time
end_time=$(date +%s)
execution_time=$((end_time - start_time))
log "Total execution time: $(date -d@$execution_time -u +%H:%M:%S)"
