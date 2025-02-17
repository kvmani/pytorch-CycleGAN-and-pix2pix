#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <shell_file_to_run> [additional_arguments to shell_file_to_run]"
  exit 1
fi

shell_file=$1
shift  # Remove the first argument from the list

echo "sbatch --time=14400 --nodes=1 \"$shell_file\" \"$@\""
#sbatch --time=14400 --exclusive --nodes=1 --gpus=4 --cpus-per-task=8  "$shell_file" "$@"
# Uncomment the following line when ready to actually invoke the sbatch command
sbatch --time=14400 --nodes=1 --gpus=1 --cpus-per-task=96  "$shell_file" "$@"

