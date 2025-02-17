#!/bin/bash

source $HOME/anaconda3/bin/activate
conda activate ml_env
# Configuration
input_dir="/home/lus04/kvmani/ml_works/kaushal_2025/input_dir_preprocess_notReduced/"
output_dir="/home/lus04/kvmani/ml_works/kaushal_2025/outputs/data11.0/"
log_file="/home/lus04/kvmani/ml_works/kaushal_2025/outputs/data11.0/progressLogs/progressLog.log"
num_processors=16  # Set the desired number of processors (parallel jobs)

# Count total files
total_files=$(find "$input_dir" -type f \( -name "*.ang" -o -name "*.ctf" \) | wc -l)

# Create or clear the log file
> "$log_file"

# Create a temporary file to store the progress count
progress_file=$(mktemp)
echo 0 > "$progress_file"  # Initialize progress count

# Display configuration
echo "Processing $total_files files using $num_processors processors..."
echo "Logs will be saved to $log_file"

# Export variables for use in xargs subshells
export output_dir log_file progress_file total_files

# Process files
find "$input_dir" -type f \( -name "*.ang" -o -name "*.ctf" \) | \
xargs -P "$num_processors" -I {} bash -c '
  {
    # Log the timestamp and number of active processors
    active_processes=$(pgrep -fc "python3 make_pix2pix_data.py")
    echo "$(date "+%Y-%m-%d %H:%M:%S") Active processors: $active_processes Processing: {}" >> "'"$log_file"'"

    # Run the Python script
    python3 make_pix2pix_data.py --input_file "{}" --output_dir "'"$output_dir"'" --forceReduceEulerAngles

    # Atomically update the progress count
    flock 200  # Ensure only one process updates the file at a time
    processed=$(($(cat "'"$progress_file"'") + 1))
    echo $processed > "'"$progress_file"'"
    flock -u 200  # Release the lock

    # Log the progress
    echo "$(date "+%Y-%m-%d %H:%M:%S") Processed $processed/$total_files" | tee -a "'"$log_file"'"
  }' 200>"$progress_file.lock"

# Clean up the temporary progress file
rm "$progress_file" "$progress_file.lock"
