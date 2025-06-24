import os
import shutil
import sys
import subprocess
from PIL import Image
import argparse
import time
import random
import platform
import getpass
import socket
import numpy as np


# Logging function

def log(message, log_file=None):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)  # Print to console
    if log_file:
        with open(log_file, 'a') as f:
            f.write(formatted_message + '\n')

# Function to create testA and testB folders and copy images
def setup_test_folders(input_folder, debug):
    log("Setting up testA and testB directories for CycleGAN inference...")

    # Create testA and testB directories
    testA_dir = os.path.join(input_folder, 'testA')
    testB_dir = os.path.join(input_folder, 'testB')

    os.makedirs(testA_dir, exist_ok=True)
    os.makedirs(testB_dir, exist_ok=True)

    # Get all image files
    images = [f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')]

    if debug:
        log("Debug mode is enabled. Processing only the first 20 images.")
        images = images[:20]

    if not images:
        log("No images found in the input folder to copy.")
        sys.exit(1)

    # Copy all images to testA folder
    for image in images:
        src = os.path.join(input_folder, image)
        dst = os.path.join(testA_dir, image)
        log(f"Copying {src} to {dst}")
        shutil.copy(src, dst)

    # Copy a random image to testB folder to simulate domain B presence
    random_image = random.choice(images)
    src = os.path.join(testA_dir, random_image)
    dst = os.path.join(testB_dir, random_image)
    log(f"Copying a random image {src} to testB folder: {dst}")
    shutil.copy(src, dst)

    return testA_dir, testB_dir


# Function to remove testA and testB folders
def clean_test_folders(testA_dir, testB_dir):
    log(f"Removing testA and testB directories: {testA_dir}, {testB_dir}")
    shutil.rmtree(testA_dir)
    shutil.rmtree(testB_dir)


# Function to activate conda environment and run CycleGAN test script
def run_cyclegan_test(input_folder, results_dir, model_name):
    log(f"Running CycleGAN test script for model: {model_name}")

    try:
        # Construct the command to run test.py
        command = [
            'python', 'test.py',
            '--results_dir', results_dir,
            '--dataroot', input_folder,
            '--name', model_name,
            '--model', 'cycle_gan',
            '--input_nc', '1',
            '--output_nc', '1',
            '--load_size', '256',
            '--crop_size', '256',
            '--num_test', '100000',
            #'--gpu_ids', '-1',
        ]

        # Check and activate conda environment on Linux
        if sys.platform.startswith('linux'):
            log("Activating conda environment for Linux...")
            conda_activate = f"source $HOME/anaconda3/bin/activate && conda activate ml_env && {' '.join(command)}"
            subprocess.run(conda_activate, shell=True, check=True)
        else:
            subprocess.run(command, check=True)

        log("CycleGAN testing completed successfully.")

    except subprocess.CalledProcessError as e:
        log(f"Error occurred during CycleGAN test: {e}")
        sys.exit(1)


# Function to process images: rename, resize, and convert to grayscale
def process_images(results_dir, model_name, target_size):
    log("Starting image processing...")

    # Set up paths
    output_path = os.path.join(results_dir, model_name, 'test_latest', 'images')

    if not os.path.exists(output_path):
        log(f"Output path {output_path} does not exist.")
        sys.exit(1)

    # Process images in the output folder
    fake_B_images = 0
    for image_file in os.listdir(output_path):
        if image_file.endswith('_fake_B.png'):
            fake_B_images += 1
            # Extract the original image name
            original_image = image_file.replace('_fake_B.png', '.png')

            # Move and rename necessary images
            src = os.path.join(output_path, image_file)
            dst = os.path.join(output_path, original_image)
            if os.path.exists(src):
                log(f"Renaming {src} to {dst}")
                shutil.move(src, dst)

            # Convert to grayscale and resize
            img_path = os.path.join(output_path, original_image)
            log(f"Processing image: {img_path}")
            img = Image.open(img_path).convert('L')  # 'L' for grayscale
            img = img.resize((target_size, target_size))
            img.save(img_path)

    # Clean up unnecessary files
    for suffix in ['_real_A.png', '_rec_A.png', '_fake_A.png', '_rec_A.png', '_rec_B.png', '_real_B.png']:
        for file in os.listdir(output_path):
            if file.endswith(suffix):
                file_path = os.path.join(output_path, file)
                log(f"Removing {file_path}")
                os.remove(file_path)

    log(f"Image processing completed. Processed {fake_B_images} fake_B images.")
    return fake_B_images

def record_run_details(log_file, args, start_time):
    with open(log_file, 'a') as f:
        f.write("========== Run Details ==========\n")
        f.write(f"Run Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Full Command Used for Run: {' '.join(sys.argv)}\n")
        f.write(f"Input Data Folder: {args.input_folder}\n")
        f.write(f"Results Directory: {args.results_dir}\n")
        f.write(f"Model Name: {args.model_name}\n")
        f.write(f"Target Size for Resizing: {args.target_size}\n")
        f.write(f"Debug Mode: {args.debug}\n")
        f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")

        # System Information
        f.write("\n========== System Information ==========\n")
        f.write(f"User: {getpass.getuser()}\n")
        f.write(f"Hostname: {socket.gethostname()}\n")
        f.write(f"Operating System: {platform.system()} {platform.release()}\n")
        f.write(f"OS Version: {platform.version()}\n")
        f.write(f"Processor: {platform.processor()}\n")
        f.write(f"Python Version: {platform.python_version()}\n")
        f.write(f"Current Working Directory: {os.getcwd()}\n")
        f.write(f"Available CPU Cores: {os.cpu_count()}\n")

        f.write("=================================\n\n")
# Main function to handle the entire workflow
def main():
    parser = argparse.ArgumentParser(description="CycleGAN Inference and Image Processing Script")
    parser.add_argument('--input_folder', required=True, help="Path to the input data folder")
    parser.add_argument('--results_dir', required=True, help="Path to the results directory")
    parser.add_argument('--model_name', required=True, help="Model name to locate results")
    parser.add_argument('--target_size', type=int, default=230, help="Target size for image resizing (default: 230X230)")
    parser.add_argument('--debug', action='store_true', help="Debug mode to process only the first 20 images")

    args = parser.parse_args()

    # Ensure the results directory exists
    os.makedirs(args.results_dir, exist_ok=True)

    # Log file path in results directory
    log_file = os.path.join(args.results_dir, 'run_details.log')

    # Measure total start time
    start_time = time.time()

    # Record initial run details
    record_run_details(log_file, args, start_time)

    # Step 1: Set up testA and testB directories, and copy images
    testA_dir, testB_dir = setup_test_folders(args.input_folder, args.debug)

    # Step 2: Run CycleGAN inference (test.py)
    log(f"Starting CycleGAN inference with model: {args.model_name}", log_file)
    run_cyclegan_test(args.input_folder, args.results_dir, args.model_name)

    # Step 3: Process images (resize, grayscale, and clean up)
    log(f"Processing images in the results directory: {args.results_dir}", log_file)
    fake_B_images = process_images(args.results_dir, args.model_name, args.target_size)

    # Step 4: Clean up testA and testB directories
    clean_test_folders(testA_dir, testB_dir)

    # Measure total end time
    end_time = time.time()
    total_time = end_time - start_time

    # Write additional run details to log file
    with open(log_file, 'a') as f:
        f.write("========== Summary ==========\n")
        if fake_B_images > 0:
            time_per_inference = total_time / fake_B_images
            f.write(f"Total Processing Time: {total_time:.2f} seconds\n")
            f.write(f"Time Per Inference: {time_per_inference:.2f} seconds\n")
            f.write(f"{np.around(60/time_per_inference,0)} number of images per minute\n")
            print(f"{np.around(60/time_per_inference,0)} number of images per minute\n")
        else:
            f.write("No fake_B images were processed.\n")
        f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
        f.write("=================================\n")

    log(f"Run details saved to: {log_file}", log_file)

if __name__ == "__main__":
    ### example run command: python .\run_kikuchi_inference.py --model_name cyclegan_kikuchi_model_weights/sim_kikuchi_no_preprocess_lr2e-4_decay_400
    #                                --input_folder C:\Users\kvman\PycharmProjects\kikuchiBandAnalyzer\exported_images\magnetite_data --results_dir debarna_magnetite_ai_processed
    main()


