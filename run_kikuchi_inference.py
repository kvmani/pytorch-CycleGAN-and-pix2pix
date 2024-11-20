import os
import shutil
import sys
import subprocess
from PIL import Image
import argparse
import time
import random


# Logging function
def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


# Function to create testA and testB folders and copy images
def setup_test_folders(input_folder):
    log("Setting up testA and testB directories for CycleGAN inference...")

    # Create testA and testB directories
    testA_dir = os.path.join(input_folder, 'testA')
    testB_dir = os.path.join(input_folder, 'testB')

    os.makedirs(testA_dir, exist_ok=True)
    os.makedirs(testB_dir, exist_ok=True)

    # Copy all images to testA folder
    images = [f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')]
    for image in images:
        src = os.path.join(input_folder, image)
        dst = os.path.join(testA_dir, image)
        log(f"Copying {src} to {dst}")
        shutil.copy(src, dst)

    # Copy a random image to testB folder to simulate domain B presence
    if images:
        random_image = random.choice(images)
        src = os.path.join(testA_dir, random_image)
        dst = os.path.join(testB_dir, random_image)
        log(f"Copying a random image {src} to testB folder: {dst}")
        shutil.copy(src, dst)
    else:
        log("No images found in the input folder to copy.")
        sys.exit(1)

    return testA_dir, testB_dir


# Function to remove testA and testB folders
def clean_test_folders(testA_dir, testB_dir):
    log(f"Removing testA and testB directories: {testA_dir}, {testB_dir}")
    shutil.rmtree(testA_dir)
    shutil.rmtree(testB_dir)


# Function to run the CycleGAN test script
def run_cyclegan_test(input_folder, results_dir, model_name):
    log(f"Running CycleGAN test script for model: {model_name}")

    try:
        # Construct the command to run test.py
        command = [
            'python', 'test.py',  # Adjust this if test.py is in a different location
            '--results_dir', results_dir,
            '--dataroot', input_folder,
            '--name', model_name,
            '--model', 'cycle_gan',
            '--input_nc', '1',
            '--output_nc', '1',
            '--load_size', '512',
            '--crop_size', '512',
            '--num_test', '10'
            # '--gpu_ids', '-1'
        ]

        # Run the command
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


# Main function to handle the entire workflow
def main():
    ### example command:  python .\run_kikuchi_inference.py --model_name cyclegan_kikuchi_model_weights\sim_kikuchi_no_preprocess_lr2e-4_decay_400
    #                             --input_folder C:\Users\kvman\Documents\ml_data\kikuchi_super_resolution\test_2x2  --results_dir results\debarna_test
    parser = argparse.ArgumentParser(description="CycleGAN Inference and Image Processing Script")
    parser.add_argument('--input_folder', required=True, help="Path to the input data folder")
    parser.add_argument('--results_dir', required=True, help="Path to the results directory")
    parser.add_argument('--model_name', required=True, help="Model name to locate results")
    parser.add_argument('--target_size', type=int, default=460,
                        help="Target size for image resizing (default: 460x460)")

    args = parser.parse_args()

    # Step 1: Set up testA and testB directories, and copy images
    testA_dir, testB_dir = setup_test_folders(args.input_folder)

    # Measure total start time
    start_time = time.time()

    # Step 2: Run CycleGAN inference (test.py)
    log(f"Starting CycleGAN inference with model: {args.model_name}")
    run_cyclegan_test(args.input_folder, args.results_dir, args.model_name)

    # Step 3: Process images (resize, grayscale, and clean up)
    log(f"Processing images in the results directory: {args.results_dir}")
    fake_B_images = process_images(args.results_dir, args.model_name, args.target_size)

    # Step 4: Clean up testA and testB directories
    clean_test_folders(testA_dir, testB_dir)

    # Measure total end time
    end_time = time.time()
    total_time = end_time - start_time
    if fake_B_images > 0:
        time_per_inference = total_time / fake_B_images
        log(f"Total processing time: {total_time:.2f} seconds.")
        log(f"Time per inference: {time_per_inference:.2f} seconds.")
    else:
        log("No fake_B images were processed.")


if __name__ == "__main__":
    main()
