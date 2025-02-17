import os
import shutil
import sys
import subprocess
from PIL import Image
import argparse
import time

# Logging function
def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

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
            '--input_nc', '3',
            '--output_nc', '3',
            '--load_size', '256',
            '--crop_size', '256',
            #'--num_test', 50000
            '--gpu_ids', '-1'
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
    for image_file in os.listdir(output_path):
        if image_file.endswith('_fake_B.png'):
            # Extract the original image name
            original_image = image_file.replace('_fake_B.png', '.png')
            
            # Move and rename necessary images
            for suffix in ['_fake_B.png']:
                src = os.path.join(output_path, image_file.replace('_fake_B.png', suffix))
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
    for suffix in ['_real_A.png', '_rec_A.png', '_fake_A.png', '_rec_A.png', '_rec_B.png', '_real_B.png', ]:
        for file in os.listdir(output_path):
            if file.endswith(suffix):
                file_path = os.path.join(output_path, file)
                log(f"Removing {file_path}")
                os.remove(file_path)
    
    log("Image processing completed.")

# Main function to handle the entire workflow
def main():
    parser = argparse.ArgumentParser(description="CycleGAN Inference and Image Processing Script")
    parser.add_argument('--input_folder', required=False, default= "/home/lus04/kvmani/ml_works/kaushal_2025/cyclegan/checkpoints/ebsd_data_2.0_lr1e-4_pool_100_batch_32_vanilla/web/real_A_images/", help="Path to the input data folder")
    parser.add_argument('--results_dir', required=False, default= "/home/lus04/kvmani/ml_works/kaushal_2025/cyclegan/checkpoints/ebsd_data_2.0_lr1e-4_pool_100_batch_32_vanilla/web/results/", help="Path to the results directory")
    parser.add_argument('--model_name', required=False,default="ebsd_data_2.0_lr1e-4_pool_100_batch_32_vanilla", help="Model name to locate results")
    parser.add_argument('--target_size', type=int, default=256, help="Target size for image resizing (default: 460x460)")

    args = parser.parse_args()

    # Step 1: Run CycleGAN inference (test.py)
    log(f"Starting CycleGAN inference with model: {args.model_name}")
    run_cyclegan_test(args.input_folder, args.results_dir, args.model_name)

    # Step 2: Process images (resize, grayscale, and clean up)
    log(f"Processing images in the results directory: {args.results_dir}")
    process_images(args.results_dir, args.model_name, args.target_size)

if __name__ == "__main__":
    main()

