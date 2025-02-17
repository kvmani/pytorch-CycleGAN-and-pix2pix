import os
import shutil
import random
import argparse
import logging

import PIL.Image
from PIL import Image
from multiprocessing import Pool, cpu_count
import time
import numpy as np


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("image_distribution.log"),
            logging.StreamHandler()
        ]
    )


def collect_images(input_folder):
    image_paths = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                image_paths.append(os.path.join(root, file))
    return image_paths


def resize_image(image_path, target_size):
    with Image.open(image_path) as img:
        resized_img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        return resized_img


def process_image(args):
    path, input_folder, target_dir,circular_mask, resize = args
    relative_path = os.path.relpath(path, input_folder)
    new_name = relative_path.replace(os.sep, '_')
    new_path = os.path.join(target_dir, new_name)

    if resize:
        resized_image = resize_image(path, resize)
    if circular_mask:
        resized_image=apply_circular_mask(resized_image)

        resized_image.save(new_path)
    else:
        shutil.copy2(path, new_path)

    return new_path


def distribute_images(input_folder, image_paths, output_folder, circular_mask, dataSetString="B", resize=None, num_workers=4):
    random.shuffle(image_paths)
    train_split = int(0.8 * len(image_paths))
    test_split = int(0.1 * len(image_paths))

    datasets = {
        f'train{dataSetString}': image_paths[:train_split],
        f'test{dataSetString}': image_paths[train_split:train_split + test_split],
        f'val{dataSetString}': image_paths[train_split + test_split:]
    }

    stats = {f'train{dataSetString}': 0, f'test{dataSetString}': 0, f'val{dataSetString}': 0}

    pool = Pool(processes=num_workers)
    logging.info(f"Using {num_workers} proceses in parallel")

    for dataset, paths in datasets.items():
        target_dir = os.path.join(output_folder, dataset)
        os.makedirs(target_dir, exist_ok=True)

        task_args = [(path, input_folder, target_dir,circular_mask, resize) for path in paths]
        for result in pool.imap_unordered(process_image, task_args):
            stats[dataset] += 1

    pool.close()
    pool.join()

    return stats

def apply_circular_mask(image_array):
        """
        Applies a circular mask to a square image array.

        Parameters:
        -----------
        image_array : numpy.ndarray
            The input image array to mask.

        Returns:
        --------
        masked_array : numpy.ndarray
            The masked image array.
        mask : numpy.ndarray
            The mask applied to the image array.
        """

        image_array = np.array(image_array)
        assert image_array.shape[0] == image_array.shape[1], "Image must be square (nXn shape)"
        size = image_array.shape[0]
        center = size // 2
        radius = center

        Y, X = np.ogrid[:size, :size]
        dist_from_center = np.sqrt((X - center) ** 2 + (Y - center) ** 2)
        mask = dist_from_center <= radius

        if image_array.ndim == 2:  # Grayscale image
            masked_array = np.zeros_like(image_array)
            masked_array[mask] = image_array[mask]
        elif image_array.ndim == 3:  # Color image
            masked_array = np.zeros_like(image_array)
            for i in range(image_array.shape[2]):  # Apply mask to each channel
                masked_array[:, :, i] = np.where(mask, image_array[:, :, i], 0)

        masked_array = PIL.Image.fromarray(masked_array)
        return masked_array


def main(args):
    start_time = time.time()
    setup_logger()
    logging.info("Starting image collection from '%s'", args.input_folder)

    image_paths = collect_images(args.input_folder)
    if not image_paths:
        logging.error("No images found in the input folder. Exiting.")
        return

    logging.info("Total images found: %d", len(image_paths))

    if args.fraction < 1.0:
        totalImages = len(image_paths)
        image_paths = random.sample(image_paths, int(len(image_paths) * args.fraction))
        logging.info("Selected %d images based on fraction %f of %d total images", len(image_paths), args.fraction, totalImages)

    stats = distribute_images(args.input_folder, image_paths, args.output_folder, args.circular_mask, args.dataSetString, args.resize,
                              args.num_workers)

    logging.info("Distribution complete.")
    logging.info("Train images: %d", stats[f'train{args.dataSetString}'])
    logging.info("Test images: %d", stats[f'test{args.dataSetString}'])
    logging.info("Validation images: %d", stats[f'val{args.dataSetString}'])

    end_time = time.time()
    total_time = end_time - start_time
    logging.info("Total time taken: %.2f seconds", total_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distribute images into train, test, and validation folders.')
    parser.add_argument('--input_folder', type=str, default=r"/home/lus04/kvmani/ml_works/kaushal_2025/outputs/data10.0/tiff/A/",
                        help='Root input folder containing images')
    parser.add_argument('--output_folder', type=str, default=r"/home/lus04/kvmani/ml_works/kaushal_2025/outputs/data10.0/tiff_for_ML/",
                        help='Target output folder to store the datasets')
    parser.add_argument('--dnataSetStrig', type=str, default="A", help='A or B')
    parser.add_argument('--resize', type=int, default=256, help='Resize images to target size (e.g., 256)')
    parser.add_argument('--circular_mask', type=bool, default=False, help='whether or not to apply circular mask')
    parser.add_argument('--fraction', type=float, default=1.0,
                        help='Fraction of total images to process (0.0 < fraction <= 1.0)')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of CPU cores to use for parallel processing')
    

    args = parser.parse_args()
    main(args)
