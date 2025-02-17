import os
import argparse
import logging
import cv2
import numpy as np
import random

def add_noise_and_blur(img, prob=0.5):
    """
    Adds random noise and Gaussian blur to an image with a given probability.

    Args:
        img (numpy.ndarray): The input image.
        prob (float): Probability of applying noise and blur. Default is 0.5.

    Returns:
        numpy.ndarray: The processed image.
    """

    if random.random() < 0.5:
        if len(img.shape) == 2:  # Grayscale image
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
        else:  # RGB image
            noise = np.random.randint(0, 50, img.shape[:2]).astype(np.uint8)  # Generate noise for each pixel
            noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)  # Repeat noise for each channel
            img = cv2.add(img, noise)

        # Apply Gaussian blur
    if random.random() < prob:
        kernel = random.choice([3,5,7])
        img = cv2.GaussianBlur(img, (kernel, kernel), 0)

    img = apply_circular_mask(img)

    return img

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

        return masked_array

def split_and_save_images(input_folder, output_folder):
    """
    Splits images into two parts and saves them in separate folders, applying noise and blur to image A with a probability of 0.5.

    Args:
        input_folder (str): Path to the input folder containing train, test, and val folders.
        output_folder (str): Path to the output folder where new folders will be created.
    """

    logging.info(f"Input folder: {input_folder}")
    logging.info(f"Output folder: {output_folder}")

    # Create output folder structure
    os.makedirs(output_folder, exist_ok=True)
    for split in ['train', 'test', 'val']:
        os.makedirs(os.path.join(output_folder, f"{split}A"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, f"{split}B"), exist_ok=True)

    # Process images
    #for split in ['train', 'test', 'val']:
    for split in [ 'test', ]:
        input_split_dir = os.path.join(input_folder, split)
        output_split_dir_A = os.path.join(output_folder, f"{split}A")
        output_split_dir_B = os.path.join(output_folder, f"{split}B")

        for filename in os.listdir(input_split_dir):
            if filename.endswith(".png"):
                img_path = os.path.join(input_split_dir, filename)
                img = cv2.imread(img_path)

                # Split image into two halves
                height, width, _ = img.shape
                img_A = img[:, :width // 2, :]
                img_B = img[:, width // 2:, :]

                # Apply noise and blur to image A with probability 0.5
                img_A = add_noise_and_blur(img_A)

                # Save images
                cv2.imwrite(os.path.join(output_split_dir_A, filename), img_A)
                cv2.imwrite(os.path.join(output_split_dir_B, filename), img_B)

                logging.info(f"Processed image: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split images into two parts for CycleGAN")
    parser.add_argument("--input_folder", type=str, default="E:\\ml_works\\data_sets\kikuchi_3Channel_data3.0", help="Path to input folder")
    parser.add_argument("--output_folder", type=str, default="E:\\ml_works\\data_sets\\kikuchi_3Channel_data3.0_CycleGan", help="Path to output folder")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=args.log_level)

    split_and_save_images(args.input_folder, args.output_folder)