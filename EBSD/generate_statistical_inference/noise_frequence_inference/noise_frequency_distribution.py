"""
Example script for analyzing and plotting the distribution of white-noise percentages
in a collection of images. This script:
1. Loads all images from a specified folder.
2. Calculates the percentage of white pixels (above a threshold) in each image.
   (All three channels must be >= threshold to count as white.)
3. Plots the distribution of these percentages as a KDE curve with frequency on the Y-axis,
   showing only the X and Y axes (no top/right spines).
"""

import os
import sys
import cv2  # Ensure opencv-python is installed
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
from scipy.stats import gaussian_kde

def compute_white_noise_percentage(image_path, threshold=250):
    """
    Computes the percentage of pixels whose R, G, and B channels
    are all >= the given threshold.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Warning: could not load image {image_path}.")
        return None

    # Convert from BGR to RGB if needed
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, _ = image_rgb.shape
    total_pixels = height * width

    # Create a boolean mask where R, G, and B channels are all >= threshold
    white_mask = (
        (image_rgb[..., 0] >= threshold) &
        (image_rgb[..., 1] >= threshold) &
        (image_rgb[..., 2] >= threshold)
    )

    # Count how many pixels satisfy the condition
    white_pixels = np.count_nonzero(white_mask)

    # Convert to percentage
    noise_percentage = (white_pixels / total_pixels) * 100
    return noise_percentage

def main():
    parser = argparse.ArgumentParser(
        description="Analyze and plot the distribution of white (noise) pixels in images."
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="/home/lus04/kvmani/ml_works/kaushal_2025/outputs/data5.0/tiff/A/",
        help="Directory containing the images."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./EBSD/generate_statistical_inference/noise_frequence_inference",
        help="Directory containing the images."
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=250,
        help="Pixel value above which all channels are considered white."
    )
    args = parser.parse_args()

    # Retrieve arguments
    image_folder = args.image_folder
    threshold = args.threshold
    output_folder = args.output_dir
    output_image_file = "noise_frequence_distribution.png"
    output_path = os.path.join(output_folder, output_image_file)
    # Gather all image files
    if not os.path.isdir(image_folder):
        print(f"Error: The folder '{image_folder}' does not exist.")
        sys.exit(1)

    all_files = os.listdir(image_folder)
    image_paths = [
        os.path.join(image_folder, f) for f in all_files
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ]

    # Compute noise percentages
    noise_percentages = []
    for img_path in image_paths:
        noise_pct = compute_white_noise_percentage(img_path, threshold)
        if noise_pct is not None:
            if noise_pct > 10:  # Optional filter for very low noise cases
                noise_percentages.append(noise_pct)

    # Check if we have valid data
    if not noise_percentages:
        print("No valid images found or unable to read images.")
        return

    # ================ STATISTICS ================
    mean_noise = np.mean(noise_percentages)
    median_noise = np.median(noise_percentages)
    std_noise = np.std(noise_percentages)

    # ================ PLOTTING ================
    # Use a style that has no grid
    plt.style.use("seaborn-v0_8-white")
    matplotlib.rcParams["axes.grid"] = False  # Disable any automatic grid

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # Generate KDE (Kernel Density Estimation)
    kde = gaussian_kde(noise_percentages, bw_method=0.2)
    x_range = np.linspace(min(noise_percentages), max(noise_percentages), 300)

    # Convert KDE from density to frequency
    bin_width = (max(noise_percentages) - min(noise_percentages)) / 20.0
    y_values = kde(x_range) * len(noise_percentages) * bin_width

    # Plot the KDE curve
    ax.plot(x_range, y_values, color="navy", linewidth=2.5, label="Frequency Curve")

    # Fill the area under the curve
    ax.fill_between(x_range, y_values, color="cornflowerblue", alpha=0.3)

    # Title and labels
    ax.set_title("Noise Distribution", fontsize=22, fontweight="bold", pad=15)
    ax.set_xlabel("% Noise", fontsize=18, labelpad=10)
    ax.set_ylabel("Frequency", fontsize=18, labelpad=10)

    # Only show left & bottom spines (axis lines)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    # Adjust tick labels
    ax.tick_params(axis="both", which="major", labelsize=14, length=6, width=1.2, direction="in")
    ax.tick_params(axis="both", which="minor", length=3, width=1.0, direction="in")

    # Statistics text box
    text_str = (
        f"Number of images: {len(noise_percentages)}\n"
        f"Mean noise: {mean_noise:.2f}%\n"
        f"Median noise: {median_noise:.2f}%\n"
        f"Std. dev.: {std_noise:.2f}%"
    )
    ax.text(
        0.98,
        0.95,
        text_str,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8, boxstyle="round,pad=0.3")
    )

    # Legend
    ax.legend(fontsize=14, loc="upper left", frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
