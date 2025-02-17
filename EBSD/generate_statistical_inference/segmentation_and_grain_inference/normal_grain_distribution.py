#!/usr/bin/env python3

"""
Generates a plot of a NORMAL (Gaussian) distribution curve for the 'Area(px)' column
based on the dataset's mean and standard deviation (not a KDE from the data itself).
User can provide CSV path, column name, output path, etc., via command-line arguments.
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
from scipy.stats import norm  # For normal distribution
import os

def main():
    parser = argparse.ArgumentParser(
        description="Plot a normal distribution curve using mean/std from a numeric column in a CSV."
    )
    # Arguments
    parser.add_argument(
        "--csv_file_path",
        type=str,
        required=False,
        default="EBSD/generate_statistical_inference/segmentation_and_grain_inference/output/grains_info/grain_info.csv",
        help="Path to the CSV file containing grain data."
    )
    parser.add_argument(
        "--column_name",
        type=str,
        default="Area(px)",
        help="Column in the CSV representing grain size (area). Default='Area(px)'."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./EBSD/generate_statistical_inference/segmentation_and_grain_inference",
        help="Directory to save the output distribution plot."
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of bins used to approximate frequency scale. Default=20."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="DPI for saving the figure. Default=600."
    )
    parser.add_argument(
        "--max_x",
        type=float,
        default=2000,
        help="Optional upper limit on the x-axis. Default=2000."
    )
    args = parser.parse_args()

    csv_file_path = args.csv_file_path
    column_name = args.column_name
    output_path = args.output_path
    bin_count = args.bins
    dpi_res = args.dpi
    max_x = args.max_x

    # ============== READ CSV & EXTRACT DATA ==============
    try:
        data = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_file_path}")
        return
    except Exception as e:
        print(f"Error: Unable to read CSV due to: {e}")
        return

    if column_name not in data.columns:
        print(f"Error: The CSV file does not contain a column named '{column_name}'.")
        return

    area_values = data[column_name].dropna().values
    if len(area_values) == 0:
        print("No valid area values found. Check your CSV file and column name.")
        return

    # ============== BASIC STATS ==============
    mean_area = np.mean(area_values)
    median_area = np.median(area_values)
    std_area = np.std(area_values)

    # ============== NORMAL DISTRIBUTION ==============
    # We'll create a normal distribution with the same mean & std as the dataset
    x_min = area_values.min()
    x_max = area_values.max()
    x_range = np.linspace(x_min, x_max, 300)

    # We'll approximate a 'Frequency' curve similarly to the KDE script:
    # 1) Compute normal PDF
    pdf_vals = norm.pdf(x_range, loc=mean_area, scale=std_area)
    # 2) Convert PDF to approximate frequency
    bin_width = (x_max - x_min) / float(bin_count)
    freq_vals = pdf_vals * len(area_values) * bin_width

    # ============== PLOTTING ==============
    plt.style.use("seaborn-v0_8-white")
    matplotlib.rcParams["axes.grid"] = False  # Disable automatic grid

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Plot the normal distribution curve
    ax.plot(x_range, freq_vals, color="navy", linewidth=2.5, label="Normal Distribution")
    ax.fill_between(x_range, freq_vals, color="cornflowerblue", alpha=0.3)

    ax.set_title("Grain Size (Normal Distribution)", fontsize=22, fontweight="bold", pad=15)
    ax.set_xlabel("Area (px)", fontsize=18, labelpad=10)
    ax.set_ylabel("Frequency", fontsize=18, labelpad=10)

    # Make all spines visible
    for spine in ax.spines.values():
        spine.set_visible(True)

    # Enable both major and minor ticks
    from matplotlib.ticker import AutoMinorLocator
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Customize tick parameters
    ax.tick_params(axis="both", which="major", labelsize=14, length=6, width=1.2, direction="in")
    ax.tick_params(axis="both", which="minor", length=3, width=1.0, direction="in")

    # Constrain the x-axis if needed
    ax.set_xlim(left=-8000, right=8000)

    # Stats text box
    text_str = (
        f"Number of grains: {len(area_values)}\n"
        f"Mean area: {mean_area:.2f}\n"
        f"Median area: {median_area:.2f}\n"
        f"Std dev: {std_area:.2f}"
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

    # Build the final output file path
    output_file_path = os.path.join(output_path, "grain_size_distribution_normal.png")

    plt.tight_layout()
    plt.savefig(output_file_path, dpi=dpi_res, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
