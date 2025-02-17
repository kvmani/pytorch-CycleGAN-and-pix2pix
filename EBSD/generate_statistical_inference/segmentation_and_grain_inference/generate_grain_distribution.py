#!/usr/bin/env python3

"""
Generates a distribution plot (KDE curve) of grain sizes (Area in pixels) from a CSV file.
User can provide CSV path, column name, output path, and other parameters via command-line arguments.
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
from scipy.stats import gaussian_kde
import os

def main():
    parser = argparse.ArgumentParser(
        description="Plot a kernel density estimate (KDE) distribution for a given numeric column in a CSV."
    )

    # Arguments
    parser.add_argument(
        "--csv_file_path",
        type=str,
        required=False,
        default="/home/lus04/kvmani/ml_works/kaushal_2025/cyclegan/EBSD/generate_statistical_inference/segmentation_and_grain_inference/output/grains_info/grain_info.csv",
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
        help="Path to save the output distribution plot."
    )
    parser.add_argument(
        "--bw_method",
        type=float,
        default=0.2,
        help="Bandwidth method for gaussian_kde. Smaller values -> narrower curves. Default=0.2."
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of bins to approximate the frequency scale. Default=20."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="DPI for saving the figure. Default=600."
    )

    args = parser.parse_args()

    csv_file_path = args.csv_file_path
    column_name = args.column_name
    output_path = args.output_path
    bw_method = args.bw_method
    bin_count = args.bins
    dpi_res = args.dpi

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

    # ============== PREPARE THE KDE ==============
    kde = gaussian_kde(area_values, bw_method=bw_method)
    x_min, x_max = area_values.min(), area_values.max()
    x_max = min(2000, x_max)
    x_range = np.linspace(x_min, x_max, 300)

    # Convert density -> approximate frequency
    bin_width = (x_max - x_min) / float(bin_count)
    y_values = kde(x_range) * len(area_values) * bin_width

    # ============== PLOTTING ==============
    plt.style.use("seaborn-v0_8-white")
    matplotlib.rcParams["axes.grid"] = False  # Disable any automatic grid

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Plot the KDE curve
    ax.plot(x_range, y_values, color="navy", linewidth=2.5, label="Frequency Curve")
    ax.fill_between(x_range, y_values, color="cornflowerblue", alpha=0.3)

    ax.set_title("Grain Size Distribution (Area in px)", fontsize=22, fontweight="bold", pad=15)
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
    ax.legend(fontsize=14, loc="best", frameon=False)

    output_file_path = os.path.join(output_path, "grain_size_distribution.png")
    plt.tight_layout()
    plt.savefig(output_file_path, dpi=dpi_res, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
