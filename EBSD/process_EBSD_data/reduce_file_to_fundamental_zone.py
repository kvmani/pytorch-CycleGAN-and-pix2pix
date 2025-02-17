#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import datetime
import random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

##############################################################################
# SETTINGS
##############################################################################
# Default settings  some of these may be overridden via command-line arguments.
SETTINGS = {
    "mask_paths": {
        #"artificial_masks": "data/programeData/ebsdMaskFolder",
        # "ebsd_masks": "/mnt/volume/EBSD_ML/EBSD_Mask/ebsd_masks/",
         "ebsd_masks": "/home/lus04/kvmani/ml_works/kaushal_2025/ebsd_masks/",
        #"no_boundary_mask": "/mnt/volume/EBSD_ML/EBSD_Mask/EBSD_mask_NoBoundaries/JPEG",
        #"with_scratches": "/mnt/volume/EBSD_ML/EBSD_Mask/EBSD_mask_With Scratches"
    },
    "output_directory": "output",  # Base output directory (can be overridden via CLI)
    "augmentations": {
        "rotation_angles": [0, 90, 180, 270],
        "flip_modes": [None, 'h', 'v'],
        "mask_count": 109,
        "crop_size": (256, 256),
        "crop_overlap": 0.3
    },
    "dir_names": {
        "target": "B",
        "source": "A"
    },
    "subdirs": {
        "npy": "npy",
        "ang": "ang",
        "tiff": "tiff"
    },
    "thresholds": {
        "file_fraction_required": 0.5,    # 50% for file-level check
        "subcrop_fraction_required": 0.85,   # 85% for subcrop-level check
        "ang_ci_min": 0.3,
        "ctf_bands_ratio": 0.95,
    },
    "logs_folder": "/home/lus04/kvmani/ml_works/kaushal_2025/ctf_ang_files_reduced/logs/",  # Change as needed #"/home/lus04/kvmani/ml_works/kaushal_2025/outputs/data1.0/logs/"
    # "logs_folder": "logs",
    "debug": False,  # Set to True for debugging (files will not be written)
}

##############################################################################
# LOGGER CONFIGURATION
##############################################################################
def configure_logger(settings, input_file):
    """
    Creates a logger that writes to console and a file named with the current date/time.
    The log file is placed in settings["logs_folder"].
    """
    logs_folder = settings.get("logs_folder", "logs")
    os.makedirs(logs_folder, exist_ok=True)
    file_name = os.path.basename(input_file)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(logs_folder, f"{file_name}_{timestamp}.log")

    logger = logging.getLogger("EBSD_Logger")
    logger.setLevel(logging.INFO)

    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    logger.info(f"Initialized logger. Writing logs to: {log_filename}")
    return logger

##############################################################################
# PATH CONFIGURATIONcd
##############################################################################
def configure_path(logger):
    """
    Ensures that the pycrystallography package can be imported.
    If the module is not found, it adds parent directories to sys.path.
    """
    try:
        from pycrystallography.ebsd.ebsd import Ebsd
    except ModuleNotFoundError:
        logger.warning("pycrystallography.ebsd module not found. Attempting to add parent directories to sys.path.")
        # Get the directory of the current file.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Add the parent and grandparent directories to sys.path.
        parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
        grandparent_dir = os.path.abspath(os.path.join(current_dir, "../.."))
        sys.path.insert(0, parent_dir)
        sys.path.insert(0, grandparent_dir)
        logger.info(f"Added {parent_dir} and {grandparent_dir} to sys.path.")
        try:
            from pycrystallography.ebsd.ebsd import Ebsd
        except ModuleNotFoundError as e:
            logger.error("pycrystallography module still not found. Please install it or adjust sys.path accordingly.")
            raise e

##############################################################################
# HELPER FUNCTIONS
##############################################################################


def check_quality(ebsd, thresholds, logger, label="file"):
    """
    Checks the EBSD data (ebsd._data) for quality.
    For .ang files, a fraction of rows must have CI >= ang_ci_min.
    For .ctf files, a fraction of rows must have (Bands / maxBands) >= ctf_bands_ratio.

    Returns:
      (bool is_good, float fraction_good, str criterion_description, float criterion_threshold)
    """
    if label == "file":
        fraction_required = thresholds.get("file_fraction_required", 0.5)
    else:
        fraction_required = thresholds.get("subcrop_fraction_required", 0.9)

    ang_ci_min = thresholds.get("ang_ci_min", 0.3)
    ctf_bands_ratio = thresholds.get("ctf_bands_ratio", 0.8)

    if "ang" in ebsd._ebsdFormat:
        criterion_description = "CI"
        criterion_threshold = ang_ci_min
        if "CI" not in ebsd._data.columns:
            logger.warning(f"No 'CI' column found in an .ang {label}. Marking as not good.")
            return False, 0.0, criterion_description, criterion_threshold
        ci_values = ebsd._data["CI"].to_numpy()
        fraction_good = np.mean(ci_values >= ang_ci_min)
        return fraction_good >= fraction_required, fraction_good, criterion_description, criterion_threshold

    elif "ctf" in ebsd._ebsdFormat:
        criterion_description = "Bands ratio"
        criterion_threshold = ctf_bands_ratio
        if "Bands" not in ebsd._data.columns:
            logger.warning(f"No 'Bands' column found in a .ctf {label}. Marking as not good.")
            return False, 0.0, criterion_description, criterion_threshold
        bands_values = ebsd._data["Bands"].to_numpy()
        max_bands = bands_values.max() if len(bands_values) > 0 else 1.0
        ratio_array = bands_values / max_bands
        fraction_good = np.mean(ratio_array >= ctf_bands_ratio)
        return fraction_good >= fraction_required, fraction_good, criterion_description, criterion_threshold

    else:
        logger.warning(f"Quality check not defined for format: {ebsd._ebsdFormat}. Marking as not good.")
        return False, 0.0, "unknown", 0.0



##############################################################################
# PROCESSING FUNCTIONS
##############################################################################
def process_single_file(input_file, settings, logger):
    """
    Processes a single EBSD file:
      1. Loads the file.
      2. Performs a global (file-level) quality check.
      3. If accepted, crops the data into sub-EBSD objects.
      4. For each crop:
           - Performs a subcrop quality check.
           - If accepted, applies random augmentation (rotation, flip) and selects a random mask.
             Then writes TARGET (unmasked) files, applies mask, and writes SOURCE (masked) files.
           - If not accepted, writes only unmasked SOURCE files.
    """
    from pycrystallography.ebsd.ebsd import Ebsd
    logger.info(f"running in debug mode = {settings['debug']}")
    file_name = os.path.basename(input_file)
    logger.info(f"Loading file: {input_file}")

    ebsd = Ebsd(logger=logger)
    try:
        if input_file.lower().endswith(".ang"):
            ebsd.fromAng(input_file)
        elif input_file.lower().endswith(".ctf"):
            ebsd.fromCtf(input_file)
        else:
            logger.error(f"Unsupported file format for file: {file_name}")
            return
    except Exception as e:
        logger.warning(f"Failed to read {file_name}: {e}")
        return

    # Global (file-level) quality check
    ebsd.reduceEulerAngelsToFundamentalZone()
    logger.info(f"Applied reduceEulerAngelsToFundamentalZone method to {file_name}.")

    output = f"/home/lus04/kvmani/ml_works/kaushal_2025/ctf_ang_files_reduced/{file_name}"
    if input_file.lower().endswith(".ang"):
        ebsd.writeAng(output)
    elif input_file.lower().endswith(".ctf"):
        ebsd.writeCtf(output)
    else:
        logger.error(f"Unsupported file format for file: {file_name}")
        return
    
    logger.info("Single file processing complete.")

##############################################################################
# MAIN
##############################################################################
if __name__ == "__main__":

    #find "/home/lus04/kvmani/ml_works/kaushal_2025/ctf_ang_files/" -type f \( -name "*.ang" -o -name "*.ctf" \) | xargs -P 16 -I {} python process_ebsd_file.py --input_file {} --output_dir "/home/lus04/kvmani/ml_works/kaushal_2025/outputs/data1.0/"
    parser = argparse.ArgumentParser(description="Process a single EBSD file.")
    parser.add_argument("--input_file", required=True, help="Path to the input .ang or .ctf file")
    parser.add_argument("--output_dir", required=False, help="Path to the output directory (default from SETTINGS)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (no files are written)")
    parser.add_argument("--forceReduceEulerAngles", action="store_true", help="force the isEulerAngleReduced variable to be true")
    args = parser.parse_args()

    # Update SETTINGS based on command-line arguments
    if args.output_dir:
        SETTINGS["output_directory"] = args.output_dir
    if args.debug:
        SETTINGS["debug"] = True
    # args.input_file = r"/home/lus04/kvmani/ml_works/kaushal_2025/MMD_ctf_files_reduced/Alloy625 Thal 2020 Sample1-Tube 70-remeasured Site 4 Map Data 150_14594.ctf"
    # args.input_file = r"/home/lus04/kvmani/ml_works/kaushal_2025/ctf_ang_files/shahshank_10-400rpm-meld B boom dot near interface.ang"
    # Setup logger and adjust the system path for pycrystallography
    logger = configure_logger(SETTINGS, args.input_file)
    configure_path(logger)


    # Process the provided single file
    process_single_file(args.input_file, SETTINGS, logger)