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
# Default settings - some of these may be overridden via command-line arguments.
SETTINGS = {
    "mask_paths": {
        # "artificial_masks": "data/programeData/ebsdMaskFolder",
        # "ebsd_masks": "/mnt/volume/EBSD_ML/EBSD_Mask/ebsd_masks/",
         "ebsd_masks": "/home/lus04/kvmani/ml_works/kaushal_2025/ebsd_masks/",
        # "no_boundary_mask": "/mnt/volume/EBSD_ML/EBSD_Mask/EBSD_mask_NoBoundaries/JPEG",
        # "with_scratches": "/mnt/volume/EBSD_ML/EBSD_Mask/EBSD_mask_With Scratches"
    },
    "output_directory": "output",  # Base output directory (can be overridden via CLI)
    "augmentations": {
        "rotation_angles": [0, 90, 180, 270],
        "flip_modes": [None, 'h', 'v'],
        "mask_count": 109,
        "crop_size": (256, 256),   # Not used to exclude files; just left in SETTINGS for reference
        "crop_overlap": 0.3        # Not used anymore, just left for reference
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
        # This threshold is used for both the "bands ratio" check and the new "Error=0" check in .ctf files.
        "subcrop_fraction_required": 0.85,
        "ang_ci_min": 0.3,
        "ctf_bands_ratio": 0.95,
    },
    "logs_folder": "/home/lus04/kvmani/ml_works/kaushal_2025/outputs/data8.0/only_ctf_files/logs/",
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
# PATH CONFIGURATION
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
        current_dir = os.path.dirname(os.path.abspath(__file__))
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
def build_filename(base_name, crop_idx, rotation_angle, flip_mode, mask_number, extension):
    """
    Constructs a filename that includes crop index, rotation, flip, and mask number.
    Example:
      base_name = 'Hematite_1' produces:
      'Hematite_1_crop0_rotation_90_flip_h_mask_3.ang'
    """
    flip_str = flip_mode if flip_mode is not None else "none"
    return (
        f"{base_name}_crop{crop_idx}_rotation_{rotation_angle}"
        f"_flip_{flip_str}_mask_{mask_number}{extension}"
    )

def check_quality(ebsd, thresholds, logger, label="subcrop"):
    """
    Checks the EBSD data (ebsd._data) for quality based on:
      - If .ang file: checks fraction of CI >= ang_ci_min
      - If .ctf file: 
         1) checks fraction of (Bands / maxBands) >= ctf_bands_ratio
         2) checks fraction of 'Error' == 0
         Both must be >= subcrop_fraction_required to pass.

    Returns:
      (bool is_good, float fraction_good, str criterion_description, float criterion_threshold)
    """
    fraction_required = thresholds.get("subcrop_fraction_required", 0.85)
    ang_ci_min = thresholds.get("ang_ci_min", 0.3)
    ctf_bands_ratio = thresholds.get("ctf_bands_ratio", 0.95)

    if "ang" in ebsd._ebsdFormat.lower():
        criterion_description = "CI"
        criterion_threshold = ang_ci_min
        if "CI" not in ebsd._data.columns:
            logger.warning(f"No 'CI' column found in an .ang {label} check. Marking as not good.")
            return False, 0.0, criterion_description, criterion_threshold

        ci_values = ebsd._data["CI"].to_numpy()
        fraction_good_ci = np.mean(ci_values >= ang_ci_min)

        is_good = fraction_good_ci >= fraction_required
        return is_good, fraction_good_ci, criterion_description, criterion_threshold

    elif "ctf" in ebsd._ebsdFormat.lower():
        # For ctf, we have TWO checks: Bands ratio AND Error=0
        if ("Bands" not in ebsd._data.columns) or ("Error" not in ebsd._data.columns):
            logger.warning(f"Either 'Bands' or 'Error' column missing in .ctf {label} check. Marking as not good.")
            return False, 0.0, "Bands/Error missing", 0.0

        # 1) Bands ratio check
        bands_values = ebsd._data["Bands"].to_numpy()
        max_bands = bands_values.max() if len(bands_values) > 0 else 1.0
        ratio_array = bands_values / max_bands
        fraction_good_bands = np.mean(ratio_array >= ctf_bands_ratio)

        # 2) Error=0 check
        error_values = ebsd._data["Error"].to_numpy()
        fraction_zero_error = np.mean(error_values == 0)

        # Combine both checks with an AND
        is_good = (fraction_good_bands >= fraction_required) and (fraction_zero_error >= fraction_required)

        # We return the minimum of the two as the "fraction_good" for logging convenience
        # or you could store them separately if you prefer. 
        fraction_good_combined = min(fraction_good_bands, fraction_zero_error)

        criterion_description = "Bands ratio & Error=0"
        criterion_threshold = fraction_required

        logger.info(
            f"CTF checks => fraction_good_bands: {fraction_good_bands*100:.2f}%, "
            f"fraction_zero_error: {fraction_zero_error*100:.2f}%, "
            f"threshold: {fraction_required*100:.1f}%"
        )

        return is_good, fraction_good_combined, criterion_description, criterion_threshold

    else:
        logger.warning(f"Quality check not defined for format: {ebsd._ebsdFormat}. Marking as not good.")
        return False, 0.0, "unknown", 0.0

def setup_output_directories(settings, logger):
    """
    Creates and returns a dictionary of output directories for TARGET and SOURCE files,
    each for npy, ang, and tiff file types.
    """
    output_dir = settings["output_directory"]
    dir_names = settings["dir_names"]
    subdirs = settings["subdirs"]

    npy_target_dir = os.path.join(output_dir, subdirs["npy"], dir_names["target"])
    ang_target_dir = os.path.join(output_dir, subdirs["ang"], dir_names["target"])
    tiff_target_dir = os.path.join(output_dir, subdirs["tiff"], dir_names["target"])

    npy_source_dir = os.path.join(output_dir, subdirs["npy"], dir_names["source"])
    ang_source_dir = os.path.join(output_dir, subdirs["ang"], dir_names["source"])
    tiff_source_dir = os.path.join(output_dir, subdirs["tiff"], dir_names["source"])

    directories = [npy_target_dir, ang_target_dir, tiff_target_dir,
                   npy_source_dir, ang_source_dir, tiff_source_dir]
    for d in directories:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Ensured existence of directory: {d}")
    return {
        "npy_target": npy_target_dir,
        "ang_target": ang_target_dir,
        "tiff_target": tiff_target_dir,
        "npy_source": npy_source_dir,
        "ang_source": ang_source_dir,
        "tiff_source": tiff_source_dir,
    }

def apply_mask_augmentation(mask_img_path):
    """
    Applies random augmentation to the given mask image.

    Returns:
        PIL.Image: Augmented PIL image object.
    """
    mask_img = Image.open(mask_img_path)
    # Rotate
    rotation_angle = random.choice([0, 90, 180, 270])
    mask_img = mask_img.rotate(rotation_angle, expand=True)
    # Flip
    flip_mode = random.choice(['horizontal', 'vertical'])
    if flip_mode == 'horizontal':
        mask_img = ImageOps.mirror(mask_img)  # Horizontal flip
    elif flip_mode == 'vertical':
        mask_img = ImageOps.flip(mask_img)    # Vertical flip
    # Random crop
    width, height = mask_img.size
    crop_width = random.randint(int(0.7 * width), width)
    crop_height = random.randint(int(0.7 * height), height)

    x_start = random.randint(0, max(0, width - crop_width))
    y_start = random.randint(0, max(0, height - crop_height))
    x_end = x_start + crop_width
    y_end = y_start + crop_height

    mask_img = mask_img.crop((x_start, y_start, x_end, y_end))
    return mask_img

##############################################################################
# PROCESSING FUNCTION
##############################################################################
def process_single_file(input_file, settings, logger, output_dirs, forceReduceEulerAngles):
    """
    Processes a single EBSD file with only one (subcrop-level) quality check on the entire EBSD data:
      1. Loads the file (no global-level check).
      2. Performs the single (subcrop-level) quality check (CI for .ang, or both Bands ratio & Error=0 for .ctf).
      3. If that check is good:
         - Applies random rotation/flip to EBSD
         - Writes TARGET files (unmasked)
         - Applies random mask
         - Writes SOURCE files (masked)
        If not good:
         - Writes only the unmasked SOURCE files.
    """
    from pycrystallography.ebsd.ebsd import Ebsd

    file_name = os.path.basename(input_file)
    logger.info(f"Loading file: {input_file}")

    ebsd = Ebsd(logger=logger)
    try:
        if input_file.lower().endswith(".ang"):
            ebsd.fromAng(input_file, forceReduceEulerAngles)
        elif input_file.lower().endswith(".ctf"):
            ebsd.fromCtf(input_file, forceReduceEulerAngles)
        else:
            logger.error(f"Unsupported file format for file: {file_name}")
            return
    except Exception as e:
        logger.warning(f"Failed to read {file_name}: {e}")
        return

    # Single (subcrop-level) check performed on the entire EBSD dataset.
    sub_is_good, sub_fraction_good, sub_crit_descr, sub_crit_val = check_quality(
        ebsd, settings["thresholds"], logger, label="subcrop"
    )
    logger.info(
        f"For '{file_name}': {sub_fraction_good*100:.2f}% of rows satisfied [{sub_crit_descr}] "
        f"({sub_crit_val}). Threshold: {settings['thresholds'].get('subcrop_fraction_required', 0.85)*100:.1f}%"
    )

    base_name, ext = os.path.splitext(file_name)

    if sub_is_good:
        logger.info(f"Quality check passed for '{file_name}'. Proceeding with rotation + masking.")
        rotation_angle = random.choice(settings["augmentations"]["rotation_angles"])
        flip_mode = random.choice(settings["augmentations"]["flip_modes"])
        mask_number = random.randint(1, settings["augmentations"]["mask_count"])
        logger.info(
            f"Chosen augmentations for '{file_name}': rotation={rotation_angle}, flip={flip_mode}, mask={mask_number}"
        )

        # Rotate/flip entire EBSD
        ebsd.rotateAndFlipData(flipMode=flip_mode, rotate=rotation_angle)
        logger.info("Applied rotation/flip to EBSD data.")

        # Build filenames for TARGET + SOURCE
        npy_filename  = build_filename(base_name, 0, rotation_angle, flip_mode, mask_number, ".npy")
        ang_filename  = build_filename(base_name, 0, rotation_angle, flip_mode, mask_number, ext)
        tiff_filename = build_filename(base_name, 0, rotation_angle, flip_mode, mask_number, ".tiff")

        npy_target_path  = os.path.join(output_dirs["npy_target"],  npy_filename)
        ang_target_path  = os.path.join(output_dirs["ang_target"],  ang_filename)
        tiff_target_path = os.path.join(output_dirs["tiff_target"], tiff_filename)

        npy_source_path  = os.path.join(output_dirs["npy_source"],  npy_filename)
        ang_source_path  = os.path.join(output_dirs["ang_source"],  ang_filename)
        tiff_source_path = os.path.join(output_dirs["tiff_source"], tiff_filename)

        # Write TARGET (unmasked) files
        if not settings["debug"]:
            ebsd.writeNpyFile(pathName=npy_target_path)
            if input_file.lower().endswith(".ang"):
                ebsd.writeAng(pathName=ang_target_path)
            elif input_file.lower().endswith(".ctf"):
                ebsd.writeCtf(pathName=ang_target_path)
            ebsd.writeEulerAsPng(pathName=tiff_target_path, showMap=False)
        logger.info("TARGET (unmasked) files written.")

        # Masking and write SOURCE (masked)
        mask_image_path = os.path.join(settings["mask_paths"]["ebsd_masks"], f"{mask_number}.png")
        augmented_mask = apply_mask_augmentation(mask_image_path)
        ebsd.applyMask(augmented_mask, displayImage=False)
        logger.info("Mask applied to EBSD data.")

        if not settings["debug"]:
            ebsd.writeNpyFile(pathName=npy_source_path)
            if input_file.lower().endswith(".ang"):
                ebsd.writeAng(pathName=ang_source_path)
            elif input_file.lower().endswith(".ctf"):
                ebsd.writeCtf(pathName=ang_source_path)
            ebsd.writeEulerAsPng(pathName=tiff_source_path, showMap=False)
        logger.info("SOURCE (masked) files written.")

    else:
        # If check fails, write unmasked SOURCE only
        logger.info(f"Quality check FAILED for '{file_name}'. Writing unmasked SOURCE only.")
        npy_unmasked_filename  = f"{base_name}_unmasked.npy"
        ang_unmasked_filename  = f"{base_name}_unmasked{ext}"
        tiff_unmasked_filename = f"{base_name}_unmasked.tiff"

        npy_source_path  = os.path.join(output_dirs["npy_source"],  npy_unmasked_filename)
        ang_source_path  = os.path.join(output_dirs["ang_source"],  ang_unmasked_filename)
        tiff_source_path = os.path.join(output_dirs["tiff_source"], tiff_unmasked_filename)

        if not settings["debug"]:
            ebsd.writeNpyFile(pathName=npy_source_path)
            if input_file.lower().endswith(".ang"):
                ebsd.writeAng(pathName=ang_source_path)
            elif input_file.lower().endswith(".ctf"):
                ebsd.writeCtf(pathName=ang_source_path)
            ebsd.writeEulerAsPng(pathName=tiff_source_path, showMap=False)
        logger.info("Unmasked SOURCE files written.")

    logger.info("Single file processing complete.")

##############################################################################
# MAIN
##############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single EBSD file with only one quality check on the entire dataset.")
    parser.add_argument("--input_file", required=True, help="Path to the input .ang or .ctf file")
    parser.add_argument("--output_dir", required=False, help="Path to the output directory (default from SETTINGS)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (no files are written)")
    parser.add_argument("--forceReduceEulerAngles", action="store_true", help="force the isEulerAngleReduced variable to be true")
    args = parser.parse_args()

    # Update SETTINGS from command-line arguments
    if args.output_dir:
        SETTINGS["output_directory"] = args.output_dir
    if args.debug:
        SETTINGS["debug"] = True

    # Setup logger and fix import paths if needed
    logger = configure_logger(SETTINGS, args.input_file)
    configure_path(logger)

    # Create all required output directories
    output_dirs = setup_output_directories(SETTINGS, logger)

    # Process the file (no global checks, only "subcrop"-style check on entire data)
    process_single_file(args.input_file, SETTINGS, logger, output_dirs, args.forceReduceEulerAngles)
