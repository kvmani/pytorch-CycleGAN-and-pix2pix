#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import datetime
import random
import numpy as np
import copy
from tqdm import tqdm
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

##############################################################################
# SETTINGS
##############################################################################
SETTINGS = {
    "mask_paths": {
        "ebsd_masks": "/home/lus04/kvmani/ml_works/kaushal_2025/ebsd_masks/",
    },
    "output_directory": "output",  # Base output directory (can be overridden via CLI)
    "augmentations": {
        "rotation_angles": [0, 90, 180, 270],  # We'll specifically use these four
        "flip_modes": [None],                 # Force flip to None so logs remain consistent but no flipping is performed
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
        "file_fraction_required": 0.5,     # 50% for file-level check
        "subcrop_fraction_required": 0.85, # 85% for subcrop-level check
        "ang_ci_min": 0.3,
        "ctf_bands_ratio": 0.95,
    },
    "logs_folder": "/home/lus04/kvmani/ml_works/kaushal_2025/outputs/data11.0/logs/",
    "debug": False,  # Set to True for debugging (files will not be written)
}

##############################################################################
# LOGGER CONFIGURATION
##############################################################################
def configure_logger(settings, input_file):
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
    Returns filename like:  baseName_crop{subcropID}_rotation_{rotationAngle}_mask_{maskNumber}.extension
    (Removed flip string from filename per new requirement.)
    """
    return (
        f"{base_name}_crop{crop_idx}_rotation_{rotation_angle}_mask_{mask_number}{extension}"
    )

def check_quality(ebsd, thresholds, logger, label="file"):
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

def setup_output_directories(settings, logger):
    output_dir = settings["output_directory"]
    dir_names = settings["dir_names"]
    subdirs = settings["subdirs"]

    npy_target_dir = os.path.join(output_dir, subdirs["npy"], dir_names["target"])
    ang_target_dir = os.path.join(output_dir, subdirs["ang"], dir_names["target"])
    tiff_target_dir = os.path.join(output_dir, subdirs["tiff"], dir_names["target"])

    npy_source_dir = os.path.join(output_dir, subdirs["npy"], dir_names["source"])
    ang_source_dir = os.path.join(output_dir, subdirs["ang"], dir_names["source"])
    tiff_source_dir = os.path.join(output_dir, subdirs["tiff"], dir_names["source"])

    directories = [
        npy_target_dir, ang_target_dir, tiff_target_dir,
        npy_source_dir, ang_source_dir, tiff_source_dir
    ]
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

def apply_mask_augmentation(mask_img_path, cropped_ebsd):
    """
    Applies random rotation & flip to the mask image, plus a random partial crop of the mask.
    """
    mask_img = Image.open(mask_img_path)

    # 1. Random rotation
    rotation_angle = random.choice([0, 90, 180, 270])
    mask_img = mask_img.rotate(rotation_angle, expand=True)

    # 2. Random flip
    flip_mode = random.choice(['horizontal', 'vertical'])
    if flip_mode == 'horizontal':
        mask_img = ImageOps.mirror(mask_img)
    elif flip_mode == 'vertical':
        mask_img = ImageOps.flip(mask_img)

    # 3. Random partial crop
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
    from pycrystallography.ebsd.ebsd import Ebsd

    logger.info(f"running in debug mode = {settings['debug']}")
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

    # 1) Global (file-level) quality check
    is_good, fraction_good, crit_descr, crit_value = check_quality(ebsd, settings["thresholds"], logger, label="file")
    action_str = "using" if is_good else "skipping"
    logger.info(
        f"[{action_str}] file '{file_name}' because {fraction_good*100:.2f}% of rows satisfied the [{crit_descr}] criterion "
        f"({crit_value}) vs. threshold {settings['thresholds'].get('file_fraction_required', 0.5)*100:.1f}%"
    )
    if not is_good:
        logger.info(f"File '{file_name}' did not pass the quality check. Exiting.")
        return

    # 2) Dimensions check
    crop_size = settings["augmentations"]["crop_size"]
    if (ebsd.nYPixcels < crop_size[0]) or (ebsd.nXPixcels < crop_size[1]):
        logger.info(
            f"File {file_name} is smaller than the crop dimension {crop_size}: {ebsd.nXPixcels}x{ebsd.nYPixcels}. Exiting."
        )
        return
    
    ebsd.reduceEulerAngelsToFundamentalZone_vectorized()

    # 3) Crop the EBSD data
    crop_overlap = settings["augmentations"]["crop_overlap"]
    cropped_ebsd_list = ebsd.crop(crop_size=crop_size, overlap=crop_overlap)
    if not cropped_ebsd_list:
        logger.info(f"No valid crops were generated from {file_name}.")
        return

    logger.info(f"Generated {len(cropped_ebsd_list)} crops from file {file_name}")

    base_name, ext = os.path.splitext(file_name)

    # 4) Process each subcrop
    for crop_idx, cropped_ebsd in enumerate(cropped_ebsd_list):
        # Subcrop-level quality check
        sub_is_good, sub_fraction_good, sub_crit_descr, sub_crit_val = check_quality(
            cropped_ebsd, settings["thresholds"], logger, label="subcrop"
        )
        sub_action_str = "using" if sub_is_good else "skipping"
        logger.info(
            f"[{sub_action_str}] subcrop {crop_idx} because {sub_fraction_good*100:.2f}% of rows satisfied the [{sub_crit_descr}] "
            f"criterion ({sub_crit_val}) vs. threshold {settings['thresholds'].get('subcrop_fraction_required', 0.9)*100:.1f}%"
        )

        # If subcrop fails, move on (no unmasked SOURCE files)
        if not sub_is_good:
            continue

        # If subcrop is good => apply *four rotations* (0, 90, 180, 270)
        for rotation_angle in [0, 90, 180, 270]:
            # We keep the existing "flip" logs but force flip_mode = None
            flip_mode = None
            mask_number = random.randint(1, settings["augmentations"]["mask_count"])

            logger.info(
                f"For subcrop {crop_idx}: rotation={rotation_angle}, flip={flip_mode}, mask={mask_number}"
            )

            # Make a copy of the subcrop EBSD data so each rotation starts fresh
            sub_ebsd_copy = copy.deepcopy(cropped_ebsd)

            # Rotate (no flipping)
            sub_ebsd_copy.rotateAndFlipData(flipMode=flip_mode, rotate=rotation_angle)
            logger.info(f"Applied rotation/flip on subcrop {crop_idx}")

            # Build filenames for TARGET & SOURCE
            npy_filename  = build_filename(base_name, crop_idx, rotation_angle, flip_mode, mask_number, ".npy")
            ang_filename  = build_filename(base_name, crop_idx, rotation_angle, flip_mode, mask_number, ext)
            tiff_filename = build_filename(base_name, crop_idx, rotation_angle, flip_mode, mask_number, ".tiff")

            # Paths for TARGET files
            npy_target_path  = os.path.join(output_dirs["npy_target"],  npy_filename)
            ang_target_path  = os.path.join(output_dirs["ang_target"],  ang_filename)
            tiff_target_path = os.path.join(output_dirs["tiff_target"], tiff_filename)

            # Paths for SOURCE files
            npy_source_path  = os.path.join(output_dirs["npy_source"],  npy_filename)
            ang_source_path  = os.path.join(output_dirs["ang_source"],  ang_filename)
            tiff_source_path = os.path.join(output_dirs["tiff_source"], tiff_filename)

            # ---- Write TARGET (unmasked) files ----
            if not settings["debug"]:
                sub_ebsd_copy.writeNpyFile(pathName=npy_target_path)
                if input_file.lower().endswith(".ang"):
                    sub_ebsd_copy.writeAng(pathName=ang_target_path)
                else:
                    sub_ebsd_copy.writeCtf(pathName=ang_target_path)
                sub_ebsd_copy.writeEulerAsPng(pathName=tiff_target_path, showMap=False)
            logger.info(f"Subcrop {crop_idx} => TARGET files written")

            # ---- Apply Mask ----
            mask_image_path = os.path.join(settings["mask_paths"]["ebsd_masks"], f"{mask_number}.png")
            augmented_mask = apply_mask_augmentation(mask_image_path, sub_ebsd_copy)
            sub_ebsd_copy.applyMask(augmented_mask, displayImage=False)
            logger.info(f"Subcrop {crop_idx} => mask applied")

            # ---- Write SOURCE (masked) files ----
            if not settings["debug"]:
                sub_ebsd_copy.writeNpyFile(pathName=npy_source_path)
                if input_file.lower().endswith(".ang"):
                    sub_ebsd_copy.writeAng(pathName=ang_source_path)
                else:
                    sub_ebsd_copy.writeCtf(pathName=ang_source_path)
                sub_ebsd_copy.writeEulerAsPng(pathName=tiff_source_path, showMap=False)
            logger.info(f"Subcrop {crop_idx} => SOURCE (masked) files written")

    logger.info("Single file processing complete.")

##############################################################################
# MAIN
##############################################################################
if __name__ == "__main__":

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

    # Setup logger and adjust the system path for pycrystallography
    logger = configure_logger(SETTINGS, args.input_file)
    configure_path(logger)

    # Setup output directories for TARGET and SOURCE files
    output_dirs = setup_output_directories(SETTINGS, logger)

    # Process the file
    process_single_file(args.input_file, SETTINGS, logger, output_dirs, args.forceReduceEulerAngles)
