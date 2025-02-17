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
        "crop_size": (512, 512),
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
    "logs_folder": "/home/lus04/kvmani/ml_works/kaushal_2025/outputs/data9.0/logs/",  # Change as needed #"/home/lus04/kvmani/ml_works/kaushal_2025/outputs/data1.0/logs/"
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


def apply_mask_augmentation(mask_img_path, cropped_ebsd):
    """
    Applies augmentation to the given mask image.

    Args:
        mask_img_path (str): Path to the mask image file.

    Returns:
        np.ndarray: Augmented image as a numpy array.
    """
    # Step 2: Read the image from the path
    mask_img = Image.open(mask_img_path)

    # Step 3: Randomly select a rotation angle and apply it
    rotation_angle = random.choice([0, 90, 180, 270])
    mask_img = mask_img.rotate(rotation_angle, expand=True)

    # Step 4: Randomly select a flip mode and apply it
    flip_mode = random.choice(['horizontal', 'vertical'])
    if flip_mode == 'horizontal':
        mask_img = ImageOps.mirror(mask_img)  # Horizontal flip
    elif flip_mode == 'vertical':
        mask_img = ImageOps.flip(mask_img)  # Vertical flip

    # Step 5: Randomly select a crop
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
# PROCESSING FUNCTIONS
##############################################################################
def process_single_file(input_file, settings, logger, output_dirs, forceReduceEulerAngles):
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
            ebsd.fromAng(input_file, forceReduceEulerAngles)
        elif input_file.lower().endswith(".ctf"):
            ebsd.fromCtf(input_file, forceReduceEulerAngles)
        else:
            logger.error(f"Unsupported file format for file: {file_name}")
            return
    except Exception as e:
        logger.warning(f"Failed to read {file_name}: {e}")
        return

    # Global (file-level) quality check
    is_good, fraction_good, crit_descr, crit_value = check_quality(ebsd, settings["thresholds"], logger, label="file")
    action_str = "using" if is_good else "skipping"
    logger.info(
        f"[{action_str}] file '{file_name}' because {fraction_good*100:.2f}% of rows satisfied the [{crit_descr}] criterion "
        f"({crit_value}) vs. threshold {settings['thresholds'].get('file_fraction_required', 0.5)*100:.1f}%"
    )
    if not is_good:
        logger.info(f"File '{file_name}' did not pass the quality check. Exiting.")
        return
        
    crop_size = settings["augmentations"]["crop_size"]
    if((ebsd.nYPixcels < crop_size[0]) or (ebsd.nXPixcels < crop_size[1])):
        logger.info(f"File {file_name} is less than the crop dimension its of size {ebsd.nXPixcels}x{ebsd.nYPixcels}. Exiting.")
        return
        
    # ebsd.reduceEulerAngelsToFundamentalZone()
    #logger.info(f"applied reduceEulerAngelsToFundamentalZone to {file_name}")
    

    # Crop the EBSD data
    
    crop_overlap = settings["augmentations"]["crop_overlap"]
    cropped_ebsd_list = ebsd.crop(crop_size=crop_size, overlap=crop_overlap)
    if not cropped_ebsd_list:
        logger.info(f"No valid crops were generated from {file_name}.")
        return
    logger.info(f"Generated {len(cropped_ebsd_list)} crops from file {file_name}")

    base_name, ext = os.path.splitext(file_name)
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

        if sub_is_good:
            # Apply random augmentations
            rotation_angle = random.choice(settings["augmentations"]["rotation_angles"])
            flip_mode = random.choice(settings["augmentations"]["flip_modes"])
            # mask_number = random.randint(1, settings["augmentations"]["mask_count"])
            mask_number = random.randint(1, settings["augmentations"]["mask_count"])
            logger.info(
                f"For subcrop {crop_idx}: rotation={rotation_angle}, flip={flip_mode}, mask={mask_number}"
            )

            # (Optional) Reduce Euler angles to the fundamental zone
            # cropped_ebsd.reduceEulerAngelsToFundamentalZone()
            # logger.info(f"Applied reduceEulerAngelsToFundamentalZone on subcrop {crop_idx}")

            cropped_ebsd.rotateAndFlipData(flipMode=flip_mode, rotate=rotation_angle)
            logger.info(f"Applied rotation/flip on subcrop {crop_idx}")

            # Build filenames for TARGET and SOURCE files
            npy_filename  = build_filename(base_name, crop_idx, rotation_angle, flip_mode, mask_number, ".npy")
            ang_filename  = build_filename(base_name, crop_idx, rotation_angle, flip_mode, mask_number, ext)
            tiff_filename = build_filename(base_name, crop_idx, rotation_angle, flip_mode, mask_number, ".tiff")

            # Paths for TARGET files (unmasked)
            npy_target_path  = os.path.join(output_dirs["npy_target"], npy_filename)
            ang_target_path  = os.path.join(output_dirs["ang_target"], ang_filename)
            tiff_target_path = os.path.join(output_dirs["tiff_target"], tiff_filename)
            # Paths for SOURCE files (masked)
            npy_source_path  = os.path.join(output_dirs["npy_source"], npy_filename)
            ang_source_path  = os.path.join(output_dirs["ang_source"], ang_filename)
            tiff_source_path = os.path.join(output_dirs["tiff_source"], tiff_filename)

            # Write TARGET files (if not in debug mode)
            if not settings["debug"]:
                cropped_ebsd.writeNpyFile(pathName=npy_target_path)
                if input_file.lower().endswith(".ang"):
                    cropped_ebsd.writeAng(pathName=ang_target_path)
                elif input_file.lower().endswith(".ctf"):
                    cropped_ebsd.writeCtf(pathName=ang_target_path)
                cropped_ebsd.writeEulerAsPng(pathName=tiff_target_path, showMap=False)
            logger.info(f"Subcrop {crop_idx} => TARGET files written")

            # Apply mask then write SOURCE files (masked)
            mask_image_path = os.path.join(settings["mask_paths"]["ebsd_masks"], f"{mask_number}.png")
            augmented_mask = apply_mask_augmentation(mask_image_path, cropped_ebsd)
            cropped_ebsd.applyMask(augmented_mask, displayImage=False)
            logger.info(f"Subcrop {crop_idx} => mask applied")
            if not settings["debug"]:
                cropped_ebsd.writeNpyFile(pathName=npy_source_path)
                if input_file.lower().endswith(".ang"):
                    cropped_ebsd.writeAng(pathName=ang_source_path)
                elif input_file.lower().endswith(".ctf"):
                    cropped_ebsd.writeCtf(pathName=ang_source_path)                
                cropped_ebsd.writeEulerAsPng(pathName=tiff_source_path, showMap=False)
            logger.info(f"Subcrop {crop_idx} => SOURCE (masked) files written")
        else:
            # For subcrops that fail quality, write only unmasked SOURCE files
            # cropped_ebsd.reduceEulerAngelsToFundamentalZone()
            # logger.info(f"Applied reduceEulerAngelsToFundamentalZone on subcrop {crop_idx}")
            npy_unmasked_filename = f"{base_name}_subcrop{crop_idx}_unmasked.npy"
            ang_unmasked_filename = f"{base_name}_subcrop{crop_idx}_unmasked{ext}"
            tiff_unmasked_filename = f"{base_name}_subcrop{crop_idx}_unmasked.tiff"
            npy_source_path  = os.path.join(output_dirs["npy_source"], npy_unmasked_filename)
            ang_source_path  = os.path.join(output_dirs["ang_source"], ang_unmasked_filename)
            tiff_source_path = os.path.join(output_dirs["tiff_source"], tiff_unmasked_filename)
            if not settings["debug"]:
                cropped_ebsd.writeNpyFile(pathName=npy_source_path)
                if input_file.lower().endswith(".ang"):
                    cropped_ebsd.writeAng(pathName=ang_source_path)
                elif input_file.lower().endswith(".ctf"):
                    cropped_ebsd.writeCtf(pathName=ang_source_path)                  
                cropped_ebsd.writeEulerAsPng(pathName=tiff_source_path, showMap=False)
            logger.info(f"Subcrop {crop_idx} => Unmasked SOURCE files written")

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
    
    # Setup logger and adjust the system path for pycrystallography
    logger = configure_logger(SETTINGS, args.input_file)
    configure_path(logger)
    # Setup output directories for TARGET and SOURCE files
    output_dirs = setup_output_directories(SETTINGS, logger)

    # Process the provided single file
    process_single_file(args.input_file, SETTINGS, logger, output_dirs, True)