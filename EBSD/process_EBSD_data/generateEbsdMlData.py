import os
import logging
import datetime
import random
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

##############################################################################
# SETTINGS
##############################################################################
SETTINGS = {
    "input_folder": "/mnt/volume/EBSD_ML/ctf_ang_files/",  # Folder containing .ang and/or .ctf files 
    "mask_paths": {
        "artificial_masks": "data/programeData/ebsdMaskFolder",
        "ebsd_masks": "/mnt/volume/EBSD_ML/EBSD_Mask/ebsd_masks",
        "no_boundary_mask": "/mnt/volume/EBSD_ML/EBSD_Mask/EBSD_mask_NoBoundaries/JPEG",
        "with_scratches": "/mnt/volume/EBSD_ML/EBSD_Mask/EBSD_mask_With Scratches"
    },
    "output_directory": "output",  # Base directory to save processed files
    "augmentations": {
        "rotation_angles": [0, 90, 180, 270],
        "flip_modes": [None, 'h', 'v'],
        "mask_count": 25,
        "crop_size": (256, 256),
        "crop_overlap": 0.3
    },
    "dir_names": {
        "target": "target",
        "source": "source"
    },
    "subdirs": {
        "npy": "npy",
        "ang": "ang",
        "tiff": "tiff"
    },
    "thresholds": {
        # Now we have two separate fraction thresholds:
        "file_fraction_required": 0.5,      # 50% for file-level check
        "subcrop_fraction_required": 0.9,   # 90% for subcrop-level check

        "ang_ci_min": 0.3,         # .ang => CI >= 0.3
        "ctf_bands_ratio": 0.8,    # .ctf => (Bands / maxBands) >= 0.8
    },
    "logs_folder": "/mnt/volume/EBSD_ML/tmp/logs/",        # Folder to store log files
    "debug": True, # Debug to run the script without outputs
}


##############################################################################
# LOGGER CONFIGURATION
##############################################################################
def configure_logger(settings):
    """
    Creates a logger that logs to both console and a file named with the current date/time.
    The file is placed in settings["logs_folder"].
    """
    logs_folder = settings.get("logs_folder", "logs")
    os.makedirs(logs_folder, exist_ok=True)

    # Generate a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(logs_folder, f"{timestamp}.log")

    # Create a logger (name it whatever you like, e.g. 'EBSD_Logger')
    logger = logging.getLogger("EBSD_Logger")
    logger.setLevel(logging.INFO)  # or DEBUG, etc.

    # Create formatters
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # FileHandler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # Also log to console (StreamHandler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    logger.info(f"Initialized logger. Writing logs to: {log_filename}")
    return logger


##############################################################################
# HELPER FUNCTIONS
##############################################################################
def configure_path(logger):
    """
    Configures sys.path to locate the pycrystallography package if not installed globally.
    """
    try:
        from pycrystallography.core.orientation import Orientation
    except ImportError:
        logger.warning("Unable to find the pycrystallography package. Adjusting system path.")
        sys.path.insert(0, os.path.abspath('.'))
        sys.path.insert(0, os.path.dirname('..'))
        sys.path.insert(0, os.path.dirname('../../pycrystallography'))
        sys.path.insert(0, os.path.dirname('../../..'))
        for path in sys.path:
            logger.debug(f"Updated Path: {path}")
        from pycrystallography.ebsd.ebsd import Ebsd


def build_filename(base_name, crop_idx, rotation_angle, flip_mode, mask_number, extension):
    """
    Constructs the new file name, incorporating crop index, rotation, flip, and mask number.

    Example:
      base_name = 'Hematite_1'
      -> 'Hematite_1_crop0_rotation_90_flip_h_mask_3.ang'
    """
    flip_str = flip_mode if flip_mode is not None else "none"
    return (
        f"{base_name}_crop{crop_idx}_rotation_{rotation_angle}"
        f"_flip_{flip_str}_mask_{mask_number}{extension}"
    )


def check_quality(ebsd, thresholds, logger, label="file"):
    """
    Checks the EBSD data 'ebsd._data' to see if it meets the quality threshold.
    The fraction required depends on whether we're checking a 'file' or a 'subcrop'.

    For .ang:  fraction of rows must have CI >= ang_ci_min
    For .ctf:  fraction of rows must have (Bands / maxBands) >= ctf_bands_ratio

    Returns:
      (bool is_good, float fraction_good, str criterion_description, float criterion_threshold)
    """
    # 1) Decide fraction_required based on label
    if label == "file":
        fraction_required = thresholds.get("file_fraction_required", 0.5)
    else:  # "subcrop"
        fraction_required = thresholds.get("subcrop_fraction_required", 0.9)

    ang_ci_min = thresholds.get("ang_ci_min", 0.3)
    ctf_bands_ratio = thresholds.get("ctf_bands_ratio", 0.8)

    if "ang" in ebsd._ebsdFormat:
        # Check CI
        criterion_description = "CI"
        criterion_threshold = ang_ci_min
        if "CI" not in ebsd._data.columns:
            logger.warning(f"No 'CI' column found in an .ang {label}. Marking as not good.")
            return False, 0.0, criterion_description, criterion_threshold
        ci_values = ebsd._data["CI"].to_numpy()
        fraction_good = np.mean(ci_values >= ang_ci_min)
        is_good = fraction_good >= fraction_required
        return is_good, fraction_good, criterion_description, criterion_threshold

    elif "ctf" in ebsd._ebsdFormat:
        # Check Bands
        criterion_description = "Bands ratio"
        criterion_threshold = ctf_bands_ratio
        if "Bands" not in ebsd._data.columns:
            logger.warning(f"No 'Bands' column found in a .ctf {label}. Marking as not good.")
            return False, 0.0, criterion_description, criterion_threshold
        bands_values = ebsd._data["Bands"].to_numpy()
        max_bands = bands_values.max() if len(bands_values) else 1.0
        ratio_array = bands_values / max_bands
        fraction_good = np.mean(ratio_array >= ctf_bands_ratio)
        is_good = fraction_good >= fraction_required
        return is_good, fraction_good, criterion_description, criterion_threshold

    else:
        logger.warning(f"Quality check not defined for format: {ebsd._ebsdFormat}. Marking as not good.")
        return False, 0.0, "unknown", 0.0


##############################################################################
# MAIN PROCESSING FUNCTION
##############################################################################
def process_ebsd_data(settings):
    """
    1) Configure logger to file & console.
    2) Configure path for pycrystallography.
    3) For each .ang/.ctf file in input_folder:
       - Perform a global quality check (50% default).
         If fails, skip entire file.
       - If passes, crop into sub-EBSD objects.
       - For each subcrop, re-check quality (90% default).
         If good => write target & mask => source,
         else => only write unmasked source.
    4) Log in the requested format:
       [using/skipping] the [file/subcrop] as [percent of rows matched]% of rows
       have satisfied the [ci/bands] criteria of [ci_criteria/bands_criteria]
       in comparison with given threshold [threshold value].
    """
    # 1) Configure logger
    logger = configure_logger(settings)
    

    ACCEPTED_FILES = 0
    ACCEPTED_FILE_PIXELS = 0
    GENERATED_SUBCROPS = 0
    ACCEPTED_SUBCROPS = 0
    
    # 2) Configure path
    configure_path(logger)
    from pycrystallography.ebsd.ebsd import Ebsd  # after path is configured

    # Unpack paths & thresholds
    input_folder = settings["input_folder"]
    mask_paths = settings["mask_paths"]
    output_dir = settings["output_directory"]
    dir_names = settings["dir_names"]
    subdirs = settings["subdirs"]
    thresholds = settings.get("thresholds", {})
    debugMode = settings['debug']
    # Create output subfolders
    npy_target_dir = os.path.join(output_dir, subdirs["npy"], dir_names["target"])
    ang_target_dir = os.path.join(output_dir, subdirs["ang"], dir_names["target"])
    tiff_target_dir = os.path.join(output_dir, subdirs["tiff"], dir_names["target"])

    npy_source_dir = os.path.join(output_dir, subdirs["npy"], dir_names["source"])
    ang_source_dir = os.path.join(output_dir, subdirs["ang"], dir_names["source"])
    tiff_source_dir = os.path.join(output_dir, subdirs["tiff"], dir_names["source"])

    for d in [
        npy_target_dir, ang_target_dir, tiff_target_dir,
        npy_source_dir, ang_source_dir, tiff_source_dir
    ]:
        os.makedirs(d, exist_ok=True)
    
    # Gather EBSD files
    all_files = sorted(os.listdir(input_folder))
    ebsd_files = [
        f for f in all_files
        if (f.lower().endswith(".ang") or f.lower().endswith(".ctf")) and
        os.path.getsize(os.path.join(input_folder, f)) < 300 * 1024 * 1024  # 300 MB in bytes
    ]

    logger.info(f"Found {len(ebsd_files)} EBSD files to process.")
    logger.info(
        f"File-level fraction_required = {thresholds.get('file_fraction_required', 0.5)*100:.1f}%, "
        f"Subcrop-level fraction_required = {thresholds.get('subcrop_fraction_required', 0.9)*100:.1f}%."
    )

    for file_name in tqdm(ebsd_files, desc="Processing EBSD files"):
        file_path = os.path.join(input_folder, file_name)
        logger.info(f"Loading file: {file_path}")

        ebsd = Ebsd(logger=logger)
        if file_name.lower().endswith(".ang"): 
            try:
                ebsd.fromAng(file_path)
            except Exception as e:
                logger.warning(f"{file_name} was not read and got this exception: {e}")
                continue
               
        elif file_name.lower().endswith(".ctf"):
            try:
                ebsd.fromCtf(file_path)
            except Exception as e:
                logger.warning(f"{file_name} was not read and got this exception: {e}")
                continue
        else:
            logger.error(f"Skipping unsupported file: {file_name}")
            continue

        # ---------------------------------------------------------
        # 1) Global check at file level (50% by default)
        # ---------------------------------------------------------
        is_good, fraction_good, crit_descr, crit_value = check_quality(
            ebsd, thresholds, logger, label="file"
        )
        action_str = "using" if is_good else "skipping"
        logger.info(
            f"[{action_str}] the file '{file_name}' as {fraction_good*100:.2f}% of rows have satisfied the "
            f"[{crit_descr}] criteria of [{crit_value}] in comparison with given threshold "
            f"[{thresholds.get('file_fraction_required', 0.5)*100:.1f}]"
        )

        if not is_good:
            continue  # skip entire file
        
        ACCEPTED_FILES +=1
        ACCEPTED_FILE_PIXELS += (ebsd.nXPixcels * ebsd.nYPixcels)
        # 2) Crop
        crop_size = settings["augmentations"]["crop_size"]
        crop_overlap = settings["augmentations"]["crop_overlap"]
        cropped_ebsd_list = ebsd.crop(crop_size=crop_size, overlap=crop_overlap)
        if not cropped_ebsd_list:
            logger.info(f"No valid crops for {file_name}. Skipping.")
            continue
        logger.info(
            f"Generated {len(cropped_ebsd_list)} crops of size {crop_size} "
            f"with overlap={crop_overlap} for file {file_name}"
        )
        GENERATED_SUBCROPS += len(cropped_ebsd_list)
        # 3) For each cropped EBSD object, subcrop check (90% by default)
        base_name, ext = os.path.splitext(file_name)
        for crop_idx, cropped_ebsd in enumerate(cropped_ebsd_list):
            # subcrop-level check
            sub_is_good, sub_fraction_good, sub_crit_descr, sub_crit_val = check_quality(
                cropped_ebsd, thresholds, logger, label="subcrop"
            )
            sub_action_str = "using" if sub_is_good else "skipping"
            logger.info(
                f"[{sub_action_str}] the subcrop {crop_idx} as {sub_fraction_good*100:.2f}% of rows have satisfied the "
                f"[{sub_crit_descr}] criteria of [{sub_crit_val}] in comparison with given threshold "
                f"[{thresholds.get('subcrop_fraction_required', 0.9)*100:.1f}]"
            )

            if sub_is_good:
                ACCEPTED_SUBCROPS+=1
                # Possibly rotate & flip
                rotation_angle = random.choice(settings["augmentations"]["rotation_angles"])
                flip_mode = random.choice(settings["augmentations"]["flip_modes"])
                # Decide on mask number
                mask_number = random.randint(1, settings["augmentations"]["mask_count"])

                logger.info(
                    f"selected augmentations for subcrop {crop_idx} => rotation={rotation_angle}, "
                    f"flip={flip_mode}, mask={mask_number}"
                )
                # cropped_ebsd.reduceEulerAngelsToFundamentalZone()
                logger.info(f"applied reduceEulerAngelsToFundamentalZone on subcrop {crop_idx}")

                cropped_ebsd.rotateAndFlipData(flipMode=flip_mode, rotate=rotation_angle)
                logger.info(f"applied rotation/flip on subcrop {crop_idx}")

                # Build filenames
                npy_filename  = build_filename(base_name, crop_idx, rotation_angle, flip_mode, mask_number, ".npy")
                ang_filename  = build_filename(base_name, crop_idx, rotation_angle, flip_mode, mask_number, ext)
                tiff_filename = build_filename(base_name, crop_idx, rotation_angle, flip_mode, mask_number, ".tiff")

                npy_target_path  = os.path.join(npy_target_dir, npy_filename)
                ang_target_path  = os.path.join(ang_target_dir, ang_filename)
                tiff_target_path = os.path.join(tiff_target_dir, tiff_filename)

                npy_source_path  = os.path.join(npy_source_dir, npy_filename)
                ang_source_path  = os.path.join(ang_source_dir, ang_filename)
                tiff_source_path = os.path.join(tiff_source_dir, tiff_filename)

                # Write TARGET
                if not debugMode:
                    cropped_ebsd.writeNpyFile(pathName=npy_target_path)
                    cropped_ebsd.writeAng(pathName=ang_target_path)
                    cropped_ebsd.writeEulerAsPng(pathName=tiff_target_path, showMap=False)
                logger.info(f"subcrop {crop_idx} => wrote TARGET files")

                # Apply mask
                mask_image_path = os.path.join(mask_paths["ebsd_masks"], f"{mask_number}.png")
                cropped_ebsd.applyMask(mask_image_path, displayImage=False)
                logger.info(f"subcrop {crop_idx} => mask applied")

                # Write SOURCE (with mask)
                if not debugMode:
                    cropped_ebsd.writeNpyFile(pathName=npy_source_path)
                    cropped_ebsd.writeAng(pathName=ang_source_path)
                    cropped_ebsd.writeEulerAsPng(pathName=tiff_source_path, showMap=False)
                logger.info(f"subcrop {crop_idx} => wrote SOURCE files (masked)")
            else:
                # Not good => only write unmasked source
                npy_unmasked_filename = f"{base_name}_subcrop{crop_idx}_unmasked.npy"
                ang_unmasked_filename = f"{base_name}_subcrop{crop_idx}_unmasked.ang"
                tiff_unmasked_filename = f"{base_name}_subcrop{crop_idx}_unmasked.tiff"

                npy_source_path  = os.path.join(npy_source_dir, npy_unmasked_filename)
                ang_source_path  = os.path.join(ang_source_dir, ang_unmasked_filename)
                tiff_source_path = os.path.join(tiff_source_dir, tiff_unmasked_filename)
                if not debugMode:
                    cropped_ebsd.writeNpyFile(pathName=npy_source_path)
                    cropped_ebsd.writeAng(pathName=ang_source_path)
                    cropped_ebsd.writeEulerAsPng(pathName=tiff_source_path, showMap=False)
                logger.info(f"subcrop {crop_idx} => wrote unmasked SOURCE only")

    logger.info("All EBSD files processed.")
    logger.info(f"accepted files: {ACCEPTED_FILES}")
    logger.info(f"accepted file pixels: {ACCEPTED_FILE_PIXELS}")
    logger.info(f"total subcrops generated: {GENERATED_SUBCROPS}")
    logger.info(f"subcrops accepted: {ACCEPTED_SUBCROPS}")


##############################################################################
# MAIN
##############################################################################
if __name__ == "__main__":
    process_ebsd_data(SETTINGS)
