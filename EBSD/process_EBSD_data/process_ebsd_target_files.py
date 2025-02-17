#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import datetime
import random
# from pycrystallography.ebsd.ebsd import Ebsd

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

def configure_logger(settings, input_file):
    logs_folder = settings.get("logs_folder", "logs")
    #os.makedirs(logs_folder, exist_ok=True)
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


def build_filename(base_name, rotation_angle, flip_mode, mask_number):
    flip_str = flip_mode if flip_mode is not None else "none"
    return f"{base_name}_rotation_{rotation_angle}_flip_{flip_str}_mask_{mask_number}"


def process_ctf_as_ang(input_file, logger):
    """
    Converts a .ctf file into an .ang file by:
    1. Replacing the header with a standard .ang header.
    2. Removing the last column from all numerical rows.
    """
    ang_header = """# HEADER: Start
# TEM_PIXperUM          1.000000
# x-star                0.499371
# y-star                0.549774
# z-star                0.609261
# WorkingDistance       13.650000
# SampleTiltAngle       69.900002
# CameraElevationAngle  8.000000
# CameraAzimuthalAngle  0.000000
#
# Phase 1
# MaterialName  	Ni (Nickel)
# Formula     	Ni
# Info		
# Symmetry              43
# PointGroupID          131
# LatticeConstants      3.560 3.560 3.560  90.000  90.000  90.000
# NumberFamilies        69
# hklFamilies   	 1 -1 -1 1 11.936152 1
# hklFamilies   	 0 -2  0 1 10.514876 1
# hklFamilies   	 0 -2  2 1 7.489179 1
# hklFamilies   	 1 -3 -1 1 6.273238 1
# hklFamilies   	 2 -2 -2 0 5.956645 0
# hklFamilies   	 0 -4  0 0 4.956584 0
# hklFamilies   	 1 -3 -3 0 4.425857 0
# hklFamilies   	 0 -4  2 0 4.271905 0
# hklFamilies   	 2 -4 -2 0 3.737260 0
# hklFamilies   	 1 -5 -1 0 3.423848 0
# hklFamilies   	 3 -3 -3 0 3.423848 0
# hklFamilies   	 0 -4  4 0 2.963146 0
# hklFamilies   	 1 -5 -3 0 2.768213 0
# hklFamilies   	 2 -4 -4 0 2.708809 0
# hklFamilies   	 0 -6  0 0 2.708809 0
# hklFamilies   	 0 -6  2 0 2.479068 0
# hklFamilies   	 3 -5 -3 0 2.335206 0
# hklFamilies   	 2 -6 -2 0 2.292190 0
# hklFamilies   	 4 -4 -4 0 2.124829 0
# hklFamilies   	 1 -5 -5 0 2.006490 0
# hklFamilies   	 1 -7 -1 0 2.006490 0
# hklFamilies   	 0 -6  4 0 1.975567 0
# hklFamilies   	 2 -6 -4 0 1.854753 0
# hklFamilies   	 1 -7 -3 0 1.766952 0
# hklFamilies   	 3 -5 -5 0 1.766952 0
# hklFamilies   	 0 -8  0 0 1.642427 0
# hklFamilies   	 3 -7 -3 0 1.578907 0
# hklFamilies   	 0 -8  2 0 1.558051 0
# hklFamilies   	 4 -6 -4 0 1.558051 0
# hklFamilies   	 0 -6  6 0 1.476123 0
# hklFamilies   	 2 -8 -2 0 1.476123 0
# hklFamilies   	 5 -5 -5 0 1.423998 0
# hklFamilies   	 1 -7 -5 0 1.423998 0
# hklFamilies   	 2 -6 -6 0 1.408157 0
# hklFamilies   	 0 -8  4 0 1.345813 0
# hklFamilies   	 1 -9 -1 0 1.300071 0
# hklFamilies   	 3 -7 -5 0 1.300071 0
# hklFamilies   	 2 -8 -4 0 1.285009 0
# hklFamilies   	 4 -6 -6 0 1.231948 0
# hklFamilies   	 1 -9 -3 0 1.196316 0
# hklFamilies   	 4 -8 -4 0 1.138211 0
# hklFamilies   	 3 -9 -3 0 1.104073 0
# hklFamilies   	 1 -7 -7 0 1.104073 0
# hklFamilies   	 5 -7 -5 0 1.104073 0
# hklFamilies   	 0 -8  6 0 1.094247 0
# hklFamilies   	 2 -8 -6 0 1.058645 0
# hklFamilies   	 3 -7 -7 0 1.032390 0
# hklFamilies   	 1 -9 -5 0 1.032390 0
# hklFamilies   	 6 -6 -6 0 1.023721 0
# hklFamilies   	 3 -9 -5 0 0.965601 0
# hklFamilies   	 4 -8 -6 0 0.958806 0
# hklFamilies   	 5 -7 -7 0 0.912034 0
# hklFamilies   	 0 -8  8 0 0.879436 0
# hklFamilies   	 5 -9 -5 0 0.861084 0
# hklFamilies   	 1 -9 -7 0 0.861084 0
# hklFamilies   	 2 -8 -8 0 0.855450 0
# hklFamilies   	 6 -8 -6 0 0.833125 0
# hklFamilies   	 3 -9 -7 0 0.816595 0
# hklFamilies   	 4 -8 -8 0 0.789438 0
# hklFamilies   	 7 -7 -7 0 0.773942 0
# hklFamilies   	 5 -9 -7 0 0.741021 0
# hklFamilies   	 1 -9 -9 0 0.708940 0
# hklFamilies   	 6 -8 -8 0 0.704986 0
# hklFamilies   	 3 -9 -9 0 0.680567 0
# hklFamilies   	 7 -9 -7 0 0.653385 0
# hklFamilies   	 5 -9 -9 0 0.628454 0
# hklFamilies   	 8 -8 -8 0 0.614170 0
# hklFamilies   	 7 -9 -9 0 0.561518 0
# hklFamilies   	 9 -9 -9 0 0.477942 0
# GRID: SqrGrid
# XSTEP: 1.000000
# YSTEP: 1.000000
# NCOLS_ODD: 256
# NCOLS_EVEN: 256
# NROWS: 256
#
# OPERATOR: 	
#
# SAMPLEID: 	
#
# SCANID: 	
#
# VERSION 7
# NOTES: Start
# Version 1: phi1, PHI, phi2, x, y, iq (x*=0.1 & y*=0.1)
# Version 2: phi1, PHI, phi2, x, y, iq, ci
# Version 3: phi1, PHI, phi2, x, y, iq, ci, phase
# Version 4: phi1, PHI, phi2, x, y, iq, ci, phase, sem
# Version 5: phi1, PHI, phi2, x, y, iq, ci, phase, sem, fit
# Version 6: phi1, PHI, phi2, x, y, iq, ci, phase, sem, fit, PRIAS Bottom Strip, PRIAS Center Square, PRIAS Top Strip, Custom Value
# Version 7: phi1, PHI, phi2, x, y, iq, ci, phase, sem, fit. PRIAS, Custom, EDS and CMV values included if valid
# Phase index: 0 for single phase, starting at 1 for multiphase
# CMV = Correlative Microscopy value
# EDS = cumulative counts over a specific range of energies
# SEM = any external detector signal but usually the secondary electron detector signal
# NOTES: End
# COLUMN_COUNT: 13
# COLUMN_HEADERS: phi1, PHI, phi2, x, y, IQ, CI, Phase index, SEM, Fit, PRIAS Bottom Strip, PRIAS Center Square, PRIAS Top Strip
# COLUMN_UNITS: radians, radians, radians, microns, microns, , , , , degrees, , , 
# COLUMN_NOTES: Start
# Column 1: phi1 [radians]
# Column 2: PHI [radians]
# Column 3: phi2 [radians]
# Column 4: x [microns]
# Column 5: y [microns]
# Column 6: IQ
# Column 7: CI
# Column 8: Phase index
# Column 9: SEM
# Column 10: Fit [degrees]
# Column 11: PRIAS Bottom Strip
# Column 12: PRIAS Center Square
# Column 13: PRIAS Top Strip
# COLUMN_NOTES: End
"""

    try:
        with open(input_file, "r") as file:
            lines = file.readlines()

        # Find the first line of numerical data
        data_start_index = next(
            i for i, line in enumerate(lines) if line.strip() and line.split()[0].replace(".", "", 1).isdigit()
        )

        # Replace header and process numerical rows
        numerical_data = []
        for line in lines[data_start_index:]:
            columns = line.split()
            if len(columns) > 1:  # Ensure it's numerical data
                numerical_data.append(" ".join(columns[:-1]))  # Remove last column

        # Prepare the .ang content
        ang_content = ang_header + "\n".join(numerical_data)
        
        #Path to save the new ang files. 
        ang_directory_path = r"/home/lus04/kvmani/ml_works/kaushal_2025/outputs/data2.0/target_for_data3.0_ctf_to_ang_files/"
        #os.makedirs(ang_directory_path, exist_ok=True) # creates the directory if not exists
        
        base_file_name = os.path.basename(input_file)
        ang_file_name = base_file_name.replace(".ctf", ".ang")
        ang_file = os.path.join(ang_directory_path,ang_file_name)

        with open(ang_file, "w") as ang_out:
            ang_out.write(ang_content)

        logger.info(f"Converted {input_file} to {ang_file}")
        exit()
        return ang_file

    except Exception as e:
        logger.error(f"Failed to process {input_file} as .ang: {e}")
        return None


def process_single_file(input_file, settings, logger, output_dirs):
    configure_path(logger=logger)
    from pycrystallography.ebsd.ebsd import Ebsd 
    file_name = os.path.basename(input_file)
    logger.info(f"Loading file: {input_file}")

    # Convert .ctf to .ang if needed
    if input_file.lower().endswith(".ctf"):
        input_file = process_ctf_as_ang(input_file, logger)
        if not input_file:
            return

    ebsd = Ebsd(logger=logger)
    try:
        ebsd.fromAng(input_file, isReducedToFundamentalZone=True)
    except Exception as e:
        logger.warning(f"Failed to read {file_name}: {e}")
        return

    # Apply random rotation and flip
    rotation_angle = random.choice(settings["augmentations"]["rotation_angles"])
    flip_mode = random.choice(settings["augmentations"]["flip_modes"])
    mask_fraction = settings["augmentations"]["masked_area_fraction"]
    mask_tolerance = settings["augmentations"]["masked_area_tolerance"]
    mask_number = random.randint(39, settings["augmentations"]["mask_number"])

    logger.info(f"Decided rotation={rotation_angle}, flip={flip_mode}, mask={mask_number}")

    # Build the new file name
    new_name = build_filename(file_name, rotation_angle, flip_mode, mask_number)

    # Write TARGET files (before applying transformations)
    npy_target_path = os.path.join(output_dirs["npy_target"], f"{new_name}.npy")
    ang_target_path = os.path.join(output_dirs["ang_target"], f"{new_name}.ang")
    tiff_target_path = os.path.join(output_dirs["tiff_target"], f"{new_name}.tiff")

    if not settings["debug"]:
        ebsd.writeNpyFile(pathName=npy_target_path)
        ebsd.writeAng(pathName=ang_target_path)
        ebsd.writeEulerAsPng(pathName=tiff_target_path, showMap=False)
    logger.info(f"TARGET files written for {file_name}")

    # Apply rotation, flip, and mask
    ebsd.rotateAndFlipData(flipMode=flip_mode, rotate=rotation_angle)
    logger.info(f"Applied rotation and flip to {file_name}")

    mask_image_path = os.path.join(settings["mask_paths"]["ebsd_masks"], f"{mask_number}.png")
    # ebsd.applyMask(mask_image_path, displayImage=False)
    ebsd.applyMaskWithPercentage(mask_image_path, desired_percent=mask_fraction, tolerance=mask_tolerance)
    logger.info(f"Applied mask to {file_name}")

    # Write SOURCE files (after applying transformations)
    npy_source_path = os.path.join(output_dirs["npy_source"], f"{new_name}.npy")
    ang_source_path = os.path.join(output_dirs["ang_source"], f"{new_name}.ang")
    tiff_source_path = os.path.join(output_dirs["tiff_source"], f"{new_name}.tiff")

    if not settings["debug"]:
        ebsd.writeNpyFile(pathName=npy_source_path)
        ebsd.writeAng(pathName=ang_source_path)
        ebsd.writeEulerAsPng(pathName=tiff_source_path, showMap=False)
    logger.info(f"SOURCE files written for {file_name}")


if __name__ == "__main__":
#find "/home/lus04/kvmani/ml_works/kaushal_2025/outputs/data2.0/target_for_data3.0_ang_and_ctf/" -type f \( -name "*.ang" -o -name "*.ctf" \) | xargs -P 10 -I {} python3 process_ebsd_target_files.py --input_file {}
    parser = argparse.ArgumentParser(description="Process a single EBSD file.")
    parser.add_argument("--input_file", required=True, help="Path to the input .ang or .ctf file")
    parser.add_argument("--output_dir", required=False, help="Path to the output directory (default from SETTINGS)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (no files are written)")

    args = parser.parse_args()

    SETTINGS = {
        "mask_paths": {
            "ebsd_masks": "/home/lus04/kvmani/ml_works/kaushal_2025/ebsd_masks/",
            
        },
        "output_directory": "output",
        "augmentations": {
            "rotation_angles": [0, 90, 180, 270],
            "flip_modes": [None, 'h', 'v'],
            "mask_number":109,
            "masked_area_fraction": .65,
            "masked_area_tolerance": 0.05,
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
        "logs_folder": "/home/lus04/kvmani/ml_works/kaushal_2025/outputs/data3.0/data3.0_noise_65/logs/",
        "debug": False
    }

    if args.output_dir:
        SETTINGS["output_directory"] = args.output_dir
    if args.debug:
        SETTINGS["debug"] = True
    
    
    logger = configure_logger(SETTINGS, args.input_file)
    output_dirs = setup_output_directories(SETTINGS, logger)
    process_single_file(args.input_file, SETTINGS, logger, output_dirs)
