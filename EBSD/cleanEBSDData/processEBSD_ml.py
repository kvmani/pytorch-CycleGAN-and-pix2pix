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
from EBSD_ml import Ebsd_ml
import shutil

##############################################################################
# SETTINGS
##############################################################################
# Default settings  some of these may be overridden via command-line arguments.
SETTINGS = {

    "output_directory": "EBSD/cleanEBSDData/temp3",  # Base output directory (can be overridden via CLI)

    "logs_folder": "EBSD/cleanEBSDData/logs/",  # Change as needed #"/home/lus04/kvmani/ml_works/kaushal_2025/outputs/data1.0/logs/"
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



def processEBSDML(input_file, settings, logger):
    ebsd = Ebsd_ml(logger=logger)
    ebsd.read_file(input_file, isReducedToFundamentalZone=True)
    output_dir = settings["output_directory"]

    # file = os.path.basename(input_file)[:-4]

    os.makedirs(output_dir, exist_ok=True)
    # ebsd.cleanUpEbsdData(output_dir=output_dir, isDeleteAuxFolders=False)
    ebsd.writeEulerAsPng(os.path.join(output_dir, "original.png"))
    bandRatio = 0.4
    mad_threshold = 1
    ci_threshold = 0
    fit_threshold = 2
    ebsd.apply_mask_to_threshold(band_ratio_threshold=bandRatio, mad_threshold=mad_threshold, ci_threshold=ci_threshold, fit_threshold=fit_threshold)
    
    ebsd.writeEulerAsPng(os.path.join(output_dir, "checkMask.png"))
    img = ebsd.get_current_euler_map()
    ebsd.cleanupEBSD_by_split_recombine(band_ratio_threshold=bandRatio, mad_threshold=mad_threshold, ci_threshold=ci_threshold, fit_threshold=fit_threshold, outputDir=output_dir)
    ebsd.writeEulerAsPng(os.path.join(output_dir, "cleaned_map.png"))
    # ebsd.cleanupEBSD_by_resizing(outputDir=output_dir, isDeleteAuxFolders=False, display_images=True, mad_threshold=0.9, band_ratio_threshold=0.)
    # shutil.rmtree(output_dir)



##############################################################################
# MAIN
##############################################################################
if __name__ == "__main__":

    #find "/home/lus04/kvmani/ml_works/kaushal_2025/ctf_ang_files/" -type f \( -name "*.ang" -o -name "*.ctf" \) | xargs -P 16 -I {} python process_ebsd_file.py --input_file {} --output_dir "/home/lus04/kvmani/ml_works/kaushal_2025/outputs/data1.0/"
    parser = argparse.ArgumentParser(description="Process a single EBSD file.")
    parser.add_argument("--input_file", required=False, help="Path to the input .ang or .ctf file")
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
    
    args.input_file = r"C:\Users\kaush\Documents\BARC\pycrystallography\data\testingData\Al-B4CModelScan.ang"
    logger = configure_logger(SETTINGS, args.input_file)
    processEBSDML(args.input_file, settings=SETTINGS, logger=logger)