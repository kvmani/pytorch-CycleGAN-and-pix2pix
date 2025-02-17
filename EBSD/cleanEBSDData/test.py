
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

    "output_directory": "/home/lus04/kvmani/ml_works/kaushal_2025/cyclegan/EBSD/cleanEBSDData/temp1",  # Base output directory (can be overridden via CLI)

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
    os.makedirs(output_dir, exist_ok=True)
    file = os.path.basename(input_file)
    img_path = os.path.join(output_dir, f"{file}.png")
    ebsd.writeEulerAsPng(img_path)
    
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
    
    # Setup logger and adjust the system path for pycrystallography
    # args.input_file = r"/home/lus04/kvmani/ml_works/kaushal_2025/MMD_ctf_files_reduced/850p01 _19497.ctf"
    # args.input_file = r"/home/lus04/kvmani/ml_works/kaushal_2025/MMD_ctf_files_reduced/690-60-DS-1100-4H-Site6_31110.ctf"
    # args.input_file = r"/home/lus04/kvmani/ml_works/kaushal_2025/MMD_ctf_files_reduced/CRS1(B)-Rolled_Aged 700 100h Site 4 Map Data 27_32163.ctf"
    # args.input_file = r"/home/lus04/kvmani/ml_works/kaushal_2/025/MMD_ctf_files_reduced/Alloy625 Thal 2020 Sample 2-Tube70-outer circumferential plane Site 2 Map Data 22_21467.ctf"
    args.input_file = r"/home/lus04/kvmani/ml_works/kaushal_2025/cyclegan/EBSD/cleanEBSDData/temp/partially_predicted_source_file.ctf"

    logger = configure_logger(SETTINGS, args.input_file)
    processEBSDML(args.input_file, settings=SETTINGS, logger=logger)