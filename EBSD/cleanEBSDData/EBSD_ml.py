#!/usr/bin/env python3
"""
Module: ebsd_ml.py

This module extends the Ebsd class from pycrystallography with additional methods
for machine learning processing. All steps in the processing pipeline are logged 
using a logger named "EBSD_ml".

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import cv2
import numpy as np
import shutil
import subprocess
import matplotlib.pyplot as plt
from PIL import Image
import logging

from pycrystallography.ebsd.ebsd import Ebsd


class Ebsd_ml(Ebsd):
    """
    Extended EBSD class for machine learning processing.

    This class provides methods to read EBSD files, apply image processing
    (resizing, masking, splitting, and combining), run CycleGAN, and update Euler angle data.
    All steps are logged to help with debugging and tracking the processing flow.
    """

    def __init__(self, filePath=None, logger=None):
        """
        Initialize the Ebsd_ml object.
        
        Parameters
        ----------
        filePath : str, optional
            Path to the EBSD file.
        logger : logging.Logger, optional
            Logger for debug and information messages.
        """
        super().__init__(filePath, logger)
        # Initialize logger; if not provided, create a logger named "EBSD_ml"
        if logger is None:
            self.logger = logging.getLogger("EBSD_ml")
            self.logger.setLevel(logging.INFO)
            # Add a StreamHandler if there are no handlers
            if not self.logger.handlers:
                ch = logging.StreamHandler()
                ch.setLevel(logging.INFO)
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)
        else:
            self.logger = logger
            

        self._originalEulerData = None


    def read_file(self, filePath, isReducedToFundamentalZone=False):
        """
        Reads an EBSD file and reduces Euler angles to the fundamental zone.
        
        Parameters
        ----------
        filePath : str
            Path to the input EBSD file (.ang or .ctf).
        isReducedToFundamentalZone : bool, optional
            Whether to reduce Euler angles to the fundamental zone.
        
        Returns
        -------
        None
        
        Raises
        ------
        Exception
            If the file cannot be read.
        """
        file_name = os.path.basename(filePath)
        self.logger.info(f"Reading file: {file_name}")
        try:
            if filePath.lower().endswith(".ang"):
                self.fromAng(filePath, isReducedToFundamentalZone)
            elif filePath.lower().endswith(".ctf"):
                self.fromCtf(filePath, isReducedToFundamentalZone)
            else:
                self.logger.error(f"Unsupported file format for file: {file_name}")
                return
        except Exception as e:
            self.logger.error(f"Failed to read {file_name}: {e}")
            return

        self.logger.info("File read successfully. Reducing Euler angles to the fundamental zone.")
        # self.reduceEulerAngelsToFundamentalZone_vectorized()
        self._originalEulerData = self._data

    def writeFile(self, output_path):
        """
        Writes the EBSD file to the specified output path.
        
        Parameters
        ----------
        output_path : str
            Destination file path.
        """
        self.logger.info(f"Writing EBSD file to: {output_path}")
        if "ang" in self._ebsdFormat:
            self.writeAng(output_path)
        elif "ctf" in self._ebsdFormat:
            self.writeCtf(output_path)
        else:
            self.logger.error("Unknown EBSD format; cannot write file.")
            raise ValueError("Unknown EBSD format for writing.")

    def resize_to_square(self, image_path, save_dir, crop=(256, 256)):
        """
        Resizes an image to a square with the specified dimensions and sets the first pixel to black.
        
        Parameters
        ----------
        image_path : str
            Path to the input image.
        save_dir : str
            Path where the resized image is saved.
        crop : tuple, optional
            Desired output dimensions (width, height), default is (256, 256).
        
        Returns
        -------
        tuple
            Original image dimensions (height, width).
        
        Raises
        ------
        AssertionError
            If the image cannot be read or saved.
        """
        self.logger.info(f"Resizing image from: {image_path} to size: {crop}; saving to: {save_dir}")
        image = cv2.imread(image_path)
        assert image is not None, f"Could not read image from {image_path}"
        h, w = image.shape[:2]
        resized = cv2.resize(image, (crop[0], crop[1]), interpolation=cv2.INTER_AREA)
        # Set the top-left pixel to black.
        resized[0, 0] = [0, 0, 0]
        success = cv2.imwrite(save_dir, resized)
        assert success, f"Failed to write resized image to {save_dir}"
        self.logger.info(f"Image resized successfully. Original dimensions: {h}x{w}")
        return h, w

    def apply_mask_on_image(self, source_path, mask_path, save_dir):
        """
        Applies a binary mask to the source image.
        
        For black pixels in the mask, the corresponding source pixels are set to white.
        
        Parameters
        ----------
        source_path : str
            Path to the source image.
        mask_path : str
            Path to the mask image.
        save_dir : str
            Path to save the masked image.
        
        Raises
        ------
        ValueError
            If the source or mask image cannot be read.
        """
        self.logger.info(f"Applying mask from: {mask_path} to source image: {source_path}")
        source = cv2.imread(source_path)
        if source is None:
            raise ValueError(f"Could not read source image from '{source_path}'")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask image from '{mask_path}'")
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        h, w = source.shape[:2]
        mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
        source[mask_bin == 0] = (255, 255, 255)
        success = cv2.imwrite(save_dir, source)
        assert success, f"Failed to save masked image to {save_dir}"
        self.logger.info(f"Masked image saved to: {save_dir}")

    def split(self, image_pth, crop_size=256, display=False, save_dir="temp_subcrops"):
        """
        Splits an image into overlapping subcrops of a given crop_size.
        
        Subcrops are saved in the specified directory with filenames based on row and column indices.
        
        Parameters
        ----------
        image_pth : str
            Path to the input image.
        crop_size : int, optional
            Size of each subcrop (default: 256).
        display : bool, optional
            If True, displays subcrops using matplotlib.
        save_dir : str, optional
            Directory where subcrop images are saved.
        
        Returns
        -------
        tuple
            (h_steps, w_steps, H, W), where h_steps and w_steps are the number of subcrops vertically and horizontally,
            and H, W are the original image dimensions.
        """
        self.logger.info(f"Splitting image: {image_pth} into subcrops of size: {crop_size}")
        os.makedirs(save_dir, exist_ok=True)
        image = cv2.imread(image_pth)
        assert image is not None, f"Could not read image from {image_pth}"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, C = image.shape

        h_steps = (H // crop_size) + 1
        w_steps = (W // crop_size) + 1
        h_overlap = ((h_steps * crop_size - H) // max(h_steps - 1, 1)) if h_steps > 1 else 0
        w_overlap = ((w_steps * crop_size - W) // max(w_steps - 1, 1)) if w_steps > 1 else 0

        subcrop_list_for_display = []

        for i in range(h_steps):
            for j in range(w_steps):
                y_start = i * (crop_size - h_overlap)
                x_start = j * (crop_size - w_overlap)
                y_end = min(y_start + crop_size, H)
                x_end = min(x_start + crop_size, W)
                y_start = max(0, y_end - crop_size)
                x_start = max(0, x_end - crop_size)
                subcrop = image[y_start:y_end, x_start:x_end]
                out_name = f"subcrop_{i}_{j}.png"
                out_path = os.path.join(save_dir, out_name)
                cv2.imwrite(out_path, cv2.cvtColor(subcrop, cv2.COLOR_RGB2BGR))
                self.logger.info(f"Subcrop saved: {out_name} at {out_path}")
                if display:
                    subcrop_list_for_display.append(((i, j), subcrop))

        if display:
            fig, axes = plt.subplots(h_steps, w_steps, figsize=(3 * w_steps, 3 * h_steps))
            axes = np.array(axes, ndmin=2)
            for ((i, j), crop) in subcrop_list_for_display:
                ax = axes[i, j]
                ax.imshow(crop.astype(np.uint8))
                ax.set_title(f"({i},{j})")
                ax.axis("off")
            plt.tight_layout()
            plt.show()

        self.logger.info(f"Image splitting complete: {h_steps} rows x {w_steps} columns.")
        return (h_steps, w_steps, H, W)

    def combine(self, subcrops_dir, h_steps, w_steps, original_shape, crop_size=256, draw_boxes=False):
        """
        Recombines overlapping subcrop images from the specified directory into an image of original_shape.
        
        Overlapping areas are overwritten in row-major order.
        
        Parameters
        ----------
        subcrops_dir : str
            Directory containing subcrop images.
        h_steps : int
            Number of subcrops vertically.
        w_steps : int
            Number of subcrops horizontally.
        original_shape : tuple
            Original image shape as (H, W, C).
        crop_size : int, optional
            Size of each subcrop (default: 256).
        draw_boxes : bool, optional
            If True, draws boxes around subcrops in the final image.
        
        Returns
        -------
        numpy.ndarray
            The recombined image.
        """
        self.logger.info(f"Combining subcrops from {subcrops_dir} into an image of shape {original_shape}.")
        H, W, C = original_shape
        recombined_image = np.zeros((H, W, C), dtype=np.uint8)
        h_overlap = ((h_steps * crop_size - H) // max(h_steps - 1, 1)) if h_steps > 1 else 0
        w_overlap = ((w_steps * crop_size - W) // max(w_steps - 1, 1)) if w_steps > 1 else 0

        for i in range(h_steps):
            for j in range(w_steps):
                subcrop_file = os.path.join(subcrops_dir, f"subcrop_{i}_{j}.png")
                if not os.path.exists(subcrop_file):
                    self.logger.warning(f"Missing subcrop file: {subcrop_file}; skipping.")
                    continue
                patch_bgr = cv2.imread(subcrop_file, cv2.IMREAD_COLOR)
                if patch_bgr is None:
                    self.logger.warning(f"Could not read subcrop file: {subcrop_file}; skipping.")
                    continue
                patch = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
                y_start = i * (crop_size - h_overlap)
                x_start = j * (crop_size - w_overlap)
                y_end = min(y_start + crop_size, H)
                x_end = min(x_start + crop_size, W)
                y_start = max(0, y_end - crop_size)
                x_start = max(0, x_end - crop_size)
                patch_height = y_end - y_start
                patch_width = x_end - x_start
                recombined_image[y_start:y_end, x_start:x_end] = patch[:patch_height, :patch_width]
        if draw_boxes:
            for i in range(h_steps):
                for j in range(w_steps):
                    y_start = i * (crop_size - h_overlap)
                    x_start = j * (crop_size - w_overlap)
                    y_end = min(y_start + crop_size, H)
                    x_end = min(x_start + crop_size, W)
                    y_start = max(0, y_end - crop_size)
                    x_start = max(0, x_end - crop_size)
                    cv2.rectangle(recombined_image, (x_start, y_start), (x_end, y_end), (255, 0, 0), thickness=2)
        self.logger.info("Subcrops recombined successfully.")
        return recombined_image

    def display_original_and_clean(self, euler_map_path, recombined_image):
        """
        Displays the original Euler map and the recombined image side by side.
        
        Parameters
        ----------
        euler_map_path : str
            Path to the original Euler color map.
        recombined_image : numpy.ndarray
            The recombined image.
        """
        self.logger.info("Displaying original and recombined images.")
        image = cv2.imread(euler_map_path)
        assert image is not None, f"Cannot read image from {euler_map_path}"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Recombined Image")
        plt.imshow(recombined_image)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def generate_random_image(save_dir, file_name="random_sample.png"):
        """
        Generates a random RGB image (256x256) and saves it.
        
        Parameters
        ----------
        save_dir : str
            Directory where the random image is saved.
        file_name : str, optional
            Name of the saved file; default is "random_sample.png".
        """
        os.makedirs(save_dir, exist_ok=True)
        random_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        save_image_path = os.path.join(save_dir, file_name)
        success = cv2.imwrite(save_image_path, cv2.cvtColor(random_image, cv2.COLOR_RGB2BGR))
        assert success, f"Failed to save random image to {save_image_path}"
        logging.getLogger("EBSD_ml").info(f"Random image generated and saved to: {save_image_path}")

    @staticmethod
    def copy_and_rename_files(source_folder, destination_folder, pattern_to_remove):
        """
        Copies files from the source folder to the destination folder, renaming each file by removing a given pattern.
        
        Parameters
        ----------
        source_folder : str
            Directory containing the source files.
        destination_folder : str
            Directory where files are copied.
        pattern_to_remove : str
            Substring to remove from filenames during renaming.
        """
        os.makedirs(destination_folder, exist_ok=True)
        for filename in os.listdir(source_folder):
            if filename.endswith(pattern_to_remove):
                new_filename = filename.replace(pattern_to_remove, ".png")
                source_path = os.path.join(source_folder, filename)
                destination_path = os.path.join(destination_folder, new_filename)
                shutil.copy2(source_path, destination_path)
                logging.getLogger("EBSD_ml").info(f"Copied and renamed file {filename} to {new_filename}")

    def apply_mask_to_threshold(self, ci_threshold=0.5, fit_threshold=0.5, band_ratio_threshold=0.85,
                                mad_threshold=0.5, label="white"):
        """
        Applies threshold-based masking on Euler data.
        
        For .ang files:
          - If CI <= ci_threshold OR FIT >= fit_threshold OR FIT == 0,
            set the Euler angles to lattice limits (for "white") or zeros (for "black").
        For .ctf files:
          - Compute band_ratio = Bands / max(Bands) and apply similar thresholds.
        
        Parameters
        ----------
        ci_threshold : float, optional
            CI threshold.
        fit_threshold : float, optional
            FIT threshold.
        band_ratio_threshold : float, optional
            Band ratio threshold.
        mad_threshold : float, optional
            MAD threshold.
        label : str, optional
            "white" or "black" to determine replacement values.
        
        Returns
        -------
        pandas.DataFrame
            The modified Euler data.
        """
        self.logger.info("Applying threshold-based masking on Euler data.")
        data = self._originalEulerData

        if label == "black":
            if "ang" in self._ebsdFormat:
                mask = (data["CI"] <= ci_threshold) | (data["FIT"] >= fit_threshold) | (data["FIT"] == 0.0)
                data.loc[mask, ["phi1", "PHI", "phi2"]] = 0.0
            elif "ctf" in self._ebsdFormat:
                max_band = data["Bands"].max()
                band_ratio = data["Bands"] / max_band
                mask = (data["MAD"] >= mad_threshold) | (band_ratio <= band_ratio_threshold) | (data["MAD"] == 0.0)
                data.loc[mask, ["Euler1", "Euler2", "Euler3"]] = 0.0
        elif label == "white":
            limits = np.asarray(self._lattice._EulerLimits)
            if "ang" in self._ebsdFormat:
                mask = (data["CI"] <= ci_threshold) | (data["FIT"] >= fit_threshold) | (data["FIT"] == 0.0)
                data.loc[mask, ["phi1", "PHI", "phi2"]] = limits
            elif "ctf" in self._ebsdFormat:
                max_band = data["Bands"].max()
                band_ratio = data["Bands"] / max_band
                mask = (data["MAD"] >= mad_threshold) | (band_ratio <= band_ratio_threshold) | (data["MAD"] == 0.0)
                data.loc[mask, ["Euler1", "Euler2", "Euler3"]] = limits

        self.logger.info("Threshold-based masking applied.")
        return data

    def get_current_euler_map(self):
        """
        Generates and returns the current Euler map as a PIL image.
        
        Returns
        -------
        PIL.Image.Image
            The Euler map image.
        """
        self.logger.info("Generating current Euler map.")
        self.call_makeEulerData()
        im = Image.fromarray(self._oriDataInt)
        return im

    def call_cycle_gan(self, input_dir, output_dir, num_test, side=256,
                       checkpoint="ebsd_data_2.0_lr1e-4_pool_100_batch_32_vanilla"):
        """
        Calls the CycleGAN test script via subprocess, then renames the output files.
        
        Parameters
        ----------
        input_dir : str
            Directory containing input images.
        output_dir : str
            Directory where CycleGAN outputs are stored.
        num_test : int
            Number of test images to process.
        side : int, optional
            (Unused here) Placeholder for image dimension.
        checkpoint : str, optional
            Name of the CycleGAN checkpoint.
        """
        self.logger.info("Starting CycleGAN inference.")
        temp_dir = os.path.join(input_dir, "temp")
        results_dir = temp_dir
        dataroot = input_dir
        name = checkpoint
        model = "cycle_gan"
        input_nc = 3
        output_nc = 3
        gpu_ids = -1
        preprocess = "none"
        norm = "none"

        os.makedirs(results_dir, exist_ok=True)
        args = [
            "--results_dir", results_dir,
            "--dataroot", dataroot,
            "--name", name,
            "--model", model,
            "--input_nc", str(input_nc),
            "--output_nc", str(output_nc),
            "--gpu_ids", str(gpu_ids),
            "--num_test", str(num_test),
            "--preprocess", str(preprocess),
        ]
        self.logger.info(f"CycleGAN arguments: {args}")
        subprocess.run(["python", "test.py"] + args, check=True)
        dir_of_output = os.path.join(results_dir, checkpoint, "test_latest", "images")
        Ebsd_ml.copy_and_rename_files(dir_of_output, output_dir, pattern_to_remove="_fake_B.png")
        shutil.rmtree(results_dir)
        self.logger.info("CycleGAN inference completed.")

    def denormalize_RGB2EulerAngles(self, reconstructedImage):
        """
        Denormalizes an RGB image (values 0-255) back to Euler angles in radians.
        
        Parameters
        ----------
        reconstructedImage : numpy.ndarray
            Image with RGB channels representing normalized Euler angles.
        
        Returns
        -------
        numpy.ndarray
            Array of Euler angles in radians (shape: (-1, 3)).
        """
        self.logger.info("Denormalizing Euler angles from predicted image.")
        if self._isEulerAnglesReduced:
            lattice_limits = np.array(self._lattice._EulerLimits)
        else:
            lattice_limits = np.array([2 * np.pi, np.pi, 2 * np.pi])
        
        angle1 = np.interp(reconstructedImage[:, :, 0], [0, 255], [0, lattice_limits[0]])
        angle2 = np.interp(reconstructedImage[:, :, 1], [0, 255], [0, lattice_limits[1]])
        angle3 = np.interp(reconstructedImage[:, :, 2], [0, 255], [0, lattice_limits[2]])
        self.logger.info("Denormalization complete.")
        return np.stack([angle1, angle2, angle3], axis=2).reshape(-1, 3)

    def replaceEulerAngles(self, denormAngles, outputDir, fit_threshold=0.5, ci_threshold=0.5,
                           bandRatio=0.5, mad_threshold=1.5):
        """
        Replaces Euler angles in self._data based on threshold conditions.
        
        For .ang files, rows that meet the condition have their Euler angles replaced by the denormalized angles.
        For .ctf files, a similar replacement is performed, and the 'Error' field is set to 0.
        
        Parameters
        ----------
        denormAngles : numpy.ndarray
            Denormalized Euler angles.
        outputDir : str
            Directory where output files are written.
        fit_threshold : float, optional
            FIT threshold for .ang files.
        ci_threshold : float, optional
            CI threshold for .ang files.
        bandRatio : float, optional
            Band ratio threshold for .ctf files.
        mad_threshold : float, optional
            MAD threshold for .ctf files.
        """
        self.logger.info("Replacing Euler angles in EBSD data based on thresholds.")
        if "ang" in self._ebsdFormat:
            mask = (self._data["FIT"] >= fit_threshold) | (self._data["CI"] <= ci_threshold) | (self._data["FIT"] == 0.0)
            self._data.loc[mask, ['phi1', 'PHI', 'phi2']] = denormAngles[mask]
        elif "ctf" in self._ebsdFormat:
            max_band = max(self._data["Bands"])
            mask = (self._data["MAD"] >= mad_threshold) | ((self._data["Bands"] / max_band) <= bandRatio) | (self._data["MAD"] == 0.0)
            self._data.loc[mask, ['Euler1', 'Euler2', 'Euler3']] = denormAngles[mask]
            self._data.loc[mask, ['Error']] = 0
        self.logger.info("Euler angles replaced successfully.")

    def replaceEulerAngles_and_write(self, denormAngles, outputDir):
        """
        Replaces Euler angles in self._data and writes both Euler color maps and EBSD source files.
        
        For .ang and .ctf files, separate partially and fully predicted files are written.
        
        Parameters
        ----------
        denormAngles : numpy.ndarray
            Denormalized Euler angles.
        outputDir : str
            Directory where output files are saved.
        """
        self.logger.info("Replacing Euler angles and writing output files.")
        fully_predicted_color_map = os.path.join(outputDir, "fully_predicted_color_map.png")
        partially_predicted_color_map = os.path.join(outputDir, "Partially_predicted_color_map.png")

        if "ang" in self._ebsdFormat:
            mask = self._data["FIT"] > 0.5
            self._data.loc[mask, ['phi1', 'PHI', 'phi2']] = denormAngles[mask]
            self.writeEulerAsPng(partially_predicted_color_map)
            partially_predicted_source_file = os.path.join(outputDir, "partially_predicted_source_file.ang")
            self.writeAng(partially_predicted_source_file)
            fully_predicted_source_file = os.path.join(outputDir, "fully_predicted_source_file.ang")
            mask = self._data["FIT"] >= 0
            self._data.loc[mask, ['phi1', 'PHI', 'phi2']] = denormAngles[mask]
            self.writeEulerAsPng(fully_predicted_color_map)
            self.writeAng(fully_predicted_source_file)
        elif "ctf" in self._ebsdFormat:
            max_band = max(self._data["Bands"])
            mask = (self._data["MAD"] > 0.8) | ((self._data["Bands"] / max_band) < 0.9)
            self._data.loc[mask, ['Euler1', 'Euler2', 'Euler3']] = denormAngles[mask]
            self._data.loc[mask, ['Error']] = 0
            self.writeEulerAsPng(partially_predicted_color_map)
            partially_predicted_source_file = os.path.join(outputDir, "partially_predicted_source_file.ctf")
            self.writeCtf(partially_predicted_source_file)
            fully_predicted_source_file = os.path.join(outputDir, "fully_predicted_source_file.ctf")
            mask = self._data["MAD"] >= 0
            self._data.loc[mask, ['Euler1', 'Euler2', 'Euler3']] = denormAngles[mask]
            self._data.loc[mask, ['Error']] = 0
            self.writeEulerAsPng(fully_predicted_color_map)
            self.writeCtf(fully_predicted_source_file)
        self.logger.info("Output files written successfully.")

    def cleanUpEbsdData(self, output_dir, isDeleteAuxFolders=True, display_images=False):
        """
        Cleans up EBSD data by cropping, masking, and generating output files.
        
        This method crops the EBSD data, writes the original Euler color map and CTF file,
        applies a mask to produce masked Euler/CTF files, resizes images, and runs CycleGAN
        for further processing.
        
        Parameters
        ----------
        output_dir : str
            Directory where output files are saved.
        isDeleteAuxFolders : bool, optional
            If True, auxiliary folders created during processing are deleted.
        display_images : bool, optional
            If True, displays images during processing.
        """
        self.logger.info("Starting EBSD cleanup process.")
        # Crop the EBSD data (selecting the 4th crop arbitrarily).
        self = self.crop(crop_size=(512, 512))[3]
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Output directory created: {output_dir}")

        # Write original Euler color map and CTF file.
        original_euler_colorMap_path = os.path.join(output_dir, "original_euler_map.png")
        self.writeEulerAsPng(original_euler_colorMap_path)
        self.logger.info(f"Original Euler color map written to: {original_euler_colorMap_path}")
        original_ctf_file = os.path.join(output_dir, "original_ctf_file.ctf")
        self.writeCtf(original_ctf_file)
        self.logger.info(f"Original CTF file written to: {original_ctf_file}")

        # Apply mask to Euler color map.
        masked_euler_colorMap = os.path.join(output_dir, "masked_euler_color_map.png")
        maskImge = r"/home/lus04/kvmani/ml_works/kaushal_2025/ebsd_masks/20.png"
        self.apply_mask_on_image(source_path=original_euler_colorMap_path, mask_path=maskImge, save_dir=masked_euler_colorMap)
        self.logger.info(f"Mask applied to Euler color map. Saved masked image to: {masked_euler_colorMap}")
        self.applyMask(maskImge=maskImge)
        masked_ctf_file = os.path.join(output_dir, "masked_ctf_file.ctf")
        self.writeCtf(masked_ctf_file)
        self.writeEulerAsPng(masked_euler_colorMap)
        self.logger.info("Masked CTF file and Euler color map written.")

        # Create auxiliary folder for ML processing.
        subDirForMl = os.path.join(output_dir, "subdir_for_ml")
        os.makedirs(subDirForMl, exist_ok=True)
        self.logger.info(f"Auxiliary ML folder created: {subDirForMl}")

        aux_path = os.path.join(subDirForMl, "testA")
        os.makedirs(aux_path, exist_ok=True)
        test_image = os.path.join(aux_path, "masked_euler_color_map.png")
        h, w = self.resize_to_square(masked_euler_colorMap, test_image)
        self.logger.info("Image resized for ML processing.")
        h_steps, w_steps, H, W = self.split(original_euler_colorMap_path, crop_size=256, save_dir=aux_path, display=display_images)
        self.logger.info(f"Image split into subcrops: {h_steps} rows x {w_steps} columns.")

        test_b_path = os.path.join(subDirForMl, "testB")
        Ebsd_ml.generate_random_image(test_b_path)
        self.logger.info(f"Random image generated for testB: {test_b_path}")

        cycle_gan_output_dir = os.path.join(subDirForMl, "Ml_output")
        self.call_cycle_gan(subDirForMl, cycle_gan_output_dir, num_test=5)
        self.logger.info(f"CycleGAN output generated in: {cycle_gan_output_dir}")

        resized_predicted_image_path = os.path.join(cycle_gan_output_dir, "masked_euler_color_map.png")
        predicted_image_path = os.path.join(output_dir, "predicted_euler_map.png")
        h, w = self.resize_to_square(resized_predicted_image_path, predicted_image_path, crop=(w, h))
        self.logger.info("Predicted image resized to original dimensions.")

        predictedImage = np.array(Image.open(predicted_image_path))
        denormalizedEulerAngles = self.denormalize_RGB2EulerAngles(reconstructedImage=predictedImage)
        self.logger.info("Euler angles denormalized from predicted image.")

        self.replaceEulerAngles_and_write(denormalizedEulerAngles, output_dir)
        self.logger.info("Euler angles replaced and output files written.")

        if isDeleteAuxFolders:
            shutil.rmtree(subDirForMl)
            self.logger.info(f"Auxiliary ML folder {subDirForMl} deleted.")

    def cleanupEBSD_by_resizing(self, ci_threshold=0.5, band_ratio_threshold=0.85, mad_threshold=0.5,
                                 fit_threshold=0.5, outputDir=None, resize=(256, 256),
                                 isDeleteAuxFolders=True, display_images=False):
        """
        Cleans up EBSD data by applying threshold-based masking, resizing the Euler map,
        and processing further via CycleGAN.
        
        Steps:
         1. Apply threshold-based masking.
         2. Save the masked Euler map as PNG.
         3. Resize the masked image.
         4. Process using CycleGAN and replace Euler angles.
        
        Parameters
        ----------
        ci_threshold : float, optional
            CI threshold.
        band_ratio_threshold : float, optional
            Band ratio threshold.
        mad_threshold : float, optional
            MAD threshold.
        fit_threshold : float, optional
            FIT threshold.
        outputDir : str, optional
            Directory to save output files; if None, defaults to a subdirectory of the current working directory.
        resize : tuple, optional
            Desired dimensions for resizing the masked image.
        isDeleteAuxFolders : bool, optional
            Whether to delete auxiliary folders after processing.
        display_images : bool, optional
            If True, displays intermediate images.
        """
        self.logger.info("Starting EBSD cleanup by resizing.")
        if outputDir is None:
            outputDir = os.path.join(os.getcwd(), "output_ebsd_resized")
        os.makedirs(outputDir, exist_ok=True)
        self.logger.info(f"Output directory: {outputDir}")

        masked_ebsd_png = os.path.join(outputDir, "masked_ebsd.png")
        self.apply_mask_to_threshold(ci_threshold=ci_threshold, fit_threshold=fit_threshold,
                                     mad_threshold=mad_threshold, band_ratio_threshold=band_ratio_threshold,
                                     label="white")
        current_image = self.get_current_euler_map()
        current_image.save(masked_ebsd_png)
        self.logger.info(f"Masked EBSD PNG saved to: {masked_ebsd_png}")

        resized_path = os.path.join(outputDir, "masked_ebsd_resized.png")
        self.resize_to_square(masked_ebsd_png, resized_path, crop=resize)
        self.logger.info(f"Resized masked EBSD image saved to: {resized_path}")

        subDirForMl = os.path.join(outputDir, "subdir_for_ml")
        os.makedirs(subDirForMl, exist_ok=True)
        self.logger.info(f"Auxiliary ML folder created: {subDirForMl}")

        aux_path = os.path.join(subDirForMl, "testA")
        os.makedirs(aux_path, exist_ok=True)
        shutil.copy2(resized_path, os.path.join(aux_path, "masked_ebsd_resized.png"))
        self.logger.info("Resized masked EBSD image copied to testA.")

        test_b_path = os.path.join(subDirForMl, "testB")
        Ebsd_ml.generate_random_image(test_b_path)
        self.logger.info(f"Random image generated for testB: {test_b_path}")

        cycle_gan_output_dir = os.path.join(subDirForMl, "Ml_output")
        self.call_cycle_gan(subDirForMl, cycle_gan_output_dir, num_test=5)
        self.logger.info(f"CycleGAN output generated in: {cycle_gan_output_dir}")

        predicted_image_path = os.path.join(outputDir, "predicted_ebsd.png")
        resized_predicted_image_path = os.path.join(cycle_gan_output_dir, "masked_ebsd_resized.png")
        original_img = cv2.imread(masked_ebsd_png)
        assert original_img is not None, f"Could not read {masked_ebsd_png} to obtain image dimensions."
        oh, ow = original_img.shape[:2]
        self.resize_to_square(resized_predicted_image_path, predicted_image_path, crop=(ow, oh))
        self.logger.info("Predicted EBSD image resized to original dimensions.")

        predictedImage = np.array(Image.open(predicted_image_path))
        denormalizedEulerAngles = self.denormalize_RGB2EulerAngles(reconstructedImage=predictedImage)
        self.logger.info("Euler angles denormalized from predicted image.")

        self.replaceEulerAngles(denormalizedEulerAngles, outputDir, fit_threshold=fit_threshold,
                                ci_threshold=ci_threshold, mad_threshold=mad_threshold, bandRatio=band_ratio_threshold)
        self.logger.info("Euler angles replaced successfully in EBSD data.")

        if isDeleteAuxFolders:
            shutil.rmtree(subDirForMl)
            self.logger.info(f"Auxiliary ML folder {subDirForMl} deleted.")

    def cleanupEBSD_by_split_recombine(self, ci_threshold=0.5, band_ratio_threshold=0.85, mad_threshold=0.5,
                                        fit_threshold=0.5, outputDir=None, resize=(256, 256),
                                        isDeleteAuxFolders=True, display_images=False):
        """
        Cleans up EBSD data by splitting the Euler map into overlapping subcrops,
        processing them via CycleGAN, recombining, and updating Euler angles.
        
        Parameters
        ----------
        ci_threshold : float, optional
            CI threshold.
        band_ratio_threshold : float, optional
            Band ratio threshold.
        mad_threshold : float, optional
            MAD threshold.
        fit_threshold : float, optional
            FIT threshold.
        outputDir : str, optional
            Output directory; if None, defaults to a subdirectory of the current working directory.
        resize : tuple, optional
            (Not used) Placeholder for resize dimensions.
        isDeleteAuxFolders : bool, optional
            Whether to delete auxiliary folders after processing.
        display_images : bool, optional
            If True, displays intermediate images.
        """
        self.logger.info("Starting EBSD cleanup by split and recombine.")
        if outputDir is None:
            outputDir = os.path.join(os.getcwd(), "output_ebsd_resized")
        os.makedirs(outputDir, exist_ok=True)
        self.logger.info(f"Output directory: {outputDir}")

        masked_ebsd_png = os.path.join(outputDir, "masked_ebsd.png")
        self.apply_mask_to_threshold(ci_threshold=ci_threshold, fit_threshold=fit_threshold,
                                     mad_threshold=mad_threshold, band_ratio_threshold=band_ratio_threshold,
                                     label="white")
        current_image = self.get_current_euler_map()
        current_image.save(masked_ebsd_png)
        self.logger.info(f"Masked EBSD PNG saved to: {masked_ebsd_png}")

        subDirForMl = os.path.join(outputDir, "subdir_for_ml")
        os.makedirs(subDirForMl, exist_ok=True)
        self.logger.info(f"Auxiliary ML folder created: {subDirForMl}")

        aux_path = os.path.join(subDirForMl, "testA")
        os.makedirs(aux_path, exist_ok=True)
        h_steps, w_steps, H, W = self.split(masked_ebsd_png, crop_size=256, save_dir=aux_path, display=display_images)
        self.logger.info(f"Image split into subcrops: {h_steps} rows x {w_steps} columns.")

        test_b_path = os.path.join(subDirForMl, "testB")
        Ebsd_ml.generate_random_image(test_b_path)
        self.logger.info(f"Random image generated for testB: {test_b_path}")

        cycle_gan_output_dir = os.path.join(subDirForMl, "Ml_output")
        self.call_cycle_gan(subDirForMl, cycle_gan_output_dir, num_test=h_steps*w_steps)
        self.logger.info(f"CycleGAN output generated in: {cycle_gan_output_dir}")

        reconstructedImg = self.combine(cycle_gan_output_dir, h_steps=h_steps, w_steps=w_steps, original_shape=(H, W, 3), crop_size=256)
        predictedImage = Image.fromarray(reconstructedImg)
        predicted_image_path = os.path.join(outputDir, "predicted_ebsd.png")
        predictedImage.save(predicted_image_path)
        self.logger.info(f"Predicted EBSD image saved to: {predicted_image_path}")

        denormalizedEulerAngles = self.denormalize_RGB2EulerAngles(reconstructedImage=reconstructedImg)
        self.logger.info("Euler angles denormalized from reconstructed image.")

        self.replaceEulerAngles(denormalizedEulerAngles, outputDir, fit_threshold=fit_threshold,
                                ci_threshold=ci_threshold, mad_threshold=mad_threshold, bandRatio=band_ratio_threshold)
        self.logger.info("Euler angles replaced successfully in EBSD data.")

        if isDeleteAuxFolders:
            shutil.rmtree(subDirForMl)
            self.logger.info(f"Auxiliary ML folder {subDirForMl} deleted.")
