import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
from typing import Tuple, List, Dict

class ImageCombiner:
    """
    Handles loading, combining, and normalizing grayscale Kikuchi images.
    """

    def __init__(self, config: Dict):
        self.input_dir = config.get("input_dir", "./input")
        self.output_dir = config.get("output_dir", "./output")
        self.debug = config.get("debug", False)
        self.visualize_each = config.get("visualize_each", False)
        self.n_samples = config.get("n_samples", 5)

        os.makedirs(self.output_dir, exist_ok=True)

    def load_images(self) -> List[np.ndarray]:
        """
        Loads all grayscale images from the input folder.
        """
        images = []
        for filename in os.listdir(self.input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                img_path = os.path.join(self.input_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
        return images

    def generate_synthetic_kikuchi_images(self, size: Tuple[int, int] = (256, 256)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Debug mode: use real Kikuchi image and rotate it.
        """
        img = cv2.imread("/mnt/data/Med_Mn_10k_4x4_00190.png", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        rotated = self.rotate_image(img, angle=45)
        return img, rotated

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotates the image by a given angle.
        """
        center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rot_mat, image.shape[::-1], flags=cv2.INTER_LINEAR)
        return rotated

    def combine_images(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Adds two images safely and normalizes the result.
        """
        added = img1.astype(np.uint16) + img2.astype(np.uint16)
        normalized = cv2.normalize(added, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)

    def generate_synthetic_dataset(self):
        """
        Generates synthetic combinations of random image pairs.
        """
        if self.debug:
            img1, img2 = self.generate_synthetic_kikuchi_images()
            combined = self.combine_images(img1, img2)
            cv2.imwrite(os.path.join(self.output_dir, "synthetic_debug.png"), combined)
            if self.visualize_each:
                Visualizer.plot_image_and_histograms(img1, img2, combined)
            return

        images = self.load_images()
        assert len(images) >= 2, "Need at least two images to combine."

        for i in range(self.n_samples):
            img1, img2 = random.sample(images, 2)
            combined = self.combine_images(img1, img2)
            filename = f"synthetic_{i+1:03d}.png"
            cv2.imwrite(os.path.join(self.output_dir, filename), combined)
            if self.visualize_each:
                Visualizer.plot_image_and_histograms(img1, img2, combined)

class Visualizer:
    """
    Handles visualization of images and their histograms in a multi-tile layout.
    """

    @staticmethod
    def plot_image_and_histograms(img1: np.ndarray, img2: np.ndarray, combined: np.ndarray):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Display images
        axes[0, 0].imshow(img1, cmap='gray')
        axes[0, 0].set_title('Image 1')
        axes[0, 1].imshow(img2, cmap='gray')
        axes[0, 1].set_title('Image 2')
        axes[0, 2].imshow(combined, cmap='gray')
        axes[0, 2].set_title('Combined Image')

        # Display histograms
        for ax, img, title in zip(axes[1], [img1, img2, combined], ['Hist 1', 'Hist 2', 'Combined Hist']):
            ax.hist(img.ravel(), bins=256, range=(2, 255), fc='black', ec='white')
            ax.set_title(title)

        for ax in axes.ravel():
            ax.axis('off') if ax.images else None

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Example configuration
    config = {
        "input_dir": r"C:\Users\kvman\Documents\ml_data\TestingDataForPrepareCyclegANCode\simulated",
        "output_dir": "./output",
        "debug": False,
        "visualize_each": False,
        "n_samples": 5
    }

    combiner = ImageCombiner(config)
    combiner.generate_synthetic_dataset()
