#!/usr/bin/env python3
"""
Segment an image (or folder of images) into distinct regions based on color similarity (flood-fill),
draw boundaries around each region, calculate area for each grain, and exclude small regions.

Usage:
  python color_flood_fill_segmentation.py --image /path/to/image_or_folder
                                         --tolerance 10
                                         --min_area 50
                                         --output_dir results
                                         --csv_output grains.csv
                                         --display

If --image points to a single file with a valid extension, the script processes just that image.
If --image is a directory, the script processes all images in that folder.

'--display' will show the result in an OpenCV window (one per processed image).
Close the window to move on to the next image in folder mode.
"""

import cv2
import numpy as np
import argparse
import sys
import os
from collections import deque
import csv

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

def is_image_file(path):
    """
    Check if the path ends with a valid image extension.
    """
    ext = os.path.splitext(path)[1].lower()
    return ext in VALID_EXTENSIONS

def color_is_close(c1, c2, tol=10):
    """
    Check if two colors c1, c2 (each a 3-element array [R, G, B])
    are within 'tol' in each channel.
    """
    diff = np.abs(c1.astype(np.int32) - c2.astype(np.int32))
    return np.all(diff <= tol)

def flood_fill_bfs(image, visited, start_y, start_x, current_label, labels, tolerance=10):
    """
    Performs a BFS flood-fill starting from (start_y, start_x).
    All connected pixels whose color is 'close enough' (within tolerance)
    to the base color are assigned 'current_label'.
    """
    height, width, _ = image.shape
    base_color = image[start_y, start_x]
    
    queue = deque()
    queue.append((start_y, start_x))
    
    while queue:
        y, x = queue.popleft()
        
        if visited[y, x]:
            continue
        
        visited[y, x] = True
        labels[y, x] = current_label
        
        # Check 4 neighbors
        for ny, nx in [(y-1,x), (y+1,x), (y,x-1), (y,x+1)]:
            if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx]:
                neighbor_color = image[ny, nx]
                if color_is_close(base_color, neighbor_color, tolerance):
                    queue.append((ny, nx))

def segment_by_color_flood_fill(image_path, tolerance=10):
    """
    Segments an RGB image into connected components based on color similarity.
    Returns (labels, num_segments) where:
      - labels: 2D integer array of segment IDs
      - num_segments: total number of distinct regions found
    """
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    # Convert BGR -> RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    height, width, _ = rgb_image.shape
    
    visited = np.zeros((height, width), dtype=bool)
    labels = np.zeros((height, width), dtype=np.int32)
    
    current_label = 1
    for y in range(height):
        for x in range(width):
            if not visited[y, x]:
                flood_fill_bfs(rgb_image, visited, y, x, current_label, labels, tolerance)
                current_label += 1
    
    num_segments = current_label - 1
    return labels, num_segments

def create_label_visualization(labels):
    """
    Maps each label to a random color for visualization (no boundaries yet).
    Labels==0 => black (ignored region).
    """
    height, width = labels.shape
    label_vis = np.zeros((height, width, 3), dtype=np.uint8)
    
    unique_labels = np.unique(labels)
    rng = np.random.default_rng(42)

    label_colors = {}
    for lbl in unique_labels:
        if lbl == 0:
            label_colors[lbl] = np.array([0, 0, 0], dtype=np.uint8)
        else:
            label_colors[lbl] = rng.integers(low=0, high=255, size=3, dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            label_vis[y, x] = label_colors[labels[y, x]]
    
    return label_vis

def draw_boundaries(labels, color_img, boundary_color=(0,0,0)):
    """
    Overlays boundaries on the color_img by checking if a pixel's label differs
    from any of its neighbors. boundary_color is (R,G,B).
    """
    height, width = labels.shape
    output_img = color_img.copy()
    
    for y in range(height):
        for x in range(width):
            current_label = labels[y, x]
            neighbors = [(y-1,x), (y+1,x), (y,x-1), (y,x+1)]
            for (ny, nx) in neighbors:
                if 0 <= ny < height and 0 <= nx < width:
                    if labels[ny, nx] != current_label:
                        # Mark this pixel as boundary
                        output_img[y, x] = boundary_color
                        break
    return output_img

def process_image(
    image_path,
    tolerance,
    min_area,
    boundary_color_tuple,
):
    """
    1) Segment the image with flood fill
    2) Exclude segments smaller than min_area
    3) Create color map
    4) Draw boundaries
    5) Return final boundary image (BGR) + list of (GrainID, LabelID, Area(px))
    """
    labels, num_segments = segment_by_color_flood_fill(image_path, tolerance)
    
    # Exclude small segments
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]  # skip label=0 if any
    for lbl in unique_labels:
        area_px = np.count_nonzero(labels == lbl)
        if area_px < min_area:
            labels[labels == lbl] = 0

    # Re-check final unique labels
    final_unique_labels = np.unique(labels)
    final_unique_labels = final_unique_labels[final_unique_labels != 0]

    # Build color visualization
    label_vis_rgb = create_label_visualization(labels)
    # Draw boundaries in RGB
    boundary_img_rgb = draw_boundaries(labels, label_vis_rgb, boundary_color=boundary_color_tuple)
    # Convert to BGR for saving/display
    boundary_img_bgr = cv2.cvtColor(boundary_img_rgb, cv2.COLOR_RGB2BGR)

    # Gather final grain info
    grain_info = []
    grain_id = 1
    for lbl in final_unique_labels:
        area_px = np.count_nonzero(labels == lbl)
        grain_info.append((grain_id, lbl, area_px))
        grain_id += 1

    return boundary_img_bgr, grain_info

def main():
    parser = argparse.ArgumentParser(
        description="Segment image(s) by color flood-fill, draw boundaries, measure areas, exclude small regions."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to an image OR a folder containing images."
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=10,
        help="Channel-wise color tolerance (0-255). Higher merges more similar colors."
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=50,
        help="Exclude segments smaller than this area (pixels)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/segmented_images/",
        help="Directory to save the segmented_with_boundaries output image(s)."
    )
    parser.add_argument(
        "--boundary_color",
        type=str,
        default="0,0,0",
        help="Boundary color in R,G,B format (default=black)."
    )
    parser.add_argument(
        "--csv_output",
        type=str,
        default="./output/grains_info/",
        help="CSV file for label area info (only valid segments)."
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="If set, displays each segmented image in a window (press any key to close)."
    )
    parser.add_argument(
        "--save_segmented_images",
        action="store_true",
        help="If set, displays each segmented image in a window (press any key to close)."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, displays each segmented image in a window (press any key to close)."
    )    
    
    args = parser.parse_args()
    input_path = args.image
    tolerance = args.tolerance
    min_area = args.min_area
    boundary_color_str = args.boundary_color
    output_dir = args.output_dir
    csv_path = args.csv_output
    do_display = args.display
    save_segmented_images = args.save_segmented_images
    debugMode = args.debug
    # Parse boundary color
    boundary_color_tuple = tuple(int(c.strip()) for c in boundary_color_str.split(","))  # e.g. "255,0,0" => (255,0,0)

    # Prepare CSV writer
    # We'll write: FileIndex, FileName, GrainID, LabelID, Area(px)
    os.makedirs(csv_path, exist_ok=True)
    csv_file_path = os.path.join(csv_path, "grain_info.csv")
    csv_file = open(csv_file_path, "w", newline="")
    writer = csv.writer(csv_file)
    if not debugMode:
        writer.writerow(["FileIndex", "FileName", "GrainID", "LabelID", "Area(px)"])

    # Helper to process a single file (returns the final image and the grains info).
    def process_and_save(file_index, img_path):
        print(f"\n[File #{file_index}] Processing: {img_path}")

        out_image_bgr, grain_info = process_image(
            image_path=img_path,
            tolerance=tolerance,
            min_area=min_area,
            boundary_color_tuple=boundary_color_tuple
        )

        # Build output name
        
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        if save_segmented_images:
            os.makedirs(output_dir, exist_ok=True)
            out_name = f"segmented_with_boundaries_{base_name}.png"
            out_path = os.path.join(output_dir, out_name)
            if not debugMode:
                cv2.imwrite(out_path, out_image_bgr)
            print(f"  -> Saved result to '{out_path}'")

        # Write grain info to CSV
        if not debugMode:
            for (grain_id, label_id, area_px) in grain_info:
                writer.writerow([file_index, base_name, grain_id, label_id, area_px])

        # (Optional) show in a window
        if do_display:
            cv2.imshow(f"Segmented: {base_name}", out_image_bgr)
            cv2.waitKey(0)
            cv2.destroyWindow(f"Segmented: {base_name}")

        print(f"  -> Found {len(grain_info)} grains (after filtering).")

    # Check if input_path is a file or a directory
    if os.path.isfile(input_path) and is_image_file(input_path):
        # Single image case
        process_and_save(file_index=1, img_path=input_path)
    elif os.path.isdir(input_path):
        # Folder case: process all image files in this directory
        all_files = os.listdir(input_path)
        image_files = sorted(f for f in all_files if is_image_file(f))
        if not image_files:
            print(f"No valid images found in folder: {input_path}")
            csv_file.close()
            sys.exit(0)

        file_index = 1
        for filename in image_files:
            img_path = os.path.join(input_path, filename)
            process_and_save(file_index=file_index, img_path=img_path)
            file_index += 1
    else:
        print(f"ERROR: '{input_path}' is not a valid image or directory.")
        csv_file.close()
        sys.exit(1)

    csv_file.close()
    print(f"\nAll processing done. CSV data written to '{csv_path}'.")
    
if __name__ == "__main__":
    main()
