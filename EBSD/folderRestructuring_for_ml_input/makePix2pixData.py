#!/usr/bin/env python3
import os
import random
import shutil
from PIL import Image

def split_matched_images(
    input_dir,
    output_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
):
    """
    Splits matched images in subfolders A and B into train, val, test sets,
    preserving one-to-one correspondence (same filenames in each set).
    
    Args:
        input_dir (str): Path to the input folder (containing 'A' and 'B').
        output_dir (str): Path to the output directory.
        train_ratio (float): Fraction of pairs that go to train set (default 0.8).
        val_ratio (float): Fraction of pairs that go to val set (default 0.1).
        test_ratio (float): Fraction of pairs that go to test set (default 0.1).
        
    Example Usage:
        split_matched_images("/path/to/input", "/path/to/output", 0.8, 0.1, 0.1)
    """
    # 1. Directories for A and B in the input
    a_input_dir = os.path.join(input_dir, "A")
    b_input_dir = os.path.join(input_dir, "B")
    
    if not os.path.isdir(a_input_dir) or not os.path.isdir(b_input_dir):
        raise ValueError("Input directory must contain subdirectories 'A' and 'B'.")
    
    # 2. Create output subdirs: A/train, A/val, A/test, B/train, B/val, B/test
    subfolders = ["train", "val", "test"]
    for subdir in ["A", "B"]:
        for sf in subfolders:
            out_path = os.path.join(output_dir, subdir, sf)
            os.makedirs(out_path, exist_ok=True)
    
    # 3. Gather matching filenames in A and B
    #    We'll compare just the filenames (e.g., 1.jpg) without extension if needed.
    a_files = sorted(f for f in os.listdir(a_input_dir) if os.path.isfile(os.path.join(a_input_dir, f)))
    b_files = sorted(f for f in os.listdir(b_input_dir) if os.path.isfile(os.path.join(b_input_dir, f)))
    
    # Convert lists to sets for quick intersection
    a_set = set(a_files)
    b_set = set(b_files)
    
    # Matched filenames appear in both subfolders
    matched_filenames = sorted(a_set.intersection(b_set))
    if not matched_filenames:
        print("No common filenames found in A and B. Exiting.")
        return
    
    print(f"Found {len(matched_filenames)} matched files in A and B.")
    
    # 4. (Optional) Filter out pairs that differ in image size
    valid_pairs = []
    for filename in matched_filenames:
        a_path = os.path.join(a_input_dir, filename)
        b_path = os.path.join(b_input_dir, filename)
        
        try:
            with Image.open(a_path) as imgA, Image.open(b_path) as imgB:
                if imgA.size == imgB.size:  # same width, height
                    valid_pairs.append(filename)
                else:
                    print(f"Skipping '{filename}' due to mismatched sizes: {imgA.size} vs {imgB.size}.")
        except Exception as e:
            print(f"Skipping '{filename}' due to error opening image: {e}")
    
    # If no valid pairs remain, exit
    if not valid_pairs:
        print("No valid image pairs with matching sizes. Exiting.")
        return
    
    print(f"{len(valid_pairs)} valid pairs remain after size check.")
    
    # 5. Shuffle and split into train, val, test
    random.shuffle(valid_pairs)
    total = len(valid_pairs)
    
    train_count = int(train_ratio * total)
    val_count   = int(val_ratio * total)
    # test_count = total - train_count - val_count  # Alternative formula
    test_count = int(test_ratio * total)
    
    # Ensure we cover rounding issues
    # e.g., if train_ratio=0.8 => 80% of 7 is 5.6 => int(5.6)=5,
    # might not sum up precisely to total. So let's handle leftover.
    leftover = total - (train_count + val_count + test_count)
    train_count += leftover  # put leftover in train or do some logic

    train_files = valid_pairs[:train_count]
    val_files   = valid_pairs[train_count:train_count + val_count]
    test_files  = valid_pairs[train_count + val_count:train_count + val_count + test_count]
    
    print(f"Splitting {total} pairs into:")
    print(f"  Train: {len(train_files)}")
    print(f"  Val:   {len(val_files)}")
    print(f"  Test:  {len(test_files)}")
    
    # 6. Copy files to the appropriate output subfolders
    def copy_pairs_to_folder(file_list, folder_name):
        for fname in file_list:
            srcA = os.path.join(a_input_dir, fname)
            srcB = os.path.join(b_input_dir, fname)
            
            dstA = os.path.join(output_dir, "A", folder_name, fname)
            dstB = os.path.join(output_dir, "B", folder_name, fname)
            
            shutil.copy2(srcA, dstA)  # copy2 preserves metadata
            shutil.copy2(srcB, dstB)
    
    copy_pairs_to_folder(train_files, "train")
    copy_pairs_to_folder(val_files,   "val")
    copy_pairs_to_folder(test_files,  "test")
    
    print("Done! Your splits are in:", output_dir)


if __name__ == "__main__":
    """
    Example usage:
      python split_matched_images.py \
        --input_dir /path/to/data \
        --output_dir /path/to/output
    
    In /path/to/data, you should have:
      /path/to/data/A
      /path/to/data/B
    
    And the script will create:
      /path/to/output/A/train, /val, /test
      /path/to/output/B/train, /val, /test
    with an 80/10/10 split of matched files.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Split matched images from subfolders A, B into train/val/test sets.")
    parser.add_argument("--input_dir", required=False, help="Path to the input directory (containing A, B).", default="/home/lus04/kvmani/ml_works/kaushal_2025/outputs/data10.0/tiff/")
    parser.add_argument("--output_dir", required=False, help="Path to the output directory.", default="/home/lus04/kvmani/ml_works/kaushal_2025/outputs/data10.0/tiff_for_ML/")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Fraction of pairs for train set.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of pairs for val set.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Fraction of pairs for test set.")
    args = parser.parse_args()
    
    split_matched_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
