import os
import sys
import argparse
import random
from PIL import Image, ImageOps

def generate_subcrops_512(image_path, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1) Load the image
    image = Image.open(image_path)
    width, height = image.size

    # 2) Calculate how many subcrops we want in x and y directions
    num_crops_x = width // 512 + 1
    num_crops_y = height // 512 + 1

    # 3) Compute the stride (overlap) in x and y
    if num_crops_x > 1:
        stride_x = (width - 512) / (num_crops_x - 1)
    else:
        stride_x = 0

    if num_crops_y > 1:
        stride_y = (height - 512) / (num_crops_y - 1)
    else:
        stride_y = 0

    # Get base name (e.g., "my_image" from "my_image.tiff")
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    subcrop_index = 0

    # 4) Loop over each grid position
    for i in range(num_crops_x):
        for j in range(num_crops_y):
            # Calculate top-left corner (x0, y0)
            x0 = int(round(i * stride_x))
            y0 = int(round(j * stride_y))
            # Bottom-right corner (x1, y1)
            x1 = x0 + 512
            y1 = y0 + 512

            # Ensure we don't exceed the image boundary
            if x1 > width:
                x1 = width
                x0 = width - 512
            if y1 > height:
                y1 = height
                y0 = height - 512

            # Crop sub-image
            sub_image = image.crop((x0, y0, x1, y1))

            # 5) Random rotation
            rotation_angle = random.choice([0, 90, 180, 270])
            # expand=True keeps all pixels after rotation, possibly producing >512 size if rotating 90/270
            # If you want to strictly keep 512x512, use expand=False (but corners will be cropped).
            sub_image = sub_image.rotate(rotation_angle, expand=True)

            # Random flip
            flip_mode = random.choice(['horizontal', 'vertical', None])
            if flip_mode == 'horizontal':
                sub_image = ImageOps.mirror(sub_image)
            elif flip_mode == 'vertical':
                sub_image = ImageOps.flip(sub_image)

            # 6) Save each subcrop
            out_filename = f"{base_name}_subcrop_{subcrop_index}.tiff"
            out_path = os.path.join(output_dir, out_filename)
            sub_image.save(out_path)

            subcrop_index += 1

    print(f"Finished generating {subcrop_index} subcrops from {image_path} in {output_dir}.")

def main():
    parser = argparse.ArgumentParser(description="Generate 512x512 subcrops of an image with random rotation/flip.")
    parser.add_argument("--input_file", required=True, help="Path to the input image.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the subcrops.")
    args = parser.parse_args()

    generate_subcrops_512(args.input_file, args.output_dir)

if __name__ == "__main__":
    main()
