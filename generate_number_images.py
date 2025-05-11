import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_number_image(
    number,
    size=(35, 35),
    base_font_size=30,
    noise_level=0.1,
    rotation_range=(-10, 10),
    size_variation=0.3,
    position_variation=0.2,
):
    """
    Create an image of a number with random variations.

    Args:
        number: The number to draw
        size: Tuple of (width, height) for the image
        base_font_size: Base size of the font
        noise_level: Amount of random noise to add (0-1)
        rotation_range: Tuple of (min_rotation, max_rotation) in degrees
        size_variation: How much to vary the size (0-1)
        position_variation: How much to vary the position (0-1)
    """
    # Create a white background
    image = Image.new("L", size, color=255)
    draw = ImageDraw.Draw(image)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", base_font_size)

    except Exception:
        font = ImageFont.load_default()

    # Add size variation
    size_factor = 1 + random.uniform(-size_variation, size_variation)
    font_size = int(base_font_size * size_factor)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)

    except Exception:
        font = ImageFont.load_default()

    # Get text size to center it
    text_bbox = draw.textbbox((0, 0), str(number), font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Calculate base position (center)
    base_x = (size[0] - text_width) // 2

    # Adjust vertical centering to account for font metrics
    # Move the text up slightly to compensate for the baseline
    base_y = (size[1] - text_height) // 2 - int(text_height * 0.1)

    # Calculate maximum allowed offsets to keep text within bounds
    max_x_offset = min(int(size[0] * position_variation), (size[0] - text_width) // 2)  # Based on position variation  # Based on text width
    max_y_offset = min(int(size[1] * position_variation), (size[1] - text_height) // 2)  # Based on position variation  # Based on text height

    # Add position variation
    x_offset = random.randint(-max_x_offset, max_x_offset)
    y_offset = random.randint(-max_y_offset, max_y_offset)

    # Calculate final position
    x = base_x + x_offset
    y = base_y + y_offset

    # Draw the number
    draw.text((x, y), str(number), font=font, fill=0)

    # Convert to numpy array for processing
    img_array = np.array(image)

    # Add random rotation
    angle = random.uniform(rotation_range[0], rotation_range[1])
    image = image.rotate(angle, resample=Image.BICUBIC, fillcolor=255)
    img_array = np.array(image)

    # Add random noise
    noise = np.random.normal(0, noise_level * 255, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(img_array)


def generate_dataset(start_num, end_num, images_per_number, output_dir, is_training=True):
    """
    Generate a dataset of number images.

    Args:
        start_num: Starting number (inclusive)
        end_num: Ending number (inclusive)
        images_per_number: Number of images to generate per number
        output_dir: Directory to save the images
        is_training: Whether this is a training set (affects variation parameters)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Adjust parameters based on whether it's training or testing
    noise_level = 0 if is_training else 0.15
    rotation_range = (-7, 7) if is_training else (-2, 2)
    size_variation = 0.2 if is_training else 0.05  # Reduced size variation
    position_variation = 0.1 if is_training else 0.05  # Reduced position variation

    # Generate images for each number
    for num in range(start_num, end_num + 1):
        for i in range(images_per_number):
            # Create image with variations
            img = create_number_image(
                num,
                size=(35, 35),
                base_font_size=28,  # Slightly reduced base font size
                noise_level=noise_level,
                rotation_range=rotation_range,
                size_variation=size_variation,
                position_variation=position_variation,
            )

            # Save image
            img_path = os.path.join(output_dir, f"imagen_{num}_{i}.png")
            img.save(img_path)

            # Convert to text file format (35x35 grid of 0s and 1s)
            # img_array = np.array(img)
            # binary_array = (img_array < 128).astype(int)  # Convert to binary

            # Save as text file
            # txt_path = os.path.join(output_dir, f"imagen_{num}_{i}.txt")
            # np.savetxt(txt_path, binary_array, fmt='%d')


def main():
    # Parameters
    start_num = 0
    end_num = 9
    training_images_per_number = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    testing_images_per_number = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    # Generate training set
    print("Generating training set...")
    generate_dataset(start_num, end_num, training_images_per_number, "assets/training_set", is_training=True)

    # Generate testing set
    print("Generating testing set...")
    generate_dataset(start_num, end_num, testing_images_per_number, "assets/testing_set", is_training=False)

    print("Done! Generated:")
    print(f"- Training set: {training_images_per_number} images per number ({start_num}-{end_num})")
    print(f"- Testing set: {testing_images_per_number} images per number ({start_num}-{end_num})")


if __name__ == "__main__":
    main()
