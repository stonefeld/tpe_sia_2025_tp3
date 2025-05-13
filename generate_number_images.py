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
    image = Image.new("L", size, color=255)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", base_font_size)

    except Exception:
        font = ImageFont.load_default()

    size_factor = 1 + random.uniform(-size_variation, size_variation)
    font_size = int(base_font_size * size_factor)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)

    except Exception:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), str(number), font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    base_x = (size[0] - text_width) // 2

    base_y = (size[1] - text_height) // 2 - int(text_height * 0.1)

    max_x_offset = min(int(size[0] * position_variation), (size[0] - text_width) // 2)
    max_y_offset = min(int(size[1] * position_variation), (size[1] - text_height) // 2)

    x_offset = random.randint(-max_x_offset, max_x_offset)
    y_offset = random.randint(-max_y_offset, max_y_offset)

    x = base_x + x_offset
    y = base_y + y_offset

    draw.text((x, y), str(number), font=font, fill=0)

    img_array = np.array(image)

    angle = random.uniform(rotation_range[0], rotation_range[1])
    image = image.rotate(angle, resample=Image.BICUBIC, fillcolor=255)
    img_array = np.array(image)

    noise = np.random.normal(0, noise_level * 255, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(img_array)


def generate_dataset(start_num, end_num, images_per_number, output_dir, is_training=True):
    os.makedirs(output_dir, exist_ok=True)

    noise_level = 0 if is_training else 0.15
    rotation_range = (-7, 7) if is_training else (-2, 2)
    size_variation = 0.2 if is_training else 0.05
    position_variation = 0.1 if is_training else 0.05

    for num in range(start_num, end_num + 1):
        for i in range(images_per_number):
            img = create_number_image(
                num,
                size=(35, 35),
                base_font_size=28,
                noise_level=noise_level,
                rotation_range=rotation_range,
                size_variation=size_variation,
                position_variation=position_variation,
            )

            img_path = os.path.join(output_dir, f"imagen_{num}_{i}.png")
            img.save(img_path)


def main():
    start_num = 0
    end_num = 9
    training_images_per_number = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    testing_images_per_number = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    print("Generating training set...")
    generate_dataset(start_num, end_num, training_images_per_number, "assets/training_set", is_training=True)

    print("Generating testing set...")
    generate_dataset(start_num, end_num, testing_images_per_number, "assets/testing_set", is_training=False)

    print("Done! Generated:")
    print(f"- Training set: {training_images_per_number} images per number ({start_num}-{end_num})")
    print(f"- Testing set: {testing_images_per_number} images per number ({start_num}-{end_num})")


if __name__ == "__main__":
    main()
