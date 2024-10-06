import os
from PIL import Image
import numpy as np

def preprocess_image(image_path, output_path):
    image = Image.open(image_path).convert('RGBA')
    image = image.resize((64, 64))
    image.save(output_path)

def preprocess_dataset(raw_dir, processed_dir):
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    for root, _, files in os.walk(raw_dir):
        for file_name in files:
            if file_name.endswith('.png'):
                raw_image_path = os.path.join(root, file_name)
                relative_path = os.path.relpath(raw_image_path, raw_dir)
                processed_image_path = os.path.join(processed_dir, relative_path)

                # Tạo thư mục con tương ứng trong thư mục processed_dir
                processed_image_dir = os.path.dirname(processed_image_path)
                if not os.path.exists(processed_image_dir):
                    os.makedirs(processed_image_dir)
                
                preprocess_image(raw_image_path, processed_image_path)

if __name__ == "__main__":
    preprocess_dataset('data/raw', 'data/processed')