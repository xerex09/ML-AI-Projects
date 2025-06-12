import os
import logging
import time
from typing import Optional, Tuple
import urllib.request
import tarfile

import cv2
import numpy as np
import tensorflow as tf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the input and output folders
input_folder = 'data'
output_folder = 'output'
model_folder = 'model'

# Create the output and model folders if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)

# Model URL and local path
model_url = "http://download.tensorflow.org/models/deeplab_v3_plus_mobilenet_v3_large_cityscapes_trainfine_2019_11_15.tar.gz"
model_file = os.path.join(model_folder, "deeplab_model.tar.gz")
extracted_model_path = os.path.join(model_folder, "deeplab_model")

# Download and extract the model if it doesn't exist
if not os.path.exists(extracted_model_path):
    logging.info("Downloading DeepLab model...")
    urllib.request.urlretrieve(model_url, model_file)
    
    logging.info("Extracting DeepLab model...")
    with tarfile.open(model_file, 'r:gz') as tar:
        tar.extractall(path=model_folder)
    
    os.rename(os.path.join(model_folder, "deeplab_v3_plus_mobilenet_v3_large_cityscapes_trainfine"), extracted_model_path)
    os.remove(model_file)

# Load DeepLab model
model = tf.saved_model.load(extracted_model_path)

def run_deeplab(image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    predictions = model.signatures['serving_default'](input_tensor)
    result_image = predictions['output_0'].numpy().squeeze()
    return result_image

# Decorator function to measure elapsed time and log it
def log_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Elapsed time for {func.__name__}: {elapsed_time:.4f} seconds")
        return result
    return wrapper

@log_time
def process_image(file_path: str, output_folder: str) -> Optional[Tuple[str, np.ndarray]]:
    """
    Process a single image file using the DeepLab model for semantic segmentation.

    Args:
        file_path (str): Path to the image file.
        output_folder (str): Path to the output folder where segmented images will be saved.

    Returns:
        Optional[Tuple[str, np.ndarray]]: A tuple containing the filename and segmented image array,
        or None if there was an error processing the image.
    """
    try:
        # Load image
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Failed to load image: {file_path}")
        
        # Convert image to RGB (since OpenCV loads BGR by default)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to 513x513 (DeepLab default input size)
        resized_image = cv2.resize(image_rgb, (513, 513))
        
        # Perform semantic segmentation using DeepLab model
        segmented_mask = run_deeplab(resized_image)
        
        # Resize the mask back to original image size
        segmented_mask = cv2.resize(segmented_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Apply a color map to the segmented mask
        colored_mask = cv2.applyColorMap(np.uint8(segmented_mask * 10), cv2.COLORMAP_JET)
        
        # Blend the original image with the colored mask
        alpha = 0.7
        blended_image = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        
        # Get the filename without extension
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Save the segmented image
        segmented_image_path = os.path.join(output_folder, f'{filename}_segmented.jpg')
        cv2.imwrite(segmented_image_path, blended_image)
        
        return filename, blended_image
    
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
        return None

def main():
    logging.info(f"Looking for image files in {input_folder}")
    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    
    if not image_files:
        logging.warning(f"No image files found in {input_folder}")
        return

    logging.info(f"Found {len(image_files)} image files")

    results = []
    for file_path in image_files:
        result = process_image(file_path, output_folder)
        if result:
            results.append(result)
    
    if not results:
        logging.warning("No images were successfully processed.")
        return
    
    logging.info(f"Successfully processed {len(results)} images")

if __name__ == "__main__":
    main()