import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import segmentation, color
import logging
import time
from typing import Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the input and output folders
input_folder = 'data'
output_folder = 'output'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

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

# Function to process and save segmented images using Felzenszwalb's method
@log_time
def process_image(file_path: str, output_folder: str) -> Optional[Tuple[str, np.ndarray]]:
    """
    Process a single image file using Felzenszwalb's method for segmentation.

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
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get the filename without extension
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Felzenszwalb's Efficient Graph-Based Segmentation
        segments_fz = segmentation.felzenszwalb(image_rgb, scale=100, sigma=0.5, min_size=50)
        
        # Create the segmented image
        segmented_image_fz = color.label2rgb(segments_fz, image_rgb, kind='avg')
        
        # Save the segmented image
        segmented_image_path = os.path.join(output_folder, f'{filename}_felzenszwalb.jpg')
        cv2.imwrite(segmented_image_path, cv2.cvtColor(np.uint8(segmented_image_fz * 255), cv2.COLOR_RGB2BGR))
        
        # Display the images
        plt.figure(figsize=(8, 8))
        plt.imshow(segmented_image_fz)
        plt.title(f'Felzenszwalb Graph-Based Segmentation: {filename}')
        plt.axis('off')  # Hide axes for better visualization
        plt.show()
        
        return filename, segmented_image_fz
    
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
