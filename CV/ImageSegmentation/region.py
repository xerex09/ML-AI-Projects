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

# Region Growing function
def region_growing(img, seed, threshold=20):
    h, w = img.shape
    segmented = np.zeros((h, w))
    segmented[seed[0], seed[1]] = 1
    grow = [seed]
    while grow:
        x, y = grow.pop()
        for i in range(-1, 2):
            for j in range(-1, 2):
                if 0 <= x + i < h and 0 <= y + j < w and segmented[x + i, y + j] == 0:
                    if abs(int(img[x + i, y + j]) - int(img[seed[0], seed[1]])) < threshold:
                        segmented[x + i, y + j] = 1
                        grow.append((x + i, y + j))
    return segmented

# Function to process and save segmented images
@log_time
def process_image(file_path: str, output_folder: str) -> Optional[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Process a single image file using SLIC superpixels and region growing.

    Args:
        file_path (str): Path to the image file.
        output_folder (str): Path to the output folder where segmented images will be saved.

    Returns:
        Optional[Tuple[str, np.ndarray, np.ndarray]]: A tuple containing the filename, 
        SLIC segmented image array, and region grown image array, or None if there 
        was an error processing the image.
    """
    try:
        # Load image
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Failed to load image: {file_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get the filename without extension
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # SLIC Superpixels
        segments = segmentation.slic(image_rgb, n_segments=250, compactness=10, start_label=1)
        segmented_image = color.label2rgb(segments, image_rgb, kind='avg')
        
        # Region Growing
        seed = (100, 100)  # Choose a seed point
        region_grown = region_growing(image_gray, seed)
        
        # Save the segmented images
        cv2.imwrite(os.path.join(output_folder, f'{filename}_slic_superpixels.jpg'), cv2.cvtColor(np.uint8(segmented_image * 255), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_folder, f'{filename}_region_growing.jpg'), np.uint8(region_grown * 255))
        
        # Display the images
        plt.figure(figsize=(10, 5))
        plt.subplot(121), plt.imshow(segmented_image), plt.title('SLIC Superpixels')
        plt.subplot(122), plt.imshow(region_grown, cmap='gray'), plt.title('Region Growing')
        plt.suptitle(filename)
        plt.show()
        
        return filename, segmented_image, region_grown
    
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
