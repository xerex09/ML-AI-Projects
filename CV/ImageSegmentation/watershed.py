import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import segmentation, color, feature
from scipy import ndimage as ndi
import logging
from typing import Tuple, Optional
import time
from functools import wraps

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the input and output folders
INPUT_FOLDER = 'data'
OUTPUT_FOLDER = 'output'

# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Decorator function to measure elapsed time and log it
def log_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Elapsed time for {func.__name__}: {elapsed_time:.4f} seconds")
        return result
    return wrapper

def preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the input image for better segmentation results.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image_rgb, image_gray

def compute_markers(image_gray: np.ndarray) -> np.ndarray:
    """
    Compute markers for watershed segmentation.
    """
    distance = ndi.distance_transform_edt(image_gray)
    local_maxi = feature.peak_local_max(distance, labels=image_gray, footprint=np.ones((3, 3)))
    markers = np.zeros(image_gray.shape, dtype=bool)
    markers[tuple(local_maxi.T)] = True
    markers = ndi.label(markers)[0]
    return markers

def segment_image(image_rgb: np.ndarray, image_gray: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """
    Perform watershed segmentation on the input image.
    """
    labels = segmentation.watershed(-image_gray, markers, mask=image_gray)
    segmented_image = color.label2rgb(labels, image_rgb, kind='avg')
    return segmented_image

@log_time
def process_image(file_path: str, output_folder: str) -> Optional[Tuple[str, np.ndarray]]:
    """
    Process a single image file.
    """
    try:
        logging.info(f"Processing image: {file_path}")
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Failed to load image: {file_path}")
        
        image_rgb, image_gray = preprocess_image(image)
        markers = compute_markers(image_gray)
        segmented_image = segment_image(image_rgb, image_gray, markers)
        
        filename = os.path.splitext(os.path.basename(file_path))[0]
        segmented_image_path = os.path.join(output_folder, f'{filename}_watershed.png')
        
        # Check if the file already exists and delete it
        if os.path.exists(segmented_image_path):
            logging.warning(f"File {segmented_image_path} already exists. Overwriting.")
            os.remove(segmented_image_path)
        
        cv2.imwrite(segmented_image_path, cv2.cvtColor(np.uint8(segmented_image * 255), cv2.COLOR_RGB2BGR))
        
        logging.info(f"Processed and saved: {segmented_image_path}")
        return filename, segmented_image
    
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
        return None

def main():
    logging.info(f"Looking for image files in {INPUT_FOLDER}")
    image_files = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    
    if not image_files:
        logging.warning(f"No image files found in {INPUT_FOLDER}")
        return

    logging.info(f"Found {len(image_files)} image files")

    results = []
    for file_path in image_files:
        result = process_image(file_path, OUTPUT_FOLDER)
        if result:
            results.append(result)
    
    if not results:
        logging.warning("No images were successfully processed.")
        return
    
    logging.info(f"Successfully processed {len(results)} images")

    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (filename, segmented_image) in enumerate(results):
        axes[i].imshow(segmented_image)
        axes[i].set_title(f'{filename}')
        axes[i].axis('off')
    
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'segmentation_results.png'))
    plt.close()

    logging.info("Script execution completed successfully")

if __name__ == "__main__":
    main()
