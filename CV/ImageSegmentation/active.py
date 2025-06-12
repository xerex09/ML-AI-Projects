import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.filters import gaussian
from skimage.segmentation import active_contour
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

# Function to process and save segmented images using Active Contour Model
@log_time
def process_image(file_path: str, output_folder: str) -> Optional[Tuple[str, np.ndarray]]:
    """
    Process a single image file using the Active Contour Model (Snake).

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
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get the filename without extension
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Initialize snake
        s = np.linspace(0, 2 * np.pi, 400)
        x = 220 + 100 * np.cos(s)
        y = 100 + 100 * np.sin(s)
        init = np.array([x, y]).T
        
        # Active Contour Model
        snake = active_contour(gaussian(image_gray, 3), init, alpha=0.015, beta=10, gamma=0.001)
        
        # Save the snake model image
        snake_image_path = os.path.join(output_folder, f'{filename}_snake.jpg')
        
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(image_gray, cmap=plt.cm.gray)
        ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
        ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        plt.title('Active Contour Model')
        plt.savefig(snake_image_path)
        plt.close(fig)
        
        return filename, snake
    
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
