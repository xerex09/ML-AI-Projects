import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
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

@log_time
def process_image(file_path: str, output_folder: str) -> Optional[Tuple[str, np.ndarray]]:
    """
    Process a single image file to perform edge detection using Canny, Sobel, and Laplacian of Gaussian methods.

    Args:
        file_path (str): Path to the image file.
        output_folder (str): Path to the output folder where processed images will be saved.

    Returns:
        Optional[Tuple[str, np.ndarray]]: A tuple containing the filename and the processed images, or None if there was an error processing the image.
    """
    try:
        # Load image in grayscale
        image = cv2.imread(file_path, 0)
        if image is None:
            raise ValueError(f"Failed to load image: {file_path}")
        
        # Get the filename without extension
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Canny Edge Detection
        canny_edges = cv2.Canny(image, 100, 200)
        
        # Sobel Edge Detection
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        sobel_edges = cv2.magnitude(sobelx, sobely)
        
        # Laplacian of Gaussian (LoG)
        log_edges = cv2.Laplacian(image, cv2.CV_64F)
        
        # Save the edge-detected images
        cv2.imwrite(os.path.join(output_folder, f'{filename}_canny_edges.jpg'), canny_edges)
        cv2.imwrite(os.path.join(output_folder, f'{filename}_sobel_edges.jpg'), np.uint8(sobel_edges))
        cv2.imwrite(os.path.join(output_folder, f'{filename}_log_edges.jpg'), np.uint8(log_edges))
        
        # Display the images
        plt.figure(figsize=(15, 5))
        plt.subplot(131), plt.imshow(canny_edges, cmap='gray'), plt.title('Canny Edge Detection')
        plt.subplot(132), plt.imshow(sobel_edges, cmap='gray'), plt.title('Sobel Edge Detection')
        plt.subplot(133), plt.imshow(log_edges, cmap='gray'), plt.title('Laplacian of Gaussian')
        plt.suptitle(filename)
        plt.show()
        
        return filename, (canny_edges, sobel_edges, log_edges)
    
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
