import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.color import label2rgb
from sklearn.cluster import KMeans
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

# Function to process and save K-means segmented images
@log_time
def process_image(file_path: str, output_folder: str) -> Optional[Tuple[str, np.ndarray]]:
    """
    Process a single image file using K-means clustering for segmentation.

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
        
        # Flatten the image
        h, w, c = image_rgb.shape
        pixel_values = image_rgb.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        
        # K-means Clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(pixel_values)
        
        # Reshape the labels back to the image shape
        labels = labels.reshape((h, w))
        
        # Create the segmented image
        segmented_image_kmeans = label2rgb(labels, image_rgb, kind='avg')
        
        # Save the segmented image
        segmented_image_path = os.path.join(output_folder, f'{filename}_kmeans.jpg')
        cv2.imwrite(segmented_image_path, cv2.cvtColor(np.uint8(segmented_image_kmeans * 255), cv2.COLOR_RGB2BGR))
        
        # Display the images
        plt.figure(figsize=(8, 8))
        plt.imshow(segmented_image_kmeans)
        plt.title(f'K-means Clustering: {filename}')
        plt.axis('off')  # Hide axes for better visualization
        plt.show()
        
        return filename, segmented_image_kmeans
    
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
