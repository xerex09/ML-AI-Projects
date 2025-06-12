import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the input and output folders
input_folder = 'data'
output_folder = 'output'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to process and save thresholded images
def process_image(file_path, output_folder):
    # Load image
    image = cv2.imread(file_path, 0)
    
    # Get the filename without extension
    filename = os.path.splitext(os.path.basename(file_path))[0]
    
    # Global Thresholding
    _, global_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
    
    # Otsu's Thresholding
    _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Save the thresholded images
    cv2.imwrite(os.path.join(output_folder, f'{filename}_global_thresh.jpg'), global_thresh)
    cv2.imwrite(os.path.join(output_folder, f'{filename}_adaptive_thresh.jpg'), adaptive_thresh)
    cv2.imwrite(os.path.join(output_folder, f'{filename}_otsu_thresh.jpg'), otsu_thresh)
    
    # Display the images
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(global_thresh, cmap='gray'), plt.title('Global Thresholding')
    plt.subplot(132), plt.imshow(adaptive_thresh, cmap='gray'), plt.title('Adaptive Thresholding')
    plt.subplot(133), plt.imshow(otsu_thresh, cmap='gray'), plt.title('Otsu\'s Thresholding')
    plt.suptitle(filename)
    plt.show()

# Process all images in the input folder
for file_name in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file_name)
    if os.path.isfile(file_path):
        process_image(file_path, output_folder)
