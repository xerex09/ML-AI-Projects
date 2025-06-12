import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csgraph, csr_matrix, diags
from scipy.sparse.linalg import eigsh
from skimage.segmentation import mark_boundaries, slic
from skimage.util import img_as_float
from sklearn.cluster import KMeans


def preprocess_image(image: np.ndarray):
    # Convert BGR to RGB if needed and normalize to float
    if image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    return img_as_float(image_rgb)

def compute_affinity_matrix(image: np.ndarray, n_segments=100, compactness=10):
    # Generate superpixels
    print("Generating superpixels...")
    segments = slic(image, n_segments=n_segments, compactness=compactness)
    num_segments = np.max(segments) + 1
    print(f"Generated {num_segments} superpixels")
    
    # Compute superpixel features (color mean and position)
    pixel_count = np.zeros(num_segments)
    color_sum = np.zeros((num_segments, 3))
    position_sum = np.zeros((num_segments, 2))
    
    height, width = segments.shape
    
    # Accumulate color and position information
    for i in range(height):
        for j in range(width):
            segment_id = segments[i, j]
            pixel_count[segment_id] += 1
            color_sum[segment_id] += image[i, j]
            position_sum[segment_id] += [i, j]
    
    # Compute means
    with np.errstate(divide='ignore', invalid='ignore'):
        color_mean = color_sum / pixel_count[:, np.newaxis]
        position_mean = position_sum / pixel_count[:, np.newaxis]
    
    # Fix any NaN values (in case a segment has no pixels)
    color_mean = np.nan_to_num(color_mean)
    position_mean = np.nan_to_num(position_mean)
    
    # Normalize position to [0, 1] range
    position_mean[:, 0] /= height
    position_mean[:, 1] /= width
    
    # Compute pairwise distances between superpixels
    print("Computing superpixel affinities...")
    row_indices = []
    col_indices = []
    values = []
    
    # Parameters for the affinity calculation
    color_weight = 2.0
    position_weight = 0.1
    
    # Compute affinities between all pairs of superpixels
    for i in range(num_segments):
        for j in range(num_segments):
            if i != j:
                # Compute color distance
                color_dist = np.sum((color_mean[i] - color_mean[j])**2)
                
                # Compute spatial distance
                spatial_dist = np.sum((position_mean[i] - position_mean[j])**2)
                
                # Combined similarity (higher value = more similar)
                similarity = np.exp(-color_weight * color_dist) * \
                             np.exp(-position_weight * spatial_dist)
                
                if similarity > 0.01:  # Threshold to keep matrix sparse
                    row_indices.append(i)
                    col_indices.append(j)
                    values.append(similarity)
    
    # Create sparse affinity matrix
    W = csr_matrix((values, (row_indices, col_indices)), 
                  shape=(num_segments, num_segments))
    
    # Make sure it's symmetric
    W = (W + W.T) / 2
    
    return W, segments

def compute_laplacian(W):
    # Compute normalized Laplacian: L = I - D^(-1/2) W D^(-1/2)
    if not hasattr(W, 'toarray'):
        W = csr_matrix(W)
    
    d = W.sum(axis=1).A1
    d_sqrt_inv = 1.0 / np.sqrt(d + 1e-10)
    D_sqrt_inv = diags(d_sqrt_inv)
    
    normalized_W = D_sqrt_inv @ W @ D_sqrt_inv
    return normalized_W, d_sqrt_inv

def eigen_segmentation(image: np.ndarray, num_segments: int = 5):
    image_float = preprocess_image(image)
    
    # Create affinity matrix from superpixels
    W, superpixels = compute_affinity_matrix(image_float, n_segments=200, compactness=20)
    normalized_W, d_sqrt_inv = compute_laplacian(W)
    
    # Compute eigendecomposition
    print("Computing eigenvectors...")
    eigenvalues, eigenvectors = eigsh(normalized_W, k=num_segments+1, which='LA')
    eigenvalues = 1 - eigenvalues
    
    # Sort and select eigenvectors
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    features = eigenvectors[:, 1:num_segments+1]
    
    # Cluster feature space
    print("Clustering feature space...")
    kmeans = KMeans(n_clusters=num_segments, random_state=0, n_init=10)
    superpixel_labels = kmeans.fit_predict(features)
    
    # Map labels back to image
    height, width = superpixels.shape
    segmented_image = np.zeros((height, width), dtype=np.int32)
    for i in range(height):
        for j in range(width):
            segmented_image[i, j] = superpixel_labels[superpixels[i, j]]
    
    # Create boundary images
    outline_image = detect_segment_boundaries(segmented_image)
    boundaries_image = mark_boundaries(image_float, segmented_image)
    
    return segmented_image, outline_image, boundaries_image

def detect_segment_boundaries(segmented_image):
    # Find segment boundaries using morphological operations
    labels = np.unique(segmented_image)
    outline = np.zeros_like(segmented_image, dtype=np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    
    for label in labels:
        mask = (segmented_image == label).astype(np.uint8)
        boundary = cv2.dilate(mask, kernel) - mask
        outline[boundary > 0] = 255
    
    return outline

def main():
    image_path = 'data/saugat.jpeg'
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot load image: {image_path}")
        return

    print(f"Processing image: {image_path}, shape: {image.shape}")
    segmented_image, outline_image, boundaries_image = eigen_segmentation(image, num_segments=5)
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(segmented_image, cmap='viridis')
    plt.title("Segmented Image")
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(outline_image, cmap='gray')
    plt.title("Segment Outlines")
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(boundaries_image)
    plt.title("Boundaries Overlay")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()