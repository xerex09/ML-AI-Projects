import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.sparse import csgraph, csr_matrix, diags
from scipy.sparse.linalg import eigsh
from skimage.segmentation import mark_boundaries, slic
from skimage.util import img_as_float
from sklearn.cluster import KMeans
import threading
import time

class ImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Image Segmentation Explorer")
        self.root.geometry("1400x900")
        
        # Variables
        self.image_path = tk.StringVar()
        self.original_image = None
        self.processed_results = None
        
        # Parameter variables
        self.n_superpixels = tk.IntVar(value=500)
        self.superpixel_compactness = tk.DoubleVar(value=30.0)
        self.color_weight = tk.DoubleVar(value=2.0)
        self.position_weight = tk.DoubleVar(value=0.1)
        self.similarity_threshold = tk.DoubleVar(value=0.1)
        self.num_segments = tk.IntVar(value=8)
        
        self.setup_gui()
        
    def setup_gui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel for controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # File selection
        file_frame = ttk.LabelFrame(control_frame, text="Image Selection", padding="5")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Select Image", command=self.select_image).pack(pady=5)
        ttk.Label(file_frame, textvariable=self.image_path, wraplength=250).pack(pady=5)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(control_frame, text="Segmentation Parameters", padding="5")
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Superpixel parameters
        superpixel_frame = ttk.LabelFrame(params_frame, text="Superpixel Generation", padding="5")
        superpixel_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.create_parameter_control(superpixel_frame, "Number of Superpixels:", 
                                    self.n_superpixels, 100, 2000, 50)
        self.create_parameter_control(superpixel_frame, "Compactness:", 
                                    self.superpixel_compactness, 1.0, 100.0, 5.0)
        
        # Affinity matrix parameters
        affinity_frame = ttk.LabelFrame(params_frame, text="Affinity Matrix", padding="5")
        affinity_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.create_parameter_control(affinity_frame, "Color Weight:", 
                                    self.color_weight, 0.1, 10.0, 0.5)
        self.create_parameter_control(affinity_frame, "Position Weight:", 
                                    self.position_weight, 0.01, 2.0, 0.05)
        self.create_parameter_control(affinity_frame, "Similarity Threshold:", 
                                    self.similarity_threshold, 0.01, 1.0, 0.05)
        
        # Segmentation parameters
        seg_frame = ttk.LabelFrame(params_frame, text="Final Segmentation", padding="5")
        seg_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.create_parameter_control(seg_frame, "Number of Segments:", 
                                    self.num_segments, 2, 20, 1)
        
        # Process button
        self.process_btn = ttk.Button(control_frame, text="Process Image", 
                                    command=self.process_image_threaded, state=tk.DISABLED)
        self.process_btn.pack(pady=10, fill=tk.X)
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Select an image to begin")
        self.status_label.pack(pady=5)
        
        # Right panel for results
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create matplotlib figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        self.fig.tight_layout(pad=3.0)
        
        # Embed matplotlib in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, results_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize empty plots
        self.clear_plots()
        
    def create_parameter_control(self, parent, label, variable, min_val, max_val, increment):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(frame, text=label, width=20).pack(side=tk.LEFT)
        
        if isinstance(variable, tk.IntVar):
            scale = ttk.Scale(frame, from_=min_val, to=max_val, 
                            orient=tk.HORIZONTAL, variable=variable)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            value_label = ttk.Label(frame, text=str(variable.get()), width=6)
            value_label.pack(side=tk.RIGHT)
            
            def update_label(*args):
                value_label.config(text=str(int(variable.get())))
            variable.trace('w', update_label)
            
        else:  # DoubleVar
            scale = ttk.Scale(frame, from_=min_val, to=max_val, 
                            orient=tk.HORIZONTAL, variable=variable)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            value_label = ttk.Label(frame, text=f"{variable.get():.2f}", width=6)
            value_label.pack(side=tk.RIGHT)
            
            def update_label(*args):
                value_label.config(text=f"{variable.get():.2f}")
            variable.trace('w', update_label)
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        
        if file_path:
            self.image_path.set(file_path)
            self.load_image(file_path)
    
    def load_image(self, file_path):
        try:
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                messagebox.showerror("Error", "Could not load the selected image")
                return
            
            # Display original image
            self.display_original_image()
            self.process_btn.config(state=tk.NORMAL)
            self.status_label.config(text=f"Image loaded: {self.original_image.shape}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {str(e)}")
    
    def display_original_image(self):
        if self.original_image is not None:
            image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.axes[0, 0].clear()
            self.axes[0, 0].imshow(image_rgb)
            self.axes[0, 0].set_title("Original Image")
            self.axes[0, 0].axis('off')
            self.canvas.draw()
    
    def clear_plots(self):
        for ax in self.axes.flat:
            ax.clear()
            ax.set_title("No results yet")
            ax.axis('off')
        self.canvas.draw()
    
    def process_image_threaded(self):
        if self.original_image is None:
            return
        
        # Disable button and start progress
        self.process_btn.config(state=tk.DISABLED)
        self.progress.start()
        self.status_label.config(text="Processing...")
        
        # Run processing in separate thread
        thread = threading.Thread(target=self.process_image)
        thread.daemon = True
        thread.start()
    
    def process_image(self):
        try:
            # Get current parameter values
            n_segments = int(self.n_superpixels.get())
            compactness = float(self.superpixel_compactness.get())
            color_weight = float(self.color_weight.get())
            position_weight = float(self.position_weight.get())
            similarity_threshold = float(self.similarity_threshold.get())
            num_segments = int(self.num_segments.get())
            
            # Process the image
            segmented_image, outline_image, boundaries_image = self.eigen_segmentation(
                self.original_image, n_segments, compactness, color_weight, 
                position_weight, similarity_threshold, num_segments
            )
            
            # Update GUI in main thread
            self.root.after(0, self.display_results, segmented_image, outline_image, boundaries_image)
            
        except Exception as e:
            self.root.after(0, self.show_error, str(e))
    
    def show_error(self, error_msg):
        self.progress.stop()
        self.process_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Error occurred")
        messagebox.showerror("Processing Error", f"Error during processing: {error_msg}")
    
    def display_results(self, segmented_image, outline_image, boundaries_image):
        try:
            # Stop progress and re-enable button
            self.progress.stop()
            self.process_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Processing complete!")
            
            # Display original image
            image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.axes[0, 0].clear()
            self.axes[0, 0].imshow(image_rgb)
            self.axes[0, 0].set_title("Original Image")
            self.axes[0, 0].axis('off')
            
            # Display segmented image
            self.axes[0, 1].clear()
            self.axes[0, 1].imshow(segmented_image, cmap='viridis')
            self.axes[0, 1].set_title("Segmented Image")
            self.axes[0, 1].axis('off')
            
            # Display outline
            self.axes[1, 0].clear()
            self.axes[1, 0].imshow(outline_image, cmap='gray')
            self.axes[1, 0].set_title("Segment Outlines")
            self.axes[1, 0].axis('off')
            
            # Display boundaries overlay
            self.axes[1, 1].clear()
            self.axes[1, 1].imshow(boundaries_image)
            self.axes[1, 1].set_title("Boundaries Overlay")
            self.axes[1, 1].axis('off')
            
            self.canvas.draw()
            
        except Exception as e:
            self.show_error(str(e))
    
    # Your original segmentation methods with parameters
    def preprocess_image(self, image: np.ndarray):
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        return img_as_float(image_rgb)

    def compute_affinity_matrix(self, image: np.ndarray, n_segments=100, compactness=10, 
                              color_weight=2.0, position_weight=0.1, similarity_threshold=0.5):
        # Generate superpixels
        segments = slic(image, n_segments=n_segments, compactness=compactness)
        num_segments = np.max(segments) + 1
        
        # Compute superpixel features
        pixel_count = np.zeros(num_segments)
        color_sum = np.zeros((num_segments, 3))
        position_sum = np.zeros((num_segments, 2))
        
        height, width = segments.shape
        
        for i in range(height):
            for j in range(width):
                segment_id = segments[i, j]
                pixel_count[segment_id] += 1
                color_sum[segment_id] += image[i, j]
                position_sum[segment_id] += [i, j]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            color_mean = color_sum / pixel_count[:, np.newaxis]
            position_mean = position_sum / pixel_count[:, np.newaxis]
        
        color_mean = np.nan_to_num(color_mean)
        position_mean = np.nan_to_num(position_mean)
        
        position_mean[:, 0] /= height
        position_mean[:, 1] /= width
        
        # Compute affinities
        row_indices = []
        col_indices = []
        values = []
        
        for i in range(num_segments):
            for j in range(num_segments):
                if i != j:
                    color_dist = np.sum((color_mean[i] - color_mean[j])**2)
                    spatial_dist = np.sum((position_mean[i] - position_mean[j])**2)
                    
                    similarity = np.exp(-color_weight * color_dist) * \
                                 np.exp(-position_weight * spatial_dist)
                    
                    if similarity > similarity_threshold:
                        row_indices.append(i)
                        col_indices.append(j)
                        values.append(similarity)
        
        W = csr_matrix((values, (row_indices, col_indices)), 
                      shape=(num_segments, num_segments))
        W = (W + W.T) / 2
        
        return W, segments

    def compute_laplacian(self, W):
        if not hasattr(W, 'toarray'):
            W = csr_matrix(W)
        
        d = W.sum(axis=1).A1
        d_sqrt_inv = 1.0 / np.sqrt(d + 1e-10)
        D_sqrt_inv = diags(d_sqrt_inv)
        
        normalized_W = D_sqrt_inv @ W @ D_sqrt_inv
        return normalized_W, d_sqrt_inv

    def eigen_segmentation(self, image: np.ndarray, n_superpixels=500, superpixel_compactness=30,
                          color_weight=2.0, position_weight=0.1, similarity_threshold=0.1, 
                          num_segments=8):
        image_float = self.preprocess_image(image)
        
        # Create affinity matrix from superpixels
        W, superpixels = self.compute_affinity_matrix(
            image_float, n_superpixels, superpixel_compactness, 
            color_weight, position_weight, similarity_threshold
        )
        normalized_W, d_sqrt_inv = self.compute_laplacian(W)
        
        # Compute eigendecomposition
        eigenvalues, eigenvectors = eigsh(normalized_W, k=min(num_segments+1, W.shape[0]-1), which='LA')
        eigenvalues = 1 - eigenvalues
        
        # Sort and select eigenvectors
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        features = eigenvectors[:, 1:min(num_segments+1, eigenvectors.shape[1])]
        
        # Cluster feature space
        actual_clusters = min(num_segments, features.shape[1])
        kmeans = KMeans(n_clusters=actual_clusters, random_state=0, n_init=10)
        superpixel_labels = kmeans.fit_predict(features)
        
        # Map labels back to image
        height, width = superpixels.shape
        segmented_image = np.zeros((height, width), dtype=np.int32)
        for i in range(height):
            for j in range(width):
                segmented_image[i, j] = superpixel_labels[superpixels[i, j]]
        
        # Create boundary images
        outline_image = self.detect_segment_boundaries(segmented_image)
        boundaries_image = mark_boundaries(image_float, segmented_image)
        
        return segmented_image, outline_image, boundaries_image

    def detect_segment_boundaries(self, segmented_image):
        labels = np.unique(segmented_image)
        outline = np.zeros_like(segmented_image, dtype=np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        
        for label in labels:
            mask = (segmented_image == label).astype(np.uint8)
            boundary = cv2.dilate(mask, kernel) - mask
            outline[boundary > 0] = 255
        
        return outline

def main():
    root = tk.Tk()
    app = ImageSegmentationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()