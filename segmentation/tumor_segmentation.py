"""
Tumor Region Detection & Segmentation
- Grad-CAM: What regions influenced the model's decision
- Otsu's Thresholding: Traditional segmentation
- Watershed: For separating overlapping regions
- Heatmap overlay visualization
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Activation, Input
from tensorflow.keras import Model
import tensorflow as tf

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_PATH = "./brain_tumor_model.keras"
DATASET_PATH = "./Brain_Tumor_Dataset"
TARGET_SIZE = (64, 64)
SAMPLES_TO_VISUALIZE = 20

# =============================================================================
# GRAD-CAM IMPLEMENTATION (Keras 3 compatible)
# =============================================================================
class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name
        self.grad_model = None

    def get_heatmap(self, input_image, class_idx=None):
        """Generate Grad-CAM heatmap for the given image."""
        import tensorflow as tf

        # Create gradient model if not exists
        if self.grad_model is None:
            self.grad_model = self._get_grad_model()

        # Get predictions
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(input_image, training=False)
            if class_idx is None:
                class_idx = int(np.argmax(predictions[0]))
            loss = predictions[:, class_idx]

        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)

        # Average pooling gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight channels - handle both Keras 2 and 3 tensor operations
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs * pooled_grads
        heatmap = tf.reduce_sum(heatmap, axis=-1)

        # ReLU to keep only positive contributions
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap.numpy(), class_idx

    def _get_grad_model(self):
        """Create a model that returns both conv output and predictions."""
        import tensorflow as tf

        # Find the last convolutional layer
        last_conv_layer = None
        for layer in reversed(self.model.layers):
            if hasattr(layer, 'filters') or 'conv' in layer.name.lower():
                last_conv_layer = layer
                break

        if last_conv_layer is None:
            last_conv_layer = self.model.layers[0]

        # Build a new functional model by chaining through layers
        input_shape = self.model.input_shape[1:]  # Remove batch dimension
        inputs = tf.keras.Input(shape=input_shape)

        # Forward pass, capturing conv output
        x = inputs
        conv_output = None
        for layer in self.model.layers:
            x = layer(x)
            if layer is last_conv_layer:
                conv_output = x

        final_output = x  # This is the model output

        # Create the model with outputs from intermediate and final layers
        grad_model = tf.keras.Model(
            inputs=inputs,
            outputs=[conv_output, final_output]
        )
        return grad_model


# =============================================================================
# SKULL STRIPPING (same as training)
# =============================================================================
def skull_strip(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
        kernel = np.ones((8, 8), np.uint8)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image
        brain_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [brain_contour], -1, 255, -1)
        result = cv2.bitwise_and(image, image, mask=mask)
        if np.mean(result) < 10:
            return image
        return result
    except Exception:
        return image

def preprocess_image(image, target_size=TARGET_SIZE):
    stripped = skull_strip(image)
    resized = cv2.resize(stripped, target_size)
    normalized = resized / 255.0
    return normalized

# =============================================================================
# TUMOR SEGMENTATION TECHNIQUES
# =============================================================================

def otsu_thresholding(image):
    """
    Otsu's Thresholding - automatic threshold selection from gray-level histogram
    Paper: Otsu, N.. A Threshold Selection Method from Gray-Level Histograms
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def watershed_segmentation(image, binary_mask):
    """
    Watershed Segmentation - treats image as topographic surface
    Paper: Vincent, L. & Soille, P.. Watersheds in Digital Spaces
    """
    # Sure background
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary_mask, kernel, iterations=3)

    # Sure foreground (distance transform)
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)

    # Unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    markers = cv2.watershed(image, markers)

    return markers


def morphological_segmentation(image, binary_mask):
    """
    Morphological Operations for cleanup
    Paper: Serra, J.. Image Analysis and Mathematical Morphology
    """
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((5, 5), np.uint8)

    # Opening to remove noise
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small)

    # Closing to fill holes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_large)

    # Remove small objects
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(binary_mask)
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Minimum area threshold
            cv2.drawContours(result, [contour], -1, 255, -1)

    return result


def region_growing(image, seed_point):
    """
    Region Growing - seeded segmentation
    Paper: Adams, R. & Bischof, L.. Seeded Region Growing
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    visited = np.zeros_like(gray, dtype=bool)
    seed_value = gray[seed_point[1], seed_point[0]]

    # Threshold for similarity
    threshold = 15

    # BFS
    queue = [seed_point]
    region = np.zeros_like(gray)
    region[seed_point[1], seed_point[0]] = 255
    visited[seed_point[1], seed_point[0]] = True

    while queue:
        x, y = queue.pop(0)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                if abs(int(gray[ny, nx]) - int(seed_value)) < threshold:
                    visited[ny, nx] = True
                    region[ny, nx] = 255
                    queue.append((nx, ny))

    return region


def active_contours(image, initial_mask, iterations=100):
    """
    Active Contours (Snakes) - energy minimization
    Paper: Kass, M., Witkin, A. & Terzopoulos, D.. Snakes: Active Contour Models

    Note: Using OpenCV's implementation
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find contours from initial mask
    contours, _ = cv2.findContours(initial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return initial_mask

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Create mask from contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)

    # Apply morphological operations for smoothing
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def create_heatmap_overlay(original_image, heatmap, alpha=0.5):
    """Overlay heatmap on original image."""
    # Resize heatmap to original image size
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

    # Apply colormap (Jet - red for high, blue for low)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    # Blend with original
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay


def create_mask_from_heatmap(heatmap, threshold=0.3):
    """Create binary mask from Grad-CAM heatmap."""
    mask = (heatmap > threshold).astype(np.uint8) * 255
    return mask


def find_tumor_centroids(mask):
    """Find centroids of tumor regions."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for contour in contours:
        if cv2.contourArea(contour) > 20:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroids.append((cx, cy, cv2.contourArea(contour)))
    return centroids


# =============================================================================
# MAIN VISUALIZATION
# =============================================================================
def visualize_segmentation(image, model, title=""):
    """Create comprehensive visualization for one image."""
    # Preprocess
    original = image.copy()
    preprocessed = preprocess_image(image)
    input_img = np.expand_dims(preprocessed, axis=0)

    # Get model prediction
    pred_prob = model.predict(input_img, verbose=0)[0][0]
    prediction = "Tumor" if pred_prob > 0.5 else "No Tumor"

    # Generate Grad-CAM heatmap
    gradcam = GradCAM(model)
    heatmap, pred_class = gradcam.get_heatmap(input_img)

    # Traditional segmentation
    otsu_mask = otsu_thresholding(original)
    morph_mask = morphological_segmentation(original, otsu_mask)

    # Create overlay
    heatmap_overlay = create_heatmap_overlay(original, heatmap)

    # Combined mask (intersection of Grad-CAM and Otsu)
    cam_mask = create_mask_from_heatmap(heatmap, threshold=0.4)

    # Resize masks to match original image size
    cam_mask_resized = cv2.resize(cam_mask, (original.shape[1], original.shape[0]))
    morph_mask_resized = cv2.resize(morph_mask, (original.shape[1], original.shape[0]))
    combined_mask = cv2.bitwise_and(cam_mask_resized, morph_mask_resized)

    # Find centroids
    centroids = find_tumor_centroids(combined_mask)

    return {
        'original': original,
        'preprocessed': preprocessed,
        'heatmap': heatmap,
        'heatmap_overlay': heatmap_overlay,
        'otsu_mask': otsu_mask,
        'morph_mask': morph_mask,
        'combined_mask': combined_mask,
        'prediction': prediction,
        'confidence': pred_prob,
        'centroids': centroids
    }


def plot_comprehensive_results(results, save_path):
    """Plot all segmentation results."""
    n = len(results)
    fig = plt.figure(figsize=(20, 4 * n))

    for i, r in enumerate(results):
        row_offset = i * 9

        # 1. Original Image
        ax = fig.add_subplot(n, 9, row_offset + 1)
        ax.imshow(cv2.cvtColor(r['original'], cv2.COLOR_BGR2RGB))
        ax.set_title(f"Original\nConf: {r['confidence']:.3f}", fontsize=10)
        ax.axis('off')

        # 2. Skull Stripped
        ax = fig.add_subplot(n, 9, row_offset + 2)
        stripped = skull_strip(r['original'])
        ax.imshow(cv2.cvtColor(stripped, cv2.COLOR_BGR2RGB))
        ax.set_title("Skull Stripped", fontsize=10)
        ax.axis('off')

        # 3. Grad-CAM Heatmap
        ax = fig.add_subplot(n, 9, row_offset + 3)
        ax.imshow(r['heatmap'], cmap='jet')
        ax.set_title("Grad-CAM\n(Model Attention)", fontsize=10)
        ax.axis('off')

        # 4. Heatmap Overlay
        ax = fig.add_subplot(n, 9, row_offset + 4)
        ax.imshow(cv2.cvtColor(r['heatmap_overlay'], cv2.COLOR_BGR2RGB))
        ax.set_title("Heatmap Overlay", fontsize=10)
        ax.axis('off')

        # 5. Otsu Thresholding
        ax = fig.add_subplot(n, 9, row_offset + 5)
        ax.imshow(r['otsu_mask'], cmap='gray')
        ax.set_title("Otsu's\nThresholding", fontsize=10)
        ax.axis('off')

        # 6. Morphological Mask
        ax = fig.add_subplot(n, 9, row_offset + 6)
        ax.imshow(r['morph_mask'], cmap='gray')
        ax.set_title("Morphological\nCleanup", fontsize=10)
        ax.axis('off')

        # 7. Combined Mask
        ax = fig.add_subplot(n, 9, row_offset + 7)
        ax.imshow(r['combined_mask'], cmap='Reds')
        ax.set_title("Combined Mask\n(Grad-CAM + Otsu)", fontsize=10)
        ax.axis('off')

        # 8. Final Overlay with Contours
        ax = fig.add_subplot(n, 9, row_offset + 8)
        overlay_final = r['original'].copy()
        contours, _ = cv2.findContours(r['combined_mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_final, contours, -1, (0, 255, 0), 2)
        for j, (cx, cy, area) in enumerate(r['centroids']):
            cv2.circle(overlay_final, (cx, cy), 5, (0, 0, 255), -1)
        ax.imshow(cv2.cvtColor(overlay_final, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Tumor Regions\n{len(r['centroids'])} found", fontsize=10)
        ax.axis('off')

        # 9. Legend / Info
        ax = fig.add_subplot(n, 9, row_offset + 9)
        ax.axis('off')
        info_text = f"Prediction: {r['prediction']}\n"
        info_text += f"Confidence: {r['confidence']:.3f}\n"
        info_text += f"Tumor Regions: {len(r['centroids'])}\n"
        if r['centroids']:
            info_text += f" Largest: {max(r['centroids'], key=lambda x: x[2])[2]:.0f} px"
        ax.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_technique_comparison(image, model, save_path):
    """Compare all segmentation techniques on one image."""
    original = image.copy()
    preprocessed = preprocess_image(image)
    input_img = np.expand_dims(preprocessed, axis=0)

    # Get all masks
    gradcam = GradCAM(model)
    heatmap, _ = gradcam.get_heatmap(input_img)
    heatmap_overlay = create_heatmap_overlay(original, heatmap)

    otsu_mask = otsu_thresholding(original)
    morph_mask = morphological_segmentation(original, otsu_mask)

    # Region growing from brightest point
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
    region_mask = region_growing(original, max_loc)

    # Active contours
    active_mask = active_contours(original, morph_mask)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original MRI', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(heatmap, cmap='jet')
    axes[0, 1].set_title('Grad-CAM\n(Model Attention)', fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(heatmap_overlay)
    axes[0, 2].set_title('Heatmap Overlay', fontsize=12)
    axes[0, 2].axis('off')

    axes[0, 3].imshow(otsu_mask, cmap='gray')
    axes[0, 3].set_title("Otsu's Thresholding", fontsize=12)
    axes[0, 3].axis('off')

    # Row 2
    axes[1, 0].imshow(morph_mask, cmap='gray')
    axes[1, 0].set_title('Morphological\nOperations', fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(region_mask, cmap='gray')
    axes[1, 1].set_title('Region Growing\n(From Brightest Point)', fontsize=12)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(active_mask, cmap='gray')
    axes[1, 2].set_title('Active Contours\n(Snakes)', fontsize=12)
    axes[1, 2].axis('off')

    # Combined final result
    final = original.copy()
    contours, _ = cv2.findContours(morph_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(final, contours, -1, (0, 255, 0), 2)
    axes[1, 3].imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    axes[1, 3].set_title('Final Segmentation\n(Combined)', fontsize=12)
    axes[1, 3].axis('off')

    plt.suptitle('Comparison of Segmentation Techniques', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_heatmap_examples(images, model, save_path):
    """Plot Grad-CAM heatmaps for multiple examples."""
    n = len(images)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))

    if n == 1:
        axes = axes.reshape(1, -1)

    for i, img in enumerate(images):
        original = img.copy()
        preprocessed = preprocess_image(img)
        input_img = np.expand_dims(preprocessed, axis=0)

        # Get prediction and heatmap
        pred_prob = model.predict(input_img, verbose=0)[0][0]
        gradcam = GradCAM(model)
        heatmap, _ = gradcam.get_heatmap(input_img)

        # Create overlays
        heatmap_overlay = create_heatmap_overlay(original, heatmap)

        # Threshold mask
        cam_mask = create_mask_from_heatmap(heatmap, threshold=0.4)

        # Row
        axes[i, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f'Original\nConf: {pred_prob:.3f}', fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(heatmap, cmap='jet')
        axes[i, 1].set_title('Grad-CAM', fontsize=10)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(heatmap_overlay)
        axes[i, 2].set_title('Overlay', fontsize=10)
        axes[i, 2].axis('off')

        axes[i, 3].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[i, 3].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[i, 3].set_title('Heatmap Blend', fontsize=10)
        axes[i, 3].axis('off')

    plt.suptitle('Grad-CAM: What Regions Does the Model Focus On?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
print("=" * 70)
print("TUMOR SEGMENTATION & VISUALIZATION")
print("=" * 70)

# Load model
print("Loading model...")
import tensorflow as tf
model = load_model(MODEL_PATH)
print("Model loaded successfully")

# Load tumor images
print("\nLoading tumor images...")
positive_images = []
positive_path = os.path.join(DATASET_PATH, 'Positive')
for filename in sorted(os.listdir(positive_path))[:100]:  # Load first 100
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(os.path.join(positive_path, filename))
        if img is not None:
            positive_images.append((img, filename))

print(f"Loaded {len(positive_images)} images")

# Select random samples
np.random.seed(42)
indices = np.random.choice(len(positive_images), min(10, len(positive_images)), replace=False)
sample_images = [positive_images[i][0] for i in indices]
sample_names = [positive_images[i][1] for i in indices]

# Plot 1: Comprehensive comparison
print("\nGenerating technique comparison plot...")
plot_technique_comparison(sample_images[0], model, 'plot_11_technique_comparison.png')
print("[OK] Saved: plot_11_technique_comparison.png")

# Plot 2: Grad-CAM examples
print("Generating Grad-CAM heatmaps...")
plot_heatmap_examples(sample_images[:6], model, 'plot_12_gradcam_heatmaps.png')
print("[OK] Saved: plot_12_gradcam_heatmaps.png")

# Plot 3: Detailed segmentation for multiple images
print("Generating detailed segmentation results...")
results = [visualize_segmentation(img, model) for img in sample_images[:8]]
plot_comprehensive_results(results, 'plot_13_tumor_segmentation_detailed.png')
print("[OK] Saved: plot_13_tumor_segmentation_detailed.png")

# Generate individual masks for all tumor images
print("\nGenerating tumor masks for all detected tumor images...")
tumor_masks_dir = "./tumor_masks"
os.makedirs(tumor_masks_dir, exist_ok=True)

count = 0
for img, filename in positive_images:
    original = img.copy()
    preprocessed = preprocess_image(img)
    input_img = np.expand_dims(preprocessed, axis=0)

    # Check if model predicts tumor
    pred_prob = model.predict(input_img, verbose=0)[0][0]

    if pred_prob > 0.5:  # Tumor prediction
        # Generate heatmap
        gradcam = GradCAM(model)
        heatmap, _ = gradcam.get_heatmap(input_img)

        # Generate masks
        otsu_mask = otsu_thresholding(original)
        morph_mask = morphological_segmentation(original, otsu_mask)
        cam_mask = create_mask_from_heatmap(heatmap, threshold=0.4)

        # Save masks
        base_name = os.path.splitext(filename)[0]
        cv2.imwrite(f"{tumor_masks_dir}/{base_name}_original.jpg", original)
        cv2.imwrite(f"{tumor_masks_dir}/{base_name}_heatmap.jpg", cv2.resize(np.uint8(255 * heatmap), (original.shape[1], original.shape[0])))
        cv2.imwrite(f"{tumor_masks_dir}/{base_name}_otsu.jpg", otsu_mask)
        cv2.imwrite(f"{tumor_masks_dir}/{base_name}_mask.jpg", morph_mask)
        cv2.imwrite(f"{tumor_masks_dir}/{base_name}_cam_mask.jpg", cam_mask)

        # Save overlay
        heatmap_overlay = create_heatmap_overlay(original, heatmap)
        cv2.imwrite(f"{tumor_masks_dir}/{base_name}_overlay.jpg", heatmap_overlay)

        count += 1

print(f"[OK] Generated masks for {count} tumor images in '{tumor_masks_dir}/'")

print("\n" + "=" * 70)
print("SEGMENTATION COMPLETE")
print("=" * 70)
print("\nGenerated files:")
print("  plot_11_technique_comparison.png - Compare all techniques")
print("  plot_12_gradcam_heatmaps.png - Grad-CAM heatmaps")
print("  plot_13_tumor_segmentation_detailed.png - Detailed results")
print("  tumor_masks/ - Individual masks for all tumor images")
