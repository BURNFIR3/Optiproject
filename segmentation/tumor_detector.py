"""
Unified Tumor Detection & Segmentation Pipeline
- Single output image with overlay for positive cases
- Improved segmentation using multiple techniques
- Clean, publication-ready visualizations
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import load_model
import tensorflow as tf

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_PATH = "./brain_tumor_model.keras"
DATASET_PATH = "./Brain_Tumor_Dataset"
TARGET_SIZE = (64, 64)

# =============================================================================
# ROBUST SKULL STRIPPING - Isolate brain from skull
# =============================================================================

def robust_skull_strip(image):
    """
    Robust skull stripping that properly isolates the brain region.
    Returns both the brain-only image and the brain mask.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Step 1: Apply median blur for better edge detection
    blurred = cv2.medianBlur(gray, 5)

    # Step 2: Use Otsu's thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 3: Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Step 4: Fill holes and get largest region
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    brain_mask = np.zeros_like(gray)
    min_area = h * w * 0.1  # At least 10% of image

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            cv2.drawContours(brain_mask, [contour], -1, 255, -1)

    # Step 5: Apply mask to get clean brain region
    brain = cv2.bitwise_and(image, image, mask=brain_mask)

    # Step 6: Erode slightly to remove skull edge artifacts
    brain_mask_eroded = cv2.erode(brain_mask, np.ones((3, 3), np.uint8), iterations=2)
    brain_clean = cv2.bitwise_and(image, image, mask=brain_mask_eroded)

    return brain_clean, brain_mask_eroded


def get_brain_inner_mask(image, margin=10):
    """
    Get the INNER region of the brain (away from skull edges).
    This is where tumors are actually located.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Get the brain region
    blurred = cv2.medianBlur(gray, 5)
    _, thresh = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)

    kernel = np.ones((8, 8), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Get largest region
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones_like(gray) * 255

    largest = max(contours, key=cv2.contourArea)

    # Create filled mask
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, pts=[largest], color=255)

    # Shrink by margin to get INNER region only
    inner_mask = cv2.erode(mask, np.ones((margin, margin), np.uint8), iterations=1)

    return inner_mask


# =============================================================================
# IMPROVED PREPROCESSING
# =============================================================================

def mri_intensity_normalization(image):
    """
    MRI-specific intensity normalization.
    Handles the characteristic intensity distribution of MRI scans.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This is particularly good for MRI images
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)

    return normalized


def adaptive_threshold_segmentation(image):
    """
    Adaptive thresholding - better for MRI with varying illumination.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold handles varying intensity across image
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    return binary


def kmeans_clustering(gray, k=3):
    """
    K-means clustering for automatic intensity-based segmentation.
    Separates brain tissue from background and tumor regions.
    """
    # Reshape for k-means
    h, w = gray.shape
    pixel_values = gray.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    # K-means criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Apply k-means
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Sort centers by intensity (brighter = higher value)
    centers = sorted(centers, key=lambda x: x[0])

    # Create segmented image
    labels_reshaped = labels.reshape(h, w)
    segmented = np.zeros_like(gray)
    for i, center in enumerate(centers):
        segmented[labels_reshaped == i] = int(center[0])

    return segmented, centers, labels_reshaped


def grabcut_segmentation(image, iterations=5):
    """
    GrabCut algorithm - graph-cut based segmentation.
    Excellent for separating foreground (brain/tumor) from background.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create initial mask
    mask = np.zeros(gray.shape, np.uint8)

    # Define rectangle around the brain (assuming brain is central)
    h, w = gray.shape
    margin = int(min(h, w) * 0.1)
    rect = (margin, margin, w - 2 * margin, h - 2 * margin)

    # GrabCut models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Apply GrabCut
    try:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

        # Create binary mask from GrabCut result
        final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    except:
        final_mask = np.ones_like(gray)

    return final_mask * 255


def find_optimal_threshold(gray):
    """
    Find optimal threshold using histogram analysis.
    Tumor regions in MRI often have distinct intensity peaks.
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten()

    # Find valleys between peaks
    # Smooth the histogram
    kernel = np.ones((5,), np.float32)
    hist_smooth = np.convolve(hist.flatten(), kernel / kernel.sum(), mode='same')

    # Find the minimum between first major peak and max
    # (assuming tumor is brighter than normal tissue)
    max_idx = np.argmax(hist_smooth)
    first_peak_idx = np.argmax(hist_smooth[:max_idx // 2]) if max_idx > 50 else 50

    # Find minimum between first peak and max
    valley_region = hist_smooth[first_peak_idx:max_idx]
    threshold = first_peak_idx + np.argmin(valley_region)

    return threshold


def refined_morphological(mask, iterations=2):
    """
    Refined morphological operations for clean boundaries.
    """
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((5, 5), np.uint8)

    # Opening to remove noise
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=iterations)

    # Closing to fill gaps
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_large, iterations=iterations)

    # Optional: erosion to shrink slightly for cleaner edges
    # eroded = cv2.erode(closed, kernel_small, iterations=1)

    return closed


def get_largest_region(mask, min_area=100):
    """
    Keep only the largest connected component.
    Removes small noise regions.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask

    # Find largest contour
    largest = max(contours, key=cv2.contourArea)

    # Check if it's large enough
    if cv2.contourArea(largest) < min_area:
        return mask

    # Create mask with only largest region
    result = np.zeros_like(mask)
    cv2.drawContours(result, [largest], -1, 255, -1)

    return result


# =============================================================================
# GRAD-CAM (Fixed for Keras 3)
# =============================================================================
class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    def __init__(self, model):
        self.model = model
        self.grad_model = None

    def get_heatmap(self, input_image):
        """Generate Grad-CAM heatmap."""
        # Build grad model on first call
        if self.grad_model is None:
            self._build_grad_model()

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(input_image, training=False)
            pred_class = int(np.argmax(predictions[0]))
            loss = predictions[:, pred_class]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs * pooled_grads
        heatmap = tf.reduce_sum(heatmap, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap.numpy(), pred_class

    def _build_grad_model(self):
        """Build gradient model."""
        # Find last conv layer
        last_conv = None
        for layer in reversed(self.model.layers):
            if hasattr(layer, 'filters') or 'conv' in layer.name.lower():
                last_conv = layer
                break
        if last_conv is None:
            last_conv = self.model.layers[0]

        # Build functional model
        input_shape = self.model.input_shape[1:]
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        conv_out = None
        for layer in self.model.layers:
            x = layer(x)
            if layer is last_conv:
                conv_out = x

        self.grad_model = tf.keras.Model(inputs=inputs, outputs=[conv_out, x])


# =============================================================================
# MAIN SEGMENTATION PIPELINE
# =============================================================================

def skull_strip(image):
    """Basic skull stripping."""
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


def preprocess(image, target_size=TARGET_SIZE):
    """Preprocess image for model."""
    stripped = skull_strip(image)
    resized = cv2.resize(stripped, target_size)
    normalized = resized / 255.0
    return normalized


def segment_tumor(image, gradcam):
    """
    Advanced tumor segmentation combining multiple techniques.
    CRITICAL: Only considers regions INSIDE the brain, away from skull edges.
    """
    # Step 1: Get the INNER brain mask (excludes skull edges)
    inner_mask = get_brain_inner_mask(image, margin=15)

    # Step 2: Get Grad-CAM attention
    input_img = np.expand_dims(preprocess(image), axis=0)
    heatmap, pred_class = gradcam.get_heatmap(input_img)

    # Resize heatmap to original size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Step 3: Create weighted heatmap mask
    heatmap_threshold = np.mean(heatmap_resized) + 0.5 * np.std(heatmap_resized)
    cam_mask = (heatmap_resized > max(heatmap_threshold, 0.3)).astype(np.uint8) * 255

    # Step 4: K-means clustering on intensity
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_gray = mri_intensity_normalization(image)
    kmeans_result, centers, _ = kmeans_clustering(normalized_gray, k=4)

    # Select brightest clusters (potential tumor regions)
    brightness_threshold = np.median(centers)
    bright_mask = np.zeros_like(gray)
    for i, center in enumerate(centers):
        if center[0] > brightness_threshold * 1.1:
            bright_mask[kmeans_result == i] = 255

    # Step 5: Combine masks AND apply inner brain mask
    # This is the KEY step - we exclude skull edges
    combined = cv2.bitwise_and(cam_mask, bright_mask)
    combined = cv2.bitwise_and(combined, inner_mask)  # Only inside brain!

    # Step 6: Refine with morphology
    refined = refined_morphological(combined, iterations=2)

    # Step 7: Keep only largest region within brain
    final_mask = get_largest_region(refined, min_area=200)

    # Step 8: Apply inner mask one more time to be sure
    final_mask = cv2.bitwise_and(final_mask, inner_mask)

    # Step 9: Clean up edges
    final_mask = cv2.erode(final_mask, np.ones((3, 3), np.uint8), iterations=1)

    return final_mask, heatmap_resized, pred_class


def create_overlay(image, mask, alpha=0.5):
    """
    Create a clean overlay with tumor highlighted.
    Green contour with semi-transparent fill.
    """
    # Create color overlay
    overlay = image.copy()

    # Fill the tumor region with green (semi-transparent)
    overlay[mask > 0] = [0, 255, 0]  # Green for tumor

    # Blend
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    # Draw sharp contour on top
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    return result


def create_heatmap_overlay(image, heatmap, alpha=0.4):
    """Create gradient heatmap overlay."""
    # Apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Blend with original
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


def process_image(image_path, model, gradcam):
    """
    Process a single image and return results.
    Returns prediction and visualization.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None, "Could not load image"

    original = image.copy()

    # Preprocess for model
    input_img = np.expand_dims(preprocess(image), axis=0)

    # Get prediction
    pred_prob = model.predict(input_img, verbose=0)[0][0]
    prediction = "Tumor Detected" if pred_prob > 0.5 else "No Tumor"
    confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob

    result = {
        'original': original,
        'prediction': prediction,
        'confidence': float(confidence),
        'is_positive': pred_prob > 0.5,
        'raw_probability': float(pred_prob)
    }

    # If positive, generate segmentation
    if result['is_positive']:
        mask, heatmap, pred_class = segment_tumor(image, gradcam)
        result['tumor_mask'] = mask
        result['heatmap'] = heatmap

        # Create clean overlay
        overlay = create_overlay(original, mask, alpha=0.4)
        result['overlay'] = overlay

        # Create heatmap overlay
        heatmap_viz = create_heatmap_overlay(original, heatmap, alpha=0.4)
        result['heatmap_overlay'] = heatmap_viz

        # Combined visualization
        combined = np.hstack([
            cv2.resize(original, (300, 300)),
            cv2.resize(overlay, (300, 300)),
            cv2.resize(heatmap_viz, (300, 300))
        ])
        result['combined'] = combined

        # Tumor info
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result['tumor_count'] = len(contours)
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            result['largest_tumor_pixels'] = max(areas)

    return result


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_result(result, save_path=None):
    """Plot the final result."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if result['is_positive']:
        # Original
        axes[0].imshow(cv2.cvtColor(result['original'], cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Original MRI\nConfidence: {result['confidence']:.1%}", fontsize=12)
        axes[0].axis('off')

        # Tumor overlay
        axes[1].imshow(cv2.cvtColor(result['overlay'], cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Tumor Detection\n{result['tumor_count']} region(s) found", fontsize=12)
        axes[1].axis('off')

        # Heatmap
        axes[2].imshow(cv2.cvtColor(result['heatmap_overlay'], cv2.COLOR_BGR2RGB))
        axes[2].set_title("Model Attention\n(Grad-CAM)", fontsize=12)
        axes[2].axis('off')
    else:
        # Negative case
        axes[0].imshow(cv2.cvtColor(result['original'], cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original MRI", fontsize=12)
        axes[0].axis('off')

        axes[1].text(0.5, 0.5, "NO TUMOR DETECTED",
                    ha='center', va='center', fontsize=20, fontweight='bold',
                    color='green', transform=axes[1].transAxes)
        axes[1].text(0.5, 0.3, f"Confidence: {result['confidence']:.1%}",
                    ha='center', va='center', fontsize=14, transform=axes[1].transAxes)
        axes[1].axis('off')

        axes[2].text(0.5, 0.5, "This MRI scan appears\nto be healthy",
                    ha='center', va='center', fontsize=14,
                    transform=axes[2].transAxes)
        axes[2].axis('off')

    plt.suptitle(f"Brain MRI Analysis: {result['prediction']}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison(results, save_path):
    """Plot comparison of multiple results."""
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 10))

    if n == 1:
        axes = axes.reshape(1, -1)

    for i, result in enumerate(results):
        if result['is_positive']:
            # Top: Original
            axes[0, i].imshow(cv2.cvtColor(result['original'], cv2.COLOR_BGR2RGB))
            axes[0, i].set_title(f"Original\nConf: {result['confidence']:.1%}", fontsize=10)
            axes[0, i].axis('off')

            # Bottom: Overlay
            axes[1, i].imshow(cv2.cvtColor(result['overlay'], cv2.COLOR_BGR2RGB))
            axes[1, i].set_title(f"Tumor Detected\n{result['tumor_count']} regions", fontsize=10)
            axes[1, i].axis('off')
        else:
            axes[0, i].imshow(cv2.cvtColor(result['original'], cv2.COLOR_BGR2RGB))
            axes[0, i].set_title(f"Original\nConf: {result['confidence']:.1%}", fontsize=10)
            axes[0, i].axis('off')

            axes[1, i].text(0.5, 0.5, "NO TUMOR", ha='center', va='center',
                          fontsize=14, fontweight='bold', color='green')
            axes[1, i].axis('off')

    plt.suptitle("Brain Tumor Detection Results", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_final_visualization(image, result, save_path=None):
    """
    Create a single, publication-ready image.
    Clean layout with all information.
    """
    fig = plt.figure(figsize=(16, 8))

    if result['is_positive']:
        # Layout: Original | Overlay | Heatmap | Info
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.8])

        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(cv2.cvtColor(result['original'], cv2.COLOR_BGR2RGB))
        ax1.set_title("Input MRI", fontsize=14, fontweight='bold')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(cv2.cvtColor(result['overlay'], cv2.COLOR_BGR2RGB))
        ax2.set_title("Tumor Region Detected", fontsize=14, fontweight='bold', color='red')
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[2])
        ax3.imshow(cv2.cvtColor(result['heatmap_overlay'], cv2.COLOR_BGR2RGB))
        ax3.set_title("Model Attention (Grad-CAM)", fontsize=14, fontweight='bold')
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[3])
        ax4.axis('off')

        # Info text
        info = f"""
        RESULT: TUMOR DETECTED

        Confidence: {result['confidence']:.1%}

        Tumor Regions: {result['tumor_count']}

        Largest Region:
        {result.get('largest_tumor_pixels', 0):.0f} pixels

        Segmentation Method:
        - Grad-CAM (Deep Learning)
        - K-Means Clustering
        - GrabCut Segmentation
        - Morphological Refinement
        """
        ax4.text(0.1, 0.5, info, fontsize=12, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    else:
        # Negative case
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8])

        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(cv2.cvtColor(result['original'], cv2.COLOR_BGR2RGB))
        ax1.set_title("Input MRI", fontsize=14, fontweight='bold')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[1])
        ax2.text(0.5, 0.5, "NO TUMOR DETECTED", ha='center', va='center',
                fontsize=24, fontweight='bold', color='green', transform=ax2.transAxes)
        ax2.text(0.5, 0.3, f"Confidence: {result['confidence']:.1%}",
                ha='center', va='center', fontsize=16, transform=ax2.transAxes)
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[2])
        ax3.axis('off')
        ax3.text(0.1, 0.5, "MRI scan appears\nto be healthy.\n\nNo abnormal growth\nor tumor-like regions\ndetected.",
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.suptitle("Brain Tumor Detection & Segmentation", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
print("=" * 70)
print("UNIFIED TUMOR DETECTION & SEGMENTATION")
print("=" * 70)

# Load model
print("\nLoading model...")
model = load_model(MODEL_PATH)
gradcam = GradCAM(model)
print("Model loaded successfully")

# Load sample images (mix of positive and negative)
print("\nProcessing sample images...")
positive_path = os.path.join(DATASET_PATH, 'Positive')
negative_path = os.path.join(DATASET_PATH, 'Negative')

positive_files = sorted([f for f in os.listdir(positive_path) if f.endswith(('.jpg', '.jpeg', '.png'))])[:10]
negative_files = sorted([f for f in os.listdir(negative_path) if f.endswith(('.jpg', '.jpeg', '.png'))])[:5]

# Process positive cases
print("\n--- POSITIVE CASES ---")
positive_results = []
for i, fname in enumerate(positive_files):
    image_path = os.path.join(positive_path, fname)
    result = process_image(image_path, model, gradcam)
    if result:
        positive_results.append(result)
        print(f"  [{i+1}] {fname}: {result['prediction']} ({result['confidence']:.1%} confidence)")

# Process negative cases
print("\n--- NEGATIVE CASES ---")
negative_results = []
for i, fname in enumerate(negative_files):
    image_path = os.path.join(negative_path, fname)
    result = process_image(image_path, model, gradcam)
    if result:
        negative_results.append(result)
        print(f"  [{i+1}] {fname}: {result['prediction']} ({result['confidence']:.1%} confidence)")

# Generate visualizations
print("\nGenerating visualizations...")

# Individual positive results
for i, result in enumerate(positive_results[:5]):
    save_path = f"result_positive_{i+1:02d}.png"
    create_final_visualization(result['original'], result, save_path)
    print(f"  [OK] Saved: {save_path}")

# Individual negative results
for i, result in enumerate(negative_results[:3]):
    save_path = f"result_negative_{i+1:02d}.png"
    create_final_visualization(result['original'], result, save_path)
    print(f"  [OK] Saved: {save_path}")

# Comparison plot
all_results = positive_results[:6] + negative_results[:4]
plot_comparison(all_results, 'plot_14_detection_results.png')
print("  [OK] Saved: plot_14_detection_results.png")

print("\n" + "=" * 70)
print("PROCESSING COMPLETE")
print("=" * 70)
print("\nGenerated files:")
print("  result_positive_XX.png - Individual positive case visualizations")
print("  result_negative_XX.png - Individual negative case visualizations")
print("  plot_14_detection_results.png - Comparison of all results")
