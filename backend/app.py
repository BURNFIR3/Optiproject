"""
NeuroScan Pro - Brain Tumor Detection
Hugging Face Spaces deployment with Gradio
"""

import numpy as np
import cv2
import base64
import io
import warnings
import os
warnings.filterwarnings('ignore')

import gradio as gr
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

# =============================================================================
# LOAD MODEL ON STARTUP
# =============================================================================
print("Loading model... this may take a moment...")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "brain_tumor_model_finetuned.keras")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# =============================================================================
# GRAD-CAM
# =============================================================================
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.grad_model = None

    def get_heatmap(self, input_image):
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
        last_conv = None
        for layer in reversed(self.model.layers):
            if hasattr(layer, 'filters') or 'conv' in layer.name.lower():
                last_conv = layer
                break
        if last_conv is None:
            last_conv = self.model.layers[0]
        input_shape = self.model.input_shape[1:]
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        conv_out = None
        for layer in self.model.layers:
            x = layer(x)
            if layer is last_conv:
                conv_out = x
        self.grad_model = tf.keras.Model(inputs=inputs, outputs=[conv_out, x])

gradcam = GradCAM(model)
gradcam._build_grad_model()

TARGET_SIZE = (64, 64)
RESIZE_FOR_SEGMENTATION = (256, 256)

# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================
def preprocess(image, target_size=TARGET_SIZE):
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
    resized = cv2.resize(result, target_size)
    return resized / 255.0

def strip_skull(image, resize_dim=None):
    img = image.copy()
    if resize_dim is not None:
        img = cv2.resize(img, resize_dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    _, thresh = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)
    kernel = np.ones((8, 8), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    brain_mask = np.zeros_like(gray)
    min_area = gray.shape[0] * gray.shape[1] * 0.1
    if contours:
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours_sorted:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(brain_mask, [cnt], -1, 255, -1)
                break
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    brain_mask = cv2.erode(brain_mask, np.ones((8, 8), np.uint8), iterations=2)
    return cv2.bitwise_and(img, img, mask=brain_mask), brain_mask

def segment_tumor(image, brain_mask):
    img = cv2.resize(image, RESIZE_FOR_SEGMENTATION, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(brain_mask, RESIZE_FOR_SEGMENTATION, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced_masked = enhanced.copy()
    enhanced_masked[mask == 0] = 0
    brain_pixels = enhanced[mask > 0]
    if len(brain_pixels) == 0:
        return np.zeros((RESIZE_FOR_SEGMENTATION[1], RESIZE_FOR_SEGMENTATION[0]), dtype=np.uint8)
    threshold_value = np.percentile(brain_pixels, 85)
    bright_mask = (enhanced > threshold_value).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    tumor_mask = cv2.bitwise_and(bright_mask, bright_mask, mask=mask)
    contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(tumor_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return clean_mask

def create_overlay_image(image, tumor_mask):
    h, w = image.shape[:2]
    overlay = image.copy()
    if np.sum(tumor_mask > 0) > 100:
        mask_resized = cv2.resize(tumor_mask, (w, h))
        overlay[mask_resized > 0] = [0, 255, 0]
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(overlay, (cx, cy), 8, (0, 0, 255), -1)
    return overlay

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================
def analyze_mri(image):
    """
    Takes a PIL/numpy image, returns analysis result.
    For Gradio: returns (markdown_result, original_img, overlay_img)
    """
    if image is None:
        return "**Please upload an MRI scan image to begin analysis.**", None, None

    # Convert to cv2 BGR
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get model prediction
    preprocessed = preprocess(image)
    input_img = np.expand_dims(preprocessed, axis=0)
    pred_prob = model.predict(input_img, verbose=0)[0][0]

    confidence = float(pred_prob)
    confidence_pct = f"{confidence * 100:.1f}%"

    if confidence > 0.5:
        # Tumor detected - run segmentation
        skull_stripped, brain_mask = strip_skull(image, RESIZE_FOR_SEGMENTATION)
        tumor_mask = segment_tumor(skull_stripped, brain_mask)
        overlay_bgr = create_overlay_image(image, tumor_mask)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)


        result_text = f"""
## Tumor Detected

**AI Analysis Result:** Tumor Present
**Confidence Level:** {confidence_pct}
**AI Model:** NeuroScan v2.1

---

### Medical Consultation Required

This is an AI-assisted preliminary analysis. The highlighted green regions indicate areas where tumor-like characteristics were detected.

**Please consult a qualified neurologist or neurosurgeon to review these findings before making any medical decisions.**

### Recommended Next Steps

1. **Schedule a consultation** with a neurologist to review these findings in person
2. **Request a secondary analysis** by sharing this report with your referring physician
3. **Do not delay seeking care** — even if the growth appears small, early evaluation is critical
"""

        return result_text, overlay_rgb

    else:
        # No tumor
        conf_neg = f"{(1 - confidence) * 100:.1f}%"
        result_text = f"""
## No Tumor Detected

**AI Analysis Result:** No Abnormality Found
**Confidence Level:** {conf_neg}

---

Our AI-assisted analysis has examined the MRI scan and found no signs of brain tumor. The scan parameters appear within normal limits.

**Continue to consult your physician for any medical concerns.**
"""
        return result_text, original_rgb

# =============================================================================
# GRADIO UI
# =============================================================================
css = """
:root {
    --primary: #0d1f3c;
    --accent: #1e5fa8;
    --success: #15803d;
    --danger: #b91c1c;
    --gray-100: #f1f5f9;
    --gray-200: #e2e8f0;
    --gray-600: #475569;
}

body { font-family: 'Source Sans 3', -apple-system, sans-serif; }
h1, h2, h3 { font-family: 'Source Serif 4', Georgia, serif; }

.gr-button-primary { background: #0d1f3c !important; color: white !important; }
.gr-button-primary:hover { background: #162952 !important; }

#banner {
    background: #0d1f3c;
    color: rgba(255,255,255,0.85);
    padding: 14px 24px;
    text-align: center;
    font-size: 14px;
    margin-bottom: 0;
}
#banner strong { color: white; font-weight: 500; }

#footer {
    border-top: 1px solid #e2e8f0;
    padding: 16px 24px;
    text-align: center;
    font-size: 12px;
    color: #94a3b8;
}

.markdown-content h2 { color: #0d1f3c; font-size: 22px; margin-bottom: 8px; }
.markdown-content h3 { color: #475569; font-size: 15px; font-weight: 600; }
.markdown-content strong { color: #0d1f3c; }
"""

with gr.Blocks(css=css, title="NeuroScan Pro - Brain Tumor Detection") as demo:
    # Banner
    gr.HTML("""
    <div id="banner">
        <strong>MRI Brain Scan Analysis</strong> &mdash; For screening purposes only. Always consult a qualified medical professional.
    </div>
    """)

    gr.Markdown("""
    # NeuroScan Pro
    ### AI-Powered Brain Tumor Detection
    Upload an MRI scan and our AI system will analyze it to detect the presence of brain tumors.
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload MRI Scan",
                type="numpy",
                height=300,
            )
            analyze_btn = gr.Button("Analyze MRI Scan", variant="primary", size="lg")
            gr.HTML("""
            <div style="margin-top:12px; font-size:12px; color:#94a3b8; text-align:center;">
                Accepted: JPG, PNG &bull; Max 16 MB
            </div>
            """)

        with gr.Column(scale=1):
            result_output = gr.Markdown("""
            *Awaiting scan...*

            Upload an MRI image on the left and click **Analyze MRI Scan** to begin.
            """)
            result_image = gr.Image(
                label="Result",
                type="numpy",
                height=300,
            )

    gr.HTML("""
    <div id="footer">
        <strong>NeuroScan Pro</strong> &bull; Brain Tumor Detection System<br>
        For authorized medical screening use only &bull; Results are preliminary &bull; Always consult a qualified healthcare provider
    </div>
    """)

    analyze_btn.click(
        fn=analyze_mri,
        inputs=[image_input],
        outputs=[result_output, result_image]
    )

# =============================================================================
# LAUNCH
# =============================================================================
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    max_file_size=16 * 1024 * 1024
)
