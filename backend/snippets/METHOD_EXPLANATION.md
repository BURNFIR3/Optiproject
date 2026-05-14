# NeuroScan Pro – Method Explanation

This document walks through the core pipeline used by **NeuroScan Pro** to analyze an MRI scan and highlight potential tumour regions.

---

## 1️⃣ Pre‑processing (`preprocess`)
- **Convert to grayscale** – simplifies the image and removes colour information which is irrelevant for tumour detection.
- **Gaussian blur** – reduces noise while preserving edges.
- **Morphological closing & opening** – fills small holes and removes spurious particles.
- **Brain contour extraction** – the largest external contour is assumed to be the brain; a mask is created and applied to keep only brain tissue.
- **Resize & normalise** – the mask‑filtered image is resized to `(64, 64)` (the model input size) and scaled to `[0, 1]`.

## 2️⃣ Model Inference
```python
preprocessed = preprocess(image)
input_img = np.expand_dims(preprocessed, axis=0)
prob = model.predict(input_img, verbose=0)[0][0]
```
- The model outputs a probability of tumour presence. A threshold of **0.5** decides *tumour* vs *no tumour*.

## 3️⃣ Skull‑stripping (`strip_skull`)
- Optional resizing to the segmentation size `(256, 256)`.
- Median blur + binary threshold creates a coarse brain mask.
- The largest contour that exceeds **10 %** of the image area is kept, then morphed/eroded to produce a clean mask.
- The mask isolates brain tissue for the subsequent segmentation step.

## 4️⃣ Tumour Segmentation (`segment_tumor`)
- The masked brain image is **CLAHE‑enhanced** to improve contrast.
- A bright‑pixel percentile (85 %) determines a threshold; bright areas are likely tumour tissue.
- Morphological opening & closing clean the binary mask.
- Contours with area > 200 px are kept, producing the final tumour mask (`256 × 256`).

## 5️⃣ Overlay Creation (`create_overlay_image`)
- The tumour mask is resized back to the original image size.
- Green colour (`[0, 255, 0]`) is over‑laid on the original MRI where the mask is positive.
- Contours are drawn with a thick green line and a red centre point for visual emphasis.

## 6️⃣ Final Result
- For a **positive** prediction the overlay image (original + highlights) is returned.
- For a **negative** prediction the original image is returned unchanged.
- The markdown text summarises the confidence and provides medical‑disclaimer advice.

---

### How the pieces fit together in `analyze_mri`
```python
# Pre‑process → model → confidence
# If tumour detected:
#   skull stripping → segmentation → overlay → return overlay
# Else:
#   return original image
```

This pipeline runs completely on **CPU** on Windows (TensorFlow‑DirectML handles the lack of GPU support). The code lives in `backend/app.py` and is exposed via a **Gradio** UI.
