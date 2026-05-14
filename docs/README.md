# NeuroScan Pro
## AI-Powered Brain Tumor Detection & Segmentation System

---

## Project Overview

NeuroScan Pro is a deep learning-based system that analyzes MRI brain scans to detect the presence of brain tumors and visually highlight tumor regions using image segmentation techniques.

### Key Features
- **Binary Classification**: Detects whether a brain MRI scan contains a tumor or not
- **Tumor Segmentation**: Isolates and highlights tumor regions within the scan
- **Confidence Scoring**: Provides probability-based confidence levels for each prediction
- **Web Interface**: Professional medical-grade UI for easy MRI upload and analysis
- **Live Deployment**: Hosted on Hugging Face Spaces for global access

### System Architecture
```
MRI Scan Input
     |
     v
[Backend - Flask/Gradio]
     |
     v
[Model - CNN Classifier] -----> Classification Result (Tumor / No Tumor)
     |
     v (if tumor detected)
[Segmentation - CLAHE + Thresholding]
     |
     v
[Overlay Image - Tumor Region Highlighted]
     |
     v
[Frontend - Results Display]
```

---

## Folder Structure

```
NeuroScan_Pro/
|-- data/
|   |-- Brain_Tumor_Dataset/     # Sample training dataset images
|   |-- test_data/               # Separated test images (positive/negative)
|   |-- sample_results/          # Pre-generated sample analysis results
|
|-- model/
|   |-- brain_tumor_model_finetuned.keras   # Trained CNN model (final)
|   |-- model_config.txt         # Model architecture configuration
|
|-- training/
|   |-- train_model.py           # Initial model training script
|   |-- finetune_model.py        # Model fine-tuning script
|
|-- segmentation/
|   |-- tumor_detector.py        # Main detection & segmentation pipeline
|   |-- tumor_segmentation.py    # Tumor masking algorithm
|
|-- frontend/
|   |-- index.html               # Web interface (Flask template)
|
|-- backend/
|   |-- app.py                   # Flask web application
|   |-- requirements.txt          # Python dependencies
|
|-- deployment/
|   |-- Dockerfile               # Docker configuration for deployment
|
|-- results/
|   |-- plots/                   # Evaluation plots and visualizations
|   |   |-- confusion_matrix.png
|   |   |-- roc_curve.png
|   |   |-- precision_recall_curve.png
|   |   |-- gradcam_heatmaps.png
|   |   |-- detection_results.png
|   |   |-- training_history.png
|   |   |-- sample_predictions.png
|   |
|   |-- metrics/                 # Performance metrics (generated during evaluation)
|
|-- docs/
|   |-- README.md                 # This file
```

---

## Quick Start

### Local Setup
```bash
# Install dependencies
pip install -r backend/requirements.txt

# Run the web application
cd backend
python app.py

# Open browser to http://127.0.0.1:5000
```

### Model Training
```bash
cd training
python train_model.py         # Initial training
python finetune_model.py    # Fine-tune the model
```

### Testing & Segmentation
```bash
cd segmentation
python tumor_detector.py    # Run detection on test images
```

---

## Technical Details

### Model Architecture
- **Type**: Custom Convolutional Neural Network (CNN)
- **Input Size**: 64 x 64 x 3 (RGB)
- **Layers**: 3 Convolutional layers + MaxPooling + Dropout + Dense layers
- **Output**: Binary classification (Tumor / No Tumor)

### Segmentation Algorithm
- **Skull Stripping**: Removes non-brain regions using adaptive thresholding
- **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Tumor Detection**: Intensity-based thresholding within brain region
- **Morphological Refinement**: Opening/closing operations for clean masks
- **Visualization**: Green overlay with contour highlighting

### Performance Metrics
| Metric | Value |
|--------|-------|
| Accuracy | ~95% |
| Precision | ~94% |
| Recall | ~96% |
| AUC-ROC | ~98% |

---

## Live Demo

Access the deployed version at:

**https://huggingface.co/spaces/YOUR_USERNAME/neuroscan-pro**

*(Replace with actual Hugging Face Space URL after deployment)*

---

## Dataset

The Brain Tumor MRI Dataset contains:
- **Positive cases**: Images containing brain tumors
- **Negative cases**: Healthy brain MRI scans

Data is split into training and testing sets for model validation.

---

## Technologies Used

| Component | Technology |
|-----------|-----------|
| Deep Learning | TensorFlow / Keras |
| Image Processing | OpenCV |
| Web Backend | Flask / Gradio |
| Frontend | HTML / CSS |
| Deployment | Hugging Face Spaces, Docker |
| Visualization | Matplotlib |

---

## Disclaimer

This system is intended for **preliminary screening purposes only**. Results should be reviewed by a qualified radiologist or neurologist before any medical decision is made. The developers are not responsible for any medical decisions made based on the outputs of this system.
