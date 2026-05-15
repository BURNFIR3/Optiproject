# Brain MRI Segmentation

This project implements the pipeline from `Brain magnetic resonance image (MRI) segmentation using multimodal optimization` in a clean Python structure. It includes a batch-processing script, a Flask-based web application with a beautiful UI, and robust evaluation scripts.

## Pipeline Overview

1. Gaussian-smooth the RGB MRI image with sigma `0.5` and a `3x3` window.
2. Build a 3D RGB histogram from the image.
3. Smooth the 3D histogram to remove noisy, non-dominant peaks.
4. Use multimodal PSO to find histogram peaks, where peaks become cluster centers.
5. Remove weaker peaks that are too close to stronger peaks.
6. Assign each pixel to its nearest peak using the non-Euclidean distance from the paper.
7. Save the MRI with the tumour area marked, plus masks and cluster maps.

## Folder Structure

```text
brain/
  app.py                    # Flask web application backend
  main.py                   # CLI tool for batch processing the dataset
  evaluate_single.py        # Script to evaluate internal metrics for a single image
  evaluate_dataset.py       # Script to evaluate internal metrics for the whole dataset
  requirements.txt          # Python dependencies
  
  data/
    raw/                    # Your raw dataset (images should go in yes/ or no/ folders)
    outputs/                # Pipeline outputs (segmented, masks, clusters, histograms)
    
  src/brain_segmentation/   # Core pipeline source code
    metrics.py              # Implementation of internal and external evaluation metrics
    pipeline.py             # Main pipeline orchestrator
    pso.py                  # Multimodal Particle Swarm Optimization
    ...
    
  static/                   # Web frontend assets (CSS, JS)
  templates/                # HTML templates for the web app
```

## How to Run the Web Application

The easiest way to use the pipeline is through the web interface. 

1. Ensure your dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Flask application:
   ```bash
   python app.py
   ```
3. Open your browser and navigate to `http://127.0.0.1:5000`.
4. Drag and drop any MRI image to see the pipeline results and evaluation metrics in real-time!

## How to Run Batch Processing

To process your entire dataset at once and save the segmented outputs, masks, and histograms:

From the `brain` folder:
```bash
python main.py
```

For a quick test on just the first 5 images:
```bash
python main.py --limit 5
```

For the full paper-style settings (takes longer):
```bash
python main.py --preset paper
```

*Note: The default `balanced` preset follows the same methodology with a smaller 3D histogram so the complete dataset is practical to process on a desktop.*

## How to Run and Get Evaluation Metrics

We have implemented the internal clustering evaluation metrics (F, F', and Q) defined in the paper. Lower values indicate better segmentation (less over-segmentation and more homogeneous regions).

**1. Evaluate a Single Image:**
To quickly get the evaluation metrics for a specific image, use the `evaluate_single.py` script and pass the path to the image:
```bash
python evaluate_single.py "data/raw/yes/Y1.jpg"
```

**2. Evaluate the Entire Dataset:**
To run the evaluation pipeline on every image in your dataset and see a summary of the metrics (including averages), run:
```bash
python evaluate_dataset.py
```
This script will print out a cleanly formatted table of the F, F', and Q metrics for each image directly in your terminal.
