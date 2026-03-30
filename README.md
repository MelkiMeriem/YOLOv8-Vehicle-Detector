#  YOLOv8 Vehicle Detector

> Real-time multi-class vehicle detection — fine-tuned from scratch on 9,559 traffic images and deployed as a live web app.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.32-0075FF?style=flat-square)
![mAP@50](https://img.shields.io/badge/mAP%4050-85.4%25-2ecc71?style=flat-square)
![FPS](https://img.shields.io/badge/Inference-~94_FPS-f39c12?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

##  Overview

This project implements an end-to-end object detection pipeline for traffic scenes using **YOLOv8s** (You Only Look Once, version 8 — small variant). Starting from a pre-trained backbone, the model was fine-tuned on a labeled vehicle dataset and deployed as an interactive Gradio web application.

The full pipeline covers:
- Dataset acquisition and train/validation splitting
- Transfer learning with YOLOv8s on a T4 GPU
- Evaluation with mAP@50, mAP@50-95, precision, recall, and confusion matrix
- Real-time inference with bounding box visualization
- Speed benchmarking (~94 FPS on Tesla T4)
- Deployment as a Gradio web app with a public shareable URL

---

##  Results

| Metric | Score |
|--------|-------|
| **mAP@50** | **85.4%** |
| mAP@50-95 | 51.1% |
| Precision | 80.1% |
| Recall | 81.4% |
| Inference speed | ~11ms / image (~94 FPS) |
| Training time | ~50 min on T4 GPU |

### Per-Class Performance (mAP@50)

| Class | mAP@50 | Grade |
|-------|--------|-------|
|  bus | 0.915 |  Excellent |
|  car | 0.912 |  Excellent |
|  pickup-van | 0.885 |  Very Good |
|  motorbike | 0.851 |  Good |
|  microbus | 0.834 |  Good |
|  truck | 0.730 |  Decent (limited training data) |

> **Note on truck class:** Only 95 validation instances vs 3,500+ for car/motorbike. More data would improve this significantly.

---

##  Model Architecture

```
Input image (640 × 640)
        │
        ▼
  ┌─────────────────────────────────────────┐
  │  BACKBONE  — CSPDarknet feature extractor│
  │  NECK      — PAN+FPN multi-scale fusion  │
  │  HEAD      — Decoupled detection head    │
  └─────────────────────────────────────────┘
        │
        ▼
  Raw predictions (bounding boxes + class logits)
        │
        ▼  Non-Maximum Suppression (NMS)
        ▼
  Final detections: [x1, y1, x2, y2, class, confidence]
```

**Model:** YOLOv8s — 11.1M parameters, 28.4 GFLOPs
**Transfer learning:** Pre-trained on COCO (80 classes) → fine-tuned on 6 vehicle classes

---

##  Dataset

| Property | Value |
|----------|-------|
| Source | [Roboflow Universe — Vehicle Detection](https://universe.roboflow.com/lynkeus/vehicle-detection-mgjdd) |
| Total images | 9,559 |
| Train split | 7,647 (80%) |
| Validation split | 1,912 (20%) |
| Classes | 6 |
| Label format | YOLOv8 (normalized xywh) |

**Classes:** `bus` · `car` · `microbus` · `motorbike` · `pickup-van` · `truck`

---

##  Project Structure

```
yolov8-vehicle-detector/
│
├── Yolov8_Project_Complete.ipynb   # Full pipeline notebook
│
├── vehicle_detector/
│   └── v1/
│       ├── weights/
│       │   ├── best.pt             # Best model checkpoint (use this)
│       │   └── last.pt             # Final epoch checkpoint
│       ├── results.csv             # Epoch-by-epoch training metrics
│       └── confusion_matrix_normalized.png
│
├── outputs/
│   ├── training_dashboard.png      # Loss & mAP curves (6-panel)
│   ├── per_class_map.png           # Per-class mAP@50 bar chart
│   ├── inference_grid.png          # Sample detection results
│   └── latency_benchmark.png       # Speed distribution histogram
│
└── README.md
```

---

##  Quick Start

### 1. Clone & install

```bash
pip install ultralytics gradio
```

### 2. Run inference on an image

```python
from ultralytics import YOLO

model = YOLO("vehicle_detector/v1/weights/best.pt")
results = model.predict("your_image.jpg", conf=0.40)
results[0].show()  # display with bounding boxes
```

### 3. Run inference on a video

```python
results = model.predict("traffic_video.mp4", conf=0.40, save=True)
# annotated video saved to runs/detect/predict/
```

### 4. Launch the Gradio web app

```python
# Run the last section of the notebook
# or:
demo.launch(share=True)  # generates a public URL valid for 72h
```

### 5. Re-train from scratch

Open `Yolov8_Project_Complete.ipynb` in Google Colab with a **T4 GPU** runtime and run all cells in order.

---

##  Training Configuration

```python
model.train(
    data        = "vehicle-detection-3/data.yaml",
    epochs      = 75,
    imgsz       = 640,
    batch       = 16,
    optimizer   = "auto",        # AdamW selected automatically
    save_period = 10,
    project     = "vehicle_detector",
    name        = "v1",
    plots       = True,
)
```

**Hardware:** Google Colab Tesla T4 (14.9 GB VRAM)
**Training time:** ~50 minutes
**Augmentations:** Random flip, HSV shift, mosaic, blur (via Albumentations)

---



##  Tech Stack

| Tool | Purpose |
|------|---------|
| [Ultralytics YOLOv8](https://docs.ultralytics.com) | Model training & inference |
| [Roboflow](https://roboflow.com) | Dataset hosting & download |
| [PyTorch](https://pytorch.org) | Deep learning framework |
| [Gradio](https://gradio.app) | Web app deployment |
| [OpenCV](https://opencv.org) | Image processing & annotation |
| [Matplotlib](https://matplotlib.org) | Training visualization |
| Google Colab | Cloud GPU training (T4) |

---

