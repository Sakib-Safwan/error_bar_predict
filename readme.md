````markdown
# Error Bar Detection Task

A comprehensive computer vision pipeline for detecting error bars in scientific plots. This project implements two complementary approaches: a Deep Learning Regression CNN (trained on synthetic data) and a Hybrid Heuristic Walker (robust to label noise).

## 📂 Repository Structure

```text
ERROR BAR DETECTION FINAL/
│
├── synthetic_generation/
│   ├── generate_data.py       # Generates synthetic dataset (Matplotlib)
│   └── audit_predictions.py   # Visual QA tool
│
├── detection_methods/
│   ├── Deep Learning/
│   │   ├── train_model_kaggle.ipynb  # GPU Training Notebook
│   │   └── predict_ml.py             # CNN Inference Script
│   │
│   └── Hybrid Walker/
│       └── Predictor.py              # Hybrid Walker Algorithm
│
├── requirements.txt           # Project dependencies
└── readme.md                  # Project documentation
```
````

## 🚀 Quick Start

### 1. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt

```

### 2. Generate Synthetic Data

Run the generator script to create the synthetic training data:

```bash
python synthetic_generation/generate_data.py

```

- **Output:** This will create a folder named `dataset_v8_backgrounds`.

### ⚠️ IMPORTANT: Dataset Setup

The detection scripts are configured to look for a production-ready folder name.
**Please rename the generated folder before running inference:**

1. **Delete** any existing `dataset_v7_production` folder if it exists.
2. **Rename** the generated `dataset_v8_backgrounds` to `dataset_v7_production`.

### 3. Training the Model

The CNN model is optimized for cloud training.

- **File:** `detection_methods/Deep Learning/train_model_kaggle.ipynb`
- **Instruction:** Upload this notebook to Kaggle or Google Colab, enable **GPU (T4)**, and upload the synthetic dataset.
- **Output:** Download the resulting `error_bar_model_ml.h5` and place it in the project root directory.

### 4. Running Inference

Both detection scripts run automatically on the `dataset_v7_production` folder.

**Run Hybrid Heuristic Walker:**

```bash
python "detection_methods/Hybrid Walker/Predictor.py"

```

_Results will be saved to: `detection_results_v6_hybrid_`

**Run Deep Learning Inference:**

```bash
python "detection_methods/Deep Learning/predict_ml.py"

```

_Results will be saved to: `detection_results_ml_`

## 🧠 Methodology

### Approach 1: Deep Learning (CNN)

- **Architecture:** Custom VGG-style regression network.
- **Strategy:** Trained on 3,000 synthetic images using Mixed Precision training. Highly effective for texture invariance (e.g., distinguishing dashed lines).

### Approach 2: Hybrid Walker (Recommended)

A rule-based algorithm developed to handle "Sim-to-Real" gaps and label noise.

- **Logic:** Uses adaptive thresholding and "Strict Cap Detection" to distinguish real error bars from axis lines.
- **Performance:** Successfully identifies error bars even when ground truth labels are inconsistent.

## 🔗 Project Links

- [Technical Report PDF](https://drive.google.com/file/d/1SSzOeEl9Wd6MqpI1-UZbB4ox1HlMNQvD/view?usp=sharing)
- [Generated Dataset (Google Drive)](https://drive.google.com/drive/folders/16upJmQ-A5x2g6ACZB2ujH0E8rEyzP352?usp=sharing)

```

```
