# Error Bar Detection Task

A comprehensive computer vision pipeline for detecting error bars in scientific plots. This project implements two complementary approaches: a Deep Learning Regression CNN (trained on synthetic data) and a Hybrid Heuristic Walker (robust to label noise).

## 📂 Repository Structure

```text
error-bar-detection/
│
├── 1_synthetic_generation/   # Part 1: Dataset Creation
│   ├── generate_dataset.py   # Procedural generation script (Matplotlib)
│   └── audit_synthetic.py    # Visual QA tool for generated data
│
├── 2_detection_methods/      # Part 2: Detection Logic
│   ├── deep_learning/
│   │   ├── train_kaggle.ipynb # Jupyter notebook for GPU training
│   │   └── inference_cnn.py   # Script to predict using the .h5 model
│   │
│   └── hybrid_walker/
│       ├── predictor.py      # The 'HybridErrorDetector' class (Main Logic)
│       └── run_walker.py     # Script to run the walker on a folder
│
├── 3_analysis_audits/        # Analysis Tools
│   ├── full_audit.py         # Generates 4-panel comparison images
│   └── audit_real_data.py    # Visualizes Ground Truth vs Predictions
│
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

🚀 Quick Start

1. Installation
   Install the required dependencies:

Bash
pip install -r requirements.txt 2. Generate Synthetic Data
To generate a fresh batch of 3,000 labeled images:

Bash
python 1_synthetic_generation/generate_dataset.py
Output will be saved to dataset_v7_production/.

3. Training the Model
   The CNN model is optimized for cloud training.

File: 2_detection_methods/deep_learning/train_kaggle.ipynb

Instruction: Upload this notebook to Kaggle or Google Colab, enable GPU Acceleration (T4), and attach the synthetic dataset.

Output: Download the resulting error_bar_model.h5 and place it in the root directory.

4. Running Inference (Hybrid Walker)
   To run the robust heuristic walker on your local machine:

Bash
python 2_detection_methods/hybrid_walker/run_walker.py --input "path/to/test_images"
🧠 Methodology
Approach 1: Deep Learning (CNN)
Architecture: Custom VGG-style regression network.

Input: 192x64 vertical patch crops.

Strategy: Trained on 3,000 synthetic images using Mixed Precision training. Highly effective for texture invariance (e.g., distinguishing dashed lines).

Approach 2: Hybrid Walker
Logic: Uses adaptive thresholding to identify ink, "walks" from the marker center, and jumps gaps (e.g., inside hollow markers).

Feature: Includes "Strict Cap Detection" to distinguish real error bars from axis lines.

📊 Results
Synthetic MAE: ~4.5 pixels (CNN)
