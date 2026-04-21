# FlameGuard: Efficient Forest Fire Recognition with MSA-Net

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.11](https://img.shields.io/badge/TensorFlow-2.11-orange.svg)](https://www.tensorflow.org/)
[![Patent: Pending](https://img.shields.io/badge/Patent-Pending-success.svg)](#-intellectual-property)

FlameGuard is a lightweight deep learning system for **real-time forest fire detection**, built using a custom Multi-Scale Attention Network (MSA-Net). Unlike conventional approaches that rely on heavy pretrained models, FlameGuard is **designed from scratch for edge deployment**, making it ideal for drones and IoT-based monitoring systems.

## 📖 The Problem: Beyond Just Computer Vision
Forest fire detection is not just a vision problem—it’s a **latency, reliability, and compute constraint problem**. Traditional detection methods fail in real-world scenarios:
* **Satellite systems:** Too high latency for early response.
* **Standard CNNs (ResNet, VGG):** Too computationally heavy for edge devices.
* **Environmental Noise:** Fog, clouds, and sunlight cause massive false alarm rates.

FlameGuard addresses all of these by focusing on a lightweight architecture, context awareness, and reducing false positives without increasing computational load.

##  Key Features & Architecture
Instead of scaling up a generic CNN, this project utilizes a custom architecture tailored for constrained hardware:

* **Lightweight & Edge-Ready:** Contains only ~2.8 million parameters, designed specifically for low-latency inference on drones.
* **Multi-Scale Feature Extraction:** Parallel 3x3, 5x5, and 7x7 convolutions capture fine smoke patterns, mid-level textures, and large fire regions simultaneously, making the model scale-invariant.
* **Channel Attention (SE Blocks):** Squeeze-and-Excitation blocks learn which feature channels matter, actively suppressing background noise (clouds, trees) and improving precision without adding heavy computation.
* **Robust Data Pipeline:** Incorporates Gaussian blur and contrast enhancement to simulate real-world atmospheric distortion and complex lighting variations.

## 📊 Performance & Results
The model was trained on a balanced dataset of fire and non-fire images and evaluated on a held-out test set. It achieved an outstanding **test accuracy of 96.10%**.

| Metric | Score | Impact |
| :--- | :--- | :--- |
| **Accuracy** | 96.10% | Overall model correctness |
| **Precision** | 0.965 | Fewer false alarms (critical for emergency response) |
| **Recall** | 0.96 | Fewer missed fires (minimizing false negatives) |
| **F1-Score** | 0.96 | Highly reliable real-world performance |
| **AUC-ROC** | 0.96 | Strong separability between fire and non-fire classes |

### Visualizing Performance
![Confusion Matrix](assets/Confusion_Matrix.png) 
![Accuracy Curve](assets/Accuracy_curve.png) 
![Loss Curve](assets/Loss_curve.png)

## 🛠️ Tech Stack
* **Deep Learning Framework:** TensorFlow / Keras
* **Computer Vision & Data Processing:** OpenCV, NumPy
* **Visualization & Metrics:** Matplotlib, Seaborn, Scikit-learn

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/jstmahi/Forest-Fire-Detection.git](https://github.com/jstmahi/Forest-Fire-Detection.git)
cd Forest-Fire-Detection
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. 
```bash
pip install -r requirements.txt
```

### 3. Usage Structure
The repository is cleanly modularized:
* `src/model.py`: Contains the MSA-Net architecture and SE-block logic.
* `src/train.py`: Script to train the model with Early Stopping and Learning Rate reduction.
* `src/evaluate.py`: Generates the confusion matrix and classification reports.
* `src/predict.py`: Inference script to test the model on a single image.

## 🎓 Academic Context
This project was developed as a B.Tech Final Year Project at the Department of Computer Science and Engineering.

* **Institution:** Sree Vidyanikethan Engineering College (Affiliated to JNTUA)
* **Project Guide:** Dr. K. Reddy Madhavi (Professor, Dept of CSE)

**Development Team:**
* Ramayanam Mahidhar
* Battala Chandralahari
* Singireddy Udayadithya Reddy
* Kankatala Nss Sukesh Kumar

## 🏆 Intellectual Property
**Patent Application Filed:** A formal patent application has been submitted for the novel MSA-Net architecture and its specific application in lightweight, edge-deployed forest fire detection systems developed in this repository.
