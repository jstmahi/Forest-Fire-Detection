# FlameGuard: Efficient Forest Fire Recognition with MSA-Net

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.11](https://img.shields.io/badge/TensorFlow-2.11-orange.svg)](https://www.tensorflow.org/)
[![Patent: Pending](https://img.shields.io/badge/Patent-Pending-success.svg)](#-intellectual-property)

FlameGuard is a lightweight deep learning system for **real-time forest fire detection**, built using a custom Multi-Scale Attention Network (MSA-Net).

Unlike conventional approaches that rely on heavy pretrained models, FlameGuard is **designed from scratch for edge deployment**, making it suitable for drones and IoT-based monitoring systems.

---

## Why This Project Exists

Forest fire detection is not just a vision problem—it’s a **latency + reliability + compute constraint problem**.

* Satellite systems → high latency
* Human monitoring → not scalable
* Standard CNNs (ResNet, VGG) → too heavy for edge devices
* Real-world noise → fog, clouds, sunlight cause false alarms

FlameGuard addresses all four.

---

## Core Idea

Instead of scaling up a generic CNN, this project focuses on:

* designing a **lightweight architecture**
* improving **context awareness**
* reducing **false positives without increasing compute**

---

## Key Features

### Lightweight & Edge-Ready

* ~2.8 million parameters
* Designed for **low-latency inference on constrained hardware**

### Multi-Scale Feature Extraction

* Parallel 3×3, 5×5, 7×7 convolutions
* Captures:

  * fine smoke patterns
  * mid-level textures
  * large fire regions
* Makes the model **scale-invariant**

### Channel Attention (SE Blocks)

* Learns which feature channels matter
* Suppresses noise (clouds, trees, sunlight)
* Improves precision **without adding heavy computation**

### Robust Data Pipeline

* Gaussian blur → atmospheric simulation
* Contrast enhancement → lighting variation
* Balanced dataset → avoids bias

---

## Performance

The model was trained on a balanced dataset of fire and non-fire images and evaluated on a held-out test set.

| Metric    | Score  |
| :-------- | :----- |
| Accuracy  | 96.10% |
| Precision | 0.965  |
| Recall    | 0.96   |
| F1-Score  | 0.96   |
| AUC-ROC   | 0.96   |

### What These Numbers Mean

* High precision → fewer false alarms
* High recall → fewer missed fires
* Balanced F1 → reliable real-world performance

---

## Visual Results

![Confusion Matrix](assets/Confusion_Matrix.png)
![Accuracy Curve](assets/Accuracy_curve.png)
![Loss Curve](assets/Loss_curve.png)

---

## Architecture Overview

MSA-Net combines:

* Multi-scale convolution blocks
* Squeeze-and-Excitation attention
* Lightweight design constraints

This allows the model to:

* detect fires at different distances
* suppress environmental noise
* remain efficient enough for edge deployment

---

## Tech Stack

* TensorFlow / Keras
* OpenCV
* NumPy
* Scikit-learn
* Matplotlib / Seaborn

---

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/jstmahi/Forest-Fire-Detection.git
cd Forest-Fire-Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Project Structure

* src/model.py → MSA-Net architecture
* src/train.py → training pipeline
* src/evaluate.py → metrics + confusion matrix
* src/predict.py → inference on new images

---

## What Makes This Different

Most projects:

* use pretrained models
* optimize for accuracy only

This project:

* designs a **custom architecture**
* optimizes for **accuracy + efficiency**
* explicitly targets **edge deployment constraints**
* reduces false positives using **attention mechanisms**

---

## Academic Context

Developed as a B.Tech Final Year Project
Department of Computer Science and Engineering

Institution: Sree Vidyanikethan Engineering College (JNTUA)

Project Guide: Dr. K. Reddy Madhavi

Team:

* Ramayanam Mahidhar
* Battala Chandralahari
* Singireddy Udayadithya Reddy
* Kankatala Nss Sukesh Kumar

---

## Intellectual Property

Patent Application Filed

A formal patent application has been submitted for the MSA-Net architecture and its application in lightweight, edge-deployed forest fire detection systems.

