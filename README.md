# Forest Fire Detection Using MSA-Net with SE Blocks

This is a B.Tech final year project demonstrating a lightweight deep learning model for forest fire detection. The model, an MSA-Net with Squeeze-and-Excitation (SE) blocks, is designed for efficiency and accuracy, making it suitable for real-time deployment on edge devices.

This model was trained on a curated dataset of fire and non-fire images and achieved a **test accuracy of 96.10%**.

[Sample Predictions](assets/sample%20output-1.png)

## 📖 Abstract

Forest fires pose a significant threat to ecosystems, biodiversity, and human life. This project proposes a novel, lightweight deep learning approach (MSA-Net with SE blocks) for rapid and accurate detection. The model uses multi-scale convolutional layers to capture varied spatial patterns and SE blocks to adaptively recalibrate feature responses, enhancing sensitivity with minimal computational overhead.

## 📊 Results

The model's performance was validated using a confusion matrix, accuracy/loss curves, and a classification report.

| Metric | Value |
| :--- | :--- |
| **Test Accuracy** | **96.10%** |
| Precision (Fire) | 0.95 |
| Recall (Fire) | 0.98 |
| F1-Score (Fire) | 0.96 |

### Performance Curves
[Training vs Validation Accuracy](assets/Accuracy%20curve.png)

[Training vs Validation Loss](assets/Loss%20curve.png)

### Confusion Matrix
[Confusion Matrix](assets/Confusion%20Matrix.png)

## 🛠️ How to Use

### 1. Prerequisites

You must have Python and the following libraries installed. You can install them using the `requirements.txt` file:

```bash
pip install -r requirements.txt

### 2. Get the Project Files

Clone the repository to get a copy of the project on your local machine.

```bash
git clone [https://github.com/jstmahi/Forest-Fire-Detection.git](https://github.com/jstmahi/Forest-Fire-Detection.git)
cd Forest-Fire-Detection
