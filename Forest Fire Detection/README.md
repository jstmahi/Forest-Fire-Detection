# 🔥 Forest Fire Detection Using MSA-Net with SE Blocks

This project presents a lightweight and efficient deep learning model for early forest fire detection. It integrates a Multi-Scale Attention Network (MSA-Net) with Squeeze-and-Excitation (SE) blocks to enhance feature extraction and focus on fire-relevant regions. The model is optimized for real-time deployment on resource-constrained devices like drones and IoT systems.

## 🚀 Features
- Achieved 96.1% test accuracy
- Real-time classification performance
- Lightweight architecture suitable for edge devices
- Robust against background noise and varied lighting conditions

## 🛠️ Technologies Used
- Python 3
- TensorFlow / Keras
- OpenCV
- Google Colab

## 📁 Project Structure

Forest Fire Detection/
├── fire_detection.ipynb # Jupyter notebook with model code and outputs
├── fire_detection.py # Python script version of the notebook
├── requirements.txt # List of Python dependencies
├── README.md # Project documentation
├── .gitignore # Specifies files/folders to ignore in Git
├── Results/ # Folder containing output/result images
## 📁 Dataset

This project uses the [Forest Fire, Smoke and Non-Fire Image Dataset](https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset?resource=download) from Kaggle.

### 🔽 How to Use

1. Visit the dataset page:  
👉 [Click here to download from Kaggle](https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset?resource=download)  
*(Kaggle login required)*

2. Download the ZIP file and place the extracted folders inside a `data/` directory within your project:

Forest Fire Detection/
└── data/
├── fire/
├── smoke/
└── non_fire/

> ⚠️ Note: The dataset is **not included** in this repository due to its large size (~7GB).

## 🎯 Trained Model

The trained model (`model.h5`, ~1.5GB) is not included in this repository due to GitHub's file size limitations.

📥 **Download the model here:**  
👉 [model.h5 – Google Drive](https://drive.google.com/file/d/1-U_XCgM0Ay_yztxzjdDrNJIOjLS_86tQ/view?usp=sharing)

### 🔧 How to Use

1. Download `model.h5` from the link above.
2. Place it in your project folder like this:

Forest Fire Detection/
├── model.h5

3. Load the model in your code using:

```python
from tensorflow.keras.models import load_model
model = load_model('model.h5')
📦 Installation
To install required dependencies, run:
pip install -r requirements.txt
📈 Results
Test Accuracy: 96.1%

Precision: 96.5%

Recall: 96%

F1-Score: 96.2%

