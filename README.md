# 🩺 AI-Powered Skin Lesion Triage Assistant  
### Deep Learning–Based High-Risk vs Low-Risk Classification with Grad-CAM

This project develops an AI-assisted triage tool for classifying skin lesions into **high-risk** or **low-risk** categories using dermoscopic images.  
It leverages **ResNet-18** and **Grad-CAM** to improve clinical decision support and interpretability.

> ⚠️ **Disclaimer:**  
> This system is a **prototype for academic purposes only**.  
> It is **not** a medical device and must **not** be used for clinical diagnosis.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Pipeline](#-project-pipeline)
- [Model Architecture](#-model-architecture)
- [Training Details](#-training-details)
- [Evaluation Results](#-evaluation-results)
- [Grad-CAM Interpretability](#-grad-cam-interpretability)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Limitations](#-limitations)
- [Future Work](#-future-work)
- [Team T08](#-team-t08)

---

## 🧠 Overview

Skin cancer is one of the most common cancers worldwide.  
Early detection of conditions such as **melanoma** is critical for improving patient outcomes, yet clinicians often struggle with triage decisions.

This project provides:

✔️ A **ResNet-18 classifier**  
✔️ A **binary high-risk vs low-risk triage system**  
✔️ **Grad-CAM heatmaps** for interpretability  

Its purpose is to **assist clinicians**, not replace them, by identifying lesions that may require urgent dermatology review.

---

## 📂 Dataset

### **HAM10000 — Human Against Machine (10,015 images)**  
A dermoscopic image dataset containing seven diagnostic classes, mapped into two categories:

#### **High-risk (1)**  
- `mel` — Melanoma  
- `bcc` — Basal Cell Carcinoma  
- `akiec` — Actinic Keratoses / Intraepithelial Carcinoma  

#### **Low-risk (0)**  
- `nv`, `bkl`, `df`, `vasc`

Metadata includes age, sex, and anatomical site.  
Dataset available publicly on **Kaggle**.

---

## 🔄 Project Pipeline

**Data Loading → Cleaning → EDA → Preprocessing → Model Training → Evaluation → Grad-CAM Visualization**

Key preprocessing steps include:

- Removing missing/duplicate images  
- Standardizing image size (160×160)  
- Creating binary triage labels  
- Handling class imbalance via `pos_weight`  

---

## 🏗️ Model Architecture

### **ResNet-18 Backbone**
- Pretrained on ImageNet  
- Final layer replaced with a **1-output unit**  
- Binary classification using sigmoid  
- Fully compatible with Grad-CAM  

**Why ResNet-18?**

- Lightweight  
- Fast training  
- Good performance on medical imaging  

---

## ⚙️ Training Details

| Component | Setting |
|----------|---------|
| Train/Val/Test Split | 70% / 15% / 15% |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Epochs | 6 |
| Batch Size | 16 |
| Loss Function | BCEWithLogitsLoss (`pos_weight`) |
| Augmentation | Flips, rotations |

---

## 📊 Evaluation Results

**Final Test Metrics**

- **Accuracy:** 0.75  
- **Precision (High-risk):** 0.69  
- **Recall (High-risk):** 0.88  
- **F1-score:** 0.78  
- **ROC-AUC:** 0.860  

**Interpretation:**  
High recall means the model is effective at identifying high-risk lesions, which is critical for triage.

---

## 🔥 Grad-CAM Interpretability

Grad-CAM heatmaps reveal which image regions influenced the model's decision.  
This allows clinicians to verify that predictions are based on meaningful dermatological features, such as:

- Asymmetry  
- Irregular borders  
- Color variation  

This improves transparency and trust.

---

## 📁 Project Structure

DS620-TEAMPROJECT/

│── backend/
  ── backend.py
  ── model.pth
  ── requirements.txt

│── data/
  ── HAM10000_images/
  ── HAM10000_metadata.csv
  
│── frontend/
  ── index.html

│── notebooks/
  ── ai-powered-skin-lesion-triage-assistant.ipynb
  ── best_resnet18_triage.pth

│── README.md
│── requirements.txt



---

## 📦 Installation

### 1. Clone the repository

git clone <your-repo-url>
cd DS620-TEAMPROJECT

### 2. Install dependencies
pip install -r requirements.txt'

### 3. Download the dataset
Place the HAM10000 images and metadata inside the data/ directory.

▶️ How to Run Backend and Frontend

This project includes two components:
Backend → Flask API that performs AI predictions
Frontend → Simple HTML/JS interface to upload images and display results

Follow the steps below.

🖥️ 1. Start the Backend (Model API)

Step 1 — Navigate to backend folder
cd DS620-TEAMPROJECT/backend

Step 2 — Install backend dependencies
pip install -r requirements.txt

Step 3 — Run backend
python3 backend.py
You should see:
Running on http://127.0.0.1:5000

Step 4 — Test backend
Open this URL:
http://127.0.0.1:5000/ping
Expected output:
{"message": "pong"}

🌐 2. Run the Frontend
Recommended Method — VS Code Live Server
Open the frontend folder in VS Code
Right-click index.html
Choose Open with Live Server
This will open:
http://127.0.0.1:5500/frontend/index.html
Alternative Method (not recommended)

Double-click index.html
⚠️ May cause CORS issues.

🔗 3. Use the Web App
Once backend and frontend are running:
Visit the frontend page
Upload an image
Click Predict
View:
Predicted class
Confidence score

Probability for each class
Raw JSON output

---
⚠️ Troubleshooting
CORS Error
Backend must include:
CORS(app, resources={r"/*": {"origins": "*"}})

Restart backend after editing.
Predict Button Not Working
Ensure backend is running:
http://127.0.0.1:5000/ping

Missing packages
Install with:
pip install flask flask-cors torch torchvision pillow

---
⚠️ Limitations
Dataset is skewed toward lighter skin tones
Only dermoscopic images — smartphone photos unsupported

Limited data used due to academic constraints
No clinical validation
Potential demographic bias

---
🚀 Future Work
Explore deeper backbones (ResNet-50, EfficientNet, ViT)

Improve fairness with diverse datasets

Evaluate demographic performance

Deploy as a mobile/web clinical assistant

Adapt to smartphone images

---
| Member             | Contribution                         |
| ------------------ | ------------------------------------ |
| **Sara Verkiyani** | Team Lead, Writing, Model Evaluation |
| **Siraphat**       | Model Training, Grad-CAM, Analysis   |
| **Akzhol**         | Data Cleaning, EDA                   |
| **Zeinep**         | Literature Review, Fairness Analysis |

