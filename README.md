# DS620-TeamProject
AI-Powered Skin Lesion Triage Assistant 

🩺 AI-Powered Skin Lesion Triage Assistant
Deep Learning–Based High-Risk vs Low-Risk Classification with Grad-CAM

This project implements an AI-assisted triage system for skin lesions using dermoscopic images.
The model classifies lesions into high-risk or low-risk categories to support early detection of potentially dangerous skin cancers such as melanoma.

⚠️ Disclaimer:
This model is a prototype for academic purposes only.
It is not a medical device and must not be used for clinical diagnosis.

📌 Table of Contents

Overview

Dataset

Project Pipeline

Model Architecture

Training Details

Evaluation Results

Grad-CAM Interpretability

Project Structure

Installation

Usage

Limitations

Future Work


🧠 Overview

Skin cancer is one of the most common cancers worldwide, and melanoma can be life-threatening if detected late.
Primary care providers often face difficulty determining which lesions require urgent dermatology referral.

This project builds:

✔️ A ResNet-18 deep learning classifier
✔️ A binary high-risk vs low-risk triage system
✔️ Grad-CAM heatmaps to visualize model decision regions

The objective is to assist, not replace, clinicians by highlighting high-risk lesions that need priority review.


📂 Dataset
HAM10000 (Human Against Machine, 10,015 images)

Dermoscopic images of pigmented skin lesions

7 diagnostic classes, mapped to 2 triage categories:

High-risk (1):

mel — Melanoma

bcc — Basal Cell Carcinoma

akiec — Actinic Keratoses / Carcinoma

Low-risk (0):

nv, bkl, df, vasc

Metadata includes: age, sex, body site.

The dataset is publicly available on Kaggle.

🔄 Project Pipeline
Data Loading → Cleaning → EDA → Preprocessing → Model Training → Evaluation → Interpretability (Grad-CAM)

Data Cleaning Includes:

Removing missing or duplicate images

Standardizing image resolution (160×160)

Creating binary labels

Handling class imbalance (via pos_weight)

🏗️ Model Architecture
ResNet-18 (pretrained on ImageNet)

Final fully connected layer replaced with 1-output unit

Sigmoid activation for binary classification

Compatible with Grad-CAM for interpretability

Why ResNet-18?

Lightweight

Fast training

Proven strong performance in medical imaging tasks

⚙️ Training Details
Component	Setting
Train/Val/Test Split	70/15/15 (stratified)
Optimizer	Adam
Learning Rate	1e-4
Epochs	6
Batch Size	16
Loss Function	BCEWithLogitsLoss (+ pos_weight)
Augmentation	Horizontal/vertical flips, rotations

Model selection is based on lowest validation loss.

📊 Evaluation Results

Final Test Performance:

Accuracy: 0.75

Precision (High-risk): 0.69

Recall (High-risk): 0.88

F1-score: 0.78

ROC-AUC: 0.860

Interpretation:
High recall means the model rarely misses high-risk lesions — desirable for a triage system.

🔥 Grad-CAM Interpretability

Grad-CAM heatmaps highlight regions of the image the model focuses on when predicting risk.
These visual explanations help clinicians understand whether the model is focusing on clinically relevant features such as:

Asymmetry

Irregular borders

Color variation

This improves transparency and trust.

📁 Project Structure

DS620-TEAMPROJECT/
│── backend/
│   ├── backend.py
│   ├── model.pth
│   └── requirements.txt
│── data/
│   ├── HAM10000_images/
│   └── HAM10000_metadata.csv
│── frontend/
│   └── index.html
│── notebooks/
│   ├── ai-powered-skin-lesion-triage-assistant.ipynb
│   └── best_resnet18_triage.pth
│── README.md
│── requirements.txt


📦 Installation
1. Clone repository
git clone <your-repo-url>
cd project

2. Install dependencies
pip install -r requirements.txt

3. Download dataset

Place the HAM10000 folder under data/.

▶️ Usage
Train model
python src/train.py

Evaluate model
python src/evaluate.py

Generate Grad-CAM heatmaps
python src/gradcam.py --image path/to/image.jpg

⚠️ Limitations

Dataset heavily skewed toward lighter skin tones

Dermoscopic images only — performance on smartphone photos is not guaranteed

Small subset used for training due to course constraints

No clinical validation

🚀 Future Work

Use deeper models (ResNet-50, EfficientNet, Vision Transformers)

Improve fairness through dataset diversification

Evaluate performance across demographic groups

Deploy as a mobile/web diagnostic-support tool

Fine-tune on consumer smartphone images

👥 Team T08

Sara Verkiyani — Team Lead, Writing, Model Evaluation

Siraphat — Model Training, Grad-CAM, Analysis

Akzhol — Data Cleaning, EDA

Zeinep — Literature Review, Fairness & Limitations