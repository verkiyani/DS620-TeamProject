# backend.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

# -------------------------
# 1. FLASK + CORS
# -------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # allow frontend on any localhost port

# -------------------------
# 2. MODEL & CLASSES
# -------------------------

# TODO: Change these names to match your real classes.
# Example: ["benign", "malignant"] or ["normal", "cancer"]
class_names = ["benign", "malignant"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build the same model architecture used in training.
# Your checkpoint shows it is a ResNet with 1 output neuron.
model = models.resnet18(weights=None)  # Use resnet18; change if you trained resnet34, resnet50, etc.

# Replace the final FC layer with 1 output for binary classification.
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 1)

# Load the trained weights.
state_dict = torch.load("model.pth", map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()

# -------------------------
# 3. IMAGE PREPROCESSING
# -------------------------
# Must match your training pipeline.
transform = transforms.Compose([
    transforms.Resize((224, 224)),       # Change if your training used a different size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],      # Adjust if your training used different normalization
        std=[0.229, 0.224, 0.225]
    ),
])

# -------------------------
# 4. TEST ROUTE
# -------------------------
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong"})

# -------------------------
# 5. PREDICTION ROUTE
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "empty filename"}), 400

    # Open image safely
    try:
        img = Image.open(f.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"cannot open image: {e}"}), 400

    # Preprocess
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(img_tensor)              # shape [1, 1]
        prob_class1 = torch.sigmoid(logits)[0, 0].item()

    prob_class0 = 1.0 - prob_class1

    # Choose predicted class
    if prob_class1 >= 0.5:
        predicted_index = 1
        confidence = prob_class1
    else:
        predicted_index = 0
        confidence = prob_class0

    predicted_label = class_names[predicted_index]

    # Return detailed response
    return jsonify({
        "filename": f.filename,
        "predicted_label": predicted_label,
        "predicted_index": int(predicted_index),
        "confidence": float(confidence),     # confidence for selected label
        "prob_class0": float(prob_class0),
        "prob_class1": float(prob_class1),
        "threshold": 0.5
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
