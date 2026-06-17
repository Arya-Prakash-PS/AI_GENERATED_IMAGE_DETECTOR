"""
AI Image Detector — Flask Backend
===================================
REST API that loads the trained EfficientNet-B0 model and serves predictions.
Also serves the web frontend.

Endpoint:
    POST /api/predict  — Upload an image, returns AI-generated probability %.

Usage:
    python app.py
"""

import os
import io
import traceback
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
import torch.nn as nn
from torchvision import transforms, models
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS


# ──────────────────── Configuration ────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "ai_detector_model.pth")
IMG_SIZE = 224
BLUR_THRESHOLD = 100.0  # Laplacian variance below this → image is blurry
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif", "gif"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ──────────────────── Flask App ────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


# ──────────────────── Image Enhancement ────────────────────

def compute_blur_score(image: Image.Image) -> float:
    """
    Compute blur score using Laplacian variance.
    Lower score = blurrier image.
    Works with any image size — no external dependencies beyond numpy.
    """
    gray = np.array(image.convert("L"), dtype=np.float64)

    # Handle very small images
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0

    # Compute discrete Laplacian using finite differences
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4 * gray[1:-1, 1:-1]
    )
    return float(laplacian.var())


def enhance_image(image: Image.Image) -> Image.Image:
    """
    Enhance a blurry or low-quality image:
    1. Sharpen (applied twice for stronger effect)
    2. Boost contrast slightly
    3. Boost sharpness via PIL enhancer
    """
    # Apply unsharp mask for detail recovery
    image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    # Additional sharpen pass
    image = image.filter(ImageFilter.SHARPEN)

    # Slight contrast boost
    contrast_enhancer = ImageEnhance.Contrast(image)
    image = contrast_enhancer.enhance(1.15)

    # Sharpness boost
    sharpness_enhancer = ImageEnhance.Sharpness(image)
    image = sharpness_enhancer.enhance(1.5)

    return image


# ──────────────────── Model Loading ────────────────────

def load_model():
    """Load the trained model from disk."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model file not found at: {MODEL_PATH}")
        print("   Run `python train.py` first to train the model.")
        return None, None, device

    print(f"Loading model from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)

    # Rebuild model architecture
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 1),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    class_names = checkpoint.get("class_names", ["FAKE", "REAL"])
    print(f"  [OK] Model loaded (Val Acc: {checkpoint.get('val_accuracy', 'N/A'):.4f})")
    print(f"  Classes: {class_names}")
    print(f"  Device: {device}")

    return model, class_names, device


# Global model instance — loaded once at startup
model, class_names, device = load_model()

# Inference transform (matches training val transform)
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ──────────────────── Prediction ────────────────────

def predict_image(image: Image.Image):
    """
    Run the full prediction pipeline on a PIL image.
    Returns a dict with prediction results.
    """
    if model is None:
        return {
            "success": False,
            "error": "Model not loaded. Train the model first with `python train.py`.",
        }

    # Step 1: Convert to RGB (handles RGBA, grayscale, palette images)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Step 2: Blur detection and enhancement
    blur_score = compute_blur_score(image)
    was_enhanced = False

    if blur_score < BLUR_THRESHOLD:
        image = enhance_image(image)
        was_enhanced = True

    # Step 3: Record original dimensions
    original_size = image.size  # (width, height)

    # Step 4: Transform for model input
    input_tensor = inference_transform(image).unsqueeze(0).to(device)

    # Step 5: Inference
    with torch.no_grad():
        logit = model(input_tensor).squeeze()
        probability_real = torch.sigmoid(logit).item()

    # Class mapping: FAKE=0, REAL=1 (alphabetical from ImageFolder)
    # sigmoid(logit) → probability of REAL (class 1)
    # So AI-generated probability = 1 - P(REAL)
    ai_generated_pct = round((1.0 - probability_real) * 100, 2)

    # Determine label and confidence level
    if ai_generated_pct >= 75:
        label = "AI Generated"
        confidence = "High"
    elif ai_generated_pct >= 50:
        label = "Likely AI Generated"
        confidence = "Medium"
    elif ai_generated_pct >= 25:
        label = "Likely Real"
        confidence = "Medium"
    else:
        label = "Real"
        confidence = "High"

    return {
        "success": True,
        "ai_generated_percentage": ai_generated_pct,
        "label": label,
        "confidence": confidence,
        "image_enhanced": was_enhanced,
        "blur_score": round(blur_score, 2),
        "original_dimensions": f"{original_size[0]}×{original_size[1]}",
    }


# ──────────────────── Routes ────────────────────

@app.route("/")
def index():
    """Serve the main web page."""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accept an image upload and return prediction results.

    Request: multipart/form-data with field 'image'
    Response: JSON with ai_generated_percentage and metadata
    """
    # Validate request
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided. Send a file with field name 'image'."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    # Check file extension
    filename = file.filename.lower()
    extension = filename.rsplit(".", 1)[-1] if "." in filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": f"Unsupported file type '.{extension}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        }), 400

    try:
        # Read and validate image
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({"success": False, "error": "Uploaded file is empty."}), 400

        image = Image.open(io.BytesIO(image_bytes))
        image.load()  # Force full decode to catch corrupt files early

        # Run prediction
        result = predict_image(image)

        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 500

    except (IOError, OSError) as e:
        return jsonify({"success": False, "error": f"Cannot read image file: {str(e)}"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device),
    })


# ──────────────────── Main ────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("   AI Image Detector - Web Application")
    print("=" * 55)

    if model is None:
        print("\n[WARNING] No trained model found!")
        print("   The app will start but predictions will fail.")
        print("   Train the model first: python train.py\n")

    print("   Open http://localhost:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
