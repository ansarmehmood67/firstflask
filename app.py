from flask import Flask, request, jsonify
import os
import torch
import numpy as np
from flask_cors import CORS
import cv2
import uuid
from datetime import datetime
from io import BytesIO

# ✅ Fix Import Error: Add ESRGAN Folder to Python Path
import sys
sys.path.append(os.path.join(os.getcwd(), "ESRGAN"))

# ✅ Import ESRGAN Model
import RRDBNet_arch as arch

app = Flask(__name__)
CORS(app)

# ✅ Set Your S3 Bucket Name
BUCKET_NAME = 'my-first-app-storage'

# ✅ Load ESRGAN Model from `ESRGAN/models/`
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get absolute project path
MODEL_PATH = os.path.join(BASE_DIR, "ESRGAN", "models", "RRDB_ESRGAN_x4.pth")
device = torch.device("cpu")  # Using CPU mode

def load_model_from_local():
    """Loads the ESRGAN model from the local 'ESRGAN/models/' folder."""
    try:
        print(f"⏳ Loading ESRGAN model from: {MODEL_PATH}")
        model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=True)
        model.eval()
        print("✅ Model loaded successfully from 'ESRGAN/models/' folder!")
        return model
    except Exception as e:
        print(f"❌ Error loading the model from local folder: {e}")
        return None

# ✅ Load the Model at Startup
model = load_model_from_local()
if model is None:
    raise Exception("Failed to load the model from local folder. Exiting.")

def enhance_image_with_esrgan(image_path):
    """Runs ESRGAN on the input image and saves the enhanced output."""
    img = cv2.imread(image_path)
    img = img * 1.0 / 255  # Normalize pixel values
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    # Convert back to an image
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)

    # ✅ Save Enhanced Image Locally (Before Upload to S3)
    unique_filename = f"enhanced_{uuid.uuid4().hex}.jpg"
    enhanced_image_path = os.path.join("enhanced_images", unique_filename)
    
    # Make sure the 'enhanced_images' directory exists
    os.makedirs(os.path.dirname(enhanced_image_path), exist_ok=True)
    
    cv2.imwrite(enhanced_image_path, output)

    # ✅ Upload the enhanced image to S3
    upload_to_s3(enhanced_image_path, unique_filename)

    return enhanced_image_path, unique_filename

def upload_to_s3(file_path, s3_filename):
    """Uploads the enhanced image to AWS S3."""
    try:
        s3.upload_file(file_path, BUCKET_NAME, s3_filename)
        print(f"✅ File uploaded successfully to S3: {file_path}")
    except Exception as e:
        print(f"❌ Error uploading to S3: {e}")

@app.route("/enhance", methods=["POST"])
def enhance_image():
    """Handles image enhancement request from frontend."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    
    # Save the uploaded image to a local path
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    image_path = os.path.join("uploads", f"{timestamp}_{file.filename}")
    
    # Ensure the upload directory exists
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    file.save(image_path)

    # ✅ Run ESRGAN on the Image
    enhanced_image_path, s3_filename = enhance_image_with_esrgan(image_path)

    # ✅ Return the Enhanced Image URL (from S3) with a pre-signed URL if necessary
    file_url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{s3_filename}"

    return jsonify({
        "message": "Image enhanced successfully!",
        "image_url": file_url
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
