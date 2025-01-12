from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import torch
from model import MRI, Resize3D, add_channel_dimension, to_tensor  # Import necessary functions
from torch import nn
from torchvision.models.video import r3d_18
from torchvision.transforms import Compose, Normalize
import nibabel as nib
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'nii', 'nii.gz'}

# Load trained model
model_path = 'best_model.pth'
model = r3d_18(weights='R3D_18_Weights.DEFAULT')
model.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Preprocessing pipeline
transform = Compose([
    Resize3D((128, 128, 128)),
    add_channel_dimension,
    to_tensor,
    Normalize(mean=[0.5], std=[0.5])
])

def allowed_file(filename):
    """Check if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_mri(file_path):
    """Load and preprocess the MRI file."""
    try:
        mri_data = nib.load(file_path).get_fdata()
        mri_data = transform(mri_data)
        mri_data = mri_data.unsqueeze(0)  # Add batch dimension
        return mri_data
    except Exception as e:
        raise ValueError(f"Error loading or processing MRI file: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'mri' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['mri']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload a .nii or .nii.gz file."}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        # Ensure upload folder exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        file.save(file_path)

        # Preprocess and predict
        input_data = preprocess_mri(file_path)
        with torch.no_grad():
            output = model(input_data)
            prediction = torch.argmax(output, dim=1).item()

        result = 'Tumor Detected' if prediction == 1 else 'No Tumor'
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500


@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_text = request.form.get('feedback')
    if feedback_text:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open('feedback_log.txt', 'a') as f:
            f.write(f"[{timestamp}] Feedback: {feedback_text}\n")
        flash('Thank you for your feedback!')
    else:
        flash('Feedback cannot be empty!')
    return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
