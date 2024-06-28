import requests
import os
from flask import request, Flask, jsonify
from model import Model
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction."""
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if 'patient_data' not in request.form:
        return jsonify({'error': 'No patient data provided'}), 400

    try:

        patient_data = request.form['patient_data']
        patient_data = np.array(eval(patient_data), dtype=np.float32)
        print("Patient data : ", patient_data)
        if patient_data.shape[0] != 22:
            return jsonify({'error': 'Patient data must contain 22 features'}), 400

        model = Model()
        predictions = model.predict(image_file, patient_data)
        
        return jsonify({
            "predictions": predictions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000)
