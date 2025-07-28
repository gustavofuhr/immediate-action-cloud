import os
import base64
import logging
import time

from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from flask_cors import CORS
import torch
torch.multiprocessing.set_start_method('spawn', force=True)

from model_pipelines import get_models_and_pipelines, model_pipelines

app = Flask(__name__)
CORS(app)

# Ensure Flask logs are propagated to gunicorn
gunicorn_logger = logging.getLogger("gunicorn.error")
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)


def load_image_from_file(file):
    image = Image.open(file)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def load_image_from_base64(encoded_data):
    decoded_data = base64.b64decode(encoded_data)
    np_data = np.frombuffer(decoded_data, np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    return image

def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def load_image_pil_from_file(file):
    return Image.open(file).convert("RGB")

def load_image_pil_from_base64(encoded_data):
    decoded_data = base64.b64decode(encoded_data)
    np_data = np.frombuffer(decoded_data, np.uint8)
    image_bgr = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)

def load_image_pil_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")



@app.route('/models', methods=['GET'])
def list_models():
    global model_pipelines
    res = {"available_models": []}
    for key, model_info in model_pipelines.items():
        res["available_models"].append({
            "name": key,
            "description": model_info.description,
            "version": model_info.version
        })
    return jsonify(res), 200


@app.route('/invocations', methods=['POST'])
@app.route('/detect/', methods=['POST'])
def detect():
    try:
        if 'image_file' in request.files:
            file = request.files['image_file']
            image_pil = load_image_pil_from_file(file)
        elif 'image_base64' in request.json:
            base64_data = request.json['image_base64']
            image_pil = load_image_pil_from_base64(base64_data)
        elif 'image_url' in request.json:
            url = request.json['image_url']
            image_pil = load_image_pil_from_url(url)
        else:
            return jsonify({'error': 'No valid image data provided'}), 400

        global model_pipelines
        models_to_run = request.json.get("models", ["object_detection"])
        if not isinstance(models_to_run, list):
            return jsonify({'error': '`models` must be a list of model names'}), 400

        overall_start = time.time()
        results = {}

        for model_name in models_to_run:
            if model_name not in model_pipelines:
                return jsonify({'error': f"Model '{model_name}' not found"}), 400

            model = model_pipelines[model_name].model

            model_start = time.time()
            detections = model.run(image_pil, 
                                threshold=request.json.get("threshold", 0.5),
                                classes_to_detect=request.json.get("classes_to_detect", ["person"]))
            model_elapsed = (time.time() - model_start) * 1000

            results[model_name] = {
                "detections": detections,
                "time_ms": round(model_elapsed, 2)
            }

        overall_elapsed = (time.time() - overall_start) * 1000

        return jsonify({
            "results": results,
            "total_time_ms": round(overall_elapsed, 2)
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/ping', methods=['GET'])
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200


@app.before_request
def lazy_initialize_detector():
    global model_pipelines
    if model_pipelines is None:
        model_pipelines = get_models_and_pipelines()


if __name__ == '__main__':
    app.logger.info(f'__main__')
    app.run(host='0.0.0.0', debug=True)

    
    
