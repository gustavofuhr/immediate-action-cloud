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
from fast_alpr import ALPR
import torch
torch.multiprocessing.set_start_method('spawn', force=True)

from dfine_controller import DFINE_Controller

app = Flask(__name__)
CORS(app)

# Ensure Flask logs are propagated to gunicorn
gunicorn_logger = logging.getLogger("gunicorn.error")
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

d_fine_detector = None
license_plate_recognizer = None

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

def _lpr_results_to_dicts(results):
    detections = []
    for r in results:
        bbox = {
            "top_left": {
                "x": r.detection.bounding_box.x1,
                "y": r.detection.bounding_box.y1
            },
            "bottom_right": {
                "x": r.detection.bounding_box.x2,
                "y": r.detection.bounding_box.y2
            }
        }
        detections.append({
            "bbox": bbox,
            "label": "license_plate",  # could also be 0 or a string depending on your handling
            "score": r.detection.confidence,
            "ocr_text": r.ocr.text,
            "ocr_confidence": r.ocr.confidence
        })
    return detections

@app.route('/invocations', methods=['POST'])
@app.route('/detect/', methods=['POST'])
def detect():
    try:
        if 'image_file' in request.files:
            file = request.files['image_file']
            image = load_image_from_file(file)
        elif 'image_base64' in request.json:
            base64_data = request.json['image_base64']
            image = load_image_from_base64(base64_data)
        elif 'image_url' in request.json:
            url = request.json['image_url']
            image = load_image_from_url(url)
        else:
            return jsonify({'error': 'No valid image data provided'}), 400
        
        model_type = request.json.get("model", "object_detection")
        global d_fine_detector, license_plate_recognizer
        image_pil = Image.fromarray(image)
        start = time.time()
        
        if model_type == "license_plate_recognition":
            results = license_plate_recognizer.predict(image)
            detections = _lpr_results_to_dicts(results)
        else: # default to object detection
            classes_to_detect = request.json.get('classes_to_detect', None)
            detections = d_fine_detector.run(image_pil, classes_to_detect=classes_to_detect)

        elapsed_time_ms = (time.time() - start) * 1000
        return jsonify({'detections': detections, 'time_ms': round(elapsed_time_ms, 2)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/ping', methods=['GET'])
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200


@app.before_request
def lazy_initialize_detector():
    config_file = os.environ.get('CONFIG_FILE', '/workspace/D-FINE/configs/dfine/objects365/dfine_hgnetv2_x_obj2coco.yml')
    checkpoint_file = os.environ.get('CHECKPOINT_FILE', '/workspace/dfine_checkpoints/dfine_x_obj2coco.pth')
    device = os.environ.get('DEVICE', 'cpu')

    global d_fine_detector
    if d_fine_detector is None:
        app.logger.info(f'Initializing detector with config file: {config_file}, checkpoint file: {checkpoint_file}, device: {device}')
        d_fine_detector = DFINE_Controller(config_file, checkpoint_file, device)

    global license_plate_recognizer
    if license_plate_recognizer is None:
        license_plate_recognizer = ALPR(
            detector_model="yolo-v9-t-640-license-plate-end2end",
            ocr_model="cct-s-v1-global-model",
            detector_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            ocr_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )


if __name__ == '__main__':
    app.logger.info(f'__main__')
    app.run(host='0.0.0.0', debug=True)

    
    
