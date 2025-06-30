import json
import base64
import time
from io import BytesIO

import boto3
from PIL import Image


COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
    9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
    24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
    31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
    37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
    52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}


class SageMakerController:

    def __init__(self, aws_region, endpoint_name):
        self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=aws_region)
        self.endpoint_name = endpoint_name

    def detect_objects(self, im_pil, classes_to_detect: list[str] = ["person"], threshold : float = 0.5, verbose: bool = False):
        """
        Run detection using the D-FINE model for the selected classes.

        image: PIL image
        classes_to_detect: list of int, classes to detect from COCO dataset (default: [0] "person")
        threshold: float, detection threshold (default: 0.5)
        """
        # encode im_pil into base64
        image_base64 = self._encode_image_to_base64(im_pil)
        start_time = time.time()
        response = self._make_aws_sagemaker_request(image_base64)
        elapsed_time = time.time() - start_time

        if verbose:
            print(f"Sagemaker inference time: {response['time_ms']:.2f} ms; Request time: {elapsed_time * 1000:.2f} ms")

        # right now, in the response of sagemaker we got labels in integers
        response["detections"] = [
            {
                "label": COCO_CLASSES[int(det["label"])],
                "score": det["score"],
                "bbox": det["bbox"]
            }
            for det in response["detections"]
        ]
        
        filtered_detections = [det for det in response["detections"] if det["label"] in classes_to_detect and det["score"] >= threshold]
        return filtered_detections, response["detections"]
    
    def detect_plates(self, im_pil, threshold: float = 0.5, verbose: bool = False):
        image_base64 = self._encode_image_to_base64(im_pil)
        start_time = time.time()
        response = self._make_aws_sagemaker_request(image_base64, model="license_plate_recognition")
        elapsed_time = time.time() - start_time
        if verbose:
            print(f"Sagemaker inference time: {response['time_ms']:.2f} ms; Request time: {elapsed_time * 1000:.2f} ms")

        filtered_detections = [det for det in response["detections"] if det["score"] >= threshold]
        return filtered_detections, response["detections"]

    def _make_aws_sagemaker_request(self, image_base64 : str, model: str = "object_detection"):
        response = self.sagemaker_runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=json.dumps({
                "image_base64": image_base64,
                "model": model
            }),
        )
        return json.loads(response["Body"].read().decode())

    def _encode_image_to_base64(self, im_pil: Image.Image) -> str:
        buffered = BytesIO()
        im_pil.save(buffered, format="JPEG")  # or "PNG" depending on your needs

        return base64.b64encode(buffered.getvalue()).decode("utf-8")