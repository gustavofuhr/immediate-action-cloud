import json
import base64
import time
from io import BytesIO

import boto3
from PIL import Image



class SageMakerController:

    def __init__(self, aws_region, endpoint_name):
        self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=aws_region)
        self.endpoint_name = endpoint_name

    def predict(self, im_pil, models, configs, verbose: bool = False):
        """
        Call sagemaker endpoint for inference, might running multiple models at once.

        image: PIL image
        models: list of str, names of the models to run
        params: dict, parameters for the model, such as:
            classes_to_detect: list of int, classes to detect from COCO dataset 
            threshold: float, detection threshold 
        """
        # encode im_pil into base64
        image_base64 = self._encode_image_to_base64(im_pil)
        start_time = time.time()
        model = "object_detection_and_ppe" if include_ppe_classification else "object_detection"
        response = self._make_aws_sagemaker_request(image_base64, model=model)
        elapsed_time = time.time() - start_time

        if verbose:
            print(f"Sagemaker inference time: {response['time_ms']:.2f} ms; Request time: {elapsed_time * 1000:.2f} ms")

        # right now, in the response of sagemaker we got labels in integers
        # response["detections"] = [
        #     {
        #         "label": COCO_CLASSES[int(det["label"])],
        #         "score": det["score"],
        #         "bbox": det["bbox"],
        #         **({"ppe": det["ppe"]} if "ppe" in det else {})
        #     }
        #     for det in response["detections"]
        # ]
        
        filtered_detections = [det for det in response["detections"] if det["label"] in classes_to_detect and det["score"] >= threshold]
        return filtered_detections, response["detections"]
    
    def detect_plates(self, im_pil, threshold: float = 0.5, ocr_theshold: float = 0.5, verbose: bool = False):
        image_base64 = self._encode_image_to_base64(im_pil)
        start_time = time.time()
        response = self._make_aws_sagemaker_request(image_base64, model="license_plate_recognition")
        elapsed_time = time.time() - start_time
        if verbose:
            print(f"Sagemaker inference time: {response['time_ms']:.2f} ms; Request time: {elapsed_time * 1000:.2f} ms")

        filtered_detections = [det for det in response["detections"] if det["score"] >= threshold and det["ocr_confidence"] >= ocr_theshold]
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
        im_pil.save(buffered, format="PNG")  # JPEG maybe be used to save data

        return base64.b64encode(buffered.getvalue()).decode("utf-8")