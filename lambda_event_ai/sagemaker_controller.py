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

    def predict(self, im_pil, models_and_configs, verbose: bool = False):
        """
        Call sagemaker endpoint for inference, might running multiple models at once.

        image: PIL image
        models_and_configs: list of dict, each dict contains "model" and some parameters such as threshold, etc.
        """
        # encode im_pil into base64
        image_base64 = self._encode_image_to_base64(im_pil)
        start_time = time.time()
        
        # TODO: this would need to be changed once we modify the sagemaker way of receiving parameters
        model_list = [m["name"] for m in models_and_configs]
        threshold = models_and_configs[0].get("threshold", 0.5)
        classes_to_detect = models_and_configs[0].get("classes_to_detect", [])

        response = self._make_aws_sagemaker_request(image_base64, models=model_list, classes_to_detect=classes_to_detect, threshold=threshold)
        elapsed_time = time.time() - start_time

        if verbose:
            print(f"Sagemaker inference time: {response['time_ms']:.2f} ms; Request time: {elapsed_time * 1000:.2f} ms")

        return response
    
    def _make_aws_sagemaker_request(self, image_base64 : str, models : list[str], classes_to_detect: list[str], threshold: float = 0.5):
        print(f"Making AWS SageMaker request for models: {models}, classes_to_detect: {classes_to_detect}, threshold: {threshold}")
        response = self.sagemaker_runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=json.dumps({
                "image_base64": image_base64,
                "models": models,
                "classes_to_detect": classes_to_detect,
                "threshold": threshold
            }),
        )
        return json.loads(response["Body"].read().decode())

    def _encode_image_to_base64(self, im_pil: Image.Image) -> str:
        buffered = BytesIO()
        im_pil.save(buffered, format="PNG")  # JPEG maybe be used to save data

        return base64.b64encode(buffered.getvalue()).decode("utf-8")