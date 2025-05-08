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

    def run(self, im_pil, classes_to_detect: list[int] = [0], threshold : float = 0.5, verbose: bool = False):
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

        filtered_detections = [det for det in response["detections"] if det["label"] in classes_to_detect and det["score"] >= threshold]
        return filtered_detections, response["detections"]


    def _make_aws_sagemaker_request(self, image_base64 : str):
        response = self.sagemaker_runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=json.dumps({
                "image_base64": image_base64
            }),
        )
        return json.loads(response["Body"].read().decode())

    def _encode_image_to_base64(self, im_pil: Image.Image) -> str:
        buffered = BytesIO()
        im_pil.save(buffered, format="JPEG")  # or "PNG" depending on your needs

        return base64.b64encode(buffered.getvalue()).decode("utf-8")