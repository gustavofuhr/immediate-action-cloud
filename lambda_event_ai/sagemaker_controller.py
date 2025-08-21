import json
import base64
import time
from io import BytesIO

import boto3
from PIL import Image

from lambda_logging import base_logger


class SageMakerController:

    def __init__(self, aws_region, endpoint_name, logger=None):
        self.logger = logger or base_logger
        self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=aws_region)
        self.endpoint_name = endpoint_name

    def predict(self, im_pil, models : list[str], per_model_params : dict = None, verbose: bool = False):
        """
        Call sagemaker endpoint for inference, might running multiple models at once.

        image: PIL image
        models_and_configs: list of dict, each dict contains "model" and some parameters such as threshold, etc.
        """
        # encode im_pil into base64
        image_base64 = self._encode_image_to_base64(im_pil)
        start_time = time.time()

        response = self._make_aws_sagemaker_request(image_base64, models=models, per_model_params=per_model_params)
        elapsed_time = time.time() - start_time

        if verbose:
            self.logger.info(f"Sagemaker total inference time: {response['total_time_ms']:.2f} ms; Request time: {elapsed_time * 1000:.2f} ms")

        return response
    
    def _make_aws_sagemaker_request(self, image_base64 : str, models : list[str], per_model_params : dict = None):
        # print(f"Making AWS SageMaker request for models: {models}, per_model_params: {per_model_params}")
        response = self.sagemaker_runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=json.dumps({
                "image_base64": image_base64,
                "models": models,
                "per_model_params": per_model_params if per_model_params else {}
            }),
        )
        return json.loads(response["Body"].read().decode())

    def _encode_image_to_base64(self, im_pil: Image.Image) -> str:
        buffered = BytesIO()
        im_pil.save(buffered, format="PNG")  # JPEG maybe be used to save data

        return base64.b64encode(buffered.getvalue()).decode("utf-8")