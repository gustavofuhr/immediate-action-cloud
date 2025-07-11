import os
import json
import base64
import boto3
import time
from botocore.config import Config
from locust import User, task, between

ENDPOINT_NAME = "sagemaker-inference-server-loadtest-endpoint"
REGION = os.getenv("AWS_REGION", "eu-west-1")

# Configure boto3 client for high-concurrency load testing (1000 users)
config = Config(
    max_pool_connections=200,  # High connection pool for 1000 users
    retries={
        'max_attempts': 3,
        'mode': 'adaptive'  # Adaptive retry mode for better handling
    },
    # Connection timeouts
    connect_timeout=10,
    read_timeout=60,
    # Regional endpoint for better performance
    region_name=REGION,
    # Signature version (optional, but explicit)
    signature_version='v4'
)
client = boto3.client("sagemaker-runtime", region_name=REGION, config=config)

with open("sample_image.jpeg", "rb") as img_file:
    img_b64 = base64.b64encode(img_file.read()).decode("utf-8")


def make_aws_sagemaker_request(image_base64, model_type):
    response = client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps({
            "image_base64": image_base64,
            "model": model_type
        }),
    )
    return json.loads(response["Body"].read().decode())

class SageMakerUser(User):
    wait_time = between(20, 30)  # Wait 20-30 seconds between requests
    
    def on_start(self):
        """Called when a user starts - initialize request counter"""
        self.request_count = 0
    
    def on_stop(self):
        """Called when a user stops"""
        print(f"User completed {self.request_count} requests")

    @task
    def invoke_endpoint(self):
        # Stop after 40 requests
        if self.request_count >= 40:
            self.stop()
            return
            
        start_time = time.time()

        try:
            # Make the actual requests
            result_1 = make_aws_sagemaker_request(img_b64, "object_detection")
            result_2 = make_aws_sagemaker_request(img_b64, "license_plate_recognition")
            
            total_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds
            
            # Calculate response length (sum of both results)
            response_length = len(json.dumps(result_1)) + len(json.dumps(result_2))
            
            # Fire success event for overall request
            self.environment.events.request.fire(
                request_type="sagemaker",
                name="invoke_endpoint_total",
                response_time=total_time,
                response_length=response_length,
                exception=None
            )
            
            # Track individual model execution times if available
            if result_1 and 'time_ms' in result_1:
                self.environment.events.request.fire(
                    request_type="model_execution",
                    name="object_detection",
                    response_time=result_1['time_ms'],
                    response_length=len(json.dumps(result_1)),
                    exception=None
                )
            
            if result_2 and 'time_ms' in result_2:
                self.environment.events.request.fire(
                    request_type="model_execution",
                    name="license_plate_recognition",
                    response_time=result_2['time_ms'],
                    response_length=len(json.dumps(result_2)),
                    exception=None
                )
            
            # Increment request counter
            self.request_count += 1
            
        except Exception as e:
            total_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds
            print(f"Error invoking endpoint: {e}")
            
            # Fire failure event
            self.environment.events.request.fire(
                request_type="sagemaker",
                name="invoke_endpoint_total",
                response_time=total_time,
                response_length=0,
                exception=e
            )
            
            # Increment request counter even on failure
            self.request_count += 1