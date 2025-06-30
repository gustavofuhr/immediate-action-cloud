FROM public.ecr.aws/lambda/python:3.10

RUN pip install \
    "numpy<2.0" \
    # onnxruntime \
    pillow \
    boto3 \
    "imageio[pyav]"

RUN yum install -y dejavu-sans-fonts

# dfine_controller_ort.py
COPY lambda_function.py event_ai_processor.py sagemaker_controller.py event_clip.py ./
# COPY onnx_models/ ./onnx_models/ 
COPY amazon_kinesis_video_consumer_library/ ./amazon_kinesis_video_consumer_library/

CMD ["lambda_function.lambda_handler"]
