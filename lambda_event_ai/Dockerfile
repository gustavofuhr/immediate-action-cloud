FROM public.ecr.aws/lambda/python:3.10

RUN yum install -y \
    gcc \
    zlib-devel \
    libjpeg-turbo-devel \
    libpng-devel \
    freetype-devel \
    dejavu-sans-fonts \
    dejavu-sans-mono-fonts
    
RUN pip install \
    "av==14.4.0" \
    "numpy==1.26.4" \
    "pillow==11.3.0" \
    "boto3==1.39.2" \
    "imageio[pyav]"


# dfine_controller_ort.py
COPY lambda_function.py event_ai_processor.py sagemaker_controller.py event_clip.py ./
# COPY onnx_models/ ./onnx_models/ 
COPY amazon_kinesis_video_consumer_library/ ./amazon_kinesis_video_consumer_library/

CMD ["lambda_function.lambda_handler"]
