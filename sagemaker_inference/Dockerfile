FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

WORKDIR /workspace/
RUN apt update && apt install -y git python3-pip libgl1 libglib2.0-0


RUN git clone https://github.com/Peterande/D-FINE.git && \
    cd D-FINE && \
    pip install -r requirements.txt && \
    pip install opencv-python tqdm 
    
RUN pip install fast-alpr[onnx-gpu]==0.2.0
RUN pip install timm==0.9.5
RUN pip install Flask opencv-python numpy requests Pillow flask-cors sagemaker-inference gunicorn 

COPY dfine_checkpoints /workspace/dfine_checkpoints
COPY models/ppe /workspace/models/ppe
COPY *.py /workspace/
COPY serve /usr/bin/serve
RUN chmod +x /usr/bin/serve

ENV PYTHONUNBUFFERED=1
ENV SAGEMAKER_PROGRAM=/workspace/app.py

EXPOSE 8080

ENTRYPOINT ["serve"]