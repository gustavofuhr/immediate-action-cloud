FROM registry.cn-hangzhou.aliyuncs.com/peterande/dfine:v1

WORKDIR /workspace/

RUN git clone https://github.com/Peterande/D-FINE.git && \
    cd D-FINE && \
    pip install -r requirements.txt && \
    pip install -r tools/inference/requirements.txt && \
    pip install opencv-python tqdm 
    
RUN pip install fast-alpr==0.2.0

RUN pip install Flask opencv-python numpy requests Pillow flask-cors sagemaker-inference gunicorn 

COPY dfine_checkpoints /workspace/dfine_checkpoints
COPY *.py /workspace/
COPY serve /usr/bin/serve
RUN chmod +x /usr/bin/serve

ENV PYTHONUNBUFFERED=1
ENV SAGEMAKER_PROGRAM=/workspace/app.py

EXPOSE 8080

ENTRYPOINT ["serve"]