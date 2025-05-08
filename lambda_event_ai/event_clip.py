import os
from typing import Optional

import boto3
import imageio.v3 as iio
from PIL import Image
import numpy as np


class EventClip:
    """
    A class to handle event movie clips for the Event AI application.
    """

    def __init__(self, aws_region : str, bucket_name : str, resize_clip_height : Optional[int] = None):
        self.frames = []
        self.fps = 2
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3', region_name=aws_region)
        self.resize_clip_height = resize_clip_height

    def add_frame(self, im_pil : Image.Image):
        if self.resize_clip_height is not None:
            width, height = im_pil.size
            new_width = int(width * (self.resize_clip_height / height))
            im_pil = im_pil.resize((new_width, self.resize_clip_height), Image.LANCZOS)

        print("Adding frame to clip. Type: ", type(im_pil), "Size: ", im_pil.size)
        self.frames.append(np.array(im_pil))

    def send_clip_to_s3(self, file_path : str):
        if len(self.frames) == 0:
            print("WARNING! No frames to save in S3, skipping.")
            return
        
        print("Saving event clip locally. Total frames: ", len(self.frames))

        clip_local_path = os.path.join("/tmp/", os.path.basename(file_path))
        iio.imwrite(
            clip_local_path,
            self.frames,
            fps=self.fps, 
            codec="libx264",
        )

        print("Sending clip to S3: ", file_path)
        with open(clip_local_path, "rb") as f:
            self.s3_client.upload_fileobj(
                Fileobj=f, 
                Bucket=self.bucket_name, 
                Key=file_path,
                ExtraArgs={
                    "ContentType": "video/mp4"
                }
            )
