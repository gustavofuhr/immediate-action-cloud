import os
from typing import Optional

import boto3
import ffmpeg
from PIL import Image
import numpy as np


class EventClip:
    """
    A class to handle event movie clips for the Event AI application,
    now using ffmpeg-python for better control over video quality.
    """

    def __init__(self, aws_region: str, bucket_name: str, resize_clip_height: Optional[int] = None):
        self.frames = []
        self.fps = 2
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3', region_name=aws_region)
        self.resize_clip_height = resize_clip_height

    def add_frame(self, im_pil: Image.Image):
        if self.resize_clip_height is not None:
            width, height = im_pil.size
            new_width = int(width * (self.resize_clip_height / height))
            im_pil = im_pil.resize((new_width, self.resize_clip_height), Image.LANCZOS)

        print("Adding frame to clip. Type:", type(im_pil), "Size:", im_pil.size)
        self.frames.append(np.array(im_pil))

    def send_clip_to_s3(self, file_path: str):
        if not self.frames:
            print("WARNING! No frames to save in S3, skipping.")
            return

        print("Saving event clip locally. Total frames:", len(self.frames))

        clip_local_path = os.path.join("/tmp/", os.path.basename(file_path))
        h, w = self.frames[0].shape[:2]
        video_data = np.stack(self.frames).astype(np.uint8).tobytes()

        try:
            (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(w, h), framerate=self.fps)
                .output(clip_local_path, vcodec='libx264', crf=18, preset='slow', pix_fmt='yuv420p')
                .overwrite_output()
                .run(input=video_data)
            )
        except ffmpeg.Error as e:
            print("FFmpeg error:", e.stderr.decode() if e.stderr else str(e))
            return

        print("Sending clip to S3:", file_path)
        with open(clip_local_path, "rb") as f:
            self.s3_client.upload_fileobj(
                Fileobj=f,
                Bucket=self.bucket_name,
                Key=file_path,
                ExtraArgs={
                    "ContentType": "video/mp4"
                }
            )
