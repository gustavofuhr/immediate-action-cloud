import os
from typing import Optional

import boto3
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from PIL import ImageDraw, ImageFont

DETECTION_CLASS_COLORS = {
    'person': (204, 0, 0),  
    'car': (0, 153, 0),  
    'bicycle': (0, 153, 0),
    'motorcycle': (0, 153, 0),  
    'bus': (0, 153, 0),  
    'train': (0, 153, 0),  
    'truck': (0, 153, 0),  
    'bird': (0, 51, 204),  
    'dog': (0, 51, 204),
    'sheep': (0, 51, 204),
    'cow': (0, 51, 204),
    'cat': (0, 51, 204),
    'horse': (0, 51, 204),
    'plate': (230, 138, 0)
}


def draw_boxes_on_frame(
    frame_pil,
    detections,
    label_fn,
    color_fn,
    font_size=14,
    text_color=(255, 255, 255),
    padding=2,
    label_position="top",  # can be "top" or "bottom"
    invisible_box=False
):
    draw = ImageDraw.Draw(frame_pil)
    try:
        font_path = "/usr/share/fonts/dejavu/DejaVuSansMono.ttf"
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        print("Warning: Custom font not found, using default font.")
        font = ImageFont.load_default()

    for det in detections:
        x1 = int(det['bbox']['top_left']['x'])
        y1 = int(det['bbox']['top_left']['y'])
        x2 = int(det['bbox']['bottom_right']['x'])
        y2 = int(det['bbox']['bottom_right']['y'])

        label = label_fn(det)
        color = color_fn(det)

        if not invisible_box:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4) # actual object bbox

        # Compute text size
        try:
            text_bbox = font.getbbox(label)  # (x_min, y_min, x_max, y_max)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            text_width, text_height = font.getsize(label)

        # Default: label on top
        if label_position == "top":
            rect_x1 = x1
            rect_y1 = y1 - text_height - 2 * padding
            rect_x2 = x1 + text_width + 2 * padding
            rect_y2 = y1

            # If top is outside image, move label below box
            if rect_y1 < 0:
                rect_y1 = y1
                rect_y2 = y1 + text_height + 2 * padding

        elif label_position == "bottom":
            rect_x1 = x1
            rect_y1 = y2
            rect_x2 = x1 + text_width + 2 * padding
            rect_y2 = y2 + text_height + 2 * padding

            # If bottom is outside image, switch to top logic
            if rect_y2 > frame_pil.height:
                rect_x1 = x1
                rect_y1 = y1 - text_height - 2 * padding
                rect_x2 = x1 + text_width + 2 * padding
                rect_y2 = y1

                # If top is outside image too, move label below box
                if rect_y1 < 0:
                    rect_y1 = y1
                    rect_y2 = y1 + text_height + 2 * padding
        else:
            raise ValueError(f"Unknown label_position: {label_position}")

        draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], fill=color)
        draw.text((rect_x1 + padding, rect_y1 + padding), label, fill=text_color, font=font)

    return frame_pil





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

        self.frames.append(np.array(im_pil))

    def clear_frames(self):
        self.frames = []

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
