import os
import json
import random
import urllib.request

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import requests

import sys
sys.path.append("../../lambda_event_ai/")
from event_clip import draw_boxes_on_frame
from event_ai_processor import DETECTION_CLASS_COLORS

def get_random_coco_images(n_images=5):
    json_path = "instances_val2017.json"
    download_url = "https://huggingface.co/datasets/merve/coco/resolve/main/annotations/instances_val2017.json"
    
    # Download JSON if not present
    if not os.path.exists(json_path):
        print(f"{json_path} not found. Downloading...")
        urllib.request.urlretrieve(download_url, json_path)
        print("Download complete.")

    with open(json_path, "r") as f:
        coco_data = json.load(f)

    base_url = "http://images.cocodataset.org/val2017/"
    image_urls = [base_url + img["file_name"] for img in coco_data["images"]]
    random.shuffle(image_urls)
    
    return image_urls[:n_images]

def get_plate_images():
    return ["https://tse1.mm.bing.net/th/id/OIP.P2JPMypZjrBWKSVL0SYhOAHaFj?r=0&w=355&h=355&c=7",
            "https://tse2.mm.bing.net/th/id/OIP.XYQfthJf3fWqW5yGRE9hYwHaFc?r=0&w=348&h=348&c=7"]


def get_local_images():
    return [
        "test_images/B8A44FB3A1F9__front_event_2025-06-18T15:38:14.946325+00:00_frame_17.png",
        "test_images/B8A44FB981BB__panoramic_trains_event_2025-06-26T16:32:17.550213+00:00_frame_19.png"
    ]


def draw_results_by_model(image_path_or_url, response: dict):
    if image_path_or_url.startswith("http"):
        image_pil = Image.open(requests.get(image_path_or_url, stream=True).raw).convert("RGB")
    else:
        image_pil = Image.open(image_path_or_url).convert("RGB")
    results = response.get("results", {})

    for model_name, model_output in results.items():
        frame = image_pil.copy()
        detections = model_output.get("detections", [])

        if model_name == "license_plate_recognition":
            # Direct OCR plate output
            frame = draw_boxes_on_frame(
                frame,
                detections,
                label_fn=lambda d: f"{d['ocr_text']} | {d['score']:.1f} | OCR: {d['ocr_confidence']:.1f}",
                color_fn=lambda d: DETECTION_CLASS_COLORS.get("plate", "blue"),
                font_size=18,
                label_position="bottom"
            )
        else:
            # Draw general objects
            frame = draw_boxes_on_frame(
                frame,
                detections,
                label_fn=lambda d: f"{d['label']}: {d.get('score', 0):.2f}",
                color_fn=lambda d: DETECTION_CLASS_COLORS.get(d['label'], "red"),
                font_size=18
            )

            # Draw nested license plates
            for det in detections:
                if 'license_plate' in det and isinstance(det['license_plate'], list):
                    frame = draw_boxes_on_frame(
                        frame,
                        det['license_plate'],
                        label_fn=lambda d: f"{d['ocr_text']} | {d['score']:.1f} | OCR: {d['ocr_confidence']:.1f}",
                        color_fn=lambda d: DETECTION_CLASS_COLORS.get("plate", "blue"),
                        font_size=18,
                        label_position="bottom"
                    )

            # Draw PPE overlays if present
            def label_fn(det):
                if det["label"] == "person" and "ppe" in det:
                    level = det["ppe"]["ppe_level"]
                    label_map = {
                        "full": "PPE: full",
                        "upper": "PPE: upper",
                        "bottom": "PPE: bottom",
                        "noppe": "no PPE",
                        "na": "PPE: n/a"
                    }
                    return label_map.get(level, "PPE: unknown") + f" ({det['ppe']['confidence']:.2f})"
                return None

            def color_fn(det):
                if det["label"] == "person" and "ppe" in det:
                    level = det["ppe"]["ppe_level"]
                    color_map = {
                        "full": (0, 255, 0),
                        "upper": (225, 165, 0),
                        "bottom": (225, 165, 0),
                        "noppe": (210, 0, 0),
                        "na": (128, 128, 128)
                    }
                    return color_map.get(level, (255, 0, 0))
                return (255, 0, 0)

            ppe_overlays = [d for d in detections if d.get("ppe")]
            if ppe_overlays:
                frame = draw_boxes_on_frame(
                    frame,
                    ppe_overlays,
                    label_fn=label_fn,
                    color_fn=color_fn,
                    font_size=18,
                    label_position="bottom"
                )

        # Show result
        plt.figure(figsize=(12, 8))
        plt.imshow(frame)
        plt.title(f"Model: {model_name}")
        plt.axis("off")
        plt.show()
