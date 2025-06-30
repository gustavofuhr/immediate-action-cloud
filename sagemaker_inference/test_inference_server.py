import os
import json
import random
import urllib.request

def get_random_coco_images(n_images=5):
    json_path = "instances_val2017.json"
    download_url = "https://huggingface.co/datasets/merve/coco/resolve/main/annotations/instances_val2017.json"
    
    # Download JSON if not present
    if not os.path.exists(json_path):
        print(f"{json_path} not found. Downloading...")
        urllib.request.urlretrieve(download_url, json_path)
        print("Download complete.")

    # Load COCO data
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    # Build image URLs
    base_url = "http://images.cocodataset.org/val2017/"
    image_urls = [base_url + img["file_name"] for img in coco_data["images"]]
    random.shuffle(image_urls)
    
    return image_urls[:n_images]

def get_plate_images():
    return ["https://tse1.mm.bing.net/th/id/OIP.P2JPMypZjrBWKSVL0SYhOAHaFj?r=0&w=355&h=355&c=7",
            "https://tse2.mm.bing.net/th/id/OIP.XYQfthJf3fWqW5yGRE9hYwHaFc?r=0&w=348&h=348&c=7"]