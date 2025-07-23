import os
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image


class PPEClassifier:
    def __init__(self, model_pt, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_pt, map_location=self.device)
        self.model.eval()

        self.DEFAULT_MEAN = (0.485, 0.456, 0.406)
        self.DEFAULT_STD = (0.229, 0.224, 0.225)

        # IMPORTANT: this is the transformation related with resize_exact policy!
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=self.DEFAULT_MEAN, std=self.DEFAULT_STD),
        ])
        # This will be replaced when you call get_val_loader
        self.classes =  ["bottom", "full", "na", "noppe", "upper"]

    def classify(self, image, min_area = 0):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        
        if min_area > 0:
            width, height = image.size
            if width * height < min_area:
                return "na", 100.0
        
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        return self.classes[predicted_class.item()], confidence.item()
    
    def classify_objects(self, image_pil, detections : list[dict]):
        """
        Crop each detection and classify it using the PPE classifier.
        """
        results = []
        for ith, det in enumerate(detections):
            bbox = det['bbox']
            cropped = image_pil.crop((bbox['top_left']['x'], bbox['top_left']['y'], bbox['bottom_right']['x'], bbox['bottom_right']['y']))
            class_name, confidence = self.classify(cropped)
            results.append({
                "ppe_level": class_name,
                "confidence": confidence
            })
        return results

class LetterboxTransform:
    def __init__(self, size, fill_color=(114, 114, 114)):
        self.size = (size, size) if isinstance(size, int) else size
        self.fill_color = fill_color

    def __call__(self, img):
        w, h = img.size
        tw, th = self.size
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.BILINEAR)
        new_img = Image.new("RGB", (tw, th), self.fill_color)
        new_img.paste(img, ((tw - nw) // 2, (th - nh) // 2))
        return new_img