import os
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image

from model_controller import ModelController

class PPEClassifierController(ModelController):

    def __init__(self, model_path, device = None):
        super().__init__(model_path, device)
        self.model = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.eval()

        self.DEFAULT_MEAN = (0.485, 0.456, 0.406)
        self.DEFAULT_STD = (0.229, 0.224, 0.225)

        # IMPORTANT: this is the transformation related with resize_exact policy!
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=self.DEFAULT_MEAN, std=self.DEFAULT_STD),
        ])
        self.classes =  ["bottom", "full", "na", "noppe", "upper"]

    def get_default_parameters(self) -> dict:
        return {
            "threshold": 0.7
        }
    
    def filter_results(self, result: dict, params: dict) -> dict:
        if params is None:
            params = self.get_default_parameters()
        return result if result['confidence'] >= params["threshold"] else None
    
    def run(self, image_pil: Image.Image, param : dict = None) -> list[dict]:
        image = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        res = {
            "ppe_level": self.classes[predicted_class.item()], 
            "confidence": confidence.item()
        }
        return self.filter_results(res, param)


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