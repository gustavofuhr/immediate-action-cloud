
import sys
import os
import json

import torch
import torch.nn as nn
import torchvision.transforms as T



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'D-FINE/')))
from src.core import YAMLConfig

class DFINE_Controller:

    def __init__(self, config_file, checkpoint_file, device):
        self.device = device
        cfg = YAMLConfig(config_file, resume=checkpoint_file)

        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # Load train mode state and convert to deploy mode
        cfg.model.load_state_dict(state)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()

            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs

        self.model = Model().to(self.device)
        self.transformation = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])


    def filter_and_format_detections(self, outputs, threshold, classes):
        labels, boxes, scores = outputs

        filtered_labels = labels[scores > threshold]
        filtered_boxes = boxes[scores > threshold]
        filtered_scores = scores[scores > threshold]

        detections = []
        for i in range(len(filtered_labels)):
            if classes is None or filtered_labels[i] in classes:
                detections.append({
                    'label': filtered_labels[i].item(),
                    'bbox': filtered_boxes[i].tolist(),
                    'bbox': {
                        'top_left': {'x': filtered_boxes[i][0].item(), 'y': filtered_boxes[i][1].item()},
                        'bottom_right': {'x': filtered_boxes[i][2].item(), 'y': filtered_boxes[i][3].item()}
                    },
                    'score': filtered_scores[i].item()
                })
                print(f"Detected {filtered_labels[i]} with confidence {filtered_scores[i].item()}")
        print(detections)
        return detections


    def run(self, image, threshold: float = 0.5, classes_to_detect: list[int] = [0]):
        """
        Run detection using the D-FINE model for the selected classes.

        image: PIL image
        threshold: float, confidence threshold
        classes_to_detect: list of int, classes to detect from COCO dataset
        """
        w, h = image.size
        orig_size = torch.tensor([[w, h]]).to(self.device)
        # TODO: check image type
        im_data = self.transformation(image).unsqueeze(0).to(self.device)

        outputs = self.model(im_data, orig_size)
        return self.filter_and_format_detections(outputs, threshold, classes_to_detect)