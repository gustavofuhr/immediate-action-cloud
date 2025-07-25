import os
from collections import namedtuple

from PIL import Image
import numpy as np

from model_controller import ModelController
from object_detection_controller import ObjectDetectionController
from lpr_controller import LPR_Controller
from ppe_classifier_controller import PPEClassifierController


MIN_LPR_AREA = 10000
MIN_PPE_AREA = 1000
LPR_CLASSES = ["car", "truck", "bus", "motorcycle"]
PPE_CLASSES = ["person"]
LPR_TARGET_FIELD = "license_plate"
PPE_TARGET_FIELD = "ppe"
    
class ChainedModel:
    def __init__(self, model: ModelController, min_area: int, filter_classes: list[str], target_field: str):
        self.model = model
        self.min_area = min_area
        self.filter_classes = filter_classes
        self.target_field = target_field

    def run_model_for_each_detection(self, image_pil: Image.Image, detections: list[dict], threshold: float) -> list[dict]:
        if not detections:
            return []


        new_detections = []
        n_calls = 0
        for i, d in enumerate(detections):
            if d['label'] in self.filter_classes and self._bbox_area(d["bbox"]) >= self.min_area:
                crop_box = d["bbox"]
                cropped_img = self._crop(image_pil, crop_box)
                output = self.model.run(cropped_img, threshold)
                n_calls += 1

                # If output is a list of dicts with bboxes, translate coordinates back
                if isinstance(output, list):
                    for item in output:
                        if isinstance(item, dict) and "bbox" in item:
                            self._translate_bbox(item["bbox"], crop_box)

                d[self.target_field] = output
            new_detections.append(d)

        return new_detections, n_calls
            
    def _bbox_area(self, bbox):  # reuse same logic
        x1, y1 = bbox["top_left"]["x"], bbox["top_left"]["y"]
        x2, y2 = bbox["bottom_right"]["x"], bbox["bottom_right"]["y"]
        return (x2 - x1) * (y2 - y1)
    
    def _crop(self, image_pil: Image.Image, bbox: dict) -> Image.Image:
        x1, y1 = bbox["top_left"]["x"], bbox["top_left"]["y"]
        x2, y2 = bbox["bottom_right"]["x"], bbox["bottom_right"]["y"]
        return image_pil.crop((x1, y1, x2, y2))
    
    def _translate_bbox(self, bbox: dict, origin_bbox: dict):
        offset_x = origin_bbox["top_left"]["x"]
        offset_y = origin_bbox["top_left"]["y"]

        bbox["top_left"]["x"] += offset_x
        bbox["top_left"]["y"] += offset_y
        bbox["bottom_right"]["x"] += offset_x
        bbox["bottom_right"]["y"] += offset_y
    
    
class ObjectDetectionMultiStagePipeline(ModelController):
    def __init__(self, 
                 object_detector: ModelController, 
                 chained_models: list[ModelController]):
        self.object_detector = object_detector
        self.chained_models = chained_models

    def run(self, image_pil: Image.Image, threshold: float = 0.5, classes_to_detect: list[str] = ["person"]) -> list[dict]:
        detections = self.object_detector.run(image_pil, threshold, classes_to_detect)
        for ith, ch_model in enumerate(self.chained_models):
            print("Running chained model", ith + 1, "of", len(self.chained_models))
            detections, n_calls = ch_model.run_model_for_each_detection(image_pil, detections, threshold)
            print(f"Total calls to chained model: {n_calls}")
        return detections
    
ModelInfo = namedtuple("ModelInfo", ["model", "description", "version"])

model_pipelines = None
def get_models_and_pipelines():
    global model_pipelines
    if model_pipelines is None:
        config_file = os.environ.get('CONFIG_FILE', '/workspace/D-FINE/configs/dfine/objects365/dfine_hgnetv2_x_obj2coco.yml')
        checkpoint_file = os.environ.get('CHECKPOINT_FILE', '/workspace/dfine_checkpoints/dfine_x_obj2coco.pth')
        device = os.environ.get('DEVICE', 'cpu')

        model_pipelines = {
            "object_detection": ModelInfo(
                model=ObjectDetectionController(
                    config_file=config_file, 
                    checkpoint_file=checkpoint_file, 
                    device=device
                ),
                description="Object Detection (COCO classes), single stage.",
                version="v1.0"
            ),
            "license_plate_recognition": ModelInfo(
                model=LPR_Controller(
                    device=device,
                    detector_model="yolo-v9-t-640-license-plate-end2end",
                    ocr_model="cct-s-v1-global-model"
                ),
                description="License Plate Recognition (ALPR), single stage.",
                version="v1.0"
            ),
            "ppe_classification": ModelInfo(
                model=PPEClassifierController(
                    model_path="/workspace/models/ppe/ppe_classifier_v1_resize_exact_224.pt",
                    device=device
                ),
                description="PPE Classification, single stage.",
                version="v1.0"
            ),
        }

        model_pipelines["object_detection_then_ppe"] = ModelInfo(
            model=ObjectDetectionMultiStagePipeline(
                object_detector=model_pipelines["object_detection"].model,
                chained_models=[
                    ChainedModel(
                        model=model_pipelines["ppe_classification"].model, 
                        min_area=MIN_PPE_AREA, 
                        filter_classes=PPE_CLASSES, 
                        target_field=PPE_TARGET_FIELD
                    )
                ]
            ),
            description="Object Detection with PPE Classification, multi-stage.",
            version="v1.0"
        )

        model_pipelines["object_detection_then_lpr"] = ModelInfo(
            model=ObjectDetectionMultiStagePipeline(
                object_detector=model_pipelines["object_detection"].model,
                chained_models=[
                    ChainedModel(
                        model=model_pipelines["license_plate_recognition"].model, 
                        min_area=MIN_LPR_AREA, 
                        filter_classes=LPR_CLASSES, 
                        target_field=LPR_TARGET_FIELD
                    )
                ]
            ),
            description="Object Detection with License Plate Recognition, multi-stage.",
            version="v1.0"
        )

        model_pipelines["object_detection_then_all"] = ModelInfo(
            model=ObjectDetectionMultiStagePipeline(
                object_detector=model_pipelines["object_detection"].model,
                chained_models=[
                    ChainedModel(
                        model=model_pipelines["ppe_classification"].model, 
                        min_area=MIN_PPE_AREA, 
                        filter_classes=PPE_CLASSES, 
                        target_field=PPE_TARGET_FIELD
                    ),
                    ChainedModel(
                        model=model_pipelines["license_plate_recognition"].model, 
                        min_area=MIN_LPR_AREA, 
                        filter_classes=LPR_CLASSES, 
                        target_field=LPR_TARGET_FIELD
                    )
                ]
            ),
            description="Object Detection with All Models, multi-stage.",
            version="v1.0"
        )

    return model_pipelines