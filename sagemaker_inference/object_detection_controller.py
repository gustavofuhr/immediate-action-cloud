from PIL import Image

from model_controller import ModelController
from dfine_controller import DFINE_Controller


COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
    9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
    24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
    31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
    37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
    52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}
COCO_CLASS_TO_IDX = {v: k for k, v in COCO_CLASSES.items()}

class ObjectDetectionController(ModelController):
    def __init__(self, config_file, checkpoint_file, device = None):
        super().__init__(checkpoint_file, device)
        self.detector = DFINE_Controller(config_file, checkpoint_file, device)

    def run(self, image_pil: Image.Image, params: dict = None) -> list[dict]:
        # DFINE_Controller works with COCO classes *integers*
        if params is None:
            params = self.get_default_parameters()
        coco_classes_indices = [COCO_CLASS_TO_IDX[c] for c in params["classes_to_detect"] if c in COCO_CLASS_TO_IDX]

        detections = self.detector.run(image_pil, params["threshold"], coco_classes_indices)
        for d in detections:
            d['label'] = COCO_CLASSES[d['label']]

        filtered_detections = self.filter_results(detections, params)
        return self._filter_duplicates(filtered_detections)
    
    def get_default_parameters(self) -> dict:
        return {
            "threshold": 0.5,
            "classes_to_detect": ["person"]
        }
    
    def filter_results(self, results: list[dict], params: dict) -> list[dict]:
        return [r for r in results if r['score'] >= params["threshold"]] # classes are already filtered in run()

    def _filter_duplicates(self, detections: list[dict], iou_threshold: float = 0.99, filter_classes: set[str] = {"car", "truck", "bus"}) -> list[dict]:
        detections = sorted(detections, key=lambda d: d["score"], reverse=True)

        kept = []
        for i, det in enumerate(detections):
            if det["label"] not in filter_classes:
                kept.append(det)
                continue

            overlap = False
            for k in kept:
                if k["label"] in filter_classes and self._iou(det["bbox"], k["bbox"]) > iou_threshold:
                    overlap = True
                    break

            if not overlap:
                kept.append(det)

        return kept

    def _iou(self, boxA: dict, boxB: dict) -> float:
        xA = max(boxA["top_left"]["x"], boxB["top_left"]["x"])
        yA = max(boxA["top_left"]["y"], boxB["top_left"]["y"])
        xB = min(boxA["bottom_right"]["x"], boxB["bottom_right"]["x"])
        yB = min(boxA["bottom_right"]["y"], boxB["bottom_right"]["y"])

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH

        boxAArea = (boxA["bottom_right"]["x"] - boxA["top_left"]["x"]) * (boxA["bottom_right"]["y"] - boxA["top_left"]["y"])
        boxBArea = (boxB["bottom_right"]["x"] - boxB["top_left"]["x"]) * (boxB["bottom_right"]["y"] - boxB["top_left"]["y"])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou


    
