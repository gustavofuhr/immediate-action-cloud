import numpy as np
from PIL import Image
from fast_alpr import ALPR

from model_controller import ModelController

class LPR_Controller(ModelController):

    def __init__(self, detector_model: str, ocr_model: str, device: str = None):
        super().__init__("", device)
        if self.device.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        

        self.model = ALPR(
            detector_model=detector_model,
            ocr_model=ocr_model,
            detector_providers=providers,
            ocr_providers=providers
        )

    def run(self, image_pil: Image.Image, threshold: float = 0.5, classes_to_detect: list[str] = None) -> list[dict]:
        image = np.array(image_pil)
        results = self.model.predict(image)
        return self._lpr_results_to_dicts(results)

    def _lpr_results_to_dicts(self, model_results: list[dict]) -> list[dict]:
        detections = []
        for r in model_results:
            bbox = {
                "top_left": {
                    "x": r.detection.bounding_box.x1,
                    "y": r.detection.bounding_box.y1
                },
                "bottom_right": {
                    "x": r.detection.bounding_box.x2,
                    "y": r.detection.bounding_box.y2
                }
            }
            detections.append({
                "bbox": bbox,
                "label": "license_plate",  # could also be 0 or a string depending on your handling
                "score": r.detection.confidence,
                "ocr_text": r.ocr.text,
                "ocr_confidence": r.ocr.confidence
            })
        return detections
