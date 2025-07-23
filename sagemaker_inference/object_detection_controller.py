
from dfine_controller import DFINE_Controller
from ppe_classifier import PPEClassifier


class ObjectDetectionController:
    def __init__(self, config_file, checkpoint_file, device, include_ppe_classification: bool = False):
        print(f'Initializing detector with config file: {config_file}, checkpoint file: {checkpoint_file}, device: {device}')
        self.detector = DFINE_Controller(config_file, checkpoint_file, device)

        self.ppe_classifier = PPEClassifier("/workspace/models/ppe/ppe_classifier_v1_resize_exact_224.pt", device) if include_ppe_classification else None

    def run(self, image_pil, threshold: float = 0.5, classes_to_detect: list[int] = [0], include_ppe_classification: bool = False):
        detections = self.detector.run(image_pil, threshold, classes_to_detect)

        if include_ppe_classification:
            people_detections = [(i, d) for i, d in enumerate(detections) if d['label'] == 0]
            ppe_classifications = self.ppe_classifier.classify_objects(image_pil, [d for _, d in people_detections])
            for i, ppe in zip([i for i, _ in people_detections], ppe_classifications):
                detections[i].update({"ppe": ppe})

        return detections
    

