import numpy as np
import onnxruntime as ort
from PIL import Image
import time

class DFINEControllerORT:

    def __init__(self, onnx_model_filepath):
        self.sess = ort.InferenceSession(onnx_model_filepath)
        print(f"Using device: {ort.get_device()}")

    def run(self, im_pil, threshold: float = 0.5, classes_to_detect: list[int] = [0], verbose: bool = False):
        """
        Run detection using the D-FINE model for the selected classes.

        image: PIL image
        threshold: float, confidence threshold
        classes_to_detect: list of int, classes to detect from COCO dataset (default: [0] "person")
        """
        # preprocess the image:
        resized_im_pil, ratio, pad_w, pad_h = self._resize_with_aspect_ratio(im_pil, 640)

        orig_size = np.array([[resized_im_pil.size[1], resized_im_pil.size[0]]], dtype=np.int64)

        im_data = np.array(resized_im_pil).astype(np.float32) / 255.0  # Normalize
        if im_data.ndim == 2:  # grayscale
            im_data = np.expand_dims(im_data, axis=-1)
        im_data = np.transpose(im_data, (2, 0, 1))  # HWC -> CHW
        im_data = np.expand_dims(im_data, axis=0)  # add batch dimension

        start_time = time.time()
        outputs = self.sess.run(
            output_names=None,
            input_feed={"images": im_data, "orig_target_sizes": orig_size},
        )
        elapsed_time = time.time() - start_time
        if verbose: 
            print(f"Model inference time: {elapsed_time * 1000:.2f} ms")

        return self.filter_and_format_detections(outputs, threshold, classes_to_detect, ratio, pad_w, pad_h, verbose)
    
    def filter_and_format_detections(self, outputs, threshold, classes, ratio, pad_w, pad_h, verbose=False):
        labels, boxes, scores = outputs

        # Filter by score threshold
        keep = scores > threshold
        filtered_labels = labels[keep]
        filtered_boxes = boxes[keep]
        filtered_scores = scores[keep]

        detections = []
        for i in range(len(filtered_labels)):
            if filtered_labels[i] in classes:
                # Reverse the resizing and padding
                box = filtered_boxes[i]
                x1 = (box[0].item() - pad_w) / ratio
                y1 = (box[1].item() - pad_h) / ratio
                x2 = (box[2].item() - pad_w) / ratio
                y2 = (box[3].item() - pad_h) / ratio

                detections.append({
                    'label': filtered_labels[i].item(),
                    'bbox': {
                        'top_left': {'x': x1, 'y': y1},
                        'bottom_right': {'x': x2, 'y': y2}
                    },
                    'score': filtered_scores[i].item()
                })

                if verbose: print(f"Detected {filtered_labels[i]} with confidence {filtered_scores[i].item()}")

        if verbose and len(detections) > 0: print(detections)
        return detections



    def _resize_with_aspect_ratio(self, image, size, interpolation=Image.BILINEAR):
        """Resizes an image while maintaining aspect ratio and pads it."""
        original_width, original_height = image.size
        ratio = min(size / original_width, size / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        image = image.resize((new_width, new_height), interpolation)

        # Create a new image with the desired size and paste the resized image onto it
        new_image = Image.new("RGB", (size, size))
        new_image.paste(image, ((size - new_width) // 2, (size - new_height) // 2))
        return new_image, ratio, (size - new_width) // 2, (size - new_height) // 2

