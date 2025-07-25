import torch
from PIL import Image

class ModelController:
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the model with the given path and device.

        Args:
            model_path (str): Path to the model file.
            device (str): Device to run the model on (e.g., 'cpu', 'cuda').
        """
        self.model_path = model_path
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self, image_pil: Image.Image) -> list[dict]:
        """
        Run the model pipeline on the input image.

        Args:
            image (Image.Image): Input image in PIL format.

        Returns:
            list[dict]: List of detection results, each containing bounding box, label, score, 
                                    and other relevant information.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
