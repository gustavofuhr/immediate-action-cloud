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

    def get_default_parameters(self) -> dict:
        """
        Get the default parameters for the model run.

        Returns:
            dict: Default parameters, such as threshold and classes to detect.
        """
        return {
            "threshold": 0.5,
            "classes_to_detect": None
        }
    
    def filter_results(self, results: list[dict], params: dict) -> list[dict]:
        """
        Filter the results based on the provided parameters.

        Args:
            results (list[dict]): List of detection results.
            params (dict): Parameters to filter the results, such as threshold and classes to detect.

        Returns:
            list[dict]: Filtered results.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def run(self, image_pil: Image.Image, parameters : dict = None) -> list[dict]:
        """
        Run the model pipeline on the input image.

        Args:
            image (Image.Image): Input image in PIL format.
            parameters (dict): Additional parameters for the model run, such as threshold and classes to detect.

        Returns:
            list[dict]: List of detection results, each containing bounding box, label, score, 
                                    and other relevant information.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    

