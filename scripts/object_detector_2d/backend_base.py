import numpy as np
from unld_msgs.msg import InstSegmArray


class ObjectDetector2DBackendBase:
    def __init__(self):
        """You should not initialize your network here. __init__ is supposed to be empty for lazy initialization."""
        pass

    def initialize(self, params: dict) -> None:
        """initialize your neural network here. loading weights, allocating memory, any kind of warming up process should be here.

        Args:
            params (dict): backend initialize parameters
        """
        pass

    def forward(self, image: np.ndarray) -> InstSegmArray:
        """forward process with given numpy image.

        Args:
            image (np.ndarray): input color image.
        """
        return InstSegmArray()
