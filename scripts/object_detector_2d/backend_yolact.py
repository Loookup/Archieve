#! /usr/bin/python3

# python built-in
from pathlib import Path
import os, sys

# pip installed
import numpy as np
import torch

from .backend_base import ObjectDetector2DBackendBase
from unld_msgs.msg import InstSegm, InstSegmArray
from unld_msgs.mask_codec import numpy_to_mask
from sensor_msgs.msg import RegionOfInterest

# append system path to the submodules(third_party.)
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../third_party")
)

# written in project
from yolact.yolact import cfg, set_cfg, Yolact
from yolact.utils.functions import SavePath
from yolact.utils.augmentations import FastBaseTransform
from yolact.layers.unloader_postproc import unld_postproc


class YolactBackend(ObjectDetector2DBackendBase):
    def initialize(self, params: dict) -> None:
        """initialize your neural network here. loading weights, allocating memory, any kind of warming up process should be here.

        Args:
            weight_path (str): path to the weight file.
        """

        weight_path = params["weight"]
        self.top_k = params["top_k"]
        self.score_threshold = params["score_threshold"]
        weight_save_path = SavePath.from_str(str(weight_path))
        config_name = weight_save_path.model_name + "_config"
        print("Configuration:{}".format(config_name))
        self.transform = FastBaseTransform()
        set_cfg(config_name)

        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = Yolact()
        self.net.load_weights(weight_path)
        self.net.eval()

    def forward(self, image: np.ndarray) -> InstSegmArray:
        """forward process with given numpy image.

        Args:
            image (np.ndarray): input color image.
        """

        with torch.no_grad():
            frame = torch.from_numpy(image).cuda().float()
            batch = self.transform(frame.unsqueeze(0))
            det_raw = self.net.forward(batch)
        h, w = image.shape[:2]
        det_pp = unld_postproc(det_raw, h, w, self.top_k, self.score_threshold)
        result = InstSegmArray()
        for det in det_pp:
            x0, y0, x1, y1 = det["bbox"]
            bbox = RegionOfInterest(x0, y0, y1 - y0 + 1, x1 - x0 + 1, False)
            mask = numpy_to_mask(det["mask"])
            segm = InstSegm(int(det["id"]), int(det["id"]), float(det["score"]), bbox, mask)
            result.data.append(segm)
        return result


# single default object. don't change the name 'backend_model'
backend_model = YolactBackend()
