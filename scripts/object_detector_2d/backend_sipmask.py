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

# mmdetection
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

# append system path to the submodules(third_party.)
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../third_party")
)

class SipmaskBackend(ObjectDetector2DBackendBase):
    def initialize(self, params: dict) -> None:
        """initialize your neural network here. loading weights, allocating memory, any kind of warming up process should be here.

        Args:
            config_name (String): config file name
            weight_path (str): path to the weight file.
        """
        config_name = params['config_name']
        weight_path = params["weight"]
        self.top_k = params["top_k"]
        self.score_threshold = params["score_threshold"]

        config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../third_party/sipmask/configs/sipmask/")
        config = config + config_name + ".py"
        self.model = init_detector(config, weight_path, device='cuda:0')

    def forward(self, image: np.ndarray) -> InstSegmArray:
        """forward process with given numpy image.

        Args:
            image (np.ndarray): input color image.
        """
        det_raw = inference_detector(self.model, image)

        if isinstance(det_raw, tuple):
            bbox_result, segm_result = det_raw
        else:
            bbox_result, segm_result = det_raw, None

        bboxes = np.vstack(bbox_result)
        classes = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
            ]
        classes = np.concatenate(classes)

        if segm_result is not None:
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > self.score_threshold)[0]

            det_pp = []
            for i in inds:
              i = int(i)
              score = bboxes[i, -1]

              det_pp.append(
                  {
                      "id": (classes[i]).astype(np.int32),
                      "score": score,
                      "bbox": bboxes[i, 0:4].astype(np.int32),
                      "mask": segms[i],
                  }
              )

        result = InstSegmArray()
        for det in det_pp:
            x0, y0, x1, y1 = det["bbox"]
            bbox = RegionOfInterest(x0, y0, y1 - y0 + 1, x1 - x0 + 1, False)
            mask = numpy_to_mask(det["mask"])
            segm = InstSegm(int(det["id"]+1), int(det["id"]+1), float(det["score"]), bbox, mask)
            result.data.append(segm)
        return result


# single default object. don't change the name 'backend_model'
backend_model = SipmaskBackend()
