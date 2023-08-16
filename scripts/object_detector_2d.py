#! /usr/bin/env python3


# ===============================================
#  Just a copy of old code. still working on this.
# ===============================================

# Python
from typing import List
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, RegionOfInterest
from unld_msgs.msg import InstSegmArray, InstSegm
from unld_msgs.mask_codec import numpy_to_mask

from importlib import import_module


class CropImage:
    """
    Crop image with the given roi
    """

    def __init__(self) -> None:
        """
        initialization
        """
        self.roi = RegionOfInterest()
        self.recent_image_size = None

    def set_roi(self, x_offset: int, y_offset: int, width: int, height: int) -> None:
        """Set the crop RoI

        Args:
            xywh (List): roi coordinates in start x, start y, width, height manner

        """
        self.roi.x_offset = x_offset
        self.roi.y_offset = y_offset
        self.roi.width = width
        self.roi.height = height

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Crop the given image with the RoI

        Args:
            image (np.ndarray): source image to crop

        Returns:
            np.ndarray: cropped image
        """

        # get height and width
        self.recent_image_size = image.shape
        h, w = image.shape[0], image.shape[1]
        bounds = [0, 0, w, h]  # x,y,x,y representation.

        # crop roi should be saturated by image area.
        self.crop_roi = [
            max(bounds[0], self.roi.x_offset),  # x lower bound
            max(bounds[1], self.roi.y_offset),  # y lower bound
            min(bounds[2], self.roi.x_offset + self.roi.width),  # x upper bound
            min(bounds[3], self.roi.y_offset + self.roi.height),  # y upper bound
        ]
        crop_area = (self.crop_roi[0] - self.crop_roi[2]) * (
            self.crop_roi[1] - self.crop_roi[3]
        )
        if crop_area > 0:
            return image[
                self.crop_roi[1] : self.crop_roi[3], self.crop_roi[0] : self.crop_roi[2]
            ]

    def to_original(self, segm: InstSegm):
        # recover bitset_mask width height
        h, w = self.recent_image_size[:2]
        segm.bitset_mask.width = w
        segm.bitset_mask.height = h
        # recover croped_roi x_offset and y_offset
        segm.bitset_mask.cropped_roi.x_offset += self.roi.x_offset
        segm.bitset_mask.cropped_roi.y_offset += self.roi.y_offset
        # recover bounding_box x_offset and y_offset
        segm.bounding_box.x_offset += self.roi.x_offset
        segm.bounding_box.y_offset += self.roi.y_offset
        return segm


class FatalException(Exception):
    def __init__(self, *args) -> None:
        """Fatal level Exception. This exception is intended not to catch. make fatal level log in ros system."""
        super(Exception, self).__init__(*args)
        rospy.logfatal(*args)


class App:
    def __init__(self) -> None:
        self.is_ready = False

    def start(self):
        """start the object_detector_2d node

        Raises:
            FatalException: this means the node is failed. don't handle this exception, just leave it kill this process.
        """
        # get ros params
        # backend model name. "yolact" and "sipmask" is currently supported.
        backend_param = rospy.get_param("~object_detector_backend_param")
        # object detector input image crop size
        crop_roi = rospy.get_param("~crop_roi")

        # cvbridge for converting ros image message into numpy array
        self.bridge = CvBridge()
        self.crop = CropImage()
        self.crop.set_roi(**crop_roi)

        # setting up publisher.
        self.pub_segm = rospy.Publisher(
            "~object_detection_2d", InstSegmArray, queue_size=10
        )

        # setting up subscribers for color image and depth image
        self.sub_color = rospy.Subscriber(
            "~input_color_image", Image, self.callback, queue_size=1
        )
        # load backend module
        backend_name = backend_param["backend_name"]
        backend_module = import_module(f"object_detector_2d.backend_{backend_name}")
        self.backend_model = backend_module.backend_model

        # backend initialization
        try:
            rospy.loginfo(f"Initializing Backend: {backend_name}")
            self.backend_model.initialize(backend_param)
            rospy.loginfo(f"Initial Memory allocation")
            sample_color_img = np.random.randint(
                0, 256, (crop_roi["height"], crop_roi["width"], 3), np.uint8
            )
            # for the case you give both the color image and depth image arguments to the forward function.
            self.backend_model.forward(sample_color_img)
            rospy.loginfo(f"Initialization Complete")
            self.is_ready = True

        except:
            raise FatalException(
                f"Backend initialization Failed: {backend_name}, {backend_param}"
            )
        rospy.spin()

    def callback(self, color_image_msg: Image):
        """ROS message filter approximate time synchronizer subscriber callback.
        if node initialization is not done, ignores that message.

        Args:
            color_image_msg (Image): Color image message. this is input source for the Yolact network

        Raises:
            FatalException: the Yolact forwarding failed.
            FatalException: the Yolact post-process failed.
        """

        rospy.logdebug("image message recieved")
        if not self.is_ready:
            rospy.logdebug("node initialization in progress. ignore message")
            return

        color_img = self.bridge.imgmsg_to_cv2(color_image_msg, desired_encoding="bgr8")
        cropped_color_img = self.crop.crop(color_img)
        h, w, _ = cropped_color_img.shape

        # some code for yolact evalutation
        try:
            # for the case you give both the color image and depth image arguments to the forward function.
            result_cropped = self.backend_model.forward(cropped_color_img)
        except:
            raise FatalException("Obejct Detection 2D backend forwarding error")

        result_recovered = InstSegmArray()
        result_recovered.header = color_image_msg.header
        for segm in result_cropped.data:
            result_recovered.data.append(self.crop.to_original(segm))

        self.pub_segm.publish(result_recovered)


def main():
    """main function. create app instance, and start. handle keyboard interrup."""
    try:
        rospy.init_node("yolact_node")
        app = App()
        app.start()
    except KeyboardInterrupt:
        exit(0)
    except FatalException:
        exit(1)


if __name__ == "__main__":
    main()
