#!/usr/bin/python3

import os
from typing import Tuple, Union

from yolov6.layers.common import DetectBackend
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer

import torch
import cv2
from torchvision.transforms import ToTensor
import numpy as np
import rospy

from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from utils import create_detection_msg
from visualizer import draw_detections


class YoloV6:
    def __init__(self, weights, conf_thresh: float = 0.5, iou_thresh: float = 0.45,
                 device: str = "cuda"):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.model = DetectBackend(weights, device=device)

    def inference(self, img: torch.Tensor):
        """
        :param img: tensor [c, h, w]
        :returns: tensor of shape [num_boxes, 6], where each item is represented as
            [x1, y1, x2, y2, confidence, class_id]
        """
        img = img.unsqueeze(0)
        pred_results = self.model(img)
        detections = non_max_suppression(
            pred_results, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh
        )
        if detections:
            detections = detections[0]
        return detections


class Yolov6Publisher:
    def __init__(self, img_topic: str, weights: str, conf_thresh: float = 0.5,
                 iou_thresh: float = 0.45, pub_topic: str = "yolov6_detections",
                 device: str = "cuda",
                 img_size: Union[Tuple[int, int], None] = (640, 640),
                 queue_size: int = 1, visualize: bool = False):
        """
        :param img_topic: name of the image topic to listen to
        :param weights: path/to/yolo_weights.pt
        :param conf_thresh: confidence threshold
        :param iou_thresh: intersection over union threshold
        :param pub_topic: name of the output topic (will be published under the
            namespace '/yolov6')
        :param device: device to do inference on (e.g., 'cuda' or 'cpu')
        :param queue_size: queue size for publishers
        :visualize: flag to enable publishing the detections visualized in the image
        :param img_size: (height, width) to which the img is resized before being
            fed into the yolo network. Final output coordinates will be rescaled to
            the original img size.
        """
        self.img_size = img_size
        self.device = device

        vis_topic = pub_topic + "visualization" if pub_topic.endswith("/") else \
            pub_topic + "/visualization"
        self.visualization_publisher = rospy.Publisher(
            vis_topic, Image, queue_size=queue_size
        ) if visualize else None

        self.bridge = CvBridge()

        self.tensorize = ToTensor()
        self.model = YoloV6(
            weights=weights, conf_thresh=conf_thresh, iou_thresh=iou_thresh,
            device=device
        )
        self.img_subscriber = rospy.Subscriber(
            img_topic, Image, self.process_img_msg
        )
        self.detection_publisher = rospy.Publisher(
            pub_topic, Detection2DArray, queue_size=queue_size
        )

    def process_img_msg(self, img_msg: Image):
        """ callback function for publisher """
        np_img_orig = self.bridge.imgmsg_to_cv2(
            img_msg, desired_encoding='passthrough'
        )

        # handle possible different img formats
        if len(np_img_orig.shape) == 2:
            np_img_orig = np.stack([np_img_orig] * 3, axis=2)

        h_orig, w_orig, c = np_img_orig.shape
        if c == 1:
            np_img_orig = np.concatenate([np_img_orig] * 3, axis=2)
            c = 3

        # automatically resize the image to the next smaller possible size
        w_scaled, h_scaled = self.img_size

        # w_scaled = w_orig - (w_orig % 8)
        np_img_resized = cv2.resize(np_img_orig, (w_scaled, h_scaled))

        # conversion to torch tensor (copied from original yolov6 repo)
        img = np_img_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(np.ascontiguousarray(img))
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.
        img = img.to(self.device)

        # inference & rescaling the output to original img size
        detections = self.model.inference(img)
        detections[:, :4] = Inferer.rescale(
            [h_scaled, w_scaled], detections[:, :4], [h_orig, w_orig])
        detections[:, :4] = detections[:, :4].round()

        # publishing
        detection_msg = create_detection_msg(img_msg, detections)
        self.detection_publisher.publish(detection_msg)

        # visualizing if required
        if self.visualization_publisher:
            bboxes = [[int(x1), int(y1), int(x2), int(y2)]
                      for x1, y1, x2, y2 in detections[:, :4].tolist()]
            classes = [int(c) for c in detections[:, 5].tolist()]
            vis_img = draw_detections(np_img_orig, bboxes, classes)
            vis_msg = self.bridge.cv2_to_imgmsg(vis_img)
            self.visualization_publisher.publish(vis_msg)


if __name__ == "__main__":
    rospy.init_node("yolov6_node")

    ns = rospy.get_name() + "/"

    weights_path = rospy.get_param(ns + "weights_path")
    img_topic = rospy.get_param(ns + "img_topic")
    out_topic = rospy.get_param(ns + "out_topic")
    conf_thresh = rospy.get_param(ns + "conf_thresh")
    iou_thresh = rospy.get_param(ns + "iou_thresh")
    queue_size = rospy.get_param(ns + "queue_size")
    img_size = rospy.get_param(ns + "img_size")
    visualize = rospy.get_param(ns + "visualize")
    device = rospy.get_param(ns + "device")

    # some sanity checks
    if not os.path.isfile(weights_path):
        raise FileExistsError("Weights not found.")

    if not ("cuda" in device or "cpu" in device):
        raise ValueError("Check your device.")

    publisher = Yolov6Publisher(
        img_topic=img_topic,
        pub_topic=out_topic,
        weights=weights_path,
        device=device,
        visualize=visualize,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        img_size=(img_size, img_size),
        queue_size=queue_size
    )

    rospy.spin()
