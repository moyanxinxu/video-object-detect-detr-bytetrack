import os

import cv2 as cv
import numpy as np
import torch
import yaml
from easydict import EasyDict as edict


class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """

    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert os.path.isfile(config_file)
            with open(config_file, "r") as fo:
                cfg_dict.update(yaml.load(fo.read()))

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, "r") as fo:
            self.update(yaml.safe_load(fo.read()))

    def merge_from_dict(self, config_dict):
        self.update(config_dict)


def get_config(config_file=None):
    return YamlParser(config_file=config_file)


def xyxy2xywh(bbox):
    """
    Converts bounding boxes from (xmin, ymin, xmax, ymax) format to (xmin, ymin, width, height) format.

    Args:
        bbox (torch.Tensor): A tensor of shape (N, 4) representing bounding boxes,
            where N is the number of bounding boxes.

    Returns:
        numpy.ndarray: A numpy array of shape (N, 4) representing the converted bounding boxes.
    """

    xmin, ymin, xmax, ymax = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    width = xmax - xmin
    height = ymax - ymin

    bbox = torch.stack((xmin, ymin, width, height), dim=1)
    return bbox.numpy()


def plot_bboxes(frame, bboxes, scores, labels, color=(255, 0, 0)):
    """
    Plots bounding boxes on the given frame with their corresponding labels and scores.

    Args:
        frame (numpy.ndarray): The image frame on which to plot the bounding boxes.
        bboxes (numpy.ndarray): An array of shape (N, 4) representing bounding boxes.
        scores (numpy.ndarray): An array of shape (N,) representing the confidence scores.
        labels (numpy.ndarray): An array of shape (N,) representing the labels.
        color (tuple): The color for the bounding boxes and text in BGR format.

    Returns:
        numpy.ndarray: The frame with plotted bounding boxes, labels, and scores.
    """
    bboxes = bboxes.astype("int")
    for bbox, score, label in zip(bboxes, scores, labels):
        xmin, ymin, centerx, centery = bbox[0], bbox[1], bbox[2], bbox[3]
        width = centerx - xmin
        height = centery - ymin

        # centerxy转化xyxy
        x1 = int(centerx - width // 2)
        y1 = int(centery - height // 2)
        x2 = int(centerx + width // 2)
        y2 = int(centery + height // 2)

        frame = cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # 绘制标签和置信度
        frame = cv.putText(
            frame,
            f"{label.item()}:{score.item():.2f}",
            (x1, y1),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return frame


def detect_objects(frame, preprocessor, model):
    """
    Detects objects in the given frame using the specified preprocessor and model.

    Args:
        frame (numpy.ndarray): The image frame in which to detect objects.
        preprocessor: The preprocessor object to preprocess the image and post-process the model outputs.
        model: The object detection model.

    Returns:
        torch.Tensor: A tensor of shape (N, 6) where N is the number of detected objects.
                      Each row contains [xmin, ymin, xmax, ymax, score, label].
    """
    h, w, _ = frame.shape
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    inputs = preprocessor(images=frame, return_tensors="pt")

    with torch.no_grad():
        ret = model(**inputs)

    processed_output = preprocessor.post_process_object_detection(
        ret, target_sizes=[(h, w)], threshold=0.9
    )[0]
    scores, labels, boxes = (
        processed_output["scores"],
        processed_output["labels"],
        processed_output["boxes"],
    )
    return torch.cat((boxes, scores.unsqueeze(1), labels.unsqueeze(1)), dim=1)
