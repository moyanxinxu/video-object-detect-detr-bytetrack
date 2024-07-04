import os
import random

import cv2 as cv
import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from moviepy.editor import ImageSequenceClip, VideoFileClip, clips_array

id2label = {
    0: "N/A",
    1: "person",
    10: "traffic light",
    11: "fire hydrant",
    12: "street sign",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    2: "bicycle",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    26: "hat",
    27: "backpack",
    28: "umbrella",
    29: "shoe",
    3: "car",
    30: "eye glasses",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    4: "motorcycle",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    45: "plate",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    5: "airplane",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    6: "bus",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    66: "mirror",
    67: "dining table",
    68: "window",
    69: "desk",
    7: "train",
    70: "toilet",
    71: "door",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    8: "truck",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    83: "blender",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    9: "boat",
    90: "toothbrush",
}


def video2frames(video):
    """
    args:
        video: str  # video path
    returns:
        frames: list  # list of frames with cv.Matrix format

    example:
        video_path = "./demo.mp4"
        video2frames(video_path)
    """
    videoCapture = cv.VideoCapture()
    videoCapture.open(video)
    frames = []

    while True:
        ret, frame = videoCapture.read()
        if ret:
            frames.append(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        else:
            break

    return frames


class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    args:
        edict: dict  # config dict
    returns:
        YamlParser  # YamlParser object
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
    """
    args:
        config_file: str  # config file path
    returns:
        YamlParser  # YamlParser object
    """
    return YamlParser(config_file=config_file)


def xyxy2xywh(bbox):
    """
    Converts bounding boxes from (xmin, ymin, xmax, ymax) format to (xmin, ymin, width, height) format.

    args:
        bbox (torch.Tensor): A tensor of shape (N, 4) representing bounding boxes,
            where N is the number of bounding boxes.

    returns:
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

    args:
        frame (numpy.ndarray): The image frame on which to plot the bounding boxes.
        bboxes (numpy.ndarray): An array of shape (N, 4) representing bounding boxes.
        scores (numpy.ndarray): An array of shape (N,) representing the confidence scores.
        labels (numpy.ndarray): An array of shape (N,) representing the labels.
        color (tuple): The color for the bounding boxes and text in BGR format.

    returns:
        frame: (numpy.ndarray): The frame with plotted bounding boxes, labels, and scores.
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

        fill_color = label2color(label.item())
        points = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], np.int32)
        frame_copy = cv.fillConvexPoly(frame.copy(), points, fill_color)
        frame = cv.rectangle(frame, (x1, y1), (x2, y2), fill_color, 2)
        frame = cv.addWeighted(frame, 0.8, frame_copy, 0.2, 0)
        # 绘制标签和置信度

        text = f"{id2label[label.item()]}:{score.item():.2f}"
        (text_width, text_height), _ = cv.getTextSize(
            text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv.rectangle(
            frame, (x1, y1 - text_height), (x1 + text_width, y1), fill_color, -1
        )
        frame = cv.putText(
            frame,
            text,
            (x1, y1),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )

    return frame


def detect_objects(frame, preprocessor, model):
    """
    Detects objects in the given frame using the specified preprocessor and model.

    args:
        frame (numpy.ndarray): The image frame in which to detect objects.
        preprocessor: The preprocessor object to preprocess the image and post-process the model outputs.
        model: The object detection model.

    returns:
        torch.Tensor: A tensor of shape (N, 6) where N is the number of detected objects.  Each row contains [xmin, ymin, xmax, ymax, score, label].
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


def mp4v2h264(in_video_path, out_video_path):
    """
    args:
        in_video_path: str  # input video path
        out_video_path: str  # output video path
    returns:
        None
    """
    frames = video2frames(in_video_path)
    cap = cv.VideoCapture(in_video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(out_video_path, codec="libx264")
    cap.release()


def concateVideo(video1, video2, out_path="./output/concated_output.mp4"):
    """
    args:
        video1: str  # video path 1
        video2: str  # video path 2
    returns:
        None
    """
    video_1 = VideoFileClip(video1)
    video_2 = VideoFileClip(video2)

    final_clip = clips_array([[video_1, video_2]])
    final_clip.write_videofile(out_path)

    return out_path


def label2color(label, seed=42):
    """
    Assigns a unique color to each label, using a random seed for consistency.

    args:
        label: int  # The label to get a color for.
        seed: int  # Random seed for consistent color generation.

    returns:
        color: tuple  # The color in BGR format.
    """

    # Ensure the same seed is used for the same label across multiple calls
    random.seed(label + seed)

    # Generate a random BGR color tuple
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return color
