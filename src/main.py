import argparse
import os

import cv2 as cv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from transformers import DetrForObjectDetection, DetrImageProcessor

from deep_sort import build_tracker
from toolkits.func import detect_objects, get_config, mp4v2h264, plot_bboxes, xyxy2xywh

# Set up environment variables for proxies
os.environ["HF_ENDPOINT"] = "http://hf-mirror.com"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# Enable CUDA benchmarking for improved performance
cudnn.benchmark = True

detector = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
preprocessor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

class VideoTracker(object):
    def __init__(self, args):
        self.args = args
        self.frame_interval = args.frame_interval

        # Set the device for computation (CPU or GPU)
        self.device = torch.device(args.device)
        if args.display:
            # Set up the display window
            cv.namedWindow("demo", cv.WINDOW_NORMAL)
            cv.resizeWindow("demo", args.display_width, args.display_height)

        # Initialize video capture (camera or video file)
        if args.cam != -1:
            self.vdo = cv.VideoCapture(args.cam)
        else:
            self.vdo = cv.VideoCapture()

        # Load DeepSORT configuration
        cfg = get_config()
        cfg.merge_from_file(args.config_deepsort)

        use_cuda = self.device.type != "cpu" and torch.cuda.is_available()
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)

        # Load DETR model and processor

        detector.to(self.device).eval()

        self.names = "facebook/detr-resnet-50"

    def __enter__(self):
        # Initialize video stream
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "error: camera error"
            self.im_width = int(self.vdo.get(cv.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv.CAP_PROP_FRAME_HEIGHT))
        else:
            assert os.path.isfile(self.args.input_path), "path error"
            self.vdo.open(self.args.input_path)
            self.im_width = int(self.vdo.get(cv.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        # Initialize video writer if saving output
        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)
            self.save_video_path = os.path.join(self.args.save_path, "output.mp4")

            fourcc = cv.VideoWriter_fourcc(*self.args.fourcc)
            self.writer = cv.VideoWriter(
                self.save_video_path,
                fourcc,
                self.vdo.get(cv.CAP_PROP_FPS),
                (self.im_width, self.im_height),
            )

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # Release video capture and writer resources
        self.vdo.release()
        self.writer.release()
        mp4v2h264(self.save_video_path, self.save_video_path)

    def run(self):
        idx_frame = 0
        last_output = None

        # Main loop to process video frames
        while self.vdo.grab():
            _, image = self.vdo.retrieve()

            # Process frames at specified interval
            if idx_frame % self.args.frame_interval == 0:
                outputs, scores, labels = self.image_track(image)
                last_output = (outputs, scores, labels)
            else:
                outputs, scores, labels = last_output

            # Plot bounding boxes if there are detected objects
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                image = plot_bboxes(image, bbox_xyxy, scores, labels)

            # Display image
            if self.args.display:
                cv.imshow("test", image)
                if cv.waitKey(1) == ord("q"):
                    cv.destroyAllWindows()
                    break

            if self.args.save_path:
                self.writer.write(image)

            idx_frame += 1

    def image_track(self, image):
        # Perform object detection
        outputs = detect_objects(image, preprocessor, detector)
        if outputs is not None and len(outputs):
            bboxes_xywh = xyxy2xywh(outputs)
            scores = outputs[:, 4]
            labels = outputs[:, 5]
            outputs = self.deepsort.update(bboxes_xywh, scores, image)
        else:
            outputs = np.zeros((0, 5))
            scores = np.zeros(0)
            labels = np.zeros(0)

        return outputs, scores, labels


def detect(in_video_path):
    args = argparse.Namespace(
        input_path=in_video_path,
        save_path="output/",
        frame_interval=2,
        fourcc="mp4v",
        device="cpu",
        display=False,
        display_width=800,
        display_height=600,
        cam=-1,
        config_deepsort="./configs/deep_sort.yaml",
    )

    # Run the video tracker
    with VideoTracker(args) as vdo_trk:
        vdo_trk.run()

    return "output/output.mp4"
