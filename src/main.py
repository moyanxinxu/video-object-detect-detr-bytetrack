import argparse
import os

import cv2 as cv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from transformers import DetrForObjectDetection, DetrImageProcessor

from deep_sort import build_tracker
from toolkits.func import detect_objects, get_config, plot_bboxes, xyxy2xywh

# Set up environment variables for proxies
os.environ["HF_ENDPOINT"] = "http://hf-mirror.com"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# Enable CUDA benchmarking for improved performance
cudnn.benchmark = True


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
        self.detector = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50"
        )
        self.preprocessor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50"
        )
        self.detector.to(self.device).eval()

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

            # Save video output
            if self.args.save_path:
                self.writer.write(image)

            idx_frame += 1

    def image_track(self, image):
        # Perform object detection
        outputs = detect_objects(image, self.preprocessor, self.detector)
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


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="demo.mp4", help="source")
    parser.add_argument(
        "--save_path", type=str, default="output/", help="output folder"
    )
    parser.add_argument("--frame_interval", type=int, default=2)
    parser.add_argument(
        "--fourcc",
        type=str,
        default="mp4v",
        help="output video codec (verify ffmpeg support)",
    )
    parser.add_argument(
        "--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument(
        "--config_deepsort", type=str, default="./configs/deep_sort.yaml"
    )

    args = parser.parse_args()
    print(args)

    # Run the video tracker
    with VideoTracker(args) as vdo_trk:
        vdo_trk.run()
