import os

import cv2 as cv
import moviepy.editor as mpe
import numpy as np
import supervision as sv
import torch
from hyper import hp
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PIL import Image
from tqdm import tqdm


def detect(frame, model, processor, confidence_threshold):
    """
    args:
        image: PIL image
        model: PreTrainedModel
        processor: PreTrainedProcessor
        confidence_threshold: float
    returns:
        results: dict with keys "boxes", "labels", "scores"


    examples:
    [
        {
            "scores": tensor([0.9980, 0.9039, 0.7575, 0.9033]),
            "labels": tensor([86, 64, 67, 67]),
            "boxes": tensor(
                [
                    [1.1582e03, 1.1893e03, 1.9373e03, 1.9681e03],
                    [2.4274e02, 1.3234e02, 2.5919e03, 1.9628e03],
                    [1.1107e-01, 1.5105e03, 3.1980e03, 2.1076e03],
                    [7.1036e-01, 1.7360e03, 3.1970e03, 2.1100e03],
                ]
            ),
        }
    ]
    """
    inputs = processor(images=frame, return_tensors="pt").to(hp.device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([frame.size[::-1]])

    results = processor.post_process_object_detection(
        outputs=outputs, threshold=confidence_threshold, target_sizes=target_sizes
    )
    return results


def get_len_frames(viedo_path):
    """
    args:
        viedo_path: str
    returns:
        int: the number of frames in the video
    examples:
        get_len_frames("../demo_video/aerial.mp4") # 1478
    """
    video_info = sv.VideoInfo.from_video_path(viedo_path)
    return video_info.total_frames


def track(detected_result, tracker: sv.ByteTrack):
    """
    args:
        detected_result: dict with keys "boxes", "labels", "scores"
        tracker: sv.ByteTrack
    returns:
        tracked_result: dict with keys "boxes", "labels", "scores"
    examples:
        from transformers import DetrImageProcessor, DetrForObjectDetection

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        tracker = sv.ByteTrack()

        image = Image.open("ZJF990.jpg")

        detected_result = detect(image, model, processor, hp.confidence_threshold)
        tracked_result = track(detected_result, tracker)

        print(detected_result)
        print(tracked_result)


        [
            {
                "scores": tensor([0.9980, 0.9039, 0.7575, 0.9033]),
                "labels": tensor([86, 64, 67, 67]),
                "boxes": tensor(
                    [
                        [1.1582e03, 1.1893e03, 1.9373e03, 1.9681e03],
                        [2.4274e02, 1.3234e02, 2.5919e03, 1.9628e03],
                        [1.1107e-01, 1.5105e03, 3.1980e03, 2.1076e03],
                        [7.1036e-01, 1.7360e03, 3.1970e03, 2.1100e03],
                    ]
                ),
            }
        ]



        Detections(
            xyxy=array(
                [
                    [1.1581914e03, 1.1892766e03, 1.9372931e03, 1.9680990e03],
                    [2.4273552e02, 1.3233553e02, 2.5918860e03, 1.9628494e03],
                    [1.1106834e-01, 1.5105106e03, 3.1980032e03, 2.1075664e03],
                    [7.1036065e-01, 1.7359819e03, 3.1970449e03, 2.1100107e03],
                ],
                dtype=float32,
            ),
            mask=None,
            confidence=array([0.9980374, 0.9038882, 0.7575455, 0.9032779], dtype=float32),
            class_id=array([86, 64, 67, 67]),
            tracker_id=array([1, 2, 3, 4]),
            data={},
        )


    """

    detections = sv.Detections.from_transformers(detected_result[0])
    detections = tracker.update_with_detections(detections)
    return detections


def annotate_image(
    frame,
    detections,
    labels,
    mask_annotator: sv.MaskAnnotator,
    bbox_annotator: sv.BoxAnnotator,
    label_annotator: sv.LabelAnnotator,
) -> np.ndarray:
    out_frame = mask_annotator.annotate(frame, detections)
    out_frame = bbox_annotator.annotate(out_frame, detections)
    out_frame = label_annotator.annotate(out_frame, detections, labels=labels)
    return out_frame


def detect_and_track(
    video_path,
    model,
    processor,
    tracker,
    confidence_threshold,
    mask_annotator: sv.MaskAnnotator,
    bbox_annotator: sv.BoxAnnotator,
    label_annotator: sv.LabelAnnotator,
):
    video_info = sv.VideoInfo.from_video_path(video_path)
    fps = video_info.fps
    len_frames = video_info.total_frames

    frames_loader = sv.get_video_frames_generator(video_path, end=len_frames)

    result_file_name = hp.result_file_name
    original_file_name = hp.original_file_name
    combined_file_name = hp.combined_file_name
    result_file_path = os.path.join(hp.output_path, result_file_name)
    original_file_path = os.path.join(hp.output_path, original_file_name)
    combined_file_name = os.path.join(hp.output_path, combined_file_name)

    concated_frames = []
    original_frames = []
    for frame in tqdm(frames_loader, total=len_frames):
        results = detect(Image.fromarray(frame), model, processor, confidence_threshold)
        tracked_results = track(results, tracker)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        original_frames.append(frame.copy())
        scores = tracked_results.confidence.tolist()
        labels = tracked_results.class_id.tolist()

        frame = annotate_image(
            frame,
            tracked_results,
            labels=[
                str(f"{model.config.id2label[label]}-{score:.2f}")
                for label, score in zip(labels, scores)
            ],
            mask_annotator=mask_annotator,
            bbox_annotator=bbox_annotator,
            label_annotator=label_annotator,
        )
        concated_frames.append(frame)  # Add the processed frame to the list
    codec = hp.codec
    # Create a MoviePy video clip from the list of frames
    original_video = mpe.ImageSequenceClip(original_frames, fps=fps)
    original_video.write_videofile(original_file_path, codec=codec, fps=fps)
    concated_video = mpe.ImageSequenceClip(concated_frames, fps=fps)
    concated_video.write_videofile(result_file_path, codec=codec, fps=fps)

    combined_video = combine_frames(original_frames, concated_frames, fps)
    combined_video.write_videofile(combined_file_name, codec=codec, fps=fps)
    return result_file_path, combined_file_name


def combine_frames(frames_list1, frames_list2, fps):
    """
    args:
        frames_list1: list of PIL images
        frames_list2: list of PIL images
    returns:
        final_clip: moviepy video clip
    """
    clip1 = ImageSequenceClip(frames_list1, fps=fps)
    clip2 = ImageSequenceClip(frames_list2, fps=fps)

    final_clip = mpe.clips_array([[clip1, clip2]])

    return final_clip
