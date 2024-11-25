import os

import gradio as gr
import supervision as sv
from func import detect_and_track
from hyper import hp
from transformers import DetrForObjectDetection, DetrImageProcessor

os.environ["HF_ENDPOINT"] = "http://hf-mirror.com"

processor = DetrImageProcessor.from_pretrained(hp.model_name)
model = DetrForObjectDetection.from_pretrained(hp.model_name)
tracker = sv.ByteTrack()

mask_annotator = sv.MaskAnnotator()
bbox_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()


def process_video(video_path, confidence_threshold):
    return detect_and_track(
        video_path,
        model,
        processor,
        tracker,
        confidence_threshold,
        mask_annotator,
        bbox_annotator,
        label_annotator,
    )


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            in_video = gr.Video(
                label="待检测视频",
                show_download_button=True,
                show_share_button=True,
            )
            slide_cofidence = gr.Slider(
                minimum=hp.slide_minimum,
                maximum=hp.slide_maximum,
                value=hp.slide_value,
                label="置信度阈值",
            )
            examples = gr.Examples(
                examples=[
                    "../demo_video/blurry.mp4",
                    "../demo_video/high-way.mp4",
                    "../demo_video/aerial.mp4",
                ],
                inputs=in_video,
                label="案例视频",
            )
        with gr.Column():
            out_video = gr.Video(
                label="检测结果视频",
                interactive=False,
                show_download_button=True,
                show_share_button=True,
            )
            combine_video = gr.Video(
                interactive=False,
                label="前后对比",
                show_download_button=True,
                show_share_button=True,
            )

            start_detect = gr.Button(value="开始检测")

    start_detect.click(
        fn=process_video,
        inputs=[in_video, slide_cofidence],
        outputs=[out_video, combine_video],
    )

demo.launch()
