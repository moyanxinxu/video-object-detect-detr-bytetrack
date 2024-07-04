import gradio as gr

from main import detect
from toolkits.func import concateVideo

title = """
# 视频目标检测系统设计与实现
"""

introduction = """
- DETR 模型是 Facebook AI Research 团队提出的一种端到端的目标检测模型
- DeepSort 是一种基于深度学习的目标跟踪算法，它通过结合目标检测和目标跟踪，实现了对视频中目标的跟踪。

本项目实现了一个交通领域视频目标检测系统，实现上传一个视频，系统将检测视频中的目标并输出检测结果。
"""

banner = """![](https://github.com/moyanxinxu/video-object-detect-detr-deepsort/raw/main/assess/banner.png)"""

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown(title)
    with gr.Row():
        with gr.Column():
            gr.Markdown(introduction)
        with gr.Column():
            gr.Markdown(banner)
    with gr.Row():
        with gr.Column():
            in_video = gr.Video(
                label="输入视频",
                sources="upload",
                interactive=True,
                show_share_button=True,
                show_download_button=True,
            )
        with gr.Column():
            out_video = gr.Video(
                format="mp4",
                label="检测结果",
                interactive=False,
                show_share_button=True,
                show_download_button=True,
            )
    with gr.Row():
        with gr.Column():
            all_video = gr.Video(
                label="所有视频",
                interactive=False,
                show_share_button=True,
                show_download_button=True,
            )

    with gr.Row():
        with gr.Column():
            gr.Examples(
                examples=[
                    "demo_video/blurry.mp4",
                    "demo_video/high-way.mp4",
                    "demo_video/aerial.mp4",
                    "demo_video/in-car-line.mp4",
                ],
                inputs=in_video,
                label="案例视频",
            )
        with gr.Column():
            btn1 = gr.Button("开始检测")
            btn2 = gr.Button("合成")

    btn1.click(detect, in_video, out_video)
    btn2.click(concateVideo, [in_video, out_video], all_video)
    demo.launch()
