class hp:
    device = "cpu"
    confidence_threshold = 0.7
    model_name = "./detr-resnet-50-finetuned-coco-cars/"  # 模型路径
    slide_minimum = 0.0
    slide_maximum = 1.0
    slide_value = 0.8
    result_file_name = "output.mp4"
    original_file_name = "original.mp4"
    combined_file_name = "combined.mp4"
    output_path = "../output/"
    codec = "libx264"
