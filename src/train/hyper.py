class hp:
    # 模型相关
    data_path = "./coco-traffic"
    model_name = "./detr-resnet-50"  # 预训练模型名称
    device = "cpu"  # 训练设备
    id2label = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
    }  # 类别映射
    # 手动创建反向字典
    label2id = {
        "person": 0,
        "bicycle": 1,
        "car": 2,
        "motorcycle": 3,
        "airplane": 4,
        "bus": 5,
        "train": 6,
        "truck": 7,
    }

    traffic_ids = id2label.keys()  # 交通类别id

    # 数据增强
    image_size = 480  # 图像尺寸
    horizontal_flip_prob = 1.0  # 水平翻转概率
    brightness_contrast_prob = 1.0  # 亮度对比度调整概率

    # 训练参数
    output_dir = "detr-resnet-50-finetuned-coco-cars"  # 模型保存路径
    per_device_train_batch_size = 4  # 每个GPU的训练batch size
    num_train_epochs = 10  # 训练轮数
    fp16 = True  # 是否使用混合精度训练
    save_steps = 200  # 每隔多少步保存模型
    logging_steps = 50  # 每隔多少步打印日志
    learning_rate = 2e-5  # 学习率
    weight_decay = 1e-4  # 权重衰减
    save_total_limit = 2  # 最多保存多少个模型
    remove_unused_columns = False  # 是否移除未使用的数据列
    evaluation_strategy = "steps"  # 评估策略
    eval_steps = 200  # 每隔多少步进行评估
    load_best_model_at_end = True  # 是否在训练结束后加载最佳模型
    num_proc = 8  # 数据处理进程数
