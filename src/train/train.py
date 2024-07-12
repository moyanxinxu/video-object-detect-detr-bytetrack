from datasets import load_from_disk
import torch
from transformers import (
    AutoImageProcessor,
    AutoConfig,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
)
import albumentations
import numpy as np
import os

os.environ["HF_ENDPOINT"] = "http://hf-mirror.com"

# 定义类别标签映射
id2label = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
}
label2id = {v: k for k, v in id2label.items()}

# 加载数据集
ds = load_from_disk("./coco-traffic")
ds = ds.train_test_split(test_size=0.1, seed=42)

# 加载预训练模型和图像处理器
checkpoint = "facebook/detr-resnet-50"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

# 定义数据增强变换
transform = albumentations.Compose(
    [
        albumentations.Resize(480, 480),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.RandomBrightnessContrast(p=1.0),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)


# 格式化标注信息
def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations


# 对批次数据进行增强和格式化
def transform_aug_ann(examples):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = transform(
            image=image, bboxes=objects["bbox"], category=objects["category"]
        )

        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")


# 对训练集应用数据增强
ds["train"] = ds["train"].with_transform(transform_aug_ann)


# 定义数据整理函数
def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch


# 加载模型配置和模型
config = AutoConfig.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
model = AutoModelForObjectDetection.from_config(config)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="detr-resnet-50-finetuned-coco-cars",
    per_device_train_batch_size=8,
    num_train_epochs=10,
    fp16=True,
    save_steps=200,
    logging_steps=50,
    learning_rate=1e-5,
    weight_decay=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
)

# 创建训练器并开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=ds["train"],
    tokenizer=image_processor,
)

trainer.train()

trainer.save_model("../infer/detr-resnet-50-finetuned-coco-cars")
