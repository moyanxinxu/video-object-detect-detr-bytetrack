import os
from hyper import hp
from datasets import load_dataset

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"


ds = load_dataset("detection-datasets/coco")


traffic_ids = hp.traffic_ids


def transform_coco_format(data, id2label):
    new_data = {}
    new_data["image_id"] = data["image_id"]
    new_data["image"] = data["image"].convert("RGB")
    new_data["width"] = data["width"]
    new_data["height"] = data["height"]

    new_objects = {"id": [], "area": [], "bbox": [], "category": []}
    for i in range(len(data["objects"]["bbox_id"])):
        category_id = data["objects"]["category"][i]
        if category_id in id2label:
            new_objects["id"].append(data["objects"]["bbox_id"][i])
            new_objects["area"].append(int(data["objects"]["area"][i]))
            x_min, y_min, x_max, y_max = data["objects"]["bbox"][i]
            width = x_max - x_min
            height = y_max - y_min
            new_objects["bbox"].append([x_min, y_min, width, height])
            new_objects["category"].append(category_id)

    new_data["objects"] = new_objects
    return new_data


transformed_ds = ds.map(
    lambda data: transform_coco_format(data, traffic_ids), num_proc=hp.num_proc
)

filtered_ds = transformed_ds.filter(lambda x: x["objects"]["area"] != [])

save_path = hp.data_path
filtered_ds.save_to_disk(save_path)
