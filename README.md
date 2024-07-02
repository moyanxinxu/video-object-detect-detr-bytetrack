# video-object-detect-detr-deepsort

![banner.png](assess/banner.png)

应用创新实践课程、课号 C208996400、视频目标检测系统设计与实现

## 环境

```bash
cv2
gradio
numpy
torch
transformers
```

## 技术路线

在实际应用中，直接将基于图像的目标检测方法应用于视频目标检测往往效果不佳。这是因为视频中目标的运动状态、遮挡情况等会带来额外的挑战。为了解决这个问题，我们打算在现有的图像目标检测方法加以模块，该模块**仅利用当前帧和历史帧信息**（不使用未来帧信息）实现视频目标检测。相比于传统的基于图像的方法，这种方法能够有效地处理**目标运动过程中的遮挡、形态变化以及方向变化**等问题。

为了实现这一目标，除了完成目标检测的基本功能外，我们还需要实现以下模块：

1. 面对多目标检测，需要对连续的帧上的同一目标的目标框进行匹配。
2. 给出能检测出**目标在运动过程中的遮挡、形态与方向变化**的算法。
3. 在目标框匹配的时候优化搜索算法，需要给出不在整个图像进行搜索，而是在上一帧的局部区域进行搜索匹配目标框的算法。

在模型架构与数据集上：

- 模型选择，时间允许的条件下会从头训练，否则会尝试预训练模型。
     1. `detr`：该模型为 FaceBook 提出的业界最新的目标检测模型，只有两部分，cnn 特征提取与 transformer 编解码网络。
     2. `yolo v5`：效果经得起检验，因为不如 `detr` 的简洁高效，所以该模型为候选模型。
- 数据集选择，主要来自 Kaggle，算力与时间是有限的，数据量控制在 3G 左右，做出大致的效果即可，在这里选择 [SODA10M 数据集](https://soda-2d.github.io/download.html)。
     1. 视频数据集：[cars-video-object-tracking](https://www.kaggle.com/datasets/trainingdatapro/cars-video-object-tracking/data)
     2. 图片数据集：[car-object-detection](https://www.kaggle.com/datasets/sshikamaru/car-object-detection)，[SODA10M 数据集](https://soda-2d.github.io/download.html)

> SODA10M 是一个半/自监督的 2D 基准数据集，其主要包含一千万张多样性丰富的无标注道路场景图片以及**两万张标注有 6 个代表性对象类别**的图片。
>
> 无标注图片从 32 个不同的城市中选取，囊括了中国的大部分区域。同时图片包含了多种不同的道路场景（城市，高速，城乡道路，园区），天气（晴天，多云，雨天，雪天），时间段（白天，晚上，凌晨/黄昏）。
>
> example：
>
> ![example.png](assess/example.png)

注：ImageNet Vid，ImageNetDET，~~MSCoco~~ 过于庞大，下载也是问题，先不考虑。

总的来说初步方案：

对每帧做目标检测，然后连续的两帧之间以使用 `DeepSort` 追踪器，融合前后的信息后, 最后拼接成视频。

### 分工

| 姓名  |           职责           |
| :-: | :--------------------: |
| 田健翔 |      整体架构、目标追踪算法       |
| 张栋梁 | 视频 demo 演示制作与网络可解释性可视化 |
| 王欣雨 |     gradio 前端界面搭建      |
| 尹潇逸 |   数据集格式转化与网络可解释性可视化    |

## 参考资料

[huggingface-object-detection](https://huggingface.co/tasks/object-detection)

[DeepSORT_YOLOv5_Pytorch](https://github.com/HowieMa/DeepSORT_YOLOv5_Pytorch)

[detr-attention](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb#scrollTo=frMO0BaCYTEr)

[gradio](https://www.gradio.app/)
