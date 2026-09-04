# Easy Perception Deployment (EPD) User Guide

## Overview

EPD workflow:

```
Camera / Images
      ↓
Dataset preparation
      ↓
Model training
      ↓
ONNX export
      ↓
Deployment
      ↓
ROS 2 perception output
```

---

# Training a Model

## 1. Prepare your dataset

A dataset contains images and annotations:

```
dataset/
├── images/
└── annotations/
```

The labels used during training must match the label list selected in EPD.

---

# Choosing an Architecture

## Faster R-CNN

Use Faster R-CNN when you need object detection using bounding boxes.

Good for:

- general object detection
- high accuracy requirements
- scenes with a limited number of objects

Output example:

```
cup
x,y,width,height
confidence
```

## Mask R-CNN

Use Mask R-CNN when you need object segmentation.

Good for:

- robot grasping
- irregular objects
- precise object boundaries

Output example:

```
object location + pixel mask
```

---

# Training Parameters

## Max Iterations

The number of learning steps performed by the model.

Too low:
- model may not learn enough

Too high:
- longer training
- possible overfitting

## Checkpoint Interval

How often the training state is saved.

Example:

```
Checkpoint = 500

500
1000
1500
2000
```

Checkpoints allow recovery if training stops.

## Learning Rate Steps

Controls when learning becomes more gradual.

## Test Period

Controls how often the model is evaluated during training.

---

# Deployment

## Model file

EPD deployment uses ONNX models.

Recommended deployment format:

```
model.onnx
```

Why ONNX?

- portable
- supported by inference engines
- independent from the original training framework

## Labels

The label list must exactly match the training classes.

Example:

```
0 cup
1 box
2 tool
```

Incorrect ordering causes wrong class names during inference.

---

# Model formats

| Extension | Purpose |
|---|---|
| .onnx | Recommended deployment format |
| .pth | PyTorch training checkpoint |
| .pt | PyTorch model |
| .engine | TensorRT optimized model |

Typical workflow:

```
Training: .pth / .pt
Deployment: .onnx
```

---

# Where to get pretrained models

Common sources:

- ONNX Model Zoo
- PyTorch model repositories
- NVIDIA model repositories

Always verify:

- architecture compatibility
- input size
- class labels
- export format

---

# Recommended beginner workflow

1. Start with a pretrained model.
2. Prepare a small labelled dataset.
3. Validate the dataset.
4. Train with moderate iterations.
5. Export/select ONNX model.
6. Deploy with the correct labels and camera topic.
7. Increase complexity only when required.
