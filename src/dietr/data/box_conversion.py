"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import numpy as np
import torch

def box_coco_to_yolo(box: list[int], img_hw: tuple[int]) -> list[float]:
    """
    Convert Coco pixel formal boxes to yolo normalized.
    Coco/Pixel format:      [x_min, y_min, width, height]
    YOLO format normalized: [x_center, y_center, width, height]
    xyxy format:            [x_min, y_min, x_max, y_max]
    This is often called cxcywh format.
    """
    x, y, w, h = box[:4]
    x /= img_hw[1]
    y /= img_hw[0]
    w /= img_hw[1]
    h /= img_hw[0]
    return [x + w / 2, y + h / 2, w, h]


def box_yolo_to_coco(box_np: np.ndarray, img_hw: tuple[int, int]) -> np.ndarray:
    """
    Convert yolo format to coco format.
    Coco/Pixel format:      [x_min, y_min, width, height]
    YOLO format normalized: [x_center, y_center, width, height]
    xyxy format:            [x_min, y_min, x_max, y_max]
    """
    x_min = (box_np[:, 0] - box_np[:, 2] / 2) * img_hw[1]
    y_min = (box_np[:, 1] - box_np[:, 3] / 2) * img_hw[0]
    w = (box_np[:, 2] * img_hw[1])
    h = (box_np[:, 3] * img_hw[0])
    return np.floor(np.vstack([x_min, y_min, w, h]).T).astype(int)


def box_tensor_yolo_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    """
    Coco/Pixel format:      [x_min, y_min, width, height]
    YOLO format normalized: [x_center, y_center, width, height]
    xyxy format:            [x_min, y_min, x_max, y_max]
    Can also be the pascal voc format!
    """
    x_c, y_c, w, h = box.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_tensor_xyxy_to_yolo(box: torch.Tensor) -> torch.Tensor:
    """

    Coco/Pixel format:      [x_min, y_min, width, height]
    YOLO format normalized: [x_center, y_center, width, height]
    xyxy format:            [x_min, y_min, x_max, y_max]
    Can also be the pascal voc format!
    """
    xmin, ymin, xmax, ymax = box.unbind(-1)
    box = [(xmax - xmin) / 2, (ymax - ymin) / 2, (xmax - xmin), (ymax - ymin)]
    return torch.stack(box, dim=-1)


def box_xyxy_area(box: torch.Tensor) -> torch.Tensor:
    """Computes the area of a box in normalized pascal format"""
    return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])


def box_xyxy_iou(box1: torch.Tensor, box2: torch.Tensor):
    area1 = box_xyxy_area(box1)
    area2 = box_xyxy_area(box2)

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def box_yolo_iou(box1: torch.Tensor, box2: torch.Tensor):
    return box_xyxy_iou(
        box_tensor_yolo_to_xyxy(box=box1), box_tensor_yolo_to_xyxy(box=box2)
    )


def box_yolo_giou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    iou, union = box_yolo_iou(box1, box2)
    box1 = box_tensor_yolo_to_xyxy(box=box1)
    box2 = box_tensor_yolo_to_xyxy(box=box2)
    lt = torch.min(box1[:, None, :2], box2[:, :2])  # [N, 2]
    rb = torch.max(box1[:, None, 2:], box2[:, 2:])  # [N, 2]
    wh = (rb - lt).clamp(min=0)  # [N, 2]
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area