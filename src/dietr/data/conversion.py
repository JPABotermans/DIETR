"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import torch
import numpy as np
from dietr.data.coco_definitions import (
    coco_label_to_normal,
    normal_label_to_coco,
    coco_classes,
)
from dietr.data.msk_conversion import (
    msk_rle_to_msk_np,
    msk_rle_to_compressed_rle,
    msk_polygon_to_msk_np,
    msk_np_to_msk_crp_np,
)

from dietr.data.box_conversion import box_coco_to_yolo


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def coco_seg_to_msk_np(coco_seg: dict, img_hw: tuple[int, int]) -> np.ndarray:
    if not isinstance(coco_seg, dict):
        msk_np = msk_polygon_to_msk_np(polygon=coco_seg, img_hw=img_hw)
    else:
        if "counts" in coco_seg:
            msk_np = msk_rle_to_msk_np(msk_rle_to_compressed_rle(coco_ann=coco_seg))
        else:
            msk_np = msk_rle_to_msk_np(rle=coco_seg)
    return msk_np.astype(np.bool_)


def coco_ann_to_sample_np(
    coco_ann: list[any],
    img_hw: tuple[int, int],
    min_visibility: float,
    coco_dataset: bool,
    load_msk: bool,
) -> dict[str, np.ndarray]:
    """Convert anntations from the coco annotation format to the internal data format, which is just a dictionary of numpy arrays..

    Args:
        coco_ann (list[coco_annotation]): list of coco annotations.
        img_hw (tuple[int, int]): the original height and with of the original image, this is need to scale the bounding boxes.

    Returns:
        sample_np (dict[str, np.ndarray]): combined targets that can be use for training.

        It has the following elements:
        box_np      (np.ndarray):  Boxes in normalized yolo format. (or cxcywh).
        cls_np      (np.ndarray):  Labels in normal format, thus 0: person, etc...
        msk_np      (np.ndarray):  Masks in np.ndarray.
        crp_np      (np.ndarray):  Cropping of the masks in a numpy array, this cropping is used when computing the loss, to stabilize training the segmentation outputs.
    """
    box_np = []
    cls_np = []
    if load_msk:
        msk_np = []
        crp_np = []
    for annotation in coco_ann:
        if annotation["iscrowd"] == 1:
            continue

        box_yolo = box_coco_to_yolo(annotation["bbox"], img_hw=img_hw)
        if (box_yolo[2] <= min_visibility) or (box_yolo[3] <= min_visibility):
            continue
        box_np.append(box_yolo)

        if coco_dataset:
            cls_np.append(label_coco_to_normal(annotation["category_id"]))
        else:
            cls_np.append(annotation["category_id"])

        if load_msk:
            msk_np.append(
                coco_seg_to_msk_np(coco_seg=annotation["segmentation"], img_hw=img_hw)
            )
            crp_np.append(msk_np_to_msk_crp_np(msk_np=msk_np[-1]))

    if len(box_np) == 0:
        if load_msk:
            return {
                "box_np": np.array([]).reshape(-1, 4),
                "msk_np": np.array([]).reshape(-1, 4, 4),
                "crp_np": np.array([]).reshape(-1, 4, 4),
                "cls_np": np.array([]),
            }
        return {
            "box_np": np.array([]).reshape(-1, 4),
            "cls_np": np.array([]),
        }
    if load_msk:
        return {
            "box_np": np.stack(box_np),
            "cls_np": np.stack(cls_np),
            "msk_np": np.stack(msk_np),
            "crp_np": np.stack(crp_np),
        }
    return {
        "box_np": np.stack(box_np),
        "cls_np": np.stack(cls_np),
    }


def label_coco_to_normal(coco_label: int) -> int:
    """In the coco format the labels are from 1 to 90, this function just maps these labels to label_id,
    with id=0, is person."""
    return coco_label_to_normal[coco_label]


def label_normal_to_coco(normal_label: int) -> int:
    return normal_label_to_coco[normal_label]


def label_normal_to_name(normal_label: int) -> str:
    return coco_classes[normal_label - 1]


def label_coco_to_name(normal_label: int) -> str:
    return coco_classes[label_coco_to_normal(normal_label) - 1]
