"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import os
import cv2
import pycocotools
import numpy as np


def load_coco_img_np(
    coco_gt: pycocotools.coco.COCO, img_id: int, coco_img_path: str
) -> tuple[np.ndarray, tuple[int, int]]:
    """Given a pycocotools and img_id,"""
    img_info = coco_gt.loadImgs(ids=[img_id])[0]
    img_path = img_info["file_name"]
    combined_img_path = os.path.join(coco_img_path, img_path)

    return (
        cv2.cvtColor(cv2.imread(combined_img_path), cv2.COLOR_RGB2BGR),
        (
            img_info["height"],
            img_info["width"],
        )    )


def load_coco_ann(coco_gt: pycocotools.coco.COCO, img_id: int) -> list[any]:
    ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
    return coco_gt.loadAnns(ann_ids)
