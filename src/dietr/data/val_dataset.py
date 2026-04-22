"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import numpy as np
import cv2
from pycocotools.coco import COCO
import torch
from dietr.data import loading
from dietr.data.transforms.base import (
    transform_scale_img_np,
)


class CocoValDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        coco_ann_file: str,
        coco_img_root: str,
        img_base_size: tuple[int],
    ) -> None:
        self.coco_gt: COCO = COCO(
            coco_ann_file,
        )
        self.coco_id: list[int] = list(sorted(self.coco_gt.imgs.keys()))
        self.coco_ann_file = coco_ann_file
        self.coco_img_root = coco_img_root
        self.img_base_size = img_base_size

    def __getitem__(self, index: int) -> dict[str, np.ndarray | tuple[int] | int]:
        img_id = self.coco_id[index]
        img_np, img_hw = loading.load_coco_img_np(
            coco_gt=self.coco_gt, img_id=img_id, coco_img_path=self.coco_img_root
        )

        img_np = cv2.resize(
            img_np, dsize=self.img_base_size, interpolation=cv2.INTER_LINEAR
        )

        return {
            "img_np": transform_scale_img_np(img_np=img_np),
            "img_id": img_id,
            "img_hw": img_hw,
        }

    def __len__(self) -> int:
        return len(self.coco_id)
