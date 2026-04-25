"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import numpy as np
from pycocotools.coco import COCO
import torch
from dietr.data import loading
from dietr.data import conversion
from dietr.data.transforms.base import (
    transform_resize_sample_np,
    transform_scale_sample_np,
    transform_top_down_flip_sample_np,
    transform_left_right_flip_sample_np,
    filter_empty_annotations_sample_np,
    filter_coco_gt_remove_empty
)
from dietr.data.transforms.photometric import random_photometric_distortion
from dietr.data.transforms.mosaic import moisaic_augmentation
from dietr.data.transforms.random_crop import transform_random_crop_sample_np
from dietr.data.transforms.random_zoom import transform_random_zoomout_sample_np


class CocoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        coco_ann_file: str,
        coco_img_root: str,
        img_base_size: tuple[int],
        msk: bool,
        mask_scaling: int = 4,
        min_visibility: float = 0.05,
        td_flip_p: float = 0.5,
        lr_flip_p: float = 0.5,
        photo_transform_p: bool = 0.5,
        filter_empty: bool = False,
        random_zoom_p: float = 0.0,
        random_crop_p: float = 0.0,
        mosaic_p: float = 0.0,
        coco_dataset: bool = True
    ) -> None:
        self.coco_gt: COCO = COCO(
            coco_ann_file,
        )
        self.msk = msk
        if filter_empty:
            self.coco_gt.dataset = filter_coco_gt_remove_empty(self.coco_gt.dataset)
            self.coco_gt.createIndex()
        self.coco_id: list[int] = list(sorted(self.coco_gt.imgs.keys()))
        self.coco_ann_file = coco_ann_file
        self.coco_img_root = coco_img_root
        self.img_base_size = img_base_size
        self.min_visibility = min_visibility
        self.td_flip_p = td_flip_p
        self.lr_flip_p = lr_flip_p
        self.mask_scaling = mask_scaling
        self.photo_transform_p = photo_transform_p
        self.filter_empty = filter_empty
        self.random_zoom_p = random_zoom_p
        self.random_crop_p = random_crop_p
        self.mosaic_p = mosaic_p
        self.coco_dataset = coco_dataset

    def __getitem__(
        self, index: int
    ) -> dict[str, np.ndarray]:

        if self.mosaic_p > 0 and self.mosaic_p >= np.random.rand():
            sample_np = moisaic_augmentation(
                coco_gt=self.coco_gt,
                coco_id=self.coco_id,
                min_visibility=self.min_visibility,
                img_size=self.img_base_size,
                coco_img_pth=self.coco_img_root,
                coco_dataset=self.coco_dataset
            )
            if self.filter_empty:
                sample_np = filter_empty_annotations_sample_np(
                    sample_np=sample_np, min_msk_size=1
                )
            return transform_scale_sample_np(sample_np=sample_np) 
            
        img_id = self.coco_id[index]
        
        img_np, img_hw = loading.load_coco_img_np(
            coco_gt=self.coco_gt, img_id=img_id, coco_img_path=self.coco_img_root
        )

        coco_ann = loading.load_coco_ann(
            coco_gt=self.coco_gt,
            img_id=img_id,
        )

        sample_np = conversion.coco_ann_to_sample_np(
            coco_ann=coco_ann,
            img_hw=img_hw,
            min_visibility=self.min_visibility,
            coco_dataset=self.coco_dataset,
            msk=self.msk
        )
        img_np = random_photometric_distortion(img_np=img_np, p=self.photo_transform_p)

        sample_np = transform_resize_sample_np(
            sample_np=sample_np | {"img_np": img_np},
            size=self.img_base_size,
            mask_scaling=self.mask_scaling,
        )
        if self.filter_empty:
            sample_np = filter_empty_annotations_sample_np(
                sample_np=sample_np, min_msk_size=1
            )


        if self.random_zoom_p > 0:
            sample_np = transform_random_zoomout_sample_np(
                sample_np=sample_np,
                size=self.img_base_size,
                msk_scale=self.mask_scaling,
                p=self.random_zoom_p,
                min_size=self.min_visibility,
            )
            if self.filter_empty:
                sample_np = filter_empty_annotations_sample_np(
                    sample_np=sample_np, min_msk_size=1
                )


        if self.random_crop_p > 0:
            sample_np = transform_random_crop_sample_np(
                sample_np=sample_np,
                msk_scale=self.mask_scaling,
                p=self.random_crop_p,
                min_size=self.min_visibility,
            )
            sample_np = transform_resize_sample_np(
                sample_np=sample_np,
                size=self.img_base_size,
                mask_scaling=self.mask_scaling,
            )
            
            if self.filter_empty:
                sample_np = filter_empty_annotations_sample_np(
                    sample_np=sample_np, min_msk_size=1
                )

        sample_np = transform_top_down_flip_sample_np(
            sample_np=sample_np, p=self.td_flip_p
        )
        sample_np = transform_left_right_flip_sample_np(
            sample_np=sample_np, p=self.lr_flip_p
        )

        sample_np =  transform_scale_sample_np(sample_np=sample_np) 
        if self.filter_empty:
            return filter_empty_annotations_sample_np(
                    sample_np=sample_np, min_msk_size=1
                )
        return sample_np

    def __len__(self) -> int:
        return len(self.coco_id)
