"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import torch
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from dietr.data.trn_dataset import CocoDataset
from dietr.data.val_dataset import CocoValDataset
from dietr.data.trn_batch_collate import BatchImageCollateFunction
from dietr.data.val_batch_collate import ValBatchImageCollateFunction

from collections.abc import Iterator
from typing import Any



def get_val_dataloader(
    config, multi_gpu: bool = False, tta: bool = False
) -> torch.utils.data.DataLoader:
    coco_dataset = CocoValDataset(
        coco_ann_file=config["val_ann_file"],
        coco_img_root=config["val_img_root"],
        img_base_size=config["val_img_base_size"],
    )
    batch_collate = ValBatchImageCollateFunction(
        base_size=config["val_img_base_size"],
        val_sizes=[config["val_img_base_size"]]
        if not tta
        else config["val_tta_img_sizes"],
    )
    return torch.utils.data.DataLoader(
        dataset=coco_dataset,
        batch_size=config["val_batch_size"],
        shuffle=False,
        collate_fn=batch_collate,
        persistent_workers=True,
        pin_memory=True,
        num_workers=config["n_val_workers"],
        sampler=DistributedSampler(coco_dataset, shuffle=False) if multi_gpu else None,
    )

def contains_targets(y_trn_batch: dict[str, list[torch.Tensor]]) -> bool:
    for box in y_trn_batch["box"]:
        if len(box) == 0:
            return False
    return True


def cycle(iterable: Any) -> Iterator[Any]:
    """Create a dataloader-safe cyclical iterator.
    From: https://github.com/pytorch/pytorch/issues/23900#issuecomment-518858050
    """
    iterator = iter(iterable)
    while True:
        try:
            (x_trn_batch, y_trn_batch) =  next(iterator)
            # while not contains_targets(y_trn_batch=y_trn_batch):
            #     (x_trn_batch, y_trn_batch) =  next(iterator)
            yield (x_trn_batch, y_trn_batch)
        except StopIteration:
            iterator = iter(iterable)

def get_trn_dataloader(config, multi_gpu: bool, cycle_dataloader: bool) -> Iterator[Any]:
    coco_dataset = CocoDataset(
        msk=config["msk"],
        coco_ann_file=config["trn_ann_file"],
        coco_img_root=config["trn_img_root"],
        img_base_size=config["trn_img_base_size"],
        min_visibility=config["min_visibility"],
        td_flip_p=config["dataset_dt_flip_p"],
        lr_flip_p=config["dataset_lr_flip_p"],
        mask_scaling=config["mask_scaling"],
        photo_transform_p=config["dataset_photo_transform_p"],
        filter_empty=config["dataset_filter_empy"],
        random_zoom_p=config["dataset_random_zoom_p"],
        random_crop_p=config["dataset_random_crop_p"],
        mosaic_p=config["dataset_mosaic_p"],
        coco_dataset=config["coco_dataset"],
    )
    batch_collate = BatchImageCollateFunction(
        base_size=config["trn_img_base_size"],
        multi_scale_p=config.get("multi_scale_p", 1.0),
        multi_input_size=config["multi_input_size"],
        double_multi_input_size=config["double_multi_input_size"],
        mask_scaling=config["mask_scaling"],
        dtype=np.float32,
        both_sides=config.get("both_sides", True),
    )
    dataloader = torch.utils.data.DataLoader(
            dataset=coco_dataset,
            batch_size=config["trn_batch_size"],
            shuffle=not multi_gpu,
            collate_fn=batch_collate,
            persistent_workers=True,
            pin_memory=True,
            num_workers=config["n_trn_workers"],
            sampler=DistributedSampler(coco_dataset) if multi_gpu else None,
        )
    if cycle_dataloader:
        return cycle(dataloader)
    return dataloader


