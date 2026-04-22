"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import numpy as np
from pycocotools.coco import COCO
from dietr.data import loading
from dietr.data import conversion
from dietr.data.transforms.base import (
    transform_resize_sample_np,
)


def get_samples_np(
    coco_gt: COCO,
    coco_id,
    min_visibility: float,
    img_size: int,
    coco_img_pth: str,
    coco_dataset: bool,
):
    samples_np = []
    for i in range(4):
        index = np.random.randint(len(coco_id))
        img_id = coco_id[index]
        img_np, img_hw = loading.load_coco_img_np(
            coco_gt=coco_gt, img_id=img_id, coco_img_path=coco_img_pth
        )
        coco_ann = loading.load_coco_ann(
            coco_gt=coco_gt,
            img_id=img_id,
        )
        ann_np = conversion.coco_ann_to_sample_np(
            coco_ann=coco_ann,
            img_hw=img_hw,
            min_visibility=min_visibility,
            coco_dataset=coco_dataset,
        )

        samples_np.append(
            transform_resize_sample_np(
                sample_np=ann_np | {"img_np": img_np},
                size=[img_size[0] // 2, img_size[1] // 2],
                mask_scaling=4,
            )
        )
    return samples_np


def combine_mosaic_box_np(samples_np: list[dict[str, np.ndarray]]) -> np.ndarray:
    samples_np[0]["box_np"][:, 0] = samples_np[0]["box_np"][:, 0] / 2
    samples_np[0]["box_np"][:, 1] = samples_np[0]["box_np"][:, 1] / 2

    samples_np[1]["box_np"][:, 0] = samples_np[1]["box_np"][:, 0] / 2 + 0.5
    samples_np[1]["box_np"][:, 1] = samples_np[1]["box_np"][:, 1] / 2

    samples_np[2]["box_np"][:, 0] = samples_np[2]["box_np"][:, 0] / 2
    samples_np[2]["box_np"][:, 1] = samples_np[2]["box_np"][:, 1] / 2 + 0.5

    samples_np[3]["box_np"][:, 0] = samples_np[3]["box_np"][:, 0] / 2 + 0.5
    samples_np[3]["box_np"][:, 1] = samples_np[3]["box_np"][:, 1] / 2 + 0.5

    all_box_np = np.vstack([sample["box_np"] for sample in samples_np])
    all_box_np[..., 2:] /= 2
    return all_box_np


def combine_mosaic_img_np(samples_np: list[dict[str, np.ndarray]]) -> np.ndarray:
    img_up_np = np.hstack([samples_np[0]["img_np"], samples_np[1]["img_np"]])
    img_down_np = np.hstack([samples_np[2]["img_np"], samples_np[3]["img_np"]])

    return np.vstack([img_up_np, img_down_np])


def combine_mosaic_msk_np(
    samples_np,
    img_size: int = 640,
):
    n_samples = sum([len(sample["box_np"]) for sample in samples_np])
    combined_msk_np = np.zeros(shape=(n_samples, img_size // 4, img_size // 4))
    combined_msk_np[
        : len(samples_np[0]["msk_np"]), : img_size // 8, : img_size // 8
    ] = samples_np[0]["msk_np"]
    combined_msk_np[
        len(samples_np[0]["msk_np"]) : len(samples_np[0]["msk_np"])
        + len(samples_np[1]["msk_np"]),
        : img_size // 8,
        img_size // 8 :,
    ] = samples_np[1]["msk_np"]
    combined_msk_np[
        len(samples_np[0]["msk_np"]) + len(samples_np[1]["msk_np"]) : len(
            samples_np[0]["msk_np"]
        )
        + len(samples_np[1]["msk_np"])
        + len(samples_np[2]["msk_np"]),
        img_size // 8 :,
        : img_size // 8,
    ] = samples_np[2]["msk_np"]
    combined_msk_np[
        len(samples_np[0]["msk_np"])
        + len(samples_np[1]["msk_np"])
        + len(samples_np[2]["msk_np"]) :,
        img_size // 8 :,
        img_size // 8 :,
    ] = samples_np[3]["msk_np"]

    return combined_msk_np


def combine_mosaic_crp_np(
    samples_np,
    img_size: int,
):
    n_samples = sum([len(sample["box_np"]) for sample in samples_np])
    combined_crp_np = np.zeros(shape=(n_samples, img_size // 4, img_size // 4))
    combined_crp_np[
        : len(samples_np[0]["crp_np"]), : img_size // 8, : img_size // 8
    ] = samples_np[0]["crp_np"]
    combined_crp_np[
        len(samples_np[0]["crp_np"]) : len(samples_np[0]["crp_np"])
        + len(samples_np[1]["crp_np"]),
        : img_size // 8,
        img_size // 8 :,
    ] = samples_np[1]["crp_np"]
    combined_crp_np[
        len(samples_np[0]["crp_np"]) + len(samples_np[1]["crp_np"]) : len(
            samples_np[0]["crp_np"]
        )
        + len(samples_np[1]["crp_np"])
        + len(samples_np[2]["crp_np"]),
        img_size // 8 :,
        : img_size // 8,
    ] = samples_np[2]["crp_np"]
    combined_crp_np[
        len(samples_np[0]["crp_np"])
        + len(samples_np[1]["crp_np"])
        + len(samples_np[2]["crp_np"]) :,
        img_size // 8 :,
        img_size // 8 :,
    ] = samples_np[3]["crp_np"]

    return combined_crp_np


def moisaic_augmentation(
    coco_gt: COCO,
    coco_id: list,
    min_visibility: float,
    img_size: int,
    coco_img_pth: str,
    coco_dataset: bool
) -> dict[str, np.ndarray]:
    samples_np = get_samples_np(
        coco_gt,
        coco_id,
        min_visibility,
        img_size=img_size,
        coco_img_pth=coco_img_pth,
        coco_dataset=coco_dataset
    )
    box_np = combine_mosaic_box_np(samples_np=samples_np)
    msk_np = combine_mosaic_msk_np(samples_np=samples_np, img_size=img_size[0])
    crp_np = combine_mosaic_crp_np(samples_np=samples_np, img_size=img_size[0])
    img_np = combine_mosaic_img_np(samples_np=samples_np)
    cls_np = np.hstack([sample["cls_np"] for sample in samples_np])

    return {
        "box_np": box_np,
        "msk_np": msk_np,
        "crp_np": crp_np,
        "img_np": img_np,
        "cls_np": cls_np,
    }
