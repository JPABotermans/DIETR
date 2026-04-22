"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import cv2
import numpy as np


def filter_coco_gt_remove_empty(coco_gt: dict) -> dict:
    """
    Remove images without annotations from COCO GT.
    """
    ann_ids = {ann["image_id"] for ann in coco_gt["annotations"]}

    img = [img for img in coco_gt["images"] if img["id"] in ann_ids]

    coco_anns = [ann for ann in coco_gt["annotations"] if ann["image_id"] in ann_ids]

    return {
        "images": img,
        "annotations": coco_anns,
        "categories": coco_gt["categories"],
    }


def transform_resize_msk_np(
    sample_np: dict[str, np.ndarray],
    size: tuple[int],
    interpolation: int = cv2.INTER_LINEAR,
    mask_scaling: int = 4,
) -> dict[str, np.ndarray]:
    new_sample_np = dict()
    if "msk_np" not in sample_np:
        return new_sample_np
    new_sample_np["msk_np"] = np.stack(
        [
            cv2.resize(
                msk_np.astype(np.uint8),
                dsize=(size[0] // mask_scaling, size[1] // mask_scaling),
                interpolation=interpolation,
            )
            for msk_np in sample_np["msk_np"]
        ]
    )
    new_sample_np["crp_np"] = np.stack(
        [
            cv2.resize(
                crp_np.astype(np.uint8),
                dsize=(size[0] // mask_scaling, size[1] // mask_scaling),
                interpolation=interpolation,
            )
            for crp_np in sample_np["crp_np"]
        ]
    )
    return new_sample_np


def transform_resize_sample_np(
    sample_np: dict[str, np.ndarray],
    size: tuple[int],
    interpolation: int = cv2.INTER_LINEAR,
    mask_scaling: int = 4,
) -> dict[str, np.ndarray]:
    assert "img_np" in sample_np
    assert "box_np" in sample_np
    assert "cls_np" in sample_np

    new_sample_np = dict()
    if len(sample_np["box_np"]) == 0:
        new_sample_np |= {
            "img_np": cv2.resize(
                sample_np["img_np"], dsize=size, interpolation=interpolation
            ),
            "box_np": np.array([]).reshape(-1, 4),
            "cls_np": np.array([]).reshape(-1, 1),
        }
        if "msk_np" in sample_np:
            new_sample_np |= {
                "msk_np": np.array([]).reshape(
                    -1, size[0] // mask_scaling, size[1] // mask_scaling
                ),
                "crp_np": np.array([]).reshape(
                    -1, size[0] // mask_scaling, size[1] // mask_scaling
                ),
            }
        return new_sample_np

    new_sample_np["box_np"] = sample_np["box_np"]
    new_sample_np["cls_np"] = sample_np["cls_np"]

    new_sample_np["img_np"] = cv2.resize(
        sample_np["img_np"], dsize=size, interpolation=interpolation
    )

    return new_sample_np | transform_resize_msk_np(
        sample_np=sample_np,
        size=size,
        interpolation=interpolation,
        mask_scaling=mask_scaling,
    )


def get_keep_indices(sample_np: dict[str, np.ndarray], min_msk_size: int) -> list[int]:
    keep_indices = []
    if "msk_np" in sample_np:
        for i in range(len(sample_np["msk_np"])):
            if (
                np.sum(sample_np["msk_np"][i]) >= min_msk_size
                and np.sum(sample_np["crp_np"][i]) >= min_msk_size
            ):
                keep_indices.append(i)
    else:
        for i in range(len(sample_np["box_np"])):
            if sample_np["box_np"][i][2] > 0.0 and sample_np["box_np"][i][3] > 0.0:
                keep_indices.append(i)
    return keep_indices


def filter_empty_annotations_sample_np(
    sample_np: dict[str, np.ndarray],
    min_msk_size: int = 5,
) -> dict[str, np.ndarray]:
    assert "img_np" in sample_np
    assert "box_np" in sample_np
    assert "cls_np" in sample_np

    if len(sample_np["box_np"]) == 0:
        if "msk_np" in sample_np:
            return {
                "img_np": sample_np["img_np"],
                "box_np": np.array([]).reshape(-1, 4),
                "cls_np": np.array([]).reshape(-1, 1),
                "msk_np": np.array([]).reshape(-1, *sample_np["msk_np"].shape[1:]),
                "crp_np": np.array([]).reshape(-1, *sample_np["crp_np"].shape[1:]),
            }
        return {
            "img_np": sample_np["img_np"],
            "box_np": np.array([]).reshape(-1, 4),
            "cls_np": np.array([]).reshape(-1, 1),
        }

    keep_indices = get_keep_indices(sample_np=sample_np, min_msk_size=min_msk_size)

    if len(keep_indices) != 0:
        new_sample_np = {
            "img_np": sample_np["img_np"],
            "box_np": sample_np["box_np"][keep_indices],
            "cls_np": sample_np["cls_np"][keep_indices],
        }
        if "msk_np" in sample_np:
            new_sample_np |= {
                "msk_np": sample_np["msk_np"][keep_indices],
                "crp_np": sample_np["crp_np"][keep_indices],
            }
    else:
        new_sample_np = {
            "img_np": sample_np["img_np"],
            "box_np": np.array([]).reshape(-1, 4),
            "cls_np": np.array([]).reshape(-1, 1),
        }
        if "msk_np" in sample_np:
            new_sample_np |= {
                "msk_np": np.array([]).reshape(0, *sample_np["msk_np"].shape[1:]),
                "crp_np": np.array([]).reshape(0, *sample_np["crp_np"].shape[1:]),
            }

    return new_sample_np


def transform_scale_sample_np(
    sample_np: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    sample_np["img_np"] = transform_scale_img_np(img_np=sample_np["img_np"])
    return sample_np


def transform_scale_img_np(img_np: np.ndarray) -> np.ndarray:
    img_np = img_np.astype(np.float16)
    img_np /= 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    return (img_np - mean) / std


def transform_top_down_flip_sample_np(
    sample_np: dict[str, np.ndarray],
    p: float = 0.25,
) -> dict[str, np.ndarray]:
    if np.random.rand() > (1 - p) and len(sample_np["box_np"]) > 0:
        sample_np["box_np"][:, 1] = 1 - sample_np["box_np"][:, 1]
        sample_np["img_np"] = sample_np["img_np"][::-1, ...]
        if "msk_np" in sample_np:
            sample_np["msk_np"] = sample_np["msk_np"][..., ::-1, :]
            sample_np["crp_np"] = sample_np["crp_np"][..., ::-1, :]
    return sample_np


def transform_left_right_flip_sample_np(
    sample_np: dict[str, np.ndarray],
    p: float = 0.25,
) -> dict[str, np.ndarray]:
    if np.random.rand() > (1 - p) and len(sample_np["box_np"]) > 0:
        sample_np["box_np"][:, 0] = 1 - sample_np["box_np"][:, 0]
        sample_np["img_np"] = sample_np["img_np"][:, ::-1, ...]
        if "msk_np" in sample_np:
            sample_np["msk_np"] = sample_np["msk_np"][..., :, ::-1]
            sample_np["crp_np"] = sample_np["crp_np"][..., :, ::-1]
    return sample_np
