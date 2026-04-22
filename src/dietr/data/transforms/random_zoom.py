"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import numpy as np
import cv2
import random


def transform_random_zoomout_sample_np(
    sample_np: dict[str, np.ndarray],
    size: tuple[int],
    interpolation: int = cv2.INTER_LINEAR,
    msk_scale: int = 1,
    zoom_range: tuple[float, float] = (0.5, 0.9),
    pad_value: int = 0,
    min_size: int = 2,
    p: float = 0.5,
) -> dict[str, np.ndarray]:
    assert "img_np" in sample_np
    assert "box_np" in sample_np
    assert "cls_np" in sample_np

    if random.random() > p:
        return sample_np

    img_h, img_w = size
    new_sample_np = dict()

    if len(sample_np["box_np"]) == 0:
        return {
            "img_np": np.full(
                (img_h, img_w, 3), pad_value, dtype=sample_np["img_np"].dtype
            ),
            "box_np": np.array([]).reshape(-1, 4),
            "msk_np": np.array([]).reshape(-1, img_h, img_w),
            "crp_np": np.array([]).reshape(-1, img_h, img_w),
            "cls_np": np.array([]).reshape(-1, 1),
        }

    scale = random.uniform(*zoom_range)
    new_w, new_h = int(img_w * scale), int(img_h * scale)

    resized_img_np = cv2.resize(
        sample_np["img_np"], (new_w, new_h), interpolation=interpolation
    )
    canvas = np.full((img_h, img_w, 3), pad_value, dtype=sample_np["img_np"].dtype)

    x_offset = random.randint(0, img_w - new_w)
    y_offset = random.randint(0, img_h - new_h)

    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_img_np

    box_np = sample_np["box_np"].astype(np.float32)

    box_np[:, 0] = (
        box_np[:, 0] * (new_w / sample_np["img_np"].shape[1])
        + x_offset / sample_np["img_np"].shape[1]
    )  # cx
    box_np[:, 1] = (
        box_np[:, 1] * (new_h / sample_np["img_np"].shape[0])
        + y_offset / sample_np["img_np"].shape[0]
    )  # cy
    box_np[:, 2] = box_np[:, 2] * (new_w / sample_np["img_np"].shape[1])  # w
    box_np[:, 3] = box_np[:, 3] * (new_h / sample_np["img_np"].shape[0])  # h

    # filter boxes by minimum size (on w,h directly)
    keep = (box_np[:, 2] >= min_size) & (box_np[:, 3] >= min_size)

    if "msk_np" in sample_np:
        msk_new = []
        crp_new = []
        for i in range(len(sample_np["msk_np"])):
            if not keep[i]:
                continue
            msk_resized = cv2.resize(
                sample_np["msk_np"][i],
                (new_w // msk_scale, new_h // msk_scale),
                interpolation=cv2.INTER_NEAREST,
            )
            crp_resized = cv2.resize(
                sample_np["crp_np"][i].astype(np.uint8),
                (new_w // msk_scale, new_h // msk_scale),
                interpolation=cv2.INTER_NEAREST,
            )
            crp_canvas = np.zeros(
                (img_h // msk_scale, img_w // msk_scale), dtype=crp_resized.dtype
            )
            msk_canvas = np.zeros(
                (img_h // msk_scale, img_w // msk_scale), dtype=msk_resized.dtype
            )
            msk_canvas[
                y_offset // msk_scale : y_offset // msk_scale + msk_resized.shape[0],
                x_offset // msk_scale : x_offset // msk_scale + msk_resized.shape[1],
            ] = msk_resized

            crp_canvas[
                y_offset // msk_scale : y_offset // msk_scale + crp_resized.shape[0],
                x_offset // msk_scale : x_offset // msk_scale + crp_resized.shape[1],
            ] = crp_resized

            msk_new.append(msk_canvas)
            crp_new.append(crp_canvas)

        new_sample_np["msk_np"] = (
            np.stack(msk_new)
            if msk_new
            else np.zeros((0, img_h // msk_scale, img_w // msk_scale))
        )
        new_sample_np["crp_np"] = (
            np.stack(crp_new)
            if crp_new
            else np.zeros((0, img_h // msk_scale, img_w // msk_scale))
        )

    new_sample_np["img_np"] = canvas
    new_sample_np["box_np"] = box_np[keep]
    new_sample_np["cls_np"] = sample_np["cls_np"][keep]

    return new_sample_np
