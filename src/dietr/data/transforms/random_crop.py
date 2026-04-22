"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import numpy as np
import random


def transform_random_crop_sample_np(
    sample_np: dict[str, np.ndarray],
    msk_scale: int = 4,
    min_size: int = 2,
    p: float = 0.5,
) -> dict[str, np.ndarray]:
    assert "img_np" in sample_np
    assert "box_np" in sample_np
    assert "cls_np" in sample_np

    if random.random() > p:
        return sample_np

    img_np = sample_np["img_np"]
    h_img, w_img = img_np.shape[:2]
    h_new, w_new = random.randint(h_img // 2, h_img), random.randint(w_img // 2, w_img)

    x0 = random.randint(0, w_img - w_new)
    y0 = random.randint(0, h_img - h_new)
    x1, y1 = x0 + w_new, y0 + h_new

    new_img_np = img_np[y0:y1, x0:x1]
    new_sample_np = {"img_np": new_img_np}

    box_np = sample_np["box_np"].astype(np.float32)
    cx, cy, bw, bh = box_np[:, 0], box_np[:, 1], box_np[:, 2], box_np[:, 3]

    x1b = cx - bw / 2
    y1b = cy - bh / 2
    x2b = cx + bw / 2
    y2b = cy + bh / 2

    x1b *= w_img
    x2b *= w_img
    y1b *= h_img
    y2b *= h_img

    x1b = (x1b - x0) / (w_new)
    x2b = (x2b - x0) / (w_new)
    y1b = (y1b - y0) / (h_new)
    y2b = (y2b - y0) / (h_new)

    x1b = np.clip(x1b, 0, w_new)
    x2b = np.clip(x2b, 0, w_new)
    y1b = np.clip(y1b, 0, h_new)
    y2b = np.clip(y2b, 0, h_new)

    bw_new = x2b - x1b
    bh_new = y2b - y1b
    cx_new = (x1b + x2b) / 2
    cy_new = (y1b + y2b) / 2

    keep = (bw_new >= min_size) & (bh_new >= min_size)

    new_sample_np["box_np"] = np.stack(
        [cx_new[keep], cy_new[keep], bw_new[keep], bh_new[keep]], axis=1
    )
    new_sample_np["cls_np"] = sample_np["cls_np"][keep]

    if "msk_np" in sample_np:
        new_msk = []
        new_crp = []
        for i in range(len(sample_np["msk_np"])):
            if not keep[i]:
                continue
            new_msk_np = sample_np["msk_np"][i][
                y0 // msk_scale : y1 // msk_scale, x0 // msk_scale : x1 // msk_scale
            ]
            new_crp_np = sample_np["crp_np"][i][
                y0 // msk_scale : y1 // msk_scale, x0 // msk_scale : x1 // msk_scale
            ]

            new_msk.append(new_msk_np)
            new_crp.append(new_crp_np)

        new_sample_np["msk_np"] = (
            np.stack(new_msk)
            if new_msk
            else np.zeros((0, h_new // msk_scale, w_new // msk_scale))
        )
        new_sample_np["crp_np"] = (
            np.stack(new_crp)
            if new_crp
            else np.zeros((0, h_new // msk_scale, w_new // msk_scale))
        )

    return new_sample_np
