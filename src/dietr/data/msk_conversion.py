"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import pycocotools
import numpy as np
import torch
import cv2

def msk_polygon_to_msk_np(
    polygon: list[list[int]],
    img_hw: tuple[int, int],
) -> np.ndarray:
    rle = pycocotools.mask.frPyObjects(polygon, h=img_hw[0], w=img_hw[1])
    rle = pycocotools.mask.merge(rle)
    return pycocotools.mask.decode(rle)


def msk_rle_to_msk_np(rle: dict[str, str]) -> np.ndarray:
    return pycocotools.mask.decode(rle)


def msk_rle_to_compressed_rle(coco_ann: dict) -> dict[str, tuple[int, int] | str]:
    rle = {"counts": coco_ann["counts"], "size": coco_ann["size"]}

    height, width = rle["size"]

    compressed_rle = pycocotools.mask.frPyObjects([rle], h=height, w=width)[0]

    return {"counts": compressed_rle["counts"], "size": compressed_rle["size"]}


def msk_np_to_msk_crp_np(msk_np: np.ndarray) -> np.ndarray:
    """Given a msk_np, returns a msk_crp_np, which is a rectangle projetion of the mask.
    This msk_crp_np is used to stabilize training.
    """
    msk_coords = np.where(msk_np)

    if len(msk_coords[0]) == 0:
        return np.zeros_like(msk_np, dtype=int)

    min_row = max(msk_coords[0].min() - 1, 0)
    max_row = min(msk_coords[0].max() + 2, msk_np.shape[0])

    min_col = max(msk_coords[1].min() - 1, 0)
    max_col = min(msk_coords[1].max() + 2, msk_np.shape[1])

    # Create the crp_np
    crp_np = np.zeros_like(msk_np, dtype=bool)
    crp_np[min_row:max_row, min_col:max_col] = 1

    return crp_np.astype(np.bool_)



def msk_np_crop_padding(msk_np: np.ndarray, pad_hw: torch.Tensor, msk_scaling: int = 4) -> np.ndarray:
    """Given a numpy array of predicted masks, and the padding values (number of pixes that are padded to both sides).
    Returns the cropped mask.
    """
    if pad_hw[0] != 0 and pad_hw[1] != 0:
        msk_hw = msk_np.shape[-2:]
        msk_np = msk_np[
            ...,
            pad_hw[0] // msk_scaling : msk_hw[0] - pad_hw[0] // msk_scaling,
            pad_hw[1] // msk_scaling : msk_hw[1] - pad_hw[1] // msk_scaling,
        ]
    return msk_np

def msk_np_crop_box_np(msk_np: np.ndarray, box_np: np.ndarray, org_wh: torch.Tensor) -> np.ndarray:
    "Given a numpy array of predicted masks, and its corresponding bounding box (xyxy format) crops the mask"
    crp_np = np.ones_like(msk_np, dtype=bool)

    crp_np[
        box_np.clip(0, org_wh[1]).astype(np.uint)[1] : box_np.clip(0, org_wh[1]).astype(
            np.uint
        )[3],
        box_np.clip(0, org_wh[0]).astype(np.uint)[0] : box_np.clip(0, org_wh[0]).astype(
            np.uint
        )[2],
    ] = False
    msk_np[crp_np] = 0
    return msk_np

def msk_np_to_rle_encoded(msk_np: np.ndarray, ) -> dict[str, list[int] | str]:
    from pycocotools.mask import encode
    rle = encode(np.asfortranarray(msk_np.astype(bool)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def msk_np_rescale_and_crop_and_encode(
    msk_np: np.ndarray, box_np: np.ndarray, org_wh: torch.Tensor, pad_hw: torch.Tensor,
) -> dict[str, list[int] | str]:
    """Given a numpy array of precdicted masks:
        1. Crops the masks first using the padding values (for tta).
        2. Resizes the masks to the original image height and width.
        3. Crops the masks using the bounding boxes. 
        4. Thresholds the values to obtain a fotran type boolean array, ready to be encoded.
    
    Returns something like this    
        >>> msk_coco_encoded = {
        >>>  'segmentation': {'size': [560, 400],
        >>>  'counts': 'VRe43\\a02O101O000001N1000O102N010O100O0100010N00O3L`Ri1'}
        >>> }
    
    """
    

    msk_np = msk_np_crop_padding(msk_np=msk_np, pad_hw=pad_hw)

    msk_np = cv2.resize(
        msk_np.astype(float),
        dsize=org_wh,
        interpolation=cv2.INTER_LINEAR,
    )

    msk_np = msk_np_crop_box_np(msk_np=msk_np, box_np=box_np, org_wh=org_wh)
    msk_np = msk_np >= 0.5
    return msk_np_to_rle_encoded(msk_np=msk_np)

def coco_seg_to_msk_np(coco_seg: dict, img_hw: tuple[int, int]) -> np.ndarray:
    if not isinstance(coco_seg, dict):
        msk_np = msk_polygon_to_msk_np(polygon=coco_seg, img_hw=img_hw)
    else:
        if "counts" in coco_seg:
            msk_np = msk_rle_to_msk_np(msk_rle_to_compressed_rle(coco_ann=coco_seg))
        else:
            msk_np = msk_rle_to_msk_np(rle=coco_seg)
    return msk_np.astype(np.bool_)