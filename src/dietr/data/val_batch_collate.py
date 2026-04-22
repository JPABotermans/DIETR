"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""

import torch
import cv2
import numpy as np


class ValBatchImageCollateFunction:
    def __init__(
        self,
        base_size: tuple[int, int] = (640, 640),
        val_sizes: list[tuple[int]] = [(640, 640), (960, 960)],
        pad_value: int = 0.0,
        dtype=np.float32,
    ) -> None:
        """Collate function, responsible for resizing and padding
        the x_trn_batch, y_trn_batch, info format.

        Args:
            base_size (int):            The base size of the x_trn_batch H and W dimensions.
            base_size_repeat (int):     What should be the change to select an image of the base size.
            tta (bool):                 Whether to apply tta on the samples.
        """

        super().__init__()
        self.base_size = base_size
        self.val_sizes = val_sizes
        self.pad_value = pad_value
        self.dtype = dtype

    def __call__(
        self, samples_np: list[dict[str, np.ndarray]]
    ) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
        """Given a list of samples, collate all samples to the correct size and preparing the info dict for this batch."""
        img_hw = list()
        img_wh = list()
        img_id = list()
        pad_hw = list()
        new_imgs = list()

        for sample_np in samples_np:
            for val_size in self.val_sizes:
                img_np = cv2.resize(sample_np["img_np"], val_size)
                canvas_np = np.zeros(
                    (self.base_size[0], self.base_size[1], 3), dtype=self.dtype
                )
                canvas_np[..., :] = np.array(
                    [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]
                )

                pad_w = self.base_size[0] - val_size[0]
                pad_h = self.base_size[1] - val_size[1]
                canvas_np[
                    pad_h // 2 : val_size[1] + pad_h // 2,
                    pad_w // 2 : val_size[0] + pad_w // 2,
                ] = img_np
                new_imgs.append(canvas_np.transpose(2, 0, 1).astype(self.dtype))

                pad_hw.append((pad_h // 2, pad_w // 2))
                img_id.append(sample_np["img_id"])
                img_hw.append(sample_np["img_hw"])
                img_wh.append(sample_np["img_hw"][::-1])

        x_trn_batch = torch.stack(
            [torch.from_numpy(new_img_np).contiguous() for new_img_np in new_imgs]
        )

        return x_trn_batch, {
            "img_hw": img_hw,
            "img_wh": img_wh,
            "img_id": img_id,
            "pad_hw": pad_hw,
        }
