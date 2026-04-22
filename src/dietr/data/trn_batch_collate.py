"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import torch
import numpy as np


class BatchImageCollateFunction:
    def __init__(
        self,
        base_size: int,
        multi_input_size: bool,
        double_multi_input_size: bool,
        mask_scaling: int,
        dtype: torch.dtype,
        multi_scale_p: float = 1.0,
        both_sides: bool = True,
    ) -> None:
        """Collate function, responsible for changing the sizes for all different images in a batch and to put them into
        the x_trn_batch, y_trn_batch, info format.

        Args:
            base_size (int):            The base size of the x_trn_batch H and W dimensions.
            base_size_repeat (int):     What should be the change to select an image of the base size.
            multi_input_size (bool):    Whether we should do multi-input training. If True it will return images and masks in a different size for each different mask.
            double_multi_input_size (bool):    Whether the height and the width dimension sould be similar.
            mask_scaling: (int):         Scale factor for the masks.
        """

        super().__init__()
        self.base_size = base_size
        self.input_sizes = self.generate_input_sizes(
            base_size=base_size[0], both_sides=both_sides
        )
        self.multi_input_size = multi_input_size
        self.double_multi_input_size = double_multi_input_size
        self.mask_scaling = mask_scaling
        self.dtype = dtype
        self.multi_scale_p = multi_scale_p

    @staticmethod
    def generate_input_sizes(base_size: int, both_sides: bool = True) -> list[int]:
        scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
        sizes = [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
        sizes += [base_size]
        if both_sides:
            sizes += [
                int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)
            ]
        return sizes

    def get_random_size(
        self,
    ) -> np.ndarray[int]:
        return (
            np.random.choice(self.input_sizes, size=2)
            if not self.double_multi_input_size
            else np.random.choice(self.input_sizes, size=1).repeat(2)
        ).tolist()

    def __call__(
        self, samples_np: list[dict[str, np.ndarray]]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Collates a batch of samples to something the model can train on.
        Either processes all images in the batch to get the same (random) H and W dimensions, or processes them to get different.


        Args:
            samples_np (list[dict[str, np.ndarray]]):
            List of processed samples in numpy array form.

        Returns:
            x_trn_batch: torch.Tensor:
                Batch of images
            y_trn_batch: dict[str, list[torch.Tensor]]:
                Targets to train on, contains:
                    box: list[torch.Tensor]
                    msk: list[torch.Tensor]
                    cls: list[torch.Tensor]
                    crp: list[torch.Tensor]
        """
        x_trn_batch = torch.stack(
            [
                torch.from_numpy(
                    sample_np["img_np"].transpose(2, 0, 1).astype(self.dtype)
                ).contiguous()
                for sample_np in samples_np
            ]
        )
        box = [
            torch.from_numpy(sample_np["box_np"].astype(self.dtype)).contiguous()
            for sample_np in samples_np
        ]
        cls = [
            torch.Tensor(sample_np["cls_np"].astype(self.dtype))
            .reshape(-1)
            .contiguous()
            for sample_np in samples_np
        ]

        y_trn_batch = {"box": box, "cls": cls}
        if "msk_np" in samples_np[0]:
            y_trn_batch["msk"] = [
                torch.from_numpy(sample_np["msk_np"].astype(bool)).contiguous()
                for sample_np in samples_np
            ]
            y_trn_batch["crp"] = [
                torch.from_numpy(sample_np["crp_np"].astype(bool)).contiguous()
                for sample_np in samples_np
            ]

        if self.multi_input_size and np.random.rand() < self.multi_scale_p:
            random_size = self.get_random_size()
            x_trn_batch = torch.nn.functional.interpolate(
                input=x_trn_batch, size=random_size, mode="nearest"
            )
            if "msk_np" in samples_np[0]:
                for key in ["msk", "crp"]:
                    for i in range(len(x_trn_batch)):
                        if len(y_trn_batch[key][i]) == 0:
                            y_trn_batch[key][i] = torch.from_numpy(
                                np.array([]).reshape(
                                    -1,
                                    random_size[0] // self.mask_scaling,
                                    random_size[1] // self.mask_scaling,
                                )
                            ).contiguous()
                        else:
                            y_trn_batch[key][i] = (
                                torch.nn.functional.interpolate(
                                    input=y_trn_batch[key][i][None, ...].to(float),
                                    size=(
                                        random_size[0] // self.mask_scaling,
                                        random_size[1] // self.mask_scaling,
                                    ),
                                    mode="nearest",
                                )[0]
                                .to(torch.bool)
                                .contiguous()
                            )

        return x_trn_batch, y_trn_batch
