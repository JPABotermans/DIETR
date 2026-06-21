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
import matplotlib.pyplot as plt
import yaml
from dietr.tools.plot_coco import plot_coco_anns
from dietr.tools.training import setup_dietr
from dietr.data.transforms.base import transform_scale_img_np
from dietr.data.dataloaders import get_val_dataloader
from dietr.tools.validation import validate, decode_y_prd_batch


class DIETR:
    def __init__(self, conf_pth: str, ckpt_pth: str = None, device: str = "cuda:0") -> None:
        with open(conf_pth, "r") as file:
            self.config = yaml.safe_load(file)
        self.device = device
        self.dietr = setup_dietr(config=self.config, device=self.device, from_scratch=False)

    def predict_on_img_tt(self, img_tt: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad() and torch.amp.autocast(
            device_type=self.device, dtype=torch.float16
        ):
            y_prd_batch = self.dietr(x_trn_batch=img_tt)
        return y_prd_batch

    def predict_on_img_np(
        self,
        img_np: np.ndarray,
        val_img_base_size: tuple[int, int],
        plot: bool = True,
        cnf_threshold: float = 0.5,
        figsize: int = 10,
    ) -> list[any]:
        if plot:
            org_img_np = img_np.copy()
        img_tt, info_batch = self.preprocess_img_np(
            img_np=img_np, val_img_base_size=val_img_base_size, device=self.device
        )
        y_prd_batch = self.predict_on_img_tt(img_tt=img_tt)
        coco_results = decode_y_prd_batch(
            y_prd_batch=y_prd_batch,
            info_batch=info_batch,
            x_trn_hw=img_tt.shape[-2:],
            cnf_threshold=cnf_threshold,
            n_cls=self.config["n_cls"],
            n_predictions=self.config["val_n_predictions"],
            coco_dataset=self.config["coco_dataset"],
            device=self.device,
        )

        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(figsize, figsize))
            axs[0].imshow(org_img_np)
            axs[1].imshow(org_img_np)
            plot_coco_anns(coco_results, img_hw=info_batch["img_hw"][0], ax=axs[1])
            _ = [ax.axis("off") for ax in axs]
            fig.show()
        return coco_results

    def predict_on_file(
        self,
        img_pth: str,
        plot: bool = True,
        cnf_threshold: float = 0.5,
        val_img_base_size: tuple[int, int] = [640, 640],
    ) -> None:
        img_np = np.ascontiguousarray(cv2.imread(img_pth)[..., ::-1])
        return self.predict_on_img_np(
            img_np=img_np,
            plot=plot,
            cnf_threshold=cnf_threshold,
            val_img_base_size=val_img_base_size,
        )

    def validate_on_coco_dataset(self, cnf_threshold: float):
        val_dataloader = get_val_dataloader(config=self.config)
        return validate(
            dietr=self.dietr,
            val_dataloader=val_dataloader,
            device=self.device,
            cnf_threshold=cnf_threshold,
            n_cls=self.config["n_cls"],
            n_predictions=self.config["val_n_predictions"],
            coco_dataset=self.config["coco_dataset"],
        )

    @staticmethod
    def preprocess_img_np(
        img_np: np.ndarray, val_img_base_size: tuple[int, int], device: str
    ) -> tuple[np.ndarray, dict[str, any]]:
        img_np = transform_scale_img_np(img_np=img_np)
        info_batch = {
            "pad_hw": [[0, 0]],
            "img_id": [0],
            "img_hw": [img_np.shape[:2]],
            "img_wh": [img_np.shape[:2][::-1]],
        }
        img_np = cv2.resize(img_np, dsize=val_img_base_size, interpolation=0)
        img_tt = torch.from_numpy(img_np.transpose(2, 0, 1))[None, ...].to(device)
        return img_tt, info_batch
