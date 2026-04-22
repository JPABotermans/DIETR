"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
Modified from rt-detr (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) lyuwenyu. 
Reference: https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_criterion.py
------------------------------------------------------------------------
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
reference: https://github.com/facebookresearch/detr/blob/main/models/detr.py
------------------------------------------------------------------------
"""
import torch
import itertools
import torch.distributed

from dietr.data.box_conversion import box_yolo_iou, box_yolo_giou
from dietr.tools.distributed import is_dist_available_and_initialized, get_world_size


class DIETRLoss(torch.nn.Module):
    """This class computes the loss for DIETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        matcher: torch.nn.Module,
        weight_dict: dict[str, float],
        losses: list[str],
        n_head_layers: int,
        alpha: float = 0.2,
        gamma: float = 2.0,
        n_cls: int = 80,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.alpha = alpha
        self.gamma = gamma
        self.n_head_layers = n_head_layers

    def loss_cls_vfl(
        self,
        y_prd_batch: dict[str, torch.Tensor],
        y_trn_batch: dict[str, list[torch.Tensor]],
        indices: tuple[torch.Tensor, torch.Tensor],
        n_box: int,
    ):
        idx = self.get_y_trn_permutation(indices)
        y_prd_box = y_prd_batch["box"][idx]
        y_trn_box = torch.cat(
            [y_trn[i] for y_trn, (_, i) in zip(y_trn_batch["box"], indices)], dim=0
        )

        y_prd_cls = y_prd_batch["cls"]
        y_trn_cls = torch.cat(
            [y_trn[i] for y_trn, (_, i) in zip(y_trn_batch["cls"], indices)]
        ).to(torch.int64)
        target_cls = torch.full(
            y_prd_cls.shape[:2],
            self.n_cls,
            dtype=torch.int64,
            device=y_prd_cls.device,
        )
        target_cls[idx] = y_trn_cls
        target = torch.nn.functional.one_hot(target_cls, num_classes=self.n_cls + 1)[
            ..., :-1
        ]

        ious, _ = box_yolo_iou(box1=y_prd_box, box2=y_trn_box)
        target_score_o = torch.zeros_like(target_cls, dtype=y_prd_cls.dtype)
        target_score_o[idx] = torch.diag(ious).detach().to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        prd_score = torch.nn.functional.sigmoid(y_prd_cls).detach()
        weight = self.alpha * prd_score.pow(self.gamma) * (1 - target) + target_score

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            y_prd_cls, target_score, weight=weight, reduction="none"
        )
        return {"cls": loss.mean(1).sum() * y_prd_cls.shape[1] / n_box}

    def loss_box_giou(
        self,
        y_prd_batch: torch.Tensor,
        y_trn_batch: list[torch.Tensor],
        indices: tuple[torch.Tensor, torch.Tensor],
        n_box: int,
    ) -> dict[str, torch.Tensor]:
        idx = self.get_y_trn_permutation(indices)
        y_prd_box = y_prd_batch["box"][idx]
        y_trn_box = torch.cat(
            [y_trn[i] for y_trn, (_, i) in zip(y_trn_batch["box"], indices)], dim=0
        )

        loss_box = torch.nn.functional.l1_loss(y_prd_box, y_trn_box, reduction="none")

        loss_giou = 1 - torch.diag(box_yolo_giou(box1=y_prd_box, box2=y_trn_box))
        return {"box": loss_box.sum() / n_box, "giou": loss_giou.sum() / n_box}

    def loss_msk_dice(
        self,
        y_prd_batch: torch.Tensor,
        y_trn_batch: list[torch.Tensor],
        indices: tuple[torch.Tensor, torch.Tensor],
        n_box: int,
        dice_smooth: float = 1e-6,
    ) -> torch.Tensor:
        """First derives the predicted masks by taking a weight average of the prototype masks.
        Then crops the masks, only take into account the masks inside the boxes (to make training stable).
        Then computes the dice loss, for each example.
        """
        idx = self.get_y_trn_permutation(indices)

        y_prd_msk = y_prd_batch["msk"][idx]
        y_prd_msk = torch.nn.functional.tanh(y_prd_msk)

        y_trn_msk = torch.cat(
            [y_trn[i] for y_trn, (_, i) in zip(y_trn_batch["msk"], indices) if len(y_trn) != 0],
            dim=0,
        )
        y_trn_crp = torch.cat(
            [y_trn[i] for y_trn, (_, i) in zip(y_trn_batch["crp"], indices) if len(y_trn) != 0],
            dim=0,
        )

        predicted_masks = y_prd_msk[..., None, None] * y_prd_batch["masks"][idx[0]]
        predicted_masks = torch.nn.functional.sigmoid(predicted_masks.mean(dim=1))

        predicted_masks = predicted_masks * y_trn_crp

        intersection = torch.sum(predicted_masks * y_trn_msk, dim=(1, 2))
        union = torch.sum(predicted_masks, dim=(1, 2)) + torch.sum(
            y_trn_msk, dim=(1, 2)
        )

        return {
            "msk": (
                1 - (2.0 * intersection + dice_smooth) / (union + dice_smooth)
            ).sum()
            / n_box
        }

    def get_loss(
        self, loss, y_prd_batch, y_trn_batch, indices, n_box: int
    ) -> dict[str, torch.Tensor]:

        return {
            "box": self.loss_box_giou,
            "cls": self.loss_cls_vfl,
            "msk": self.loss_msk_dice,
        }[loss](y_prd_batch, y_trn_batch, indices, n_box)

    @staticmethod
    def get_y_prd_head(
        y_prd_batch: dict[str, torch.Tensor], group: str
    ) -> dict[str, torch.Tensor]:
        y_prd_group = {
            k.replace(group, ""): v for k, v in y_prd_batch.items() if group in k
        }
        if "masks" in y_prd_batch:
            y_prd_group |= {"masks": y_prd_batch["masks"]}
        return y_prd_group

    def forward(
        self,
        y_prd_batch: dict[str, torch.Tensor],
        y_trn_batch: dict[str, list[torch.Tensor]],
    ) -> dict[str, torch.Tensor]:

        n_box = torch.as_tensor(
            [sum([len(y) for y in y_trn_batch["cls"]])],
            dtype=torch.float,
            device=y_trn_batch["cls"][0].device,
        )
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(n_box)

        n_box = torch.clamp(n_box / get_world_size(), min=1).item()

        loss_groups = [f"_{i}" for i in range(self.n_head_layers)]
        loss_groups += ["_enc"]

        indices = {}
        instances = {}
        for head in loss_groups:
            indices[head] = self.matcher(
                y_prd_box=y_prd_batch[f"box{head}"],
                y_prd_cls=y_prd_batch[f"cls{head}"],
                y_trn_box=y_trn_batch["box"],
                y_trn_cls=y_trn_batch["cls"],
            )
            instances[head] = n_box

        if "qdn_meta" in y_prd_batch:
            n_qdn_box = int(n_box * y_prd_batch["qdn_meta"]["n_qdn_group"] * 2)
            qdn_indices = self.get_qdn_matched_indices(
                qdn_meta=y_prd_batch["qdn_meta"], y_trn_batch=y_trn_batch
            )
            loss_groups += [f"_qdn_{i}" for i in range(self.n_head_layers)]

            indices |= {f"_qdn_{i}": qdn_indices for i in range(self.n_head_layers)}
            instances |= {f"_qdn_{i}": n_qdn_box for i in range(self.n_head_layers)}

        losses = {"total": 0.0}
        for loss, group in itertools.product(self.losses, loss_groups):
            y_prd_group = self.get_y_prd_head(y_prd_batch, group)
            l_dict = self.get_loss(
                loss,
                y_prd_group,
                y_trn_batch,
                indices[group],
                instances[group],
            )

            losses.update({k + group: v for k, v in l_dict.items()})
            losses["total"] += sum([self.weight_dict[k] * v for k, v in l_dict.items()])

        return losses

    @staticmethod
    def get_qdn_matched_indices(
        qdn_meta: dict[str, torch.Tensor], y_trn_batch: dict[str, torch.Tensor]
    ):
        qdn_positive_idx, qdn_n_group = (
            qdn_meta["qdn_indexes"],
            qdn_meta["n_qdn_group"],
        )
        n_instances_batch = [len(t) for t in y_trn_batch["cls"]]
        device = y_trn_batch["cls"][0].device

        qdn_match_indices = []
        for i, n_instances_sample in enumerate(n_instances_batch):
            if n_instances_sample > 0:
                gt_idx = torch.arange(
                    n_instances_sample, dtype=torch.int64, device=device
                )
                gt_idx = gt_idx.tile(qdn_n_group)
                qdn_match_indices.append((qdn_positive_idx[i], gt_idx))
            else:
                qdn_match_indices.append(
                    (
                        torch.zeros(0, dtype=torch.int64, device=device),
                        torch.zeros(0, dtype=torch.int64, device=device),
                    )
                )

        return qdn_match_indices

    @staticmethod
    def get_y_trn_permutation(indices):
        """Given indices of the matched pairs, returns the permutation to align the predictions with the targets."""
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
