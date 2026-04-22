"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
Modified from rt-detr (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) lyuwenyu. 
Reference: https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/src/zoo/rtdetr/denoising.py
------------------------------------------------------------------------
"""
import torch
from dietr.data.conversion import (
    inverse_sigmoid,
)
from dietr.data.box_conversion import (
    box_tensor_yolo_to_xyxy,
    box_tensor_xyxy_to_yolo,
)


class QueryDenoiser(torch.nn.Module):
    def __init__(
        self,
        n_cls: int,
        n_query: int,
        hidden_dim: int,
        n_qdn_query: int = 100,
        cls_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
    ):
        """Module that adds noise to queries which increases convergence speed for DIETR models.
        """
        super().__init__()
        self.n_cls = n_cls
        self.n_query = n_query
        self.hidden_dim = hidden_dim
        self.n_qdn_query = n_qdn_query
        self.cls_noise_ratio = cls_noise_ratio
        self.box_noise_scale = box_noise_scale
        if n_qdn_query <= 0:
            return
        self.cls_qdn_embedding = torch.nn.Embedding(
            n_cls + 1, hidden_dim, padding_idx=n_cls
        )
        torch.nn.init.normal_(self.cls_qdn_embedding.weight[:-1])

    @staticmethod
    def get_qdn_attn_mask(
        n_total_query: int,
        n_max_samples: int,
        n_qdn_query: int,
        n_qdn_group: int,
        device: str,
    ) -> torch.Tensor:
        qdn_attn_msk = torch.full(
            [n_total_query, n_total_query], False, dtype=torch.bool, device=device
        )

        qdn_attn_msk[n_qdn_query:, :n_qdn_query] = True

        for i in range(n_qdn_group):
            qdn_attn_msk[
                n_max_samples * 2 * i : n_max_samples * 2 * (i + 1),
                n_max_samples * 2 * (i + 1) : n_qdn_query,
            ] = True
            qdn_attn_msk[
                n_max_samples * 2 * i : n_max_samples * 2 * (i + 1),
                : n_max_samples * 2 * i,
            ] = True
        return qdn_attn_msk

    @staticmethod
    def get_gt_msk(
        y_trn_batch: dict[str, list[torch.Tensor]],
        batch_size: int,
        cls_qdn: torch.Tensor,
        box_qdn: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        n_gts = [len(y) for y in y_trn_batch["cls"]]
        pad_gt_msk = torch.zeros(
            [batch_size, max(n_gts)], dtype=torch.bool, device=device
        )
        for i, n_gt in enumerate(n_gts):
            if n_gt > 0:
                cls_qdn[i, :n_gt] = y_trn_batch["cls"][i]
                box_qdn[i, :n_gt] = y_trn_batch["box"][i]
                pad_gt_msk[i, :n_gt] = 1
        return pad_gt_msk, cls_qdn, box_qdn

    def forward(
        self,
        y_trn_batch: dict[str, list[torch.Tensor]] | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int | tuple[int, int]]
    ]:
        if y_trn_batch is None or self.n_qdn_query <= 0:
            return None, None, None, None

        device = y_trn_batch["cls"][0].device
        n_gts = [len(y) for y in y_trn_batch["cls"]]
        if max(n_gts) == 0:
            return None, None, None, None

        batch_size = len(y_trn_batch["cls"])

        n_qdn_group = self.n_qdn_query // max(n_gts)
        n_qdn_group = 1 if n_qdn_group == 0 else n_qdn_group

        cls_qdn = torch.full(
            [batch_size, max(n_gts)], self.n_cls, dtype=torch.int32, device=device
        )
        box_qdn = torch.zeros([batch_size, max(n_gts), 4], device=device)

        gt_msk_padded, cls_qdn, box_qdn = self.get_gt_msk(
            y_trn_batch=y_trn_batch,
            batch_size=batch_size,
            cls_qdn=cls_qdn,
            box_qdn=box_qdn,
            device=device,
        )

        # each group has positive and negative queries.
        cls_qdn = cls_qdn.tile([1, 2 * n_qdn_group])
        box_qdn = box_qdn.tile([1, 2 * n_qdn_group, 1])
        gt_msk_padded = gt_msk_padded.tile([1, 2 * n_qdn_group])

        # positive and negative query denoising indexes
        qdn_idx_negative = torch.zeros([batch_size, max(n_gts) * 2, 1], device=device)
        qdn_idx_negative[:, max(n_gts) :] = 1
        qdn_idx_negative = qdn_idx_negative.tile([1, n_qdn_group, 1])
        qdn_idx_positive = 1 - qdn_idx_negative

        # query denoising training indexes
        qdn_idx_positive = qdn_idx_positive.squeeze(-1) * gt_msk_padded
        qdn_indexes = torch.nonzero(qdn_idx_positive)[:, 1]
        qdn_indexes = torch.split(qdn_indexes, [n * n_qdn_group for n in n_gts])

        # total denoising query
        n_qdn_query = int(max(n_gts) * 2 * n_qdn_group)

        cls_qdn = self.noise_cls(cls_qdn=cls_qdn, gt_msk_padded=gt_msk_padded)
        box_qdn = self.noise_box(box_qdn=box_qdn, qdn_idx_negative=qdn_idx_negative)

        n_total_query = n_qdn_query + self.n_query

        qdn_attn_mask = self.get_qdn_attn_mask(
            n_total_query=n_total_query,
            n_max_samples=max(n_gts),
            n_qdn_query=n_qdn_query,
            n_qdn_group=n_qdn_group,
            device=device,
        )
        qdn_meta = {
            "qdn_indexes": qdn_indexes,
            "n_qdn_group": n_qdn_group,
            "n_qdn_split": [n_qdn_query, self.n_query],
        }

        return cls_qdn, box_qdn, qdn_attn_mask, qdn_meta

    def noise_box(
        self, box_qdn: torch.Tensor, qdn_idx_negative: torch.Tensor
    ) -> torch.Tensor:
        if self.box_noise_scale <= 0:
            return box_qdn
        box_xyxy = box_tensor_yolo_to_xyxy(box_qdn)
        box_diff = torch.tile(box_qdn[..., 2:] * 0.5, [1, 1, 2]) * self.box_noise_scale
        rand_sign = torch.randint_like(box_qdn, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(box_qdn)
        rand_part = (rand_part + 1.0) * qdn_idx_negative + rand_part * (
            1 - qdn_idx_negative
        )
        box_xyxy += rand_sign * rand_part * box_diff
        box_xyxy = torch.clip(box_xyxy, min=0.0, max=1.0)

        box_qdn = box_tensor_xyxy_to_yolo(box_xyxy)
        return inverse_sigmoid(box_qdn)

    def noise_cls(
        self, cls_qdn: torch.Tensor, gt_msk_padded: torch.Tensor
    ) -> torch.Tensor:
        if self.cls_noise_ratio >= 0:
            cls_qdn_index = torch.rand_like(cls_qdn, dtype=torch.float) < (
                self.cls_noise_ratio * 0.5
            )
            cls_new = torch.randint_like(
                cls_qdn_index, 0, self.n_cls, dtype=cls_qdn.dtype
            )
            cls_qdn = torch.where(cls_qdn_index & gt_msk_padded, cls_new, cls_qdn)

        return self.cls_qdn_embedding(cls_qdn)
