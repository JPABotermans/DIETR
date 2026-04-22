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

from scipy.optimize import linear_sum_assignment


from dietr.data.box_conversion import box_yolo_giou


class HungarianMatcher(torch.nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        weight_dict: dict[str, torch.Tensor],
        alpha: float = 0.2,
        gamma: float = 0.2,
    ) -> None:
        super().__init__()
        self.cost_cls = weight_dict["cls"]
        self.cost_box = weight_dict["box"]
        self.cost_iou = weight_dict["giou"]
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(
        self,
        y_prd_box: torch.Tensor,
        y_prd_cls: torch.Tensor,
        y_trn_box: list[torch.Tensor],
        y_trn_cls: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Performs the matching

        Params:
            y_prd_box: torch.Tensor
            y_prd_cls: torch.Tensor
            y_trn_box: list[torch.Tensor]
            y_trn_cls: list[torch.Tensor]:
            
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, n_queries = y_prd_cls.shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = torch.nn.functional.sigmoid(y_prd_cls.flatten(0, 1))

        y_prd_box_flat = y_prd_box.flatten(0, 1) 


        sizes = [len(v) for v in y_trn_box]
        y_trn_cls = torch.cat(y_trn_cls).to(int)
        y_trn_box = torch.cat(y_trn_box)
        cost_box = torch.cdist(y_prd_box_flat, y_trn_box, p=1)

        out_prob = out_prob[:, y_trn_cls]
        neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-6).log())
        pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-6).log())
        cost_cls = pos_cost_class - neg_cost_class

        # Compute the iou
        cost_giou = -box_yolo_giou(y_prd_box_flat, y_trn_box)

        # Final cost matrix
        cost_matrix = (
            self.cost_box * cost_box
            + self.cost_cls * cost_cls
            + self.cost_iou * cost_giou
        )
        cost_matrix = cost_matrix.view(batch_size, n_queries, -1).cpu()
        cost_matrix = torch.nan_to_num(cost_matrix, nan=1.0)
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))
        ]


        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
    