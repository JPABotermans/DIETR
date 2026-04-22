"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
Modified from rt-detr (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) lyuwenyu. 
Reference: https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py
------------------------------------------------------------------------
"""


import torch
import einops
from dietr.modeling.base.layers import ProjectionLayer
from dietr.modeling.base.projections import MLP
from dietr.modeling.base.deformable_attention import (
    MSDAttentionLayer,
    bias_init_with_prob,
)
from dietr.modeling.base.query_denoising import QueryDenoiser


class DIETRHead(torch.nn.Module):
    def __init__(
        self,
        predict_msk: bool,
        channels: dict[str, tuple[int, int]],
        n_prototypes: int,
        n_cls: int,
        n_query: int,
        n_sample_points: int,
        n_attn_head: int,
        n_layers: int,
        dim_feedforward: int,
        act: str,
        mlp_act: str,
        n_qdn_query: int,
        cls_noise_ratio: float,
        box_noise_scale: float,
        location_eps: float,
        scale_dependent_query: bool,
        n_mlp_box_layers: int,
        n_mlp_msk_layers: int,
        prj_kernel_size: int,
        prj_padding: int,
        prj_stride: int,
        prj_act: str | bool,
        prj_nrm: str | bool,
    ):
        super().__init__()
        self.predict_msk = predict_msk
        self.channels = channels
        hidden_dim = list(self.channels.values())[0][0]
        self.hidden_dim = hidden_dim
        self.n_attn_head = n_attn_head

        self.num_levels = len(channels)
        self.n_cls = n_cls
        self.n_query = n_query
        self.location_eps = location_eps
        self.n_layers = n_layers
        self.n_mlp_box_layers = n_mlp_box_layers
        self.n_mlp_msk_layers = n_mlp_msk_layers
        self.scale_dependent_query = scale_dependent_query

        # Feature projection
        self.prj_layer = ProjectionLayer(
            channels=channels,
            kernel_size=prj_kernel_size,
            padding=prj_padding,
            stride=prj_stride,
            act=prj_act,
            nrm=prj_nrm,
        )

        self.decoder = MSDAttentionLayer(
            hidden_dim,
            dim_feedforward=dim_feedforward,
            act=act,
            n_head=n_attn_head,
            n_scales=len(channels),
            n_sample_points=n_sample_points,
            n_layers=n_layers,
        )

        self.query_denoiser = QueryDenoiser(
            n_cls=n_cls,
            n_query=n_query,
            hidden_dim=hidden_dim,
            n_qdn_query=n_qdn_query,
            cls_noise_ratio=cls_noise_ratio,
            box_noise_scale=box_noise_scale,
        )

        # decoder embedding
        self.query_pos_head = MLP(
            input_dim=4,
            hidden_dim=2 * hidden_dim,
            output_dim=hidden_dim,
            n_layers=2,
            act=mlp_act,
        )

        # encoder head
        self.enc_output = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(
                hidden_dim,
            ),
        )

        self.enc_cls_head = torch.nn.Linear(hidden_dim, self.n_cls)
        self.enc_box_head = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=4,
            n_layers=self.n_mlp_box_layers,
            act=mlp_act,
        )

        self.heads = torch.nn.ModuleDict(
            {
                "box": torch.nn.ModuleList(
                    [
                        MLP(
                            input_dim=hidden_dim,
                            hidden_dim=hidden_dim,
                            output_dim=4,
                            n_layers=self.n_mlp_box_layers,
                            act=mlp_act,
                        )
                        for _ in range(n_layers)
                    ]
                ),
                "cls": torch.nn.ModuleList(
                    [torch.nn.Linear(hidden_dim, self.n_cls) for _ in range(n_layers)]
                ),
            }
        )
        if self.predict_msk:
            self.heads["msk"] = torch.nn.ModuleList(
                    [
                        MLP(
                            input_dim=hidden_dim,
                            hidden_dim=hidden_dim,
                            output_dim=n_prototypes,
                            n_layers=self.n_mlp_msk_layers,
                            act=mlp_act,
                        )
                        for _ in range(n_layers)
                    ]
                )

            self.enc_msk_head = MLP(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=n_prototypes,
                n_layers=self.n_mlp_msk_layers,
                act=mlp_act,
            )

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)

        torch.nn.init.constant_(self.enc_cls_head.bias, bias)
        torch.nn.init.constant_(self.enc_box_head.layers[-1].weight, 0)
        torch.nn.init.constant_(self.enc_box_head.layers[-1].bias, 0)

        for cls_head, box_head in zip(self.heads["cls"], self.heads["box"]):
            torch.nn.init.constant_(cls_head.bias, bias)
            torch.nn.init.constant_(box_head.layers[-1].weight, 0)
            torch.nn.init.constant_(box_head.layers[-1].bias, 0)

        torch.nn.init.xavier_uniform_(self.enc_output[0].weight)
        torch.nn.init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        torch.nn.init.xavier_uniform_(self.query_pos_head.layers[1].weight)

    @staticmethod
    def get_features_melted(
        encoder_features: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given a dictionary of features, melts both the scale and height and width dimensions and returns a sequence of features.
        Project the encoder features to the correct dimensions and melt the width and height dimensions
        And concatenate them to one multi-scale feature
        """
        features_melted = []

        for feat_scale in list(encoder_features.values()):
            features_melted.append(einops.rearrange(feat_scale, "N C H W -> N (H W) C"))

        return torch.concat(features_melted, 1)

    @staticmethod
    def generate_box_offset(
        features_spatial_shapes: torch.Tensor,
        location_eps: float,
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        box_offset = []
        for scale, (h, w) in enumerate(features_spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=h, dtype=dtype),
                torch.arange(end=w, dtype=dtype),
                indexing="ij",
            )
            grid_wh = torch.tensor([w, h], dtype=dtype)
            grid_xy = torch.stack([grid_x, grid_y], -1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / grid_wh
            grid_wh = torch.ones_like(grid_xy) * grid_size * (2.0**scale)
            box_offset.append(
                torch.concat([grid_xy, grid_wh], -1).reshape(-1, h * w, 4)
            )

        box_offset = torch.concat(box_offset, 1).to(device)
        valid_mask = (
            (box_offset > location_eps) * (box_offset < 1 - location_eps)
        ).all(-1, keepdim=True)
        box_offset = torch.log(box_offset / (1 - box_offset))
        box_offset = torch.where(valid_mask, box_offset, torch.inf)

        return box_offset, valid_mask

    def select_topk_index(
        self, cls_enc_outputs: torch.Tensor, features_shapes: list[tuple[int, int]]
    ) -> torch.Tensor:
        """Given the encoder output logits, selects either;
        1. the top self.n_query indices.
        2. Selects for each scale self.n_query//len(features_shapes), queries. This means that for each
            scale queries are selected and learning is forced on each scale.
        """
        if self.scale_dependent_query and self.training:
            split_shape = [h * w for h, w in features_shapes]
            cls_enc_outputs_cls_ = cls_enc_outputs.split(split_shape, dim=1)
            topk_ind_scale = []
            for i in range(len(features_shapes)):
                topk_ind_scale.append(
                    sum(split_shape[:i])
                    + torch.topk(
                        cls_enc_outputs_cls_[i].max(-1).values,
                        self.n_query // len(features_shapes),
                        dim=1,
                    )[1].unsqueeze(-1)
                )

            topk_ind = torch.concat(topk_ind_scale, dim=1)
        else:
            topk_ind = torch.topk(cls_enc_outputs.max(-1).values, self.n_query, dim=1)[
                1
            ].unsqueeze(-1)
        return topk_ind

    def get_input_queries(
        self,
        features_melted: torch.Tensor,
        features_shapes: list[torch.Tensor],
        box_offset: torch.Tensor,
        cls_qdn: torch.Tensor | None = None,
        box_qdn: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Process the encoder features into input queries that will be refined using the multi-scale deformable attention layers..


        """
        features_melted = self.enc_output(features_melted)

        cls_enc_outputs = self.enc_cls_head(features_melted)
        topk_ind = self.select_topk_index(
            cls_enc_outputs=cls_enc_outputs, features_shapes=features_shapes
        )
        cls_enc_topk = cls_enc_outputs.gather(
            dim=1,
            index=topk_ind.repeat(1, 1, cls_enc_outputs.shape[-1]),
        )

        queries = features_melted.gather(
            dim=1,
            index=topk_ind.repeat(1, 1, features_melted.shape[-1]),
        )
        sampled_box_offsets = box_offset.repeat(topk_ind.shape[0], 1, 1).gather(
            dim=1, index=topk_ind.repeat(1, 1, box_offset.shape[-1])
        )
        box_enc_topk = self.enc_box_head(queries) + sampled_box_offsets

        if self.predict_msk:
            msk_enc_topk = self.enc_msk_head(queries)

        queries = queries.detach()

        if cls_qdn is not None:
            queries = torch.concat([cls_qdn, queries], 1)
            box_ref_unact = torch.concat([box_qdn, box_enc_topk], 1)
        else:
            box_ref_unact = box_enc_topk

        enc_output = {
            "box_enc": torch.nn.functional.sigmoid(box_enc_topk),
            "cls_enc": cls_enc_topk,
        }
        if self.predict_msk:
            enc_output["msk_enc"] = msk_enc_topk

        return (queries, box_ref_unact.detach()), enc_output

    @staticmethod
    def get_features_shapes(
        encoder_features: dict[str, torch.Tensor],
    ) -> list[list[int, int]]:
        features_spatial_shapes = []
        for _, scale_features in encoder_features.items():
            _, _, H, W = scale_features.shape
            features_spatial_shapes.append([H, W])
        return features_spatial_shapes

    def forward(
        self,
        features: dict[str, torch.Tensor],
        y_trn_batch: dict[str, list[torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
            1. First projects the encoder features into a the self.hidden dimension.
            2. Melts both the scales and height and width dimensions into one big feature sequence.
            3. Derives the query denoising queries, which are appended to the "normal" queries.
            4. Then predicts for each self.n_layers outputs.

        Args:
            features:   (Dict[str, torch.Tensor]):          Contains the features derived by the encoder.
            y_trn_batch (Dict[str, torch.Tensor]) | None:   Contains the examples for this batch, used to derive query denoising queries.

        """
        features = self.prj_layer(features)
        features_melted = self.get_features_melted(encoder_features=features)
        features_shapes = self.get_features_shapes(encoder_features=features)

        box_offsets, valid_mask = self.generate_box_offset(
            features_shapes,
            device=features_melted.device,
            location_eps=self.location_eps,
        )
        features_melted = valid_mask.to(features_melted.dtype) * features_melted

        (
            cls_qdn,
            box_qdn,
            qdn_attn_mask,
            qdn_meta,
        ) = self.query_denoiser(y_trn_batch=y_trn_batch)

        (queries, box_ref_unact), y_prd_enc = self.get_input_queries(
            features_melted=features_melted,
            features_shapes=features_shapes,
            box_offset=box_offsets,
            cls_qdn=cls_qdn,
            box_qdn=box_qdn,
        )
        y_prd_dec = self.decoder(
            queries=queries,
            ref_box_unact=box_ref_unact,
            features_melted=features_melted,
            features_shapes=features_shapes,
            heads=self.heads,
            query_pos_head=self.query_pos_head,
            attn_mask=qdn_attn_mask,
        )
        if not self.training:
            return {f"{k}_{self.n_layers - 1}": v[-1] for k, v in y_prd_dec.items()}

        y_prd_batch = {}
        if qdn_meta is not None:
            y_prd_batch["qdn_meta"] = qdn_meta
            for i in range(self.n_layers):
                for key, value in y_prd_dec.items():
                    y_prd_batch[f"{key}_qdn_{i}"] = torch.split(
                        value[i], qdn_meta["n_qdn_split"], dim=1
                    )[0]
                    y_prd_batch[f"{key}_{i}"] = torch.split(
                        value[i], qdn_meta["n_qdn_split"], dim=1
                    )[1]

        for i in range(self.n_layers):
            for key, value in y_prd_dec.items():
                y_prd_batch[f"{key}_{i}"] = value[i]

        return y_prd_batch | y_prd_enc
