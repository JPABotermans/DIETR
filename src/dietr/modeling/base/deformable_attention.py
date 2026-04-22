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
import math
from dietr.modeling.base.projections import get_act
from dietr.data.conversion import inverse_sigmoid


def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init


class MSDAttention(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_scales: int = 4,
        n_points: int = 4,
    ):
        """
        Multi-Scale Deformable Attention Module
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_scales = n_scales
        self.n_points = n_points
        self.total_points = n_heads * n_scales * n_points

        self.head_dim = embed_dim // n_heads

        self.sampling_offsets = torch.nn.Linear(
            embed_dim,
            self.total_points * 2,
        )
        self.attention_weights = torch.nn.Linear(embed_dim, self.total_points)
        self.features_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.output_proj = torch.nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.n_heads, 1, 1, 2).tile(
            [1, self.n_scales, self.n_points, 1]
        )
        scaling = torch.arange(1, self.n_points + 1, dtype=torch.float32).reshape(
            1, 1, -1, 1
        )
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        torch.nn.init.constant_(self.attention_weights.weight, 0)
        torch.nn.init.constant_(self.attention_weights.bias, 0)

        # proj
        torch.nn.init.xavier_uniform_(self.features_proj.weight)
        torch.nn.init.constant_(self.features_proj.bias, 0)
        torch.nn.init.xavier_uniform_(self.output_proj.weight)
        torch.nn.init.constant_(self.output_proj.bias, 0)

    def forward(
        self,
        query: torch.Tensor,
        ref_box: torch.Tensor,
        features: torch.Tensor,
        features_shapes: list[int],
        features_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            ref_box (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            features (Tensor): [bs, feature_length, C]
            features_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            features_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            features_mask (Tensor): [bs, feature_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """

        features = self.features_proj(features)

        if features_mask is not None:
            features *= features_mask.astype(features.dtype).unsqueeze(-1)
        features = einops.rearrange(
            features, "B F (h d) -> B F h d", h=self.n_heads, d=self.head_dim
        )

        sampling_offsets = einops.rearrange(
            self.sampling_offsets(query),
            "B F (h l p o) -> B F h l p o",
            h=self.n_heads,
            l=self.n_scales,
            p=self.n_points,
        )

        attention_weights = einops.rearrange(
            self.attention_weights(query),
            "N F (h l p) -> N F h (l p)",
            l=self.n_scales,
            p=self.n_points,
            h=self.n_heads,
        )
        attention_weights = einops.rearrange(
            torch.nn.functional.softmax(attention_weights, dim=-1),
            "N q h (l p) -> N q h l p",
            l=self.n_scales,
            p=self.n_points,
        )
        sample_locations = (
            ref_box[:, :, None, :, None, :2]
            + sampling_offsets / self.n_points * ref_box[:, :, None, :, None, 2:] * 0.5
        )

        return self.deformable_attention_core_func(
            features, features_shapes, sample_locations, attention_weights
        )

    def deformable_attention_core_func(
        self,
        features: torch.Tensor,
        features_shapes: list[torch.Tensor],
        sample_locations: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
            Given Sequece of features, samples for each n_head x n_points, n_query_length samples,
            which it mutilplies with the attention weight to get the output of the deformable attention.
        
        Args:
            features (torch.Tensor):        [N, value_length, n_head, c].  
            features_shapes (Tensor|List):  [n_levels, 2], explains how to split the values.  
            sample_location (Tensor):       [N, query_length, n_head, n_levels, n_points, 2].  
            attn_weights (Tensor):          [N, query_length, n_head, n_levels, n_points].  

        Returns:
            output (Tensor):                [N, Length_{query}, C]
        """

        split_shape = [h * w for h, w in features_shapes]
        features_list = features.split(split_shape, dim=1)
        sample_grids = 2 * sample_locations - 1
        features_sampled_list = []

        for scale, (h, w) in enumerate(features_shapes):
            features_scale_rearranged = einops.rearrange(
                features_list[scale],
                "N (h w) n d -> (N n) d w h",
                h=h,
                w=w,
                n=self.n_heads,
            )
            sample_locations_scale = einops.rearrange(
                sample_grids[:, :, :, scale], "N F h p x -> (N h) F p x"
            )
            features_sampled = torch.nn.functional.grid_sample(
                features_scale_rearranged,
                sample_locations_scale,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            features_sampled_list.append(features_sampled)

        attention_weights = einops.rearrange(
            attention_weights, "N Q h l p -> (N h) 1 Q (l p)"
        )
        features_sampled_list = einops.rearrange(
            torch.stack(features_sampled_list, dim=-2), "... p l -> ... (p l)"
        )
        return einops.rearrange(
            (features_sampled_list * attention_weights).sum(-1),
            "(N h) c Q -> N Q (h c)",
            h=self.n_heads,
        )


class MSDAttentionBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        dim_feedforward: int = 1024,
        act: str = "relu",
        n_head: int = 8,
        n_scales: int = 4,
        n_sample_points: int = 4,
    ) -> None:
        super().__init__()

        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_head, batch_first=True
        )

        self.msd_attn = MSDAttention(
            hidden_dim, n_heads=n_head, n_scales=n_scales, n_points=n_sample_points
        )

        # ffn
        self.forward_ffn = torch.nn.Sequential(
            *[
                torch.nn.Linear(hidden_dim, dim_feedforward),
                get_act(act=act),
                torch.nn.Linear(dim_feedforward, hidden_dim),
            ]
        )

        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)
        self.norm3 = torch.nn.LayerNorm(hidden_dim)

    @staticmethod
    def with_pos_embed(x: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        return x if pos is None else x + pos

    def forward(
        self,
        queries: torch.Tensor,
        ref_box: torch.Tensor,
        features: torch.Tensor,
        features_shapes: torch.Tensor,
        attn_mask: torch.Tensor = None,
        features_mask: torch.Tensor = None,
        query_pos_embed: torch.Tensor = None,
    ):
        q = k = self.with_pos_embed(queries, query_pos_embed)

        queries = self.self_attn(q, k, value=queries, attn_mask=attn_mask)[0] + queries
        queries = self.norm1(queries)

        queries = (
            self.msd_attn(
                self.with_pos_embed(queries, query_pos_embed),
                ref_box=ref_box,
                features=features,
                features_shapes=features_shapes,
                features_mask=features_mask,
            )
            + queries
        )
        queries = self.norm2(queries)

        # ffn
        queries = queries + self.forward_ffn(queries)

        return self.norm3(queries.clamp(min=-65504, max=65504))


class MSDAttentionLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dim_feedforward: int,
        act: str | bool,
        n_head: int,
        n_scales: int,
        n_sample_points: int,
        n_layers: int,
    ):
        super().__init__()
        self.attn_layers = torch.nn.ModuleList(
            [
                MSDAttentionBlock(
                    hidden_dim=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    act=act,
                    n_head=n_head,
                    n_scales=n_scales,
                    n_sample_points=n_sample_points,
                )
                for _ in range(n_layers)
            ]
        )
    def forward(
        self,
        queries: torch.Tensor,
        ref_box_unact: torch.Tensor,
        features_melted: torch.Tensor,
        features_shapes: torch.Tensor,
        heads: torch.nn.ModuleList,
        query_pos_head: torch.nn.Module,
        attn_mask: torch.Tensor | None = None,
        features_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            queries (Tensor):
                [bs, query_length, hidden_dim]
                These are the queries that will be refined in each layer.

            ref_box_unact (Tensor)
                These are the offset for the box predictions.

            features_melted (Tensor):
                [bs, features_length, hidden_dim]
            features_shapes (list[tuple[int]]):

        For each layer;
            1. Derive reference points.
            2. Forward the Transfomer block (MultiScaleDeformable attention).
            3. Derive the auxiliarly outputs using the supplied heads.

            
        """
        y_prd_batch = { key: list() for key, _ in heads.items() }
        
        ref_box_detach = torch.nn.functional.sigmoid(ref_box_unact).detach()
        ref_box_pre_sigmoid = ref_box_unact.detach()

        for i, attn_layer in enumerate(self.attn_layers):
            query_pos_embed = query_pos_head(ref_box_detach)

            queries = attn_layer(
                queries=queries,
                ref_box=ref_box_detach.unsqueeze(2),
                features=features_melted,
                features_shapes=features_shapes,
                attn_mask=attn_mask,
                features_mask=features_mask,
                query_pos_embed=query_pos_embed,
            )

            inter_ref_box = torch.nn.functional.sigmoid(
                heads["box"][i](queries) + ref_box_pre_sigmoid
            )

            if self.training:
                y_prd_batch["cls"].append(heads["cls"][i](queries))
                if "msk" in y_prd_batch:
                    y_prd_batch["msk"].append(heads["msk"][i](queries))
                y_prd_batch["box"].append(inter_ref_box)

            elif i == len(self.attn_layers) - 1:
                y_prd_batch["cls"].append(heads["cls"][i](queries))
                y_prd_batch["box"].append(inter_ref_box)
                if "msk" in y_prd_batch:
                    y_prd_batch["msk"].append(heads["msk"][i](queries))
                break

            ref_box_pre_sigmoid = inverse_sigmoid(inter_ref_box).detach()
            ref_box_detach = inter_ref_box.detach() if self.training else inter_ref_box

        return {
            k: torch.stack(v) for k, v in y_prd_batch.items()
        }