import torch
from dietr.modeling.dietr.back_convnext import DIETRConvNext
from dietr.modeling.dietr.neck import DIETRNeck
from dietr.modeling.dietr.head import DIETRHead
from dietr.modeling.dietr.mask import DIETRMask


class DIETR(torch.nn.Module):
    """Detection and InstancE TRansformer."""

    def __init__(
        self,
        predict_msk: bool,
        n_cls: int,
        n_prototypes: int,
        
        back_size: str,
        back_freeze_first_layer: bool,
        back_freeze_bn: bool,
        
        neck_channels: dict[str, tuple[int, int]],
        neck_n_blocks: int,
        neck_expansion: float,
        neck_prj_kernel_size: int,
        neck_prj_padding: int,
        neck_prj_stride: int,
        neck_prj_act: int,
        neck_prj_nrm: int,
        neck_latteral_conv: bool,
        neck_latteral_conv_size: int,
        neck_latteral_conv_act: str,
        neck_latteral_conv_nrm: str,
        neck_cnv_act: str,
        
        head_channels: int,
        head_n_query: int,
        head_n_sample_points: int,
        head_n_layers: int,
        head_n_attn_head: int,
        head_act: str,
        head_mlp_act: str,
        head_n_qdn_query: int,
        head_location_eps: float,
        head_scale_dependent_query: bool,
        head_n_mlp_box_layers: int, 
        head_n_mlp_msk_layers: int, 
        head_cls_noise_ratio: float,
        head_dim_feedfoward: int,
        head_box_noise_scale: float,
        head_prj_kernel_size: int,
        head_prj_padding: int,
        head_prj_stride: int,
        head_prj_act: int,
        head_prj_nrm: int,
        
        mask_channels: dict[str, tuple[int, int]],
        mask_prj_channels: dict[str, tuple[int, int]],
        mask_latteral_conv: bool,
        mask_latteral_conv_size: int,
        mask_latteral_conv_act: str,
        mask_latteral_conv_nrm: str,
        mask_prj_kernel_size: int,
        mask_prj_padding: int,
        mask_prj_stride: int,
        mask_prj_act: int,
        mask_prj_nrm: int,
    ) -> None:
        """Initialize all submodules.

        Args:
            n_cls (int): Number of classes to predict.
            n_prototypes (int): Number of mask prototypes that will be linearly combined to output final instance masks.
            back_size (str): Size of the backbone, either [T]iny or [S]mall.
            back_freeze_first_layer (bool): Freeze the first layer (funel) of the backbone.
            back_freeze_bn (bool): Freeze the batch normalization of the backbone.
            neck_channels (dict[str, tuple[int, int]]):
            neck_n_blocks (int): Number of convlutional blocks in the feature pyramid of the neck
            neck_latteral_conv (bool): Doing lateral convolutions?
            neck_latteral_conv_size (int): size of the lateral convolutions of the FPN in the neck
            neck_latteral_conv_act (str): The activation type of the lateral convolutoins of the nekc.
            neck_latteral_conv_nrm (str): The normalization type of the lateral convolutions of the neck.
            neck_cnv_act (str): activation type of the convolutions in the neck.
            neck_projection_layer_cfg (dict[str, any]): Projection dimension for the neck. {"scale_2": [192, 193]}
            head_channels (dict[str, any]): Porjection dimension of the head. {"scale_2": [192, 192]}
            n_head_layers (int): Number of Multi-scale attention layers in the head
            head_act (str): Activation type of the head
            head_mlp_act (str): Activation type of the head in the mlps.
            head_n_qdn_query (int): Number of queries to denoise for the head.
            head_projection_layer_cfg (dict[str, any]):
            head_scale_dependent_query (bool): Whether or not to select equal number of queries for each scale.
            head_cls_noise_ratio (float): Noise scale for query denoising.
            head_dim_feedfoward (int): _description_
            head_box_noise_scale (float): _description_
            mask_channels (dict[str, tuple[int, int]]): _description_
            mask_prj_channels (dict[str, tuple[int, int]]): _description_
            mask_latteral_conv (bool): _description_
            mask_latteral_conv_size (int): _description_
            mask_latteral_conv_act (str): _description_
            mask_latteral_conv_nrm (str): _description_
            mask_block_cfg (dict[str, any]): _description_
        """
        super().__init__()
        self.predict_msk = predict_msk

        self.back = DIETRConvNext(
            size=back_size,
            freeze_first_layer=back_freeze_first_layer,
            freeze_bn=back_freeze_bn,
        )
        self.neck = DIETRNeck(
            channels=neck_channels,
            n_blocks=neck_n_blocks,
            expansion=neck_expansion,
            latteral_conv=neck_latteral_conv,
            latteral_conv_size=neck_latteral_conv_size,
            latteral_conv_act=neck_latteral_conv_act,
            latteral_conv_nrm=neck_latteral_conv_nrm,
            cnv_act=neck_cnv_act,
            prj_kernel_size=neck_prj_kernel_size,
            prj_padding=neck_prj_padding,
            prj_stride=neck_prj_stride,
            prj_act=neck_prj_act,
            prj_nrm=neck_prj_nrm,
        )
        self.head = DIETRHead(
            predict_msk=predict_msk,
            channels=head_channels,
            n_cls=n_cls,
            n_query=head_n_query,
            n_sample_points=head_n_sample_points,
            n_attn_head=head_n_attn_head,
            n_prototypes=n_prototypes,
            n_layers=head_n_layers,
            act=head_act,
            mlp_act=head_mlp_act,
            n_qdn_query=head_n_qdn_query,
            location_eps=head_location_eps,
            scale_dependent_query=head_scale_dependent_query,
            n_mlp_box_layers=head_n_mlp_box_layers,
            n_mlp_msk_layers=head_n_mlp_msk_layers,
            cls_noise_ratio=head_cls_noise_ratio,
            dim_feedforward=head_dim_feedfoward,
            box_noise_scale=head_box_noise_scale,
            prj_kernel_size=head_prj_kernel_size,
            prj_padding=head_prj_padding,
            prj_stride=head_prj_stride,
            prj_act=head_prj_act,
            prj_nrm=head_prj_nrm,
        )
        if predict_msk:
            self.mask = DIETRMask(
                n_prototypes=n_prototypes,
                channels=mask_channels,
                prj_channels=mask_prj_channels,
                latteral_conv=mask_latteral_conv,
                latteral_conv_size=mask_latteral_conv_size,
                latteral_conv_act=mask_latteral_conv_act,
                latteral_conv_nrm=mask_latteral_conv_nrm,
                prj_kernel_size=mask_prj_kernel_size,
                prj_padding=mask_prj_padding,
                prj_stride=mask_prj_stride,
                prj_act=mask_prj_act,
                prj_nrm=mask_prj_nrm,
            )

    def forward(
        self,
        x_trn_batch: torch.Tensor,
        y_trn_batch: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        back_features = self.back(x_trn_batch)
        neck_features = self.neck(back_features)
        if self.predict_msk:
            return self.head(neck_features, y_trn_batch=y_trn_batch) | self.mask(
                back_features
            )
        return self.head(neck_features, y_trn_batch=y_trn_batch)