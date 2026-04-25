"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import torch
import pathlib
from dietr.modeling.ema import ModelEMA
from dietr.modeling.loss.loss import DIETRLoss
from dietr.modeling.loss.matcher import HungarianMatcher
from dietr.modeling.dietr.dietr import DIETR
from torch.optim.lr_scheduler import LambdaLR


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_ckpt(
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    ckpt_path,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict() if optimizer else None,
        "ema": ema_model.state_dict() if ema_model is not None else None,
        "scheduler": scheduler.state_dict(),
        "step": step,
    }

    torch.save(checkpoint, ckpt_path)


def load_ckpt(
    ckpt: str,
    dietr: torch.nn.Module,
    dietr_ema: torch.nn.Module,
    optim: torch.optim.Optimizer | None,
    device: str,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    new_training_start: bool = False,
) -> tuple[
    torch.nn.Module,
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    int,
]:
    ckpt = torch.load(ckpt, map_location=device)
    if "model" not in ckpt:
        dietr.load_state_dict(ckpt)
    else:
        dietr.load_state_dict(ckpt["model"])

    if optim and "optim" in ckpt:
        optim.load_state_dict(ckpt["optim"])

    if dietr_ema and "ema" in ckpt:
        dietr_ema.load_state_dict(ckpt["ema"])
        if "step" in ckpt:
            dietr_ema.updates = ckpt["step"]
    else:
        dietr_ema = dietr_ema
    if "scheduler" in ckpt and not new_training_start and scheduler:
        scheduler.load_state_dict(ckpt["scheduler"])

    step = ckpt["step"] if "step" in ckpt else 0
    return dietr, dietr_ema, optim, scheduler, step


def setup_dietr(config: dict[str, any], device: str, from_scratch) -> torch.nn.Module:
    dietr =  DIETR(
        msk=config["msk"],
        n_cls=config["n_cls"],
        n_prototypes=config["n_prototypes"],
        back_size=config["back_size"],
        back_freeze_first_layer=config["back_freeze_first_layer"],
        back_freeze_bn=config["back_freeze_bn"],
        neck_prj_kernel_size=config["neck_prj_kernel_size"],
        neck_expansion=config["neck_expansion"],
        neck_prj_padding=config["neck_prj_padding"],
        neck_prj_stride=config["neck_prj_stride"],
        neck_prj_act=config["neck_prj_act"],
        neck_prj_nrm=config["neck_prj_nrm"],
        neck_channels=config["neck_channels"],
        neck_n_blocks=config["neck_n_blocks"],
        neck_latteral_conv=config["neck_latteral_conv"],
        neck_latteral_conv_size=config["neck_latteral_conv_size"],
        neck_latteral_conv_act=config["neck_latteral_conv_act"],
        neck_latteral_conv_nrm=config["neck_latteral_conv_nrm"],
        neck_cnv_act=config["neck_cnv_act"],
        head_channels=config["head_channels"],
        head_n_layers=config["head_n_layers"],
        head_n_query=config["head_n_query"],
        head_n_sample_points=config["head_n_sample_points"],
        head_n_attn_head=config["head_n_attn_head"],
        head_prj_kernel_size=config["head_prj_kernel_size"],
        head_prj_padding=config["head_prj_padding"],
        head_prj_stride=config["head_prj_stride"],
        head_prj_act=config["head_prj_act"],
        head_prj_nrm=config["head_prj_nrm"],
        head_act=config["head_act"],
        head_mlp_act=config["head_mlp_act"],
        head_n_qdn_query=config["head_n_qdn_query"],
        head_scale_dependent_query=config["head_scale_dependent_query"],
        head_n_mlp_msk_layers=config["head_n_mlp_msk_layers"],
        head_n_mlp_box_layers=config["head_n_mlp_box_layers"],
        head_cls_noise_ratio=config["head_cls_noise_ratio"],
        head_box_noise_scale=config["head_box_noise_scale"],
        head_dim_feedfoward=config["head_dim_feedfoward"],
        head_location_eps=config["head_location_eps"],
        mask_channels=config["mask_channels"],
        mask_prj_kernel_size=config["mask_prj_kernel_size"],
        mask_prj_padding=config["mask_prj_padding"],
        mask_prj_stride=config["mask_prj_stride"],
        mask_prj_act=config["mask_prj_act"],
        mask_prj_nrm=config["mask_prj_nrm"],
        mask_prj_channels=config["mask_prj_channels"],
        mask_latteral_conv=config["mask_latteral_conv"],
        mask_latteral_conv_size=config["mask_latteral_conv_size"],
        mask_latteral_conv_act=config["mask_latteral_conv_act"],
        mask_latteral_conv_nrm=config["mask_latteral_conv_nrm"],
    ).to(device)
    
    if config["compiling"]:
        dietr = torch.compile(dietr)
    
    if not from_scratch:
        ckpt = torch.load(config["pre-trained-model"], map_location=device)
        keys_to_remove = [k for k in ckpt["model"].keys() if 'enc_cls_head' in k or 'heads.cls' in k]
        for k in keys_to_remove:
            del ckpt["model"][k]

        dietr.load_state_dict(ckpt["model"], strict=False)

    return dietr


def setup_modeling(
    config: dict,
    device: str,
    wandb_run: None = None,
    ckpt: str = None,
    from_scratch: bool = False
):
    dietr = setup_dietr(config, device=device, from_scratch=from_scratch)
    
    matcher = HungarianMatcher(
        weight_dict=config["matcher_weight_dict"],
        alpha=config["matcher_alpha"],
        gamma=config["matcher_gamma"],
    )
    loss = DIETRLoss(
        matcher=matcher,
        weight_dict=config["loss_weight_dict"],
        losses=config["losses"],
        n_head_layers=config["head_n_layers"],
        alpha=config["loss_alpha"],
        gamma=config["loss_gamma"],
        n_cls=config["n_cls"],
    )

    params = [
        {
            "params": dietr.back.parameters(),
            "lr": config["lr"] * config["lr_back_multi"],
        },
        {"params": dietr.head.parameters(), "lr": config["lr"]},
        {"params": dietr.neck.parameters(), "lr": config["lr"]},
    ]
    if config["msk"]:
        params.append(
            {"params": dietr.mask.parameters(), "lr": config["lr"]},
        )

    if config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(
            params=params,
            weight_decay=config.get("weight_decay", 0.000125),
        )
    else:
        optimizer = torch.optim.Adam(
            params=params,
        )

    if config["model_ema"]:
        dietr_ema = ModelEMA(
            model=dietr,
            decay=config.get("model_ema_decay", 0.999),
            warmups=config.get("model_ema_warmups", 1000),
            start=config.get("model_ema_start", 0),
        )
    else:
        dietr_ema = None

    if config["lr_scheduler"] == "one_cycle_lr":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["lr"],
            total_steps=config["total_trn_steps"],
            pct_start=config["pct_start"],
        )
    else:
        warmup_steps = int(config["total_trn_steps"]) * config["pct_start"]
        phase_two_start = config.get("phase_two_start", 1_000_000)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(
                    max(1, warmup_steps)
                )  # linear warmup
            if current_step > phase_two_start:
                return 0.1
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda)

    if ckpt is not None and pathlib.Path(ckpt).exists():
        dietr, dietr_ema, optimizer, scheduler, step = load_ckpt(
            ckpt=ckpt,
            dietr=dietr,
            dietr_ema=dietr_ema,
            optim=optimizer,
            device=device,
            scheduler=scheduler,
            new_training_start=config.get("new_training_start", False),
        )
    else:
        step = 0
    if wandb_run is not None:
        wandb_run.watch(dietr, log="all", log_freq=config["wandb-gradient-steps"])

    grad_scaler = torch.amp.grad_scaler.GradScaler(device="cuda")
    return dietr, dietr_ema, loss, optimizer, scheduler, grad_scaler, step


def setup_val_modeling(
    config: dict,
    device: str,
    wandb_run: None = None,
    ckpt: str = None,
):
    dietr = setup_dietr(config, device=device)
    if config.get("compiling", False):
        dietr = torch.compile(dietr)

    if config["model_ema"]:
        dietr_ema = ModelEMA(
            model=dietr,
            decay=config.get("model_ema_decay", 0.999),
            warmups=config.get("model_ema_warmups", 1000),
            start=config.get("model_ema_start", 0),
        )
    else:
        dietr_ema = None


    if ckpt is not None and pathlib.Path(ckpt).exists():
        dietr, dietr_ema, optimizer, scheduler, step = load_ckpt(
            ckpt=ckpt,
            dietr=dietr,
            dietr_ema=dietr_ema,
            optim=None,
            device=device,
            scheduler=None,
            new_training_start=config.get("new_training_start", False),
        )
    else:
        step = 0

    if wandb_run is not None:
        wandb_run.watch(dietr, log="all", log_freq=config["wandb-gradient-steps"])

    return dietr, dietr_ema,  step
