"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import json
from dietr.data.dataloaders import get_val_dataloader
from dietr.tools.training import (
    setup_val_modeling,
)
from dietr.tools.logging import (
    setup_val_env,
    get_args,
)
from dietr.tools.validation import validate


def val(config_pth: str, device: str = "cuda:0", ckpt: str = None):
    config, experiment_dir = setup_val_env(
        config_path=config_pth,
        ckpt=args.ckpt,
    )
    dietr, dietr_ema, step = setup_val_modeling(
        config=config,
        device=device,
        wandb_run=None,
        ckpt=ckpt,
    )
    val_dataloader = get_val_dataloader(config=config, multi_gpu=False)

    coco, coco_prd_data = validate(
        dietr=dietr_ema if dietr_ema is not None else dietr,
        msk=config["msk"],
        val_dataloader=val_dataloader,
        device=device,
        cnf_threshold=0.2,
        n_cls=config["n_cls"],
        n_predictions=config["val_n_predictions"],
        coco_dataset=config["coco_dataset"],
    )
    print(coco)
    if config["store_json"]:
        with open(f"{experiment_dir}/step-{step}.json", "w") as file:
            json.dump(coco_prd_data, file)


if __name__ == "__main__":
    args = get_args()
    val(config_pth=args.config, device=args.device, ckpt=args.ckpt)
