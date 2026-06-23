"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""

import os
import torch
import shutil
import yaml
import argparse
import logging
from pathlib import Path
import wandb
from torch.utils.tensorboard import SummaryWriter

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--device", default="cuda:0", help="The device to use")
    parser.add_argument("--ckpt", default=None, help="Continue from training.")
    return parser.parse_args()


def get_logger(
    experiment_dir: str,
) -> logging.Logger:
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter("%(asctime)s  %(levelname)5s  %(message)s")
    console = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=experiment_dir + "/log.log")

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    logger.setLevel(logging.INFO)
    return logger


def create_new_experiment_dir(base_dir: str, prefix: str) -> str:
    if not Path(base_dir).exists():
        Path.mkdir(base_dir)

    existing_dirs = [
        d
        for d in os.listdir(base_dir)
        if d.startswith(prefix) and d[len(prefix) :].isdigit()
    ]

    if existing_dirs:
        existing_nums = [int(d[len(prefix) :]) for d in existing_dirs]
        new_num = max(existing_nums) + 1
    else:
        new_num = 0

    new_dir = Path(base_dir) / f"{prefix}{new_num}"
    new_dir.mkdir(parents=True)
    return str(new_dir)


def flatten_dict(
    loss_dict_tt: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    final_dict = dict()
    for key, value in loss_dict_tt.items():
        if isinstance(value, dict):
            temp_dict = flatten_dict(loss_dict_tt=value)
            for key2, value2 in temp_dict.items():
                final_dict[f"{key}-{key2}"] = value2
        else:
            final_dict[key] = value
    return final_dict


def log_loss_dict_tt(
    loss_dict: dict[str, torch.Tensor],
    summary_writer: SummaryWriter,
    step: int,
    wandb_run=None,
) -> None:
    loss = flatten_dict(loss_dict_tt=loss_dict)
    for key, value in loss.items():
        summary_writer.add_scalar(
            tag=f"trn/{key}",
            scalar_value=value,
            global_step=step,
        )
        if wandb_run is not None:
            wandb_run.log({f"trn/{key}": value}, step=step)


def setup_trn_env(config_path: str, ckpt: str = None, rank: int = 0) -> tuple[dict, str, logging.Logger]:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    config["trn_ann_file"] = Path(config["coco_data_dir"]).expanduser() /  config["trn_ann_file"]
    config["trn_img_root"] = Path(config["coco_data_dir"]).expanduser() /  config["trn_img_root"]
    config["val_ann_file"] = Path(config["coco_data_dir"]).expanduser() /  config["val_ann_file"]
    config["val_img_root"] = Path(config["coco_data_dir"]).expanduser() /  config["val_img_root"]
    config["base_dir"] = str(Path(config["base_dir"]).expanduser())

    if ckpt is None:
        experiment_dir = create_new_experiment_dir(
            base_dir=config["base_dir"], prefix=config["prefix"]
        )
    else:
        experiment_dir = str(Path(ckpt).parent)
        if not Path(ckpt).parent.exists():
            Path(ckpt).parent.mkdir(parents=True)

    if rank == 0:
        shutil.copy(config_path, experiment_dir + "/config.yaml")
        summary_writer = SummaryWriter(log_dir=experiment_dir)
        summary_writer.add_text(tag="config", text_string=str(config))

    if config.get("wandb", False) and rank == 0:
        wanddb_run = wandb.init(
            project=config["wandb-project"],
            config=config,
            dir=experiment_dir,
            resume="auto",
            name=experiment_dir,
        )
    else:
        wanddb_run = None
    return (
        config,
        experiment_dir,
        get_logger(experiment_dir=experiment_dir),
        SummaryWriter(log_dir=experiment_dir) if rank == 0 else None,
        wanddb_run,
    )

def setup_val_env(config_path: str, ckpt: str = None, rank: int = 0) -> tuple[dict, str, logging.Logger]:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    config["trn_ann_file"] = Path(config["coco_data_dir"]).expanduser() /  config["trn_ann_file"]
    config["trn_img_root"] = Path(config["coco_data_dir"]).expanduser() /  config["trn_img_root"]
    config["val_ann_file"] = Path(config["coco_data_dir"]).expanduser() /  config["val_ann_file"]
    config["val_img_root"] = Path(config["coco_data_dir"]).expanduser() /  config["val_img_root"]
    config["base_dir"] = str(Path(config["base_dir"]).expanduser())

    experiment_dir = str(Path(ckpt).parent)

    return (
        config,
        experiment_dir,
    )

def config_dict_to_hparams(config: dict) -> dict[str, int | float | str | bool]:
    """
    Convert the configuration dictionary to a format suitable for add_hparams.
    """
    hparams = {}
    for k, v in config.items():
        if isinstance(v, int | float | str | bool):
            hparams[k] = v
        else:
            hparams[k] = str(v)
    return hparams


def log_config_with_metrics(
    summary_writer: torch.utils.tensorboard.SummaryWriter,
    config: dict,
    metrics: dict[str, float],
    global_step: int,
    wandb_run=None,
) -> None:
    summary_writer.add_hparams(
        hparam_dict=config_dict_to_hparams(config=config),
        metric_dict=metrics,
        global_step=global_step,
        run_name=summary_writer.log_dir,
    )
    summary_writer.flush()
    if wandb_run is not None:
        wandb_run.log(
            metrics,
            step=global_step,
        )

def combine_results_dicts(
    coco_eval_results: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Combine two coco results on based on segmentation the other on bounding boxes"""
    combined_results_dict = {}
    for mode, results_dict in coco_eval_results.items():
        for key, value in results_dict.items():
            combined_results_dict[f"{mode}/{key}"] = value
    return combined_results_dict