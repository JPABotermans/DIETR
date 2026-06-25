"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""

import torch
import json
import argparse
from dietr.data.dataloaders import get_trn_dataloader, get_val_dataloader
from dietr.tools.training import setup_modeling, save_ckpt, count_parameters, setup_dietr
from dietr.tools.logging import (
    setup_trn_env,
    log_loss_dict_tt,
    log_config_with_metrics,
)
from dietr.tools.validation import validate

from timeit import default_timer as timer


def train(config_pth: str, device: str = "cuda:0", ckpt: str = None, from_scratch: bool = False):
    config, experiment_dir, logger, summary_writer, wandb_run = setup_trn_env(
        config_path=config_pth,
        ckpt=ckpt,
    )
    dietr = setup_dietr(config, device=device, from_scratch=from_scratch)
    
    dietr_ema, loss, optimizer, scheduler, grad_scaler, step = setup_modeling(
        dietr=dietr,
        config=config,
        device=device,
        wandb_run=wandb_run,
        ckpt=ckpt,
    )
    logger.info(f"We have dietr.back: {count_parameters(model=dietr.back):,d}")
    logger.info(f"We have dietr.neck: {count_parameters(model=dietr.neck):,d}")
    if config["msk"]:
        logger.info(f"We have dietr.mask: {count_parameters(model=dietr.mask):,d}")
    logger.info(f"We have dietr.head: {count_parameters(model=dietr.head):,d}")
    logger.info(f"We have dietr FULL: {count_parameters(model=dietr):,d}")

    trn_dataloader = get_trn_dataloader(
        config=config, multi_gpu=False, cycle_dataloader=True
    )
    val_dataloader = get_val_dataloader(config=config, multi_gpu=False)

    while step <= (config["total_trn_steps"] -1):
        start_time = timer()
        step += 1
        (x_trn_batch, y_trn_batch) = next(trn_dataloader)
        with torch.amp.autocast(
            device_type=device,
            cache_enabled=True,
            dtype=torch.float16,
        ):
            y_trn_batch = {k: [t.to(device) for t in v] for k, v in y_trn_batch.items()}
            y_prd_batch = dietr(
                x_trn_batch=x_trn_batch.to(device), y_trn_batch=y_trn_batch
            )
            losses_dict = loss(y_prd_batch=y_prd_batch, y_trn_batch=y_trn_batch)

        grad_scaler.scale(losses_dict["total"]).backward()
        grad_scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            dietr.parameters(), max_norm=config["gradient_norm"]
        )
        grad_scaler.step(optimizer)
        grad_scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        if dietr_ema is not None:
            dietr_ema.update(dietr)

        end_time = timer()

        if step % config["trn_log_steps"] == 0 and step != 0:
            losses_dict["step_time"] = end_time - start_time
            losses_dict["grad_norm"] = grad_norm.item()
            losses_dict["learning_rate"] = scheduler.get_last_lr()[-1]
            logger.info(f"Step {step} loss:  {losses_dict['total'].item()}")
            log_loss_dict_tt(
                loss_dict=losses_dict,
                summary_writer=summary_writer,
                step=int(step),
                wandb_run=wandb_run,
            )
            save_ckpt(
                model=dietr,
                ema_model=dietr_ema,
                optimizer=optimizer,
                scheduler=scheduler,
                ckpt_path=experiment_dir + "/latest.pth",
                step=step,
            )

        if step % config["val_log_steps"] == 0 and step != 0:
            logger.info("Evaluating")
            results, coco_prd_data = validate(
                msk=config["msk"],
                dietr=dietr_ema if dietr_ema is not None else dietr,
                val_dataloader=val_dataloader,
                device=device,
                cnf_threshold=config["cnf_threshold"],
                n_cls=config["n_cls"],
                n_predictions=config["val_n_predictions"],
                coco_dataset=config["coco_dataset"],
            )
            logger.info(f"Step {step} coco results:  {results}")
            log_config_with_metrics(
                summary_writer=summary_writer,
                config=config,
                metrics=results,
                global_step=step,
                wandb_run=wandb_run,
            )
            if config["store_json"]:
                with open(f"{experiment_dir}/step-{step}.json", "w") as file:
                    json.dump(coco_prd_data, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--device", default="cuda:0", help="The device to use")
    parser.add_argument("--ckpt", default=None, help="Continue from training.")
    parser.add_argument(
        "--from_scratch", action="store_true", help="Start training from scratch"
    )
    args = parser.parse_args()

    train(
        config_pth=args.config,
        device=args.device,
        ckpt=args.ckpt,
        from_scratch=args.from_scratch,
    )
