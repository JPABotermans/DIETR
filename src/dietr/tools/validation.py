"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import torch
import tqdm
import numpy as np
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from dietr.data.box_conversion import box_tensor_yolo_to_xyxy
from dietr.data.coco_definitions import coco_result_names
from dietr.data.conversion import label_normal_to_coco
from dietr.tools.distributed import all_gather, get_world_size
from dietr.tools.logging import combine_results_dicts
from dietr.data.msk_conversion import msk_np_rescale_and_crop_and_encode


def decode_y_prd_to_msk_np(
    y_prd_batch: dict[str, torch.Tensor],
    index: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    msk_prot = y_prd_batch["masks"]  # shape [N, n_prototypes, H/4, W/4]
    msk_coef = y_prd_batch["msk_5"]  # shape [N, n_queries, n_prototypes]
    msk_coef = torch.nn.functional.tanh(msk_coef)
    if index is not None:
        msk_coef = msk_coef.gather(
            dim=1, index=index.unsqueeze(dim=-1).repeat(1, 1, msk_coef.shape[-1])
        )
    predicted_masks = msk_prot[:, None, ...] * msk_coef[..., None, None]
    predicted_masks = predicted_masks.mean(dim=2)

    return torch.nn.functional.sigmoid(predicted_masks).detach().cpu().numpy()


def decode_y_prd_to_cls_cnf_index(
    y_prd_batch: dict[str, torch.Tensor],
    n_predictions: int,
    n_cls: int,
    coco_dataset: bool,
):
    y_prd_cls = y_prd_batch["cls_5"].detach()
    y_prd_cnf = torch.nn.functional.sigmoid(y_prd_cls)
    y_prd_cnf, index = torch.topk(y_prd_cnf.flatten(1), n_predictions, dim=-1)
    y_prd_cnf = y_prd_cnf.cpu().numpy()

    y_prd_cls = index - index // n_cls * n_cls
    index = index // n_cls
    if coco_dataset:
        y_prd_cls = torch.tensor(
            [label_normal_to_coco(int(y_cls.item())) for y_cls in y_prd_cls.flatten()]
        ).reshape(y_prd_cls.shape)
    else:
        y_prd_cls = torch.tensor(
            [int(y_cls.item()) for y_cls in y_prd_cls.flatten()]
        ).reshape(y_prd_cls.shape)
    return y_prd_cls, y_prd_cnf, index


def decode_y_prd_to_box_np(
    y_prd_batch: dict[str, torch.Tensor],
    x_trn_hw: tuple[int, int],
    info_batch: dict[str, np.ndarray],
    index: torch.Tensor,
    device: str,
) -> np.ndarray:
    y_prd_box = y_prd_batch["box_5"].detach()
    y_prd_box = box_tensor_yolo_to_xyxy(box=y_prd_box)
    org_wh = [(w, h) for h, w in info_batch["img_hw"]]
    pad_values = torch.Tensor(info_batch["pad_hw"]).repeat(1, 2) / torch.Tensor(
        tuple(x_trn_hw)
    ).repeat(2)
    y_prd_box = y_prd_box - pad_values[:, None, :].to(device)
    y_prd_box *= (
        torch.Tensor(org_wh).to(device).repeat(1, 2).unsqueeze(1)
        * 1
        / (1 - 2 * pad_values[:, None, :].to(device))
    )

    return (
        y_prd_box.gather(
            dim=1, index=index.unsqueeze(-1).repeat(1, 1, y_prd_box.shape[-1])
        )
        .cpu()
        .numpy()
    )


def sample_np_to_coco_segmentation_ann(
    sample_np: dict[str, np.ndarray], idx: int
) -> dict[str, np.ndarray]:
    if not "msk_np" in sample_np:
        return dict()
    return {
        "segmentation": msk_np_rescale_and_crop_and_encode(
            msk_np=sample_np["msk_np"][idx],
            box_np=sample_np["box_np"][idx],
            org_wh=sample_np["org_wh"][idx],
            pad_hw=sample_np["pad_hw"][idx],
        )
    }


def batch_sample_np_to_coco_anns(
    batch_sample_np: list[dict[str, np.ndarray]], cnf_threshold: float = 0.1
):
    """Given al list of decoded predictions in internal numpy format"""
    coco_results = []
    for i, sample_np in enumerate(batch_sample_np):
        org_wh = sample_np["org_wh"][i]
        coco_results.extend(
            [
                {
                    "iscrowd": 0,
                    "image_id": sample_np["img_id"][i],
                    "category_id": int(cls_np),
                    "bbox": [
                        float((box_np[0]).clip(0, org_wh[0])),
                        float((box_np[1]).clip(0, org_wh[1])),
                        float((box_np[2] - box_np[0]).clip(0, org_wh[0])),
                        float((box_np[3] - box_np[1]).clip(0, org_wh[1])),
                    ],
                    "score": float(cnf_np),
                    "area": float(
                        (box_np[2].clip(0, org_wh[0]) - box_np[0].clip(0, org_wh[0]))
                        * (box_np[3].clip(0, org_wh[1]) - box_np[1].clip(0, org_wh[1]))
                    ),
                }
                | sample_np_to_coco_segmentation_ann(sample_np, idx=idx)
                for idx, (cls_np, box_np, cnf_np) in enumerate(
                    zip(
                        sample_np["cls_np"],
                        sample_np["box_np"],
                        sample_np["cnf_np"],
                    )
                )
                if cnf_np >= cnf_threshold
            ]
        )
    return coco_results


def decode_y_prd_batch(
    y_prd_batch: dict[str, torch.Tensor],
    info_batch: dict[str, any],
    x_trn_hw: tuple[int, int],
    n_predictions: int = 100,
    n_cls: int = 80,
    coco_dataset: bool = True,
    device: str = "cuda:0",
    cnf_threshold: float = 0.5,
) -> list[any]:
    """Given a whole batch prediction, decodes this batch to a list of sample_np, to internal format, then"""
    y_prd_cls, y_prd_cnf, index = decode_y_prd_to_cls_cnf_index(
        y_prd_batch=y_prd_batch,
        n_predictions=n_predictions,
        n_cls=n_cls,
        coco_dataset=coco_dataset,
    )
    y_prd_box = decode_y_prd_to_box_np(
        y_prd_batch=y_prd_batch,
        x_trn_hw=x_trn_hw,
        info_batch=info_batch,
        index=index,
        device=device,
    )
    if "masks" in y_prd_batch:
        msk_np = decode_y_prd_to_msk_np(y_prd_batch=y_prd_batch, index=index)

    batch_sample_np = []
    for i, (cls_np, box_np, cnf_np) in enumerate(zip(y_prd_cls, y_prd_box, y_prd_cnf)):
        sample_np = {
            "cls_np": cls_np,
            "box_np": box_np,
            "cnf_np": cnf_np,
            "img_id": [info_batch["img_id"][i]] * len(cls_np),
            "pad_hw": [info_batch["pad_hw"][i]] * len(cls_np),
            "org_wh": [info_batch["img_wh"][i]] * len(cls_np),
        }
        if "masks" in y_prd_batch:
            sample_np["msk_np"] = msk_np[i]

        batch_sample_np.append(sample_np)
    return batch_sample_np_to_coco_anns(
        batch_sample_np=batch_sample_np, cnf_threshold=cnf_threshold
    )


def get_coco_anns(
    dataloader: torch.utils.data.DataLoader,
    dietr: torch.nn.Module,
    device: str,
    n_cls: int = 80,
    n_predictions: int = 300,
    cnf_threshold: float = 0.1,
    coco_dataset: bool = True,
) -> dict[str, float]:
    dietr.eval()
    coco_results = []
    for x_val_batch, info_batch in tqdm.tqdm(dataloader):
        with torch.no_grad() and torch.amp.autocast(
            device_type="cuda", cache_enabled=True, dtype=torch.float16
        ):
            y_prd_batch = dietr(x_val_batch.to(device))
        coco_results.extend(
            decode_y_prd_batch(
                y_prd_batch=y_prd_batch,
                info_batch=info_batch,
                x_trn_hw=x_val_batch.shape[-2:],
                n_predictions=n_predictions,
                n_cls=n_cls,
                coco_dataset=coco_dataset,
                cnf_threshold=cnf_threshold,
                device=device,
            )
        )
    dietr.train()
    return coco_results


def get_empty_coco_results(predict_msk: bool) -> tuple[dict[str, float], any]:
    box_results = {f"bbox/{k}": 0.0 for k in coco_result_names}
    if predict_msk:
        return {f"segm/{k}": 0.0 for k in coco_result_names} | box_results, None
    return box_results, None


@torch.no_grad()
def validate(
    dietr: torch.nn.Module,
    predict_msk: bool,
    val_dataloader: torch.utils.data.DataLoader,
    device: str | int,
    n_cls: int,
    cnf_threshold: float = 0.1,
    n_predictions: int = 100,
    coco_dataset: bool = True,
) -> tuple[dict[str, float], any]:

    coco_results = get_coco_anns(
        dataloader=val_dataloader,
        dietr=dietr,
        device=device,
        cnf_threshold=cnf_threshold,
        n_cls=n_cls,
        n_predictions=n_predictions,
        coco_dataset=coco_dataset,
    )
    coco_results_gathered = all_gather(coco_results)
    coco_results = []
    for coco_results_gpu in coco_results_gathered:
        coco_results.extend(coco_results_gpu)

    if len(coco_results) == 0:
        return get_empty_coco_results(predict_msk=predict_msk)

    if get_world_size() != 1:
        if device not in [0, "cuda:0", "cuda"]:
            return get_empty_coco_results(predict_msk=predict_msk)

    coco_eval_results: dict[str, float] = {}
    coco_gt = COCO(annotation_file=val_dataloader.dataset.coco_ann_file)

    modes = ["bbox", "segm"] if predict_msk else ["bbox"]
    coco_dt = coco_gt.loadRes(coco_results)
    for mode in modes:
        coco_eval = COCOeval(cocoDt=coco_dt, cocoGt=coco_gt, iouType=mode)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_eval_results[mode] = dict(
            zip(coco_result_names, coco_eval.stats, strict=False)
        )
    return combine_results_dicts(coco_eval_results=coco_eval_results), coco_dt.dataset
