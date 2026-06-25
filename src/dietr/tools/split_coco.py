"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import json


def split_coco_dataset(
    base_file_pth: str,
    split_size: int,
) -> None:
    """Given a base annotation file path, split the dataset into a smaller size, for debugging purposes.

    Args:
        base_file_pth (str): Base pth e.g. coco/annotations/instances_train2017.json
        split_size (int): Split size e.g. 16
    """

    with open(f"{base_file_pth}", "r") as file:
        data = json.load(file)

    coco_img_ids = []
    for i in range(split_size):
        coco_img_ids.append(data["images"][i]["id"])
    all_anns = []
    for ann in data["annotations"]:
        if ann["image_id"] in coco_img_ids:
            all_anns.append(ann)

    new_anns = dict()

    new_anns["categories"] = data["categories"]
    new_anns["images"] = [
        img_info for img_info in data["images"] if img_info["id"] in coco_img_ids
    ]
    new_anns["licenses"] = data["licenses"]
    new_anns["annotations"] = all_anns
    new_anns["info"] = ""

    with open(f"{base_file_pth}_{split_size}.json", "w") as file:
        json.dump(new_anns, fp=file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split COCO dataset into smaller size")
    parser.add_argument(
        "--base_file_pth",
        type=str,
        default="coco/annotations/instances_train2017",
        help="Base file path for COCO annotations (without .json extension)",
    )
    parser.add_argument(
        "--split_size",
        type=int,
        default=16,
        help="Number of images to include in the split dataset",
    )
    args = parser.parse_args()

    split_coco_dataset(args.base_file_pth, args.split_size)