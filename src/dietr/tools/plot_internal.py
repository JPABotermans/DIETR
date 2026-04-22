"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import matplotlib.collections
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from dietr.data.box_conversion import box_yolo_to_coco


def plot_sample_np(
    sample_np: dict[str, np.ndarray],
    plot_msk: bool = False,
    ax: matplotlib.figure.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.figure.Axes]:
    if not isinstance(ax, matplotlib.figure.Axes):
        fig, ax = plt.subplots()

    img_np = sample_np["img_np"].copy()
    img_hw = img_np.shape[:2]
    box_coco_np = box_yolo_to_coco(box_np=sample_np["box_np"], img_hw=img_hw)
    for i in range(len(box_coco_np)):
        c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        xmin, ymin, w, h = box_coco_np[i]
        rect = [matplotlib.patches.Rectangle(xy=[xmin, ymin], width=w, height=h)]
        rect_collection = matplotlib.collections.PatchCollection(
            rect, facecolor="none", edgecolors=c, linewidths=2
        )
        if plot_msk:
            img_np[sample_np["msk_np"][i] == 1] = np.array(c) * 255
        ax.add_collection(rect_collection)
    ax.imshow(img_np)
