"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import numpy as np
import matplotlib
import pycocotools
import matplotlib.collections
import matplotlib.figure
import matplotlib.pyplot as plt

def plot_coco_anns(
    anns: list[any],
    img_hw: tuple[int, int],
    fig: matplotlib.figure.Axes | None = None,
    ax: matplotlib.figure.Axes | None = None,
    plot_box: bool = True,
    plot_msk: bool = True,
) -> tuple[
    matplotlib.figure.Figure,
    matplotlib.figure.Axes,
]:
    if not isinstance(ax, matplotlib.figure.Axes):
        fig, ax = plt.subplots()

    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in anns:
        c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        if "segmentation" in ann and plot_msk:
            if isinstance(ann["segmentation"], list):
                for seg in ann["segmentation"]:
                    poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                    polygons.append(matplotlib.patches.Polygon(poly))
                    color.append(c)
            else:
                if isinstance(ann["segmentation"]["counts"], list):
                    rle = pycocotools.mask.frPyObjects(
                        [ann["segmentation"]], img_hw[0], img_hw[1]
                    )
                else:
                    rle = [ann["segmentation"]]

                m = pycocotools.mask.decode(rle)
                img = np.ones((m.shape[0], m.shape[1], 3))

                if ann["iscrowd"] == 1:
                    color_mask = np.array([2.0, 166.0, 101.0]) / 255
                if ann["iscrowd"] == 0:
                    color_mask = np.random.random((1, 3)).tolist()[0]
                for i in range(3):
                    img[:, :, i] = color_mask[i]

                ax.imshow(np.dstack((img, m * 0.5)))
            p = matplotlib.collections.PatchCollection(
                polygons, facecolor=color, linewidths=0, alpha=0.4
            )
            ax.add_collection(p)
            p = matplotlib.collections.PatchCollection(
                polygons, facecolor="none", edgecolors=color, linewidths=2
            )
            ax.add_collection(p)

        if "bbox" in ann and plot_box:
            xmin, ymin, w, h = ann["bbox"]
            rect = [matplotlib.patches.Rectangle(xy=[xmin, ymin], width=w, height=h)]
            rect_collection = matplotlib.collections.PatchCollection(
                rect, facecolor="none", edgecolors=c, linewidths=2
            )
            ax.add_collection(rect_collection)

    return fig, ax
