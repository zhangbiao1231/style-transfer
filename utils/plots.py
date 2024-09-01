"""Plotting utils."""

import contextlib
import math
import os
from copy import copy
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image, ImageDraw
from scipy.ndimage.filters import gaussian_filter1d
from ultralytics.utils.plotting import Annotator

# from utils import TryExcept, threaded
from utils.general import LOGGER, increment_path
# from utils.metrics import fitness

# Settings
RANK = int(os.getenv("RANK", -1))
matplotlib.rc("font", **{"size": 11})
matplotlib.use("Agg")  # for writing to files only

def imshow_cls(im, labels=None, pred=None, names=None, nmax=25, verbose=False, f=Path("images.jpg")):
    """Displays a grid of images with optional labels and predictions, saving to a file."""
    from utils.augmentations import denormalize

    names = names or [f"class{i}" for i in range(1000)]
    blocks = torch.chunk(
        denormalize(im.clone()).cpu().float(), len(im), dim=0
    )  # select batch index 0, block by channels
    n = min(len(blocks), nmax)  # number of plots
    m = min(8, round(n**0.5))  # 8 x 8 default
    fig, ax = plt.subplots(math.ceil(n / m), m)  # 8 rows x n/8 cols
    ax = ax.ravel() if m > 1 else [ax]
    # plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for i in range(n):
        ax[i].imshow(blocks[i].squeeze().permute((1, 2, 0)).numpy().clip(0.0, 1.0))
        ax[i].axis("off")
        if labels is not None:
            s = names[labels[i]] + (f"—{names[pred[i]]}" if pred is not None else "")
            # s = f"true: {names[labels[i]]}" + "\n" + (f"pred: {names[pred[i]]}" if pred is not None else "")
            ax[i].set_title(s, fontsize=8, verticalalignment="top")
    plt.savefig(f, dpi=300, bbox_inches="tight")
    plt.close()
    if verbose:
        LOGGER.info(f"Saving {f}")
        if labels is not None:
            LOGGER.info("True:     " + " ".join(f"{names[i]:3s}" for i in labels[:nmax]))
        if pred is not None:
            LOGGER.info("Predicted:" + " ".join(f"{names[i]:3s}" for i in pred[:nmax]))
    return f
def feature_visualization(x, module_type, stage, n=32, save_dir=Path("runs/detect/exp")):
    """
    x:              Features to be visualized
    module_type:    Module type
    stage:          Module stage within model
    n:              Maximum number of feature maps to plot
    save_dir:       Directory to save results
    """
    if ("Detect" not in module_type) and (
        "Segment" not in module_type
    ):  # 'Detect' for Object Detect task,'Segment' for Segment task
        batch, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # filename

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
            n = min(n, channels)  # number of plots
            fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis("off")

            LOGGER.info(f"Saving {f}... ({n}/{channels})")
            plt.savefig(f, dpi=300, bbox_inches="tight")
            plt.close()
            np.save(str(f.with_suffix(".npy")), x[0].cpu().numpy())  # npy save
