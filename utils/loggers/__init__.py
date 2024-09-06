# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Logging utils."""


import os
import warnings
from pathlib import Path
import glob
# import matplotlib.image as mpimg

import torch

from utils.general import LOGGER, colorstr, cv2
from utils.torch_utils import de_parallel

LOGGERS = ("csv", "tb", "wandb", "clearml", "comet")  # *.csv, TensorBoard, Weights & Biases, ClearML
RANK = int(os.getenv("RANK", -1))

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:

    def SummaryWriter(*args):
        """Fall back to SummaryWriter returning None if TensorBoard is not installed."""
        return None  # None = SummaryWriter(str)
class Loggers:
    # YOLOv5 Loggers class
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        """Initializes loggers for YOLOv5 training and validation metrics, paths, and options."""
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.plots = not opt.noplots  # plot results
        self.logger = logger  # for printing results to console
        self.include = include
        self.keys = [
            "train/box_loss",
            "train/obj_loss",
            "train/cls_loss",  # train loss
            "metrics/precision",
            "metrics/recall",
            "metrics/mAP_0.5",
            "metrics/mAP_0.5:0.95",  # metrics
            "val/box_loss",
            "val/obj_loss",
            "val/cls_loss",  # val loss
            "x/lr0",
            "x/lr1",
            "x/lr2",
        ]  # params
        self.best_keys = ["best/epoch", "best/precision", "best/recall", "best/mAP_0.5", "best/mAP_0.5:0.95"]
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv
        self.ndjson_console = "ndjson_console" in self.include  # log ndjson to console
        self.ndjson_file = "ndjson_file" in self.include  # log ndjson to file

        # Messages
        # TensorBoard
        s = self.save_dir
        if "tb" in self.include and not self.opt.evolve:
            prefix = colorstr("TensorBoard: ")
            self.logger.info(f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/")
            self.tb = SummaryWriter(str(s))
class GenericLogger:
    """
    YOLOv5 General purpose logger for non-task specific logging
    Usage: from utils.loggers import GenericLogger; logger = GenericLogger(...)
    Arguments
        opt:             Run arguments
        console_logger:  Console logger
        include:         loggers to include
    """

    def __init__(self, opt, console_logger, include=("tb", "wandb", "clearml")):
        """Initializes a generic logger with optional TensorBoard, W&B, and ClearML support."""
        self.save_dir = Path(opt.save_dir)
        self.include = include
        self.console_logger = console_logger
        self.csv = self.save_dir / "results.csv"  # CSV logger
        if "tb" in self.include:
            prefix = colorstr("TensorBoard: ")
            self.console_logger.info(
                f"{prefix}Start with 'tensorboard --logdir {self.save_dir.parent}', view at http://localhost:6006/"
            )
            self.tb = SummaryWriter(str(self.save_dir))
    def log_metrics(self, metrics, epoch):
        """Logs metrics to CSV, TensorBoard, W&B, and ClearML; `metrics` is a dict, `epoch` is an int."""
        if self.csv:
            keys, vals = list(metrics.keys()), list(metrics.values())
            n = len(metrics) + 1  # number of cols
            s = "" if self.csv.exists() else (("%23s," * n % tuple(["epoch"] + keys)).rstrip(",") + "\n")  # header
            with open(self.csv, "a") as f:
                f.write(s + ("%23.5g," * n % tuple([epoch] + vals)).rstrip(",") + "\n")

        if self.tb:
            for k, v in metrics.items():
                self.tb.add_scalar(k, v, epoch)

    def log_images(self, path, epoch=0):
        """Logs images to all loggers with optional naming and epoch specification."""
        # files = [Path(f) for f in (files if isinstance(files, (tuple, list)) else [files])]  # to Path
        # files = [f for f in files if f.exists()]  # filter by exists
        files=[]
        p = str(Path(path).resolve())
        files.extend(sorted(glob.glob(os.path.join(p, "*.*"))))  # dir

        from matplotlib import image as mpimg

        if self.tb:
            for f in files:
                self.tb.add_image(tag=Path(path).stem,
                                  # img_tensor=cv2.imread(str(f))[..., ::-1],
                                  img_tensor=mpimg.imread(str(f)),
                                  global_step=epoch,
                                  dataformats="HWC")

    def log_graph(self, model, imgsz=(640, 640)):
        """Logs model graph to all configured loggers with specified input image size."""
        if self.tb:
            log_tensorboard_graph(self.tb, model, imgsz)
def log_tensorboard_graph(tb, model, imgsz=(640, 640)):
    """Logs the model graph to TensorBoard with specified image size and model."""
    try:
        p = next(model.parameters())  # for device, type
        imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz  # expand
        im = torch.zeros((1, 3, *imgsz)).to(p.device).type_as(p)  # input image (WARNING: must be zeros, not empty)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress jit trace warning
            tb.add_graph(de_parallel(model), im)
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ TensorBoard graph visualization failure {e}")
def web_project_name(project):
    """Converts a local project name to a standardized web project name with optional suffixes."""
    if not project.startswith("runs/train"):
        return project
    suffix = "-Classify" if project.endswith("-cls") else "-Segment" if project.endswith("-seg") else ""
    return f"YOLOv5{suffix}"