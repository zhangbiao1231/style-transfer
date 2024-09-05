
import argparse
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from utils.general import LOGGER, colorstr, print_args
from utils.torch_utils import (
    initialize_weights,
    model_info,
    select_device,
)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None
import torchvision

class StyleTransModel(nn.Module):
    # YOLOv5 classification model
    def __init__(self, opt):
        """Initializes YOLOv5 model with config file `cfg`, input channels `ch`, number of classes `nc`, and `cuttoff`
        index.
        """
        super(StyleTransModel,self).__init__()
        self.content_layers = opt.content_layers
        self.style_layers = opt.style_layers
        self.cutoff = max(self.content_layers + self.style_layers)+1
        vgg19 = torchvision.models.__dict__[opt.model](weights="DEFAULT")
        for param in vgg19.parameters():
            param.requires_grad = False  # 冻结预训练权重
        self.model = vgg19.features[:self.cutoff]  # cutoff

    def forward(self, x):
        contents, styles = None, []
        for i,layer in enumerate(self.model):
            x = layer(x)
            if i in self.content_layers:
                contents =x
            elif i in self.style_layers:
                styles.append(x)
            else:continue
        return contents, styles

Model = StyleTransModel

class SynthesizedImage(nn.Module):
    def __init__(self,imgsz,**kwargs):
        super(SynthesizedImage,self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*imgsz))
    def forward(self):
        return self.weight

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vgg19", help="model.yaml")
    parser.add_argument("--content_layers", default=[0, 5, 10, 19, 28], help="content_layers")
    parser.add_argument("--style_layers", default=[25], help="content_layers")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--profile", action="store_true", help="profile model speed")
    parser.add_argument("--line-profile", action="store_true", help="profile model speed layer by layer")
    parser.add_argument("--test", action="store_true", help="test all yolo*.yaml")
    opt = parser.parse_args()
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.zeros((opt.batch_size, 3, 224, 224)).to(device)
    model = Model(opt).to(device)
    contents, styles = model(im)

    print(contents[-1].shape)
    print(styles[-1].shape)


    #Attention! According to CHATGPT's suggestion:use PYTHONPATH=$(pwd) python models/transfer.py --cfg models/vgg19.yaml