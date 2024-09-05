import argparse
import sys
import platform
import os
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from models.transfer import (
    Model,
    SynthesizedImage,
)

from torch.cuda import amp #混合精度训练

from utils.loggers import GenericLogger
from utils.general import (
    DATASETS_DIR,
    WorkingDirectory,
    LOGGER,
    colorstr,
    print_args,
    increment_path,
    init_seeds,
    yaml_save,
)
from utils.torch_utils import (
    ModelEMA,
    model_info,
    select_device,
    smart_resume,
    torch_distributed_zero_first,
    EarlyStopping,
)
from utils.dataloaders import LoadImages
from utils.augmentaions import (
    preprocess_transforms,
    postprocess,
)
from utils.loss import (
    Contentloss,
    Styleloss,
    TotalVariationLoss,
)

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
def train(opt,device):
    """Trains a YOLOv5 model, managing datasets, model optimization, logging, and saving checkpoints."""
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    save_dir, weights, source, epochs, resume, nw, imgsz, loss_weight = (
        opt.save_dir,
        opt.weights,
        Path(opt.source),
        opt.epochs,
        opt.resume,
        min(os.cpu_count() - 1, opt.workers),
        opt.imgsz,
        opt.loss_weight,
    )
    cuda = device.type != "cpu"
    # Directories
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    imgdir = save_dir / "Transfer Images"
    imgdir.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = wdir / "last.pt", wdir / "best.pt"

    # Save run settings
    yaml_save(save_dir / "opt.yaml", vars(opt))

    # Logger
    logger = GenericLogger(opt=opt, console_logger=LOGGER) if RANK in {-1, 0} else None

    # DataLoaders
    content_dataset= LoadImages(source / "content-image", img_size=imgsz, transforms=preprocess_transforms(imgsz))
    content_path, content_img, content_img0, s0 = next(iter(content_dataset))
    style_dataset = LoadImages(source / "style-image", img_size=imgsz, transforms=preprocess_transforms(imgsz))
    style_path, style_img, style_img0, s1 = next(iter(style_dataset))

    # Model
    pretrained = str(weights).endswith(".pth")
    if pretrained:
        model = nn.ModuleList()
        file = Path(str(weights).strip().replace("'", ""))
        ckpt = torch.load(file, map_location="cpu")  # load
        csd = ckpt["model"].to(device).float()  # FP32 model
        model.append(csd.eval())
        model = model[-1]
    else:
        with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT):
            model = Model(opt).to(torch.float)  # create

    # Info
    if RANK in {-1, 0}:
        model_info(model)
        if opt.verbose:
            LOGGER.info(model)
        file = Path(source / "content-image")
        logger.log_images(file, name="Content Image")
        logger.log_graph(model, imgsz)  # log model

    # get content_Y, style_Y
    content_target, _ = model(content_img)  # content_X
    _, style_features = model(style_img)  # style_X


    # SynthesizedImage
    gen_img = SynthesizedImage(content_img.shape)
    gen_img.weight.data.copy_(content_img)
    generated_image = gen_img()
    # generated_image = content_img.clone().requires_grad_(True)
    # Optimizer
    optimizer = torch.optim.Adam(gen_img.parameters(), lr=opt.lr)
    # optimizer = torch.optim.Adam([generated_image], lr=opt.lr)
    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.8)

    # loss function
    content_loss = Contentloss(content_target)
    style_losses = [Styleloss(sf) for sf in style_features]
    tv_loss = TotalVariationLoss()

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume
    best_fitness, start_epoch, final_epoch = 0.0, 0, None  # initialize
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs, optimizer, scheduler, generated_image = smart_resume(ckpt, optimizer, optimizer,scheduler, generated_image,
                                                                                                    ema, weights, epochs, resume)
        del ckpt, csd

    # Train
    t0 = time.time()
    best_fitness = 0.0
    scaler = amp.GradScaler(enabled=cuda)
    stopper, stop = EarlyStopping(patience=opt.patience, min_delta=opt.min_delta), False
    LOGGER.info(
        f'Image sizes {imgsz} content-image, {imgsz} style-image\n'
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f"Transfer Image save to {colorstr('bold', imgdir)}\n"
        f'Starting {opt.model} training for {epochs} epochs...\n\n'
        f"{'Epoch':>10}{'GPU_mem':>10}{'c_loss':>12}{f's_loss':>12}{'tv_loss':>12}{'total_loss':>12}"
    )

    content_weight, style_weight, tv_weight = [w for w in loss_weight]
    for epoch in range(start_epoch,epochs):
        model.eval()

        # Forward
        content_feature, style_features = model(generated_image)

        cl = content_loss(content_feature) * content_weight
        sl = sum([sl(sf) for sl, sf in zip(style_losses, style_features)]) * style_weight
        tvl = tv_loss(generated_image) * tv_weight
        total_loss = cl + sl + tvl

        # Backward
        scaler.scale(total_loss).backward()

        # Optimize
        scaler.unscale_(optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if ema:
            ema.update(model)
        mem = "%.3gG" % (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)  # (GB)
        s = f"{f'{epoch + 1}/{epochs}':>10}{mem:>10}{cl:>12.3g}{sl:>12.3g}{tvl:>12.3g}{total_loss:>12.3g}"
        LOGGER.info(s)
        fitness = total_loss #2.08

        # Scheduler
        scheduler.step()
        stop = stopper(epoch=epoch, fitness=total_loss)  # early stop check
        # Log metrics
        if RANK in {-1, 0}:
            # Best fitness
            if epoch ==0 :
                best_fitness = fitness # Initialize best_fitness
            if fitness <= best_fitness - opt.min_delta:
                best_fitness = fitness
            # Log
            metrics = {
                f"Content/loss": cl,
                f"Style/loss": sl,
                f"TotalVariation/loss": tvl,
                f"Total/loss": total_loss,
                "lr/0": optimizer.param_groups[0]["lr"],
            }  # learning rate
            logger.log_metrics(metrics, epoch)

            if epoch != 0 and epoch % 20 == 0:
                img = postprocess(generated_image)
                img.save(imgdir / f'train-{epoch}-step.jpg')
                logger.log_images(path=imgdir, epoch=epoch)

            # Save model
            final_epoch = epoch + 1 == epochs
            if (not opt.nosave) or final_epoch:
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(ema.ema).half(),  # deepcopy(de_parallel(model)).half(),
                    "ema": deepcopy(ema.ema).half(),
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "generated_image": generated_image,
                    "opt": vars(opt),
                    "date": datetime.now().isoformat(),
                }
                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fitness:
                    torch.save(ckpt, best)
                del ckpt
        # EarlyStopping
        if stop:
            break  # must break all DDP ranks
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    # Train complete
    if RANK in {-1, 0} and final_epoch or stop:
        LOGGER.info(
            f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours."
            f"\nResults saved to {colorstr('bold', save_dir)}"
            f'\ntrain:          python trian.py --weights {best} --source im.jpg'
            f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{best}')"
            f'\nVisualize:       https://netron.app\n'
        )

def parse_opt(known=False):
    """Parses command line arguments for YOLOv5 training including model path, dataset, epochs, and more, returning
        parsed arguments.
        """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vgg19", help="model")
    parser.add_argument("--style_layers", default=[1, 6, 11, 20, 29], help="content_layers")
    parser.add_argument("--content_layers", default=[26], help="content_layers")
    parser.add_argument("--loss_weight", default=[1,10,1e-1], help="content_layers")

    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--weights", nargs="+", type=str, default="",
                        help="model.pt path(s)")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir")
    parser.add_argument("--epochs", type=int, default=500, help="total training epochs")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--imgsz", "--img", "--img-size", type = int,default=(960,1280), help="train, val image size (pixels)")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help='--cache images in "ram" (default) or "disk"')
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default=ROOT / "runs/train-cls", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--pretrained", nargs="?", const=True, default=False, help="start from i.e. --pretrained False")
    parser.add_argument("--optimizer", choices=["SGD", "Adam", "AdamW", "RMSProp"], default="Adam", help="optimizer")
    parser.add_argument("--lr", type=float, default=0.3, help="initial learning rate")
    parser.add_argument("--decay", type=float, default=5e-5, help="weight decay")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--patience", type=int, default=50, help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--min-delta", type=float, default=0.001,
                        help="EarlyStopping Minimum Delta (epochs without improvement)")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")
    return parser.parse_known_args()[0] if known else parser.parse_args()
def main(opt):
    """Executes YOLOv5 training with given options, handling device setup and DDP mode; includes pre-training checks."""
    if RANK in {-1, 0}:
        print_args(vars(opt))
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Parameters
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run

    # Train
    train(opt, device)
def run(**kwargs):
    """
       Executes YOLOv5 model training or inference with specified parameters, returning updated options.

       Example: from yolov5 import classify; classify.train.run(data=mnist, imgsz=320, model='yolov5m')
       """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
