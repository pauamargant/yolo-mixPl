# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolo11n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

import gc
import math
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics import __version__
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    callbacks,
    clean_url,
    colorstr,
    emojis,
    yaml_save,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
)
from ultralytics.utils.ops import non_max_suppression, xyxy2xywhn,xyxy2xywh

import albumentations as A

class BaseTrainer:
    """
    A base class for creating trainers.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        trainset (torch.utils.data.Dataset): Training dataset.
        testset (torch.utils.data.Dataset): Testing dataset.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        resume (bool): Resume training from a checkpoint.
        lf (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
        metrics (dict): Dictionary of metrics.
        plots (dict): Dictionary of plots.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
            _callbacks (list, optional): List of callback functions. Defaults to None.
        """
        print("STAARTING")
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / "args.yaml", vars(self.args))  # save run args
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths
        self.source_best = self.wdir / 'source_best.pt'
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100  # in case users accidentally pass epochs=None with timed training
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataset
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolo11n -> yolo11n.pt
        self.ema = None
        with torch_distributed_zero_first(LOCAL_RANK):  # avoid auto-downloading dataset multiple times
            self.trainset, self.testset = self.get_dataset()
            self.target_trainset, self.target_testset = self._get_target_dataset()




        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.best_target_fitness = None
        self.fitness = None
        self.target_fitness = None
        self.loss = None
        self.tloss = None
        self.s_loss = None
        self.u_loss = None
        self.u_tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [0, 1, 2]

        # HUB
        self.hub_session = None

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in {-1, 0}:
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        """Append the given callback to the event's callback list."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Override the existing callbacks with the given callback for the specified event."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:  # i.e. device='cpu' or 'mps'
            world_size = 0
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device=None or device=''
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                LOGGER.warning(
                    "'batch<1' for AutoBatch is incompatible with Multi-GPU training, setting default 'batch=16'"
                )
                self.args.batch = 16

            # Command
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f"{colorstr('DDP:')} debug command {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))

        else:
            self._do_train(world_size)

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self, world_size):
        """Initialize and set the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        # LOGGER.info(f'DDP info: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=world_size,
        )

    def _setup_train(self, world_size):
        """Build dataloaders and optimizer on correct rank process."""
        # Model
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp.int(), src=0)  # broadcast from rank 0 to all other ranks; gloo errors with boolean
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")
        if self.target_trainset:
            self.target_train_loader = self.get_dataloader(
                self.target_trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train", augm_type ='weak'
            )
        
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size * 2, rank=-1, mode="val"
            )
            self.validator = self.get_validator()
            if self.target_testset:
                # Ensure target_test_loader uses the target dataset configuration
                self.target_test_loader = self.get_dataloader(
                    self.target_testset, batch_size=batch_size, rank=LOCAL_RANK, mode="val"
                )
                # Ensure target_validator uses the target dataset configuration (handled in get_validator subclass)
                self.target_validator =  self.get_validator(target_loader=True)
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

    def build_pseudo_batch(
        self,
        teacher_preds,
        target_imgs,
        nms_iou=0.45, 
        pseudolabel_conf=0.1,
        agnostic=False,
        max_det=300,
        device='cpu'
        ):
        """
        Convert teacher model predictions into a pseuflabeled, augmented batch dict.

        Returns:
            dict with keys:
                - 'img': Tensor [B, C, H, W] of augmented images
                - 'batch_idx': LongTensor [N] of batch indices
                - 'cls': LongTensor    [N] of class labels
                - 'bboxes': Tensor     [N,4] of normalized xywh boxes
        """
        # 1) NMS on teacher preds
        results = non_max_suppression(
            teacher_preds,
            conf_thres=pseudolabel_conf,
            iou_thres=nms_iou,
            agnostic=agnostic,
            max_det=max_det
        )

        bs, _, img_h, img_w = target_imgs.shape

        # 2) Build an Albumentations Compose matching your strong_pipeline
        strong_transform_basic = A.Compose(
            [
                # # — weak parts are already handled upstream, so we just ensure the same resize+flip —
                # A.Resize(img_h, img_w, interpolation=1),  # like RandomResize
                # A.HorizontalFlip(p=0.5),                  # like RandomFlip

                # one random color op
                A.OneOf(
                    [
                        A.CLAHE(p=1.0),                   # ≈ AutoContrast
                        A.Equalize(p=1.0),                # Equalize
                        A.RandomBrightnessContrast(p=1.0),# Brightness/Contrast
                        A.Sharpen(p=1.0),                 # Sharpness
                        A.Posterize(num_bits=4, p=1.0),   # Posterize
                        A.Solarize(p=1.0), # Solarize
                        A.HueSaturationValue(p=1.0),      # ColorTransform
                    ],
                    p=1.0,
                    # readd 
                ),
                # one random geometric op
                A.OneOf(
                    [
                        A.Rotate(limit=15, p=1.0),                     # Rotate
                        A.Affine(shear={'x':(-20,20)}, p=1.0),         # ShearX
                        A.Affine(shear={'y':(-20,20)}, p=1.0),         # ShearY
                        A.Affine(translate_percent={'x':(-0.1,0.1)}, p=1.0),  # TranslateX
                        A.Affine(translate_percent={'y':(-0.1,0.1)}, p=1.0),  # TranslateY
                    ],
                    p=1.0,
                )
            ],
            bbox_params=A.BboxParams(
                format='yolo',            # normalized xywh
                label_fields=['labels'],
                filter_invalid_bboxes=True,
                check_each_transform=False   # clamp only at the end

            )
        )
        strong_transform_aug = A.Compose(
            [
            # blur & noise ops
                A.OneOf(
                    [
                        # A.MotionBlur(p=0.5),
                        A.GaussianBlur(blur_limit=(3,7), p=0.5),
                        A.MedianBlur(blur_limit=3, p=0.5),
                        A.GaussNoise(p=0.5),
                        A.ISONoise(color_shift=(0.01,0.05), intensity=(0.1,0.5), p=0.5),
                    ],
                    p=0.75,
                ),
                # distortion ops
                A.OneOf(
                    [
                        A.OpticalDistortion(distort_limit=(-0.05,0.05), mode='camera', p=0.5),
                        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.5),
                        A.ElasticTransform(p=0.5),
                        A.Perspective(scale=(0.05,0.1), p=0.5),
                    ],
                    p=0.75,
                ),
                # channel shuffle & gamma
                A.OneOf(
                    [
                        A.ChannelShuffle(p=0.5),
                        A.RandomGamma(gamma_limit=(80,120), p=0.5),
                        A.HueSaturationValue(p=0.5),
                    ],
                    p=0.75,
                ),
               
            ],
            bbox_params=A.BboxParams(
                format='yolo',            # normalized xywh
                label_fields=['labels'],
                filter_invalid_bboxes=True,
                check_each_transform=True   # clamp only at the end
            )
        )

            

        images_aug = []
        batch_idxs, classes, bboxes = [], [], []
        # 3) Loop per-image: filter pseudo-labels, then augment
        for batch_idx, (img_tensor, det) in enumerate(zip(target_imgs, results)):
            # --- filter pseudo-labels as before ---
            if det is None or det.shape[0] == 0:
                continue
            else:
                mask = det[:, 4] >= pseudolabel_conf
                det = det[mask]
                if det.shape[0] == 0:
                    bboxes_list, labels_list = [], []
                else:
                    xywh = xyxy2xywh(det[:, :4])
                    # normalize
                    xywh[:, 0] /= img_w; xywh[:, 1] /= img_h
                    xywh[:, 2] /= img_w; xywh[:, 3] /= img_h
                    bboxes_list = [tuple(x.tolist()) for x in xywh]
                    labels_list = det[:, 5].long().tolist()
            # --- prepare image for Albumentations (HWC uint8) ---
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            # if your imgs are floats [0,1]:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).round().astype(np.uint8)

            # --- apply the strong pipeline ---
            augmented = strong_transform_basic(
                image=img_np,
                bboxes=bboxes_list,
                labels=labels_list
            )
            # filter annotations with 0 area
            # Remove boxes with zero area or with x/y out of bounds (within 1e-7)
            # filtered = [
            #     (box, lbl)
            #     for box, lbl in zip(augmented['bboxes'], augmented['labels'])
            #     if box[2] > 0 and box[3] > 0
            #     and 1e-6 < box[0] < 1 - 1e-6
            #     and 1e-6 < box[1] < 1 - 1e-6
            # ]
            # if filtered:
            #     augmented['bboxes'], augmented['labels'] = zip(*filtered)
            #     augmented['bboxes '], augmented['labels'] = list(augmented['bboxes']), list(augmented['labels'])
            # else:
            #     augmented['bboxes'], augmented['labels'] = [], []

            augmented = strong_transform_aug(
                image=augmented['image'],
                bboxes=augmented['bboxes'],
                labels=augmented['labels']
            )

            # --- collect augmented image ---
            img_aug = augmented['image']
            # back to CHW float [0,1]
            img_aug_tensor = (
                torch.from_numpy(img_aug)
                .permute(2, 0, 1)
                .to(device)
                .float()
                .div(255.0)
            )
            images_aug.append(img_aug_tensor.unsqueeze(0))

            # --- collect any augmented bboxes/labels ---
            for lbl, box in zip(augmented['labels'], augmented['bboxes']):
                batch_idxs.append(batch_idx)
                classes.append(lbl)
                bboxes.append(box)
            # from utils.plotting import plot_images
            # plot_images(
            #     images = images_aug,
                
            # )
            break
        # 4) stack everything and return
        if images_aug:
            imgs_tensor = torch.cat(images_aug, dim=0)  # [B, C, H, W]
        else:
            # fallback to original
            imgs_tensor = target_imgs.to(device)

        if batch_idxs:
            batch_idx_tensor = torch.tensor(batch_idxs, dtype=torch.long, device=device)
            cls_tensor       = torch.tensor(classes,   dtype=torch.long, device=device)
            bboxes_tensor    = torch.tensor(bboxes,    dtype=torch.float, device=device)
        else:
            batch_idx_tensor = torch.zeros((0,), dtype=torch.long, device=device)
            cls_tensor       = torch.zeros((0,), dtype=torch.long, device=device)
            bboxes_tensor    = torch.zeros((0, 4), dtype=torch.float, device=device)
        return {
            'img':       imgs_tensor,
            'batch_idx': batch_idx_tensor,
            'cls':       cls_tensor,
            'bboxes':    bboxes_tensor,
        }
    def _do_train(self, world_size=1):
        """Train the model with the specified world size."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)
                

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            self.u_tloss = None
            self.s_tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                if self.args.s_lambda>0:
                    with autocast(self.amp):
                        batch = self.preprocess_batch(batch)
                        loss, self.s_loss_items = self.model(batch)
                        self.s_loss = loss.sum()
                        if RANK != -1:
                            self.s_loss *= world_size
                        self.s_tloss = (
                            (self.s_tloss * i + self.s_loss_items) / (i + 1) if self.s_tloss is not None else self.s_loss_items
                        )
                else:
                    self.s_loss = torch.tensor(0.0, device=self.device) # Initialize supervised loss
                    self.s_tloss = torch.zeros(3, device=self.device) # Initialize supervised loss items
                    self.s_loss_items = torch.zeros(3, device=self.device) # Initialize supervised loss items

                    
                # Pseudolableing target domain 
                if self.target_train_loader and self.args.u_lambda>0:
                    try:
                        # Get the next batch from the target dataset iterator
                        target_batch = next(target_iterator)
                    except StopIteration:
                        # If the target dataset is exhausted, reset the iterator
                        target_iterator = iter(self.target_train_loader)
                        target_batch = next(target_iterator)
                    except NameError:
                        # Initialize iterator if it doesn't exist (first epoch)
                        target_iterator = iter(self.target_train_loader)
                        target_batch = next(target_iterator)
                    target_batch = self.preprocess_batch(target_batch)
                    target_imgs = target_batch["img"].to(self.device)

                    with torch.no_grad():
                        teacher_preds = self.ema.ema(target_imgs)
                    
                    pseudo_batch = self.build_pseudo_batch(
                        teacher_preds=teacher_preds,
                        target_imgs=target_imgs,
                        pseudolabel_conf = self.args.pseudolabel_conf,
                        device = self.device
                    )
                    with autocast(self.amp):
                        u_loss, self.u_loss_items = self.model(pseudo_batch)
                        self.u_loss = u_loss.sum()
                        self.u_tloss = (
                            (self.u_tloss * i + self.u_loss_items) / (i + 1) if self.u_tloss is not None else self.u_loss_items
                        )
                        if RANK != -1:
                            self.u_loss *= world_size
                else:
                    self.u_loss = torch.tensor(0.0, device=self.device) # Initialize unsupervised loss
                    self.u_tloss = torch.zeros(3, device=self.device) # Initialize unsupervised loss items
                    self.u_loss_items = torch.zeros(3, device=self.device) # Initialize unsupervised loss items
                # Combine supervised and unsupervised loss
                self.loss = self.args.s_lambda*self.s_loss + self.args.u_lambda * self.u_loss
                # sum loss items
                self.loss_items = self.args.s_lambda*self.s_loss_items + self.args.u_lambda * self.u_loss_items
                self.tloss = (
                    (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                )                    
                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                            batch["cls"].shape[0],  # batch size, i.e. 8
                            batch["img"].shape[-1],  # imgsz, i.e 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)
                        if self.args.u_lambda>0:
                            self.plot_training_samples(pseudo_batch, ni)
                if self.args.batch_len > 0 and i%self.args.batch_len == 0: 
                    LOGGER.info(
                        f"Train {epoch}/{self.epochs} "
                        f"Batch {i}/{len(self.train_loader)} "
                        f"Loss: {self.loss:.4f} "
                        f"Supervised Loss: {self.s_loss:.4f} "
                        f"Unsupervised Loss: {self.u_loss:.4f} "
                        f"Memory: {self._get_memory():.3g}G"
                    )
                    break


                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()

                if self.args.target_data:
                    print("Target Domain Validation")
                    LOGGER.info(f"Target domain validation...")
                    tgt_metrics, self.target_fitness = self.validate(target_data=True)
                    # prefix and merge
                    tgt_metrics = {f"tgt/{k}": v for k, v in tgt_metrics.items()}
                    self.metrics.update(tgt_metrics)
                    # Restore original loader
                # Target Domain validation 

                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")
                    
            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            if self._get_memory(fraction=True) > 0.5:
                self._clear_memory()  # clear if memory utilization > 50%

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        if RANK in {-1, 0}:
            # Do final val with best.pt
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            # self.ema = ModelEMA(self.model, decay=self.args.ema_decay)  # EMA model
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

        
    

    def auto_batch(self, max_num_obj=0):
        """Calculate optimal batch size based on model and device memory constraints."""
        return check_train_batch_size(
            model=self.model,
            imgsz=self.args.imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
        )  # returns batch size

    def _get_memory(self, fraction=False):
        """Get accelerator memory utilization in GB or as a fraction of total memory."""
        memory, total = 0, 0
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
            if fraction:
                return __import__("psutil").virtual_memory().percent / 100
        elif self.device.type != "cpu":
            memory = torch.cuda.memory_reserved()
            if fraction:
                total = torch.cuda.get_device_properties(self.device).total_memory
        return ((memory / total) if total > 0 else 0) if fraction else (memory / 2**30)

    def _clear_memory(self):
        """Clear accelerator memory by calling garbage collector and emptying cache."""
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

    def read_results_csv(self):
        """Read results.csv into a dictionary using pandas."""
        import pandas as pd  # scope for faster 'import ultralytics'

        return pd.read_csv(self.csv).to_dict(orient="list")

    def _model_train(self):
        """Set model in training mode."""
        self.model.train()
        # Freeze BN stat
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer_teacher = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": deepcopy(self.ema.ema).half() if self.ema.ema else None,  # resume and final checkpoints derive from EMA
                "ema":  deepcopy(self.ema.ema).half() if self.ema else None,
                # "teacher_model": deepcopy(self.teacher_model.ema).half() if self.teacher_model else None,
                # "teacher_updates": self.teacher_model.updates  if self.teacher_model else None,
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.target_fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer_teacher,
        )
        serialized_ckpt_teacher = buffer_teacher.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt_teacher)  # save last.pt
        if self.best_target_fitness <= self.target_fitness:
            self.best.write_bytes(serialized_ckpt_teacher)  # save best.pt
        if self.best_fitness <= self.fitness:
            self.source_best.write_bytes(serialized_ckpt_teacher)
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt_teacher)  # save epoch, i.e. 'epoch3.pt'
    def _get_target_dataset(self):
        """
        Get train and validation datasets from the target data source specified in args.target_data.

        Returns:
            (tuple): A tuple containing the target dataset info dict, target training dataset,
                     and target validation/test dataset. Returns (None, None, None) if args.target_data is not set.
        """
        if not self.args.target_data:
            return  None, None

        LOGGER.info(f"Loading target dataset from '{clean_url(self.args.target_data)}'...")
        try:
            # Use the same task type as the primary dataset for checking
            if self.args.task == "classify":
                target_data = check_cls_dataset(self.args.target_data)
            elif self.args.target_data.split(".")[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                target_data = check_det_dataset(self.args.target_data)
                if "yaml_file" in target_data:
                    self.args.target_data = target_data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
            else:
                raise ValueError(f"Unsupported dataset format for target data: {self.args.target_data}")

        except Exception as e:
            raise RuntimeError(emojis(f"Target dataset '{clean_url(self.args.target_data)}' error ❌ {e}")) from e
        self.target_data = target_data
        return target_data.get("train"), target_data.get("val") or target_data.get("test")
    def get_dataset(self):
        """
        Get train and validation datasets from data dictionary.

        Returns:
            (tuple): A tuple containing the training and validation/test datasets.
        """
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif self.args.data.split(".")[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        self.data = data
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            self.data["names"] = {0: "item"}
            self.data["nc"] = 1
        return data["train"], data.get("val") or data.get("test")

    def setup_model(self):
        """
        Load, create, or download model for any task.

        Returns:
            (dict): Optional checkpoint to resume training from.
        """
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            if self.args.teacher_model: 
                self.ema = ModelEMA(self.model,decay = self.args.teacher_ema_decay, tau=1)
                self.ema.ema=self.ema.ema.to(self.device)
                # self.ema.ema.eval()

            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        if self.args.teacher_model:
            self.ema = ModelEMA(self.model,decay = self.args.teacher_ema_decay)
            self.ema.ema=self.ema.ema.to(self.device)
            # self.ema.ema.eval()

        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.args.teacher_model:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """Allows custom preprocessing model inputs and ground truths depending on task type."""
        return batch

    def validate(self, target_data=False):
        """
        Run validation on test set using self.validator.

        Returns:
            (tuple): A tuple containing metrics dictionary and fitness score.
        """
        if self.ema:
            student=self.model
            self.model = self.ema.ema
        if target_data:
            metrics = self.target_validator(self)
        else:
            metrics = self.validator(self)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not target_data and (not self.best_fitness or self.best_fitness < fitness):
            self.best_fitness = fitness
        if target_data and (not self.best_target_fitness or self.best_target_fitness < fitness):
            self.best_target_fitness = fitness
        if self.ema:
            self.model = student
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train",augm_type='default'):
        """Returns dataloader derived from torch.data.Dataloader."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Note:
            This is not needed for classification but necessary for segmentation & detection
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """Set or update model parameters before training."""
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        """Builds target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Returns a string describing training progress."""
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Save training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2  # number of cols
        s = "" if self.csv.exists() else (("%s," * n % tuple(["epoch", "time"] + keys)).rstrip(",") + "\n")  # header
        t = time.time() - self.train_time_start
        with open(self.csv, "a", encoding="utf-8") as f:
            f.write(s + ("%.6g," * n % tuple([self.epoch + 1, t] + vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        """Plot and display metrics visually."""
        pass

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)."""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Perform final evaluation and validation for object detection YOLO model."""
        ckpt = {}
        for f in self.last, self.best:
            if f.exists():
                if f is self.last:
                    ckpt = strip_optimizer(f)
                elif f is self.best:
                    k = "train_results"  # update best.pt train_metrics from last.pt
                    strip_optimizer(f, updates={k: ckpt[k]} if k in ckpt else None)
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                ckpt_args = attempt_load_weights(last).args
                if not isinstance(ckpt_args["data"], dict) and not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)  # reinstate model
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                ):  # allow arg updates to reduce memory or update device on resume
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None or not self.resume:
            return
        best_fitness = 0.0
        start_epoch = ckpt.get("epoch", -1) + 1
        if ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            best_fitness = ckpt["best_fitness"]
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
            self.ema.updates = ckpt["updates"]
        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Construct an optimizer for the given model.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = self.data.get("nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn) or "logit_scale" in fullname:  # weight (no decay)
                    # ContrastiveHead and BNContrastiveHead included here with 'logit_scale'
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )
        return optimizer