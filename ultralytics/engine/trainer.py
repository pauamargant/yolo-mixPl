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

import albumentations as A
import cv2
from ultralytics.utils.ops import xywhn2xyxy, xyxy2xywhn
from ultralytics.utils import LOGGER, colorstr

from ultralytics import __version__
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    RANK,
    TQDM,
    callbacks,
    clean_url,
    emojis,
    yaml_save,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    de_parallel,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
)


class BaseTrainer:
    """
    A base class for creating trainers.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        teacher_model (nn.Module): Teacher model instance.
        teacher_ema (ModelEMA): Teacher EMA instance.
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
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100  # in case users accidentally pass epochs=None with timed training
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolo11n -> yolo11n.pt

        # Teacher-Student Framework attributes
        self.teacher_model = None
        self.teacher_ema = None
        self.args.teacher_ema_decay = getattr(self.args, 'teacher_ema_decay', 0.9998)

        # Datasets
        self.data = None  # Main dataset info dict
        self.trainset = None
        self.testset = None
        self.target_data_info = None  # Target dataset info dict
        self.target_trainset = None
        self.target_train_loader_iter = None  # Add iterator for target train loader
        self.target_testset = None
        self.target_test_loader = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [0, 1, 2]

        # HUB
        self.hub_session = None

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in {-1, 0}:
            callbacks.add_integration_callbacks(self)

        # Strong Augmentation
        self.args.strong_augment = getattr(self.args, 'strong_augment', False)  # Add config flag
        self.strong_augment_pipeline = None  # Initialize pipeline attribute

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
        with torch_distributed_zero_first(LOCAL_RANK):  # Load datasets on rank 0 first
            self.data, self.trainset, self.testset = self.get_dataset()  # Load main dataset
            if self.args.target_data:
                self.target_data_info, self.target_trainset, self.target_testset = self._get_target_dataset()  # Load target dataset

        batch_size = self.batch_size // max(world_size, 1)
        # Main Dataloaders
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size * 2, rank=-1, mode="val"
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)

            # --- Teacher Model Setup ---
            LOGGER.info(f"Setting up Teacher model with EMA decay {self.args.teacher_ema_decay}...")
            self.teacher_model = deepcopy(self.model).eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_ema = ModelEMA(self.teacher_model, decay=self.args.teacher_ema_decay)
            
            self.teacher_ema.ema = deepcopy(self.model).eval()
            self.teacher_ema.updates = 0
            # --- End Teacher Model Setup ---

            if self.args.plots:
                self.plot_training_labels()

        # Target Dataloaders (if target data exists)
        self.target_train_loader = None
        self.target_test_loader = None
        if self.args.target_data and self.target_trainset:
            self.target_train_loader = self.get_dataloader(
                self.target_trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train"
            )
            LOGGER.info(f"Target dataset '{self.args.target_data}' train loader created.")
        if self.args.target_data and self.target_testset and RANK in {-1, 0}:
            self.target_test_loader = self.get_dataloader(
                self.target_testset, batch_size=batch_size * 2, rank=-1, mode="val"
            )
            LOGGER.info(f"Target dataset '{self.args.target_data}' val loader created.")

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

        # Build strong augmentation pipeline if enabled (only on rank 0/-1 as it's used there)
        if RANK in {-1, 0} and self.args.strong_augment:
            LOGGER.info(f"{colorstr('Strong Augment:')} Enabling strong augmentations for target domain.")
            self.strong_augment_pipeline = self._build_strong_augment_pipeline()

    def _build_strong_augment_pipeline(self):
        """Builds the Albumentations pipeline for strong augmentations."""
        pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.85, 1.15), translate_percent=(0.05, 0.05), rotate=(-10, 10), shear=(-5, 5), p=0.7, keep_ratio=True, fit_output=True),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        ], bbox_params=A.BboxParams(format='pascal_voc',
                                   label_fields=['class_labels'],
                                   min_visibility=0.1,
                                   min_area=10))
        return pipeline

    def _apply_strong_augmentations(self, target_img_tensor, target_pseudo_labels):
        """
        Applies strong augmentations using Albumentations to target images and pseudo-labels.

        Args:
            target_img_tensor (torch.Tensor): Batch of target images (BCHW, float, 0-1).
            target_pseudo_labels (torch.Tensor | None): Pseudo-labels ([N, 6] tensor [b_idx, cls, xywhn]).

        Returns:
            (tuple): Tuple containing:
                - augmented_img_tensor (torch.Tensor): Augmented images (BCHW, float, 0-1).
                - augmented_pseudo_labels (torch.Tensor | None): Augmented pseudo-labels ([M, 6] tensor [b_idx, cls, xywhn]).
        """
        if target_pseudo_labels is None or len(target_pseudo_labels) == 0 or self.strong_augment_pipeline is None:
            return target_img_tensor, target_pseudo_labels

        device = target_img_tensor.device
        dtype = target_img_tensor.dtype
        bs, _, h, w = target_img_tensor.shape

        # 1. Convert images to list of NumPy arrays (HWC, uint8)
        images_np = []
        for i in range(bs):
            img = target_img_tensor[i].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).round().astype(np.uint8)
            images_np.append(img)

        # 2. Convert pseudo-labels to Albumentations format (list of lists per image)
        labels_by_image = [[] for _ in range(bs)]
        class_labels_by_image = [[] for _ in range(bs)]

        if target_pseudo_labels is not None and len(target_pseudo_labels) > 0:
            original_indices = target_pseudo_labels[:, 0].long()
            original_classes = target_pseudo_labels[:, 1]
            original_boxes_xywhn = target_pseudo_labels[:, 2:]
            original_boxes_xyxy = xywhn2xyxy(original_boxes_xywhn, w=w, h=h, padw=0, padh=0)

            for i in range(len(target_pseudo_labels)):
                img_idx = original_indices[i]
                if img_idx >= bs: continue

                cls_id = original_classes[i].item()
                box_xyxy = original_boxes_xyxy[i].cpu().numpy().tolist()
                x_min, y_min, x_max, y_max = box_xyxy
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w, x_max), min(h, y_max)

                if x_max > x_min and y_max > y_min:
                    labels_by_image[img_idx].append([x_min, y_min, x_max, y_max])
                    class_labels_by_image[img_idx].append(cls_id)

        # 3. Apply augmentations image by image
        augmented_images_np = []
        augmented_labels_list = []

        for i in range(bs):
            img_np = images_np[i]
            bboxes = labels_by_image[i]
            class_labels = class_labels_by_image[i]

            try:
                transformed = self.strong_augment_pipeline(image=img_np, bboxes=bboxes, class_labels=class_labels)
                aug_img_np = transformed['image']
                aug_bboxes_xyxy = transformed['bboxes']
                aug_class_labels = transformed['class_labels']

                # Ensure augmented image is resized back to target size if needed (e.g., after Affine(fit_output=True))
                target_h, target_w = self.args.imgsz, self.args.imgsz
                if aug_img_np.shape[0] != target_h or aug_img_np.shape[1] != target_w:
                    aug_img_np = cv2.resize(aug_img_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                augmented_images_np.append(aug_img_np)

                # 4. Convert augmented labels back
                if aug_bboxes_xyxy:
                    aug_h, aug_w = aug_img_np.shape[:2]
                    aug_bboxes_xyxy_tensor = torch.tensor(aug_bboxes_xyxy, device=device, dtype=dtype)
                    aug_cls_tensor = torch.tensor(aug_class_labels, device=device, dtype=dtype).unsqueeze(1)

                    # Clip boxes strictly within image boundaries before normalization
                    aug_bboxes_xyxy_tensor[:, [0, 2]] = aug_bboxes_xyxy_tensor[:, [0, 2]].clamp(0, aug_w)
                    aug_bboxes_xyxy_tensor[:, [1, 3]] = aug_bboxes_xyxy_tensor[:, [1, 3]].clamp(0, aug_h)

                    aug_bboxes_xywhn = xyxy2xywhn(aug_bboxes_xyxy_tensor, w=aug_w, h=aug_h, clip=False, eps=1e-3)

                    # Filter out invalid boxes (width or height <= 0 after clipping/conversion)
                    valid_boxes_mask = (aug_bboxes_xywhn[:, 2] > 0) & (aug_bboxes_xywhn[:, 3] > 0)
                    if valid_boxes_mask.any():
                        aug_bboxes_xywhn = aug_bboxes_xywhn[valid_boxes_mask]
                        aug_cls_tensor = aug_cls_tensor[valid_boxes_mask]
                        batch_idx_tensor = torch.full((len(aug_bboxes_xywhn), 1), i, device=device, dtype=dtype)
                        augmented_labels_list.append(torch.cat([batch_idx_tensor, aug_cls_tensor, aug_bboxes_xywhn], dim=1))

            except Exception as e:
                LOGGER.warning(f"Skipping strong augmentation for target image {i} due to error: {e}")
                img_np_resized = cv2.resize(images_np[i], (self.args.imgsz, self.args.imgsz), interpolation=cv2.INTER_LINEAR)
                augmented_images_np.append(img_np_resized)

        # 5. Convert augmented images back to tensor
        if not augmented_images_np:
            return target_img_tensor, target_pseudo_labels

        augmented_img_tensor = torch.stack([torch.from_numpy(img).permute(2, 0, 1) for img in augmented_images_np]).to(device).float() / 255.0

        # 6. Concatenate all augmented labels
        if augmented_labels_list:
            augmented_pseudo_labels = torch.cat(augmented_labels_list, dim=0)
        else:
            augmented_pseudo_labels = torch.empty((0, 6), device=device, dtype=dtype)

        return augmented_img_tensor, augmented_pseudo_labels

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
        if hasattr(self, 'target_train_loader') and self.target_train_loader:
            self.target_train_loader_iter = iter(self.target_train_loader)
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

                # --- Get Target Batch and Generate Pseudo-Labels ---
                target_batch_data = None
                if RANK in {-1, 0} and hasattr(self, 'target_train_loader_iter') and self.target_train_loader_iter and \
                   hasattr(self, 'teacher_ema') and self.teacher_ema:
                    try:
                        target_batch = next(self.target_train_loader_iter)
                    except StopIteration:
                        self.target_train_loader_iter = iter(self.target_train_loader)
                        target_batch = next(self.target_train_loader_iter)

                    target_batch = self.preprocess_batch(target_batch)
                    with torch.no_grad():
                        teacher = self.teacher_ema.ema.to(self.device).eval()
                        teacher_preds_raw = teacher(target_batch['img'])

                        pseudo_labels = self._format_pseudo_labels(
                            preds_raw=teacher_preds_raw,
                            batch=target_batch,
                            conf_thres=self.args.pseudo_label_conf_thres,
                            iou_thres=self.args.nms_iou_thres
                        )

                        target_batch_data = {
                            'img': target_batch['img'],
                            'teacher_preds_raw': teacher_preds_raw,
                            'pseudo_labels': pseudo_labels,
                            'is_augmented': False
                        }

                    # --- Apply Strong Augmentation ---
                    if self.args.strong_augment and self.strong_augment_pipeline:
                        if target_batch_data['pseudo_labels'] is not None and len(target_batch_data['pseudo_labels']) > 0:
                            try:
                                aug_img, aug_labels = self._apply_strong_augmentations(
                                    target_img_tensor=target_batch_data['img'].clone(),
                                    target_pseudo_labels=target_batch_data['pseudo_labels'].clone()
                                )
                                target_batch_data['img'] = aug_img
                                target_batch_data['pseudo_labels'] = aug_labels
                                target_batch_data['is_augmented'] = True

                            except Exception as e:
                                LOGGER.warning(f"Strong augmentation failed for batch step {i}: {e}")
                    # --- End Strong Augmentation ---

                # --- End Pseudo-Label Generation & Augmentation ---

                # Forward pass with student model
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    if target_batch_data:
                        batch['target_batch_data'] = target_batch_data

                    loss, self.loss_items = self.model(batch)
                    self.loss = loss.sum()
                    if RANK != -1:
                        self.loss *= world_size
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
                            f"{self._get_memory():.3g}G",
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                            batch["cls"].shape[0],
                            batch["img"].shape[-1],
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                teacher_metrics = {}
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()

                    # --- Teacher Validation ---
                    if hasattr(self, 'teacher_ema') and self.teacher_ema:
                        LOGGER.info("Validating Teacher model...")
                        teacher_src_metrics = self.validator(model=self.teacher_ema.ema)
                        teacher_metrics["teacher/src_precision"] = teacher_src_metrics.get('metrics/precision(B)', 0)
                        teacher_metrics["teacher/src_recall"] = teacher_src_metrics.get('metrics/recall(B)', 0)
                        teacher_metrics["teacher/src_map50-95"] = teacher_src_metrics.get('metrics/mAP50-95(B)', 0)
                        teacher_metrics["teacher/src_map50"] = teacher_src_metrics.get('metrics/mAP50(B)', 0)

                        if hasattr(self, 'target_test_loader') and self.target_test_loader:
                            LOGGER.info("Validating Teacher model on Target dataset...")
                            original_dataloader = self.validator.dataloader
                            self.validator.dataloader = self.target_test_loader
                            teacher_tgt_metrics = self.validator(model=self.teacher_ema.ema)
                            teacher_metrics["teacher/tgt_precision"] = teacher_tgt_metrics.get('metrics/precision(B)', 0)
                            teacher_metrics["teacher/tgt_recall"] = teacher_tgt_metrics.get('metrics/recall(B)', 0)
                            teacher_metrics["teacher/tgt_map50-95"] = teacher_tgt_metrics.get('metrics/mAP50-95(B)', 0)
                            teacher_metrics["teacher/tgt_map50"] = teacher_tgt_metrics.get('metrics/mAP50(B)', 0)
                            self.validator.dataloader = original_dataloader
                        LOGGER.info("Teacher validation finished.")
                    # --- End Teacher Validation ---

                all_metrics_to_save = {**self.label_loss_items(self.tloss), **self.metrics, **self.lr, **teacher_metrics}
                self.save_metrics(metrics=all_metrics_to_save)
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch
                self.stop |= epoch >= self.epochs
            self.run_callbacks("on_fit_epoch_end")
            if self._get_memory(fraction=True) > 0.5:
                self._clear_memory()

            if RANK != -1:
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                self.stop = broadcast_list[0]
            if self.stop:
                break
            epoch += 1

        if RANK in {-1, 0}:
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

    def _format_pseudo_labels(self, preds_raw, batch, conf_thres, iou_thres):
        """
        Formats raw teacher predictions into pseudo-labels for detection.

        Applies NMS using the provided confidence and IoU thresholds.

        Args:
            preds_raw (list | torch.Tensor): Raw predictions from the teacher model (before NMS).
            batch (dict): The target batch dictionary (must contain 'img' and 'batch_idx').
            conf_thres (float): Confidence threshold for NMS (used directly from pseudo_label_conf_thres).
            iou_thres (float): IoU threshold for NMS.

        Returns:
            (torch.Tensor | None): Formatted pseudo-labels in [batch_idx, cls, xywhn] format, or None.
        """
        formatted_labels = []
        img_idx = batch.get('batch_idx')
        if img_idx is None:
            LOGGER.warning("Batch index ('batch_idx') not found in target batch for pseudo-label formatting.")
            return None
        img_h, img_w = batch['img'].shape[2:]

        preds_nms = non_max_suppression(preds_raw,
                                        conf_thres=conf_thres,
                                        iou_thres=iou_thres,
                                        multi_label=False,
                                        max_det=300)

        for si, pred in enumerate(preds_nms):
            if pred is None or len(pred) == 0:
                continue

            batch_idx_tensor = torch.full((pred.shape[0], 1), img_idx[si], device=self.device, dtype=pred.dtype)
            cls_tensor = pred[:, 5:6]
            xywhn_tensor = xyxy2xywhn(pred[:, :4], w=img_w, h=img_h, clip=True)

            formatted_labels.append(torch.cat((batch_idx_tensor, cls_tensor, xywhn_tensor), dim=1))

        if not formatted_labels:
            return None

        return torch.cat(formatted_labels, 0)

    def auto_batch(self, max_num_obj=0):
        """Calculate optimal batch size based on model and device memory constraints."""
        return check_train_batch_size(
            model=self.model,
            imgsz=self.args.imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
        )

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
        import pandas as pd

        return pd.read_csv(self.csv).to_dict(orient="list")

    def _model_train(self):
        """Set model in training mode."""
        self.model.train()
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        buffer = io.BytesIO()
        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": None,
            "ema": deepcopy(self.ema.ema).half(),
            "updates": self.ema.updates,
            "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
            "train_args": vars(self.args),
            "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
            "train_results": self.read_results_csv() if self.csv.exists() else None,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }
        if self.teacher_ema:
            ckpt["teacher_ema"] = deepcopy(self.teacher_ema.ema).half()
            ckpt["teacher_updates"] = self.teacher_ema.updates

        torch.save(ckpt, buffer)
        serialized_ckpt = buffer.getvalue()

        self.last.write_bytes(serialized_ckpt)
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)

    def get_dataset(self):
        """
        Get train and validation datasets from the primary data source specified in args.data.

        Returns:
            (tuple): A tuple containing the dataset info dict, training dataset, and validation/test dataset.
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
                    self.args.data = data["yaml_file"]
            else:
                raise ValueError(f"Unsupported dataset format for primary data: {self.args.data}")
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e

        self.data = data
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class for primary dataset.")
            data["names"] = {0: "item"}
            data["nc"] = 1
        return data, data["train"], data.get("val") or data.get("test")

    def _get_target_dataset(self):
        """
        Get train and validation datasets from the target data source specified in args.target_data.

        Returns:
            (tuple): A tuple containing the target dataset info dict, target training dataset,
                     and target validation/test dataset. Returns (None, None, None) if args.target_data is not set.
        """
        if not self.args.target_data:
            return None, None, None

        LOGGER.info(f"Loading target dataset from '{clean_url(self.args.target_data)}'...")
        try:
            if self.args.task == "classify":
                target_data = check_cls_dataset(self.args.target_data)
            elif self.args.target_data.split(".")[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                target_data = check_det_dataset(self.args.target_data)
            else:
                raise ValueError(f"Unsupported dataset format for target data: {self.args.target_data}")

        except Exception as e:
            raise RuntimeError(emojis(f"Target dataset '{clean_url(self.args.target_data)}' error ❌ {e}")) from e

        return target_data, target_data.get("train"), target_data.get("val") or target_data.get("test")

    def setup_model(self):
        """
        Load, create, or download model for any task.

        Returns:
            (dict): Optional checkpoint to resume training from.
        """
        if isinstance(self.model, torch.nn.Module):
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        if not self.data:
            self.data, _, _ = self.get_dataset()

        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)
        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model.module if hasattr(self.model, 'module') else self.model)

        if self.teacher_ema and RANK in {-1, 0}:
            student_model_state = self.model.module if hasattr(self.model, 'module') else self.model
            self.teacher_ema.update(student_model_state)

    def preprocess_batch(self, batch):
        """Allows custom preprocessing model inputs and ground truths depending on task type."""
        return batch

    def validate(self):
        """
        Run validation on test set using self.validator.

        Returns:
            (tuple): A tuple containing metrics dictionary and fitness score.
        """
        metrics = self.validator(self)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        if not hasattr(self, "data") or not self.data:
            raise RuntimeError("Primary dataset information (self.data) must be loaded before setting up the model.")
        raise NotImplementedError("get_model function not implemented in BaseTrainer. Implement in subclass.")

    def get_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
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
        if not self.data:
            raise RuntimeError("Primary dataset information (self.data) must be loaded before setting model attributes.")
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        """Builds target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Returns a string describing training progress."""
        return ""

    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Save training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2
        s = "" if self.csv.exists() else (("%s," * n % tuple(["epoch", "time"] + keys)).rstrip(",") + "\n")
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
                    k = "train_results"
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

                ckpt_args = attempt_load_weights(last).args
                if not isinstance(ckpt_args["data"], dict) and not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                ):
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
            self.optimizer.load_state_dict(ckpt["optimizer"])
            best_fitness = ckpt["best_fitness"]
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
            self.ema.updates = ckpt["updates"]

        if hasattr(self, 'teacher_ema') and self.teacher_ema and ckpt.get("teacher_ema"):
            LOGGER.info("Resuming teacher EMA state...")
            self.teacher_ema.ema.load_state_dict(ckpt["teacher_ema"].float().state_dict())
            self.teacher_ema.updates = ckpt.get("teacher_updates", self.teacher_ema.updates)
        elif hasattr(self, 'teacher_ema') and self.teacher_ema:
            LOGGER.warning("Resuming from checkpoint without teacher EMA state. Re-initializing teacher EMA from current student EMA.")
            self.teacher_ema.ema.load_state_dict(self.ema.ema.float().state_dict())
            self.teacher_ema.updates = self.ema.updates

        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]
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
        g = [], [], []
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = self.data.get("nc", 10)
            lr_fit = round(0.002 * 5 / (4 + nc), 6)
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:
                    g[2].append(param)
                elif isinstance(module, bn) or "logit_scale" in fullname:
                    g[1].append(param)
                else:
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

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )
        return optimizer
