import argparse
import os
from pathlib import Path

import torch
from anomalib import TaskType
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.loggers import AnomalibTensorBoardLogger
from anomalib.models import (Cfa, Cflow, Csflow, Draem, Dsr, EfficientAd,
                             Fastflow, Padim, Patchcore, ReverseDistillation,
                             Stfpm, Uflow)
from lightning.pytorch.callbacks import ModelCheckpoint
from utils import Engine_ModifiedV1

torch.set_float32_matmul_precision("high")

MODELS = {
    "Cfa": Cfa,
    "Cflow": Cflow,
    "Csflow": Csflow,
    "Dsr": Dsr,
    "Draem": Draem,
    "EfficientAd": EfficientAd,
    "Fastflow": Fastflow,
    "Padim": Padim,
    "Patchcore": Patchcore,
    "ReverseDistillation": ReverseDistillation,
    "Stfpm": Stfpm,
    "Uflow": Uflow,
}


def create_data(args):

    args_model = args.model
    args_name = args.data_root_path.split("/")[-2]
    args_root = args.data_root_path
    args_normal_dir = args.data_normal_dir
    args_image_size = args.image_size
    args_seed = args.seed


    if args_model == "Draem":
        datamodule = Folder(
            name=args_name,
            root=args_root,
            normal_dir=args_normal_dir,
            task=TaskType.SEGMENTATION,
            image_size=args_image_size,
            train_batch_size=16,
            eval_batch_size=1,
            num_workers=0,
            seed=args_seed,
            val_split_mode=ValSplitMode.SYNTHETIC,
            val_split_ratio=0.1,
            test_split_mode=TestSplitMode.SYNTHETIC,
            test_split_ratio=0.1,
        )
    else:
        datamodule = Folder(
            name=args_name,
            root=args_root,
            normal_dir=args_normal_dir,
            task=TaskType.SEGMENTATION,
            image_size=args_image_size,
            train_batch_size=32,
            eval_batch_size=1,
            num_workers=0,
            seed=args_seed,
            val_split_mode=ValSplitMode.SYNTHETIC,
            val_split_ratio=0.1,
            test_split_mode=TestSplitMode.SYNTHETIC,
            test_split_ratio=0.1,
        )

    datamodule.setup()

    return datamodule


def create_model(args):

    args_model = args.model

    model_backbone = "resnet18"

    if args_model == "Cfa":
        model = MODELS[args_model](backbone=model_backbone,
                                   gamma_c=1,
                                   gamma_d=1,
                                   num_nearest_neighbors=3,
                                   num_hard_negative_features=3,
                                   radius=1.0e-05)

    elif args_model == "Cflow":
        model = MODELS[args_model](backbone=model_backbone,
                                   layers=["layer1", "layer2", "layer3"],
                                   pre_trained=True,
                                   fiber_batch_size=64,
                                   decoder="freia-cflow",
                                   condition_vector=128,
                                   coupling_blocks=8,
                                   clamp_alpha=1.9,
                                   permute_soft=False,
                                   lr=0.0001)

    elif args_model == "Csflow":
        model = MODELS[args_model](cross_conv_hidden_channels=1024,
                                   n_coupling_blocks=4,
                                   clamp=3,
                                   num_channels=3)

    elif args_model == "Dsr":
        model = MODELS[args_model](latent_anomaly_strength=0.2,
                                   upsampling_train_ratio=0.7)

    elif args_model == "Draem":
        model = MODELS[args_model](beta=(0.1,1.0),
                                   enable_sspcab=False,
                                   sspcab_lambda=0.1,
                                   anomaly_source_path=None)

    elif args_model == "EfficientAd":
        model = MODELS[args_model](teacher_out_channels=384,
                                   model_size="small",
                                   lr=0.0001,
                                   weight_decay=1.0e-05,
                                   padding=False,
                                   pad_maps=True,
                                   batch_size=1)

    elif args_model == "Fastflow":
        model = MODELS[args_model](backbone=model_backbone,
                                   pre_trained=True,
                                   flow_steps=8,
                                   conv3x3_only=False,
                                   hidden_ratio=1.0)

    elif args_model == "Padim":
        model = MODELS[args_model](backbone=model_backbone,
                                   layers=["layer1", "layer2", "layer3"],
                                   pre_trained=True,
                                   n_features=None)

    elif args_model == "Patchcore":
        model = MODELS[args_model](backbone=model_backbone,
                                   layers=["layer1", "layer2", "layer3"],
                                   pre_trained=True,
                                   coreset_sampling_ratio=0.1,
                                   num_neighbors=9)

    elif args_model == "ReverseDistillation":
        model = MODELS[args_model](backbone=model_backbone,
                                   layers=["layer1", "layer2", "layer3"],
                                   pre_trained=True,)

    elif args_model == "Stfpm":
        model = MODELS[args_model](backbone=model_backbone,
                                   layers=["layer1", "layer2", "layer3"])

    elif args_model == "Uflow":
        model = MODELS[args_model](backbone=model_backbone,
                                   flow_steps=4,
                                   permute_soft=False,
                                   affine_clamp=2.0,
                                   affine_subnet_channels_ratio=1.0)

    return model


def create_engine(args):

    args_model = args.model
    args_experiment_path = args.experiment_path

    engine_task = "segmentation"
    engine_callbacks = [
        ModelCheckpoint(
            mode="max",
            monitor="pixel_AUROC",
            filename="best",
        ),
    ]
    engine_image_metrics = ["AUROC", "Accuracy", "Recall", "Specificity"]
    engine_pixel_metrics = ["AUROC", "Accuracy", "Recall", "Specificity"]
    engine_accelerator = "gpu"
    engine_devices = 1
    engine_max_epochs = 1000
    engine_logger = AnomalibTensorBoardLogger(Path(args.experiment_path) / args.model / args.data_root_path.split("/")[-2], name="logs")

    if args_model == "Cfa":
        engine = Engine_ModifiedV1(task=engine_task,
                                   callbacks=engine_callbacks,
                                   image_metrics=engine_image_metrics,
                                   pixel_metrics=engine_pixel_metrics,
                                   default_root_dir=args_experiment_path,
                                   accelerator=engine_accelerator,
                                   devices=engine_devices,
                                   max_epochs=engine_max_epochs,
                                   logger=engine_logger)

    elif args_model == "Cflow":
        engine = Engine_ModifiedV1(task=engine_task,
                                   callbacks=engine_callbacks,
                                   image_metrics=engine_image_metrics,
                                   pixel_metrics=engine_pixel_metrics,
                                   default_root_dir=args_experiment_path,
                                   accelerator=engine_accelerator,
                                   devices=engine_devices,
                                   max_epochs=engine_max_epochs,
                                   logger=engine_logger)

    elif args_model == "Csflow":
        engine = Engine_ModifiedV1(task=engine_task,
                                   callbacks=engine_callbacks,
                                   image_metrics=engine_image_metrics,
                                   pixel_metrics=engine_pixel_metrics,
                                   default_root_dir=args_experiment_path,
                                   accelerator=engine_accelerator,
                                   devices=engine_devices,
                                   max_epochs=engine_max_epochs,
                                   logger=engine_logger)

    elif args_model == "Dsr":
        engine = Engine_ModifiedV1(task=engine_task,
                                   callbacks=engine_callbacks,
                                   image_metrics=engine_image_metrics,
                                   pixel_metrics=engine_pixel_metrics,
                                   default_root_dir=args_experiment_path,
                                   accelerator=engine_accelerator,
                                   devices=engine_devices,
                                   max_epochs=engine_max_epochs,
                                   logger=engine_logger)

    elif args_model == "Draem":
        engine = Engine_ModifiedV1(task=engine_task,
                                   callbacks=engine_callbacks,
                                   image_metrics=engine_image_metrics,
                                   pixel_metrics=engine_pixel_metrics,
                                   default_root_dir=args_experiment_path,
                                   accelerator=engine_accelerator,
                                   devices=engine_devices,
                                   max_epochs=engine_max_epochs,
                                   logger=engine_logger)

    elif args_model == "EfficientAd":
        engine = Engine_ModifiedV1(task=engine_task,
                                   callbacks=engine_callbacks,
                                   image_metrics=engine_image_metrics,
                                   pixel_metrics=engine_pixel_metrics,
                                   default_root_dir=args_experiment_path,
                                   accelerator=engine_accelerator,
                                   devices=engine_devices,
                                   max_epochs=engine_max_epochs,
                                   max_steps=350000,
                                   logger=engine_logger)

    elif args_model == "Fastflow":
        engine = Engine_ModifiedV1(task=engine_task,
                                   callbacks=engine_callbacks,
                                   image_metrics=engine_image_metrics,
                                   pixel_metrics=engine_pixel_metrics,
                                   default_root_dir=args_experiment_path,
                                   accelerator=engine_accelerator,
                                   devices=engine_devices,
                                   max_epochs=engine_max_epochs,
                                   logger=engine_logger)

    elif args_model == "Padim":
        engine = Engine_ModifiedV1(task=engine_task,
                                   callbacks=engine_callbacks,
                                   image_metrics=engine_image_metrics,
                                   pixel_metrics=engine_pixel_metrics,
                                   default_root_dir=args_experiment_path,
                                   accelerator=engine_accelerator,
                                   devices=engine_devices,
                                   max_epochs=engine_max_epochs,
                                   logger=engine_logger)

    elif args_model == "Patchcore":
        engine = Engine_ModifiedV1(task=engine_task,
                                   callbacks=engine_callbacks,
                                   image_metrics=engine_image_metrics,
                                   pixel_metrics=engine_pixel_metrics,
                                   default_root_dir=args_experiment_path,
                                   accelerator=engine_accelerator,
                                   devices=engine_devices,
                                   max_epochs=engine_max_epochs,
                                   logger=engine_logger)

    elif args_model == "ReverseDistillation":
        engine = Engine_ModifiedV1(task=engine_task,
                                   callbacks=engine_callbacks,
                                   image_metrics=engine_image_metrics,
                                   pixel_metrics=engine_pixel_metrics,
                                   default_root_dir=args_experiment_path,
                                   accelerator=engine_accelerator,
                                   devices=engine_devices,
                                   max_epochs=engine_max_epochs,
                                   check_val_every_n_epoch=engine_max_epochs,
                                   logger=engine_logger)

    elif args_model == "Stfpm":
        engine = Engine_ModifiedV1(task=engine_task,
                                   callbacks=engine_callbacks,
                                   image_metrics=engine_image_metrics,
                                   pixel_metrics=engine_pixel_metrics,
                                   default_root_dir=args_experiment_path,
                                   accelerator=engine_accelerator,
                                   devices=engine_devices,
                                   max_epochs=engine_max_epochs,
                                   logger=engine_logger)

    elif args_model == "Uflow":
        engine = Engine_ModifiedV1(task=engine_task,
                                   callbacks=engine_callbacks,
                                   image_metrics=engine_image_metrics,
                                   pixel_metrics=engine_pixel_metrics,
                                   default_root_dir=args_experiment_path,
                                   accelerator=engine_accelerator,
                                   devices=engine_devices,
                                   max_epochs=engine_max_epochs,
                                   logger=engine_logger)


    return engine


def load_model(args):

    lightning_checkpoint_path = Path(args.experiment_path) / args.model / args.data_root_path.split("/")[-2] / "logs/version_0/checkpoints/best.ckpt"
    model=MODELS[args.model].load_from_checkpoint(os.path.abspath(lightning_checkpoint_path))

    return model


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Anomalib on MVTec AD dataset")
    parser.add_argument("--model", type=str, default="Stfpm")
    parser.add_argument("--experiment_path", type=str,
                        default="../../Experiments/Anomalib/")
    parser.add_argument("--data_root_path", type=str,
                        default="../../Datasets/MVTec/bottle/")
    parser.add_argument("--data_normal_dir", type=str, default="train/good/")
    parser.add_argument("--image_size", type=str, default="(256,256)")
    parser.add_argument("--seed", type=int, default=16)

    args = parser.parse_args()
    args.image_size = eval(args.image_size)

    # Create data, model and engine
    data = create_data(args)
    model = create_model(args)
    engine = create_engine(args)

    # Train model
    engine.train(datamodule=data, model=model)

    # Export model
    model = load_model(args)
    engine.export(model=model, export_type="torch")

if __name__ == "__main__":
    main()