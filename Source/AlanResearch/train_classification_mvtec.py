import argparse
import os
from pathlib import Path

import torch
from anomalib import TaskType
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.loggers import AnomalibTensorBoardLogger
from anomalib.models import Dfkde, Dfm, Ganomaly
from lightning.pytorch.callbacks import ModelCheckpoint
from utils import Engine_ModifiedV1

torch.set_float32_matmul_precision("high")

MODELS = {
    "Dfkde": Dfkde,
    "Dfm": Dfm,
    "Ganomaly": Ganomaly,
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
            task=TaskType.CLASSIFICATION,
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
            task=TaskType.CLASSIFICATION,
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

    if args_model == "Dfkde":
        model = MODELS[args_model](backbone=model_backbone,
                                   layers=["layer4"],
                                   pre_trained=True,
                                   n_pca_components=16,
                                   max_training_points=40000)

    elif args_model == "Dfm":
        model = MODELS[args_model](backbone=model_backbone,
                                   layer="layer3",
                                   pre_trained=True,
                                   pooling_kernel_size=4,
                                   pca_level=0.97)

    elif args_model == "Ganomaly":
        model = MODELS[args_model](batch_size=32,
                                   n_features=64,
                                   latent_vec_size=100,
                                   extra_layers=0,
                                   add_final_conv_layer=True,
                                   wadv=1,
                                   wcon=50,
                                   wenc=1,
                                   lr=0.0002,
                                   beta1=0.5,
                                   beta2=0.999,)

    return model


def create_engine(args):

    args_model = args.model
    args_experiment_path = args.experiment_path

    engine_task = "classification"
    engine_callbacks = [
        ModelCheckpoint(
            save_top_k=-1,
            filename="best",
        )
    ]
    engine_image_metrics = ["AUROC", "Accuracy", "Recall", "Specificity"]
    engine_accelerator = "gpu"
    engine_devices = 1
    engine_logger = AnomalibTensorBoardLogger(Path(args.experiment_path) / args.model / args.data_root_path.split("/")[-2], name="logs")


    if args_model == "Dfkde":
        engine = Engine_ModifiedV1(task=engine_task,
                                   callbacks=engine_callbacks,
                                   image_metrics=engine_image_metrics,
                                   default_root_dir=args_experiment_path,
                                   accelerator=engine_accelerator,
                                   devices=engine_devices,
                                   logger=engine_logger)

    elif args_model == "Dfm":
        engine = Engine_ModifiedV1(task=engine_task,
                                   callbacks=engine_callbacks,
                                   image_metrics=engine_image_metrics,
                                   default_root_dir=args_experiment_path,
                                   accelerator=engine_accelerator,
                                   devices=engine_devices,
                                   logger=engine_logger)

    elif args_model == "Ganomaly":
        engine = Engine_ModifiedV1(task=engine_task,
                                   callbacks=engine_callbacks,
                                   image_metrics=engine_image_metrics,
                                   default_root_dir=args_experiment_path,
                                   accelerator=engine_accelerator,
                                   devices=engine_devices,
                                   max_epochs=1000,
                                   logger=engine_logger)

    return engine


def load_model(args):

    lightning_checkpoint_path = Path(args.experiment_path) / args.model / args.data_root_path.split("/")[-2] / "logs/version_0/checkpoints/best.ckpt"
    model=MODELS[args.model].load_from_checkpoint(os.path.abspath(lightning_checkpoint_path))

    return model


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Anomalib on MVTec AD dataset")
    parser.add_argument("--model", type=str, default="Dfkde")
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