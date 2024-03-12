import argparse

from anomalib import TaskType
from anomalib.data import Folder
from anomalib.data.utils.split import TestSplitMode
from anomalib.engine import Engine
from anomalib.models import (Cfa, Cflow, Csflow, Dfkde, Dfm, Draem, Dsr,
                             EfficientAd, Fastflow, Ganomaly, Padim, Patchcore,
                             ReverseDistillation, Rkde, Stfpm, Uflow, WinClip)
from lightning.pytorch.callbacks import ModelCheckpoint

MODELS = {
    "CFA": Cfa,
    "CFlow": Cflow,
    "CSFlow": Csflow,
    "DFKDE": Dfkde,
    "DFM": Dfm,
    "DRAEM": Draem,
    "DSR": Dsr,
    "EFFICIENTAD": EfficientAd,
    "FASTFLOW": Fastflow,
    "GANOMALY": Ganomaly,
    "PADIM": Padim,
    "PATCHCORE": Patchcore,
    "REVERSEDISTILLATION": ReverseDistillation,
    "RKDE": Rkde,
    "STFPM": Stfpm,
    "UFLOW": Uflow,
    "WINCLIP": WinClip,
}


def create_data(args):

    args_name = args.data_root_path.split("/")[-2]
    args_root = args.data_root_path
    args_normal_dir = args.data_normal_dir
    args_abnormal_dir = args.data_abnormal_dir
    args_mask_dir = args.data_mask_dir
    args_normal_test_dir = args.data_normal_test_dir
    args_image_size = args.image_size
    args_batch_size = args.batch_size
    args_seed = args.seed
    args_test_split_mode = TestSplitMode.SYNTHETIC if args.is_testsplitmode_synthetic else TestSplitMode.FROM_DIR

    datamodule = Folder(
        name=args_name,
        root=args_root,
        normal_dir=args_normal_dir,
        abnormal_dir=args_abnormal_dir,
        mask_dir=args_mask_dir,
        normal_test_dir=args_normal_test_dir,
        task=TaskType.SEGMENTATION,
        image_size=args_image_size,
        train_batch_size=args_batch_size,
        eval_batch_size=1,
        normal_split_ratio=0.00001,
        val_split_ratio=0.99999,
        test_split_ratio=0.00001,
        num_workers=0,
        seed=args_seed,
        test_split_mode=args_test_split_mode,
    )

    datamodule.setup()

    return datamodule


def create_model(args):

    args_model = args.model

    model = MODELS[args_model]()

    return model


def create_engine(args):

    args_experiment_path = args.experiment_path

    engine = Engine(task="segmentation",
                    callbacks=[
                        ModelCheckpoint(
                            mode="max",
                            monitor="pixel_F1Score"
                        )
                    ],
                    default_root_dir=args_experiment_path
                    )

    return engine


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Anomalib on DeltaData")
    parser.add_argument("--model", type=str, default="FASTFLOW")
    parser.add_argument("--experiment_path", type=str,
                        default="../../Experiments/Anomalib/")
    parser.add_argument("--data_root_path", type=str,
                        default="../../Datasets/DeltaDataV1_AnomalibForm/P1_V1/")
    parser.add_argument("--data_normal_dir", type=str, default="trn/OK/X/")
    parser.add_argument("--data_abnormal_dir", type=str, default="val/NG/X/")
    parser.add_argument("--data_mask_dir", type=str, default="val/NG/Y/")
    parser.add_argument("--data_normal_test_dir",
                        type=str, default="val/OK/X/")
    parser.add_argument("--is_testsplitmode_synthetic",
                        type=bool, default=False)
    parser.add_argument("--image_size", type=str, default="(256,256)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=16)

    args = parser.parse_args()
    args.image_size = eval(args.image_size)

    # Create data, model and engine
    data = create_data(args)
    model = create_model(args)
    engine = create_engine(args)

    # Train model
    try:
        engine.train(datamodule=data, model=model)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
