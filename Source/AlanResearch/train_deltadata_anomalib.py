import argparse
from collections.abc import Iterable
from pathlib import Path

from anomalib import TaskType
from anomalib.callbacks.metrics import _MetricsCallback
from anomalib.callbacks.normalization import get_normalization_callback
from anomalib.callbacks.post_processor import _PostProcessorCallback
from anomalib.callbacks.thresholding import _ThresholdCallback
from anomalib.callbacks.timer import TimerCallback
from anomalib.callbacks.visualizer import _VisualizationCallback
from anomalib.data import Folder
from anomalib.data.utils.split import TestSplitMode
from anomalib.engine import Engine
from anomalib.models import (AnomalyModule, Cfa, Cflow, Csflow, Dfkde, Dfm,
                             Draem, Dsr, EfficientAd, Fastflow, Ganomaly,
                             Padim, Patchcore, ReverseDistillation, Rkde,
                             Stfpm, Uflow, WinClip)
from anomalib.utils.normalization import NormalizationMethod
from anomalib.utils.types import NORMALIZATION, THRESHOLD
from anomalib.utils.visualization import ImageVisualizer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import Logger

MODELS = {
    "Cfa": Cfa,
    "Cflow": Cflow,
    "Csflow": Csflow,
    "Dfkde": Dfkde,
    "Dfm": Dfm,
    "Draem": Draem,
    "Dsr": Dsr,
    "EfficientAd": EfficientAd,
    "Fastflow": Fastflow,
    "Ganomaly": Ganomaly,
    "Padim": Padim,
    "Patchcore": Patchcore,
    "ReverseDistillation": ReverseDistillation,
    "Rkde": Rkde,
    "Stfpm": Stfpm,
    "Uflow": Uflow,
    "WinClip": WinClip,
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

    engine = Engine_Alan(task="segmentation",
                         callbacks=[
                             ModelCheckpoint(
                                 mode="min",
                                 monitor="pixel_Accuracy",
                             )
                         ],
                         image_metrics=["Recall", "Precision",
                                        "Accuracy", "Specificity"],
                         pixel_metrics=["Recall", "Precision",
                                        "Accuracy", "Specificity"],
                         default_root_dir=args_experiment_path,)

    return engine


class _ThresholdCallback_Alan(_ThresholdCallback):
    def __init__(self, threshold: THRESHOLD = "F1AdaptiveThreshold") -> None:
        super().__init__(threshold)

    def _compute(self, pl_module: AnomalyModule) -> None:

        pl_module.image_threshold.compute()
        pl_module.pixel_threshold.value = pl_module.image_threshold.value


class Engine_Alan(Engine):
    def __init__(
        self,
        callbacks: list[Callback] | None = None,
        normalization: NORMALIZATION = NormalizationMethod.MIN_MAX,
        threshold: THRESHOLD = "F1AdaptiveThreshold",
        task: TaskType | str = TaskType.SEGMENTATION,
        image_metrics: str | list[str] | None = None,
        pixel_metrics: str | list[str] | None = None,
        logger: Logger | Iterable[Logger] | bool | None = None,
        default_root_dir: str | Path = "results",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

    def _setup_anomalib_callbacks(self) -> None:
        """Set up callbacks for the trainer."""
        _callbacks: list[Callback] = []

        # Add ModelCheckpoint if it is not in the callbacks list.
        has_checkpoint_callback = any(isinstance(
            c, ModelCheckpoint) for c in self._cache.args["callbacks"])
        if has_checkpoint_callback is False:
            _callbacks.append(
                ModelCheckpoint(
                    dirpath=self._cache.args["default_root_dir"] /
                    "weights" / "lightning",
                    filename="model",
                    auto_insert_metric_name=False,
                ),
            )

        # Add the post-processor callbacks.
        _callbacks.append(_PostProcessorCallback())

        # Add the the normalization callback.
        normalization_callback = get_normalization_callback(self.normalization)
        if normalization_callback is not None:
            _callbacks.append(normalization_callback)

        # Add the thresholding and metrics callbacks.
        _callbacks.append(_ThresholdCallback_Alan(self.threshold))
        _callbacks.append(_MetricsCallback(
            self.task, self.image_metric_names, self.pixel_metric_names))

        _callbacks.append(
            _VisualizationCallback(
                visualizers=ImageVisualizer(task=self.task),
                save=True,
                root=self._cache.args["default_root_dir"] / "images",
            ),
        )

        _callbacks.append(TimerCallback())

        # Combine the callbacks, and update the trainer callbacks.
        self._cache.args["callbacks"] = _callbacks + \
            self._cache.args["callbacks"]


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Anomalib on DeltaData")
    parser.add_argument("--model", type=str, default="Stfpm")
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
        engine.export(model=model, export_type="torch")

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
