from collections.abc import Iterable
from pathlib import Path

from anomalib import TaskType
from anomalib.callbacks.metrics import _MetricsCallback
from anomalib.callbacks.normalization import get_normalization_callback
from anomalib.callbacks.post_processor import _PostProcessorCallback
from anomalib.callbacks.thresholding import _ThresholdCallback
from anomalib.callbacks.timer import TimerCallback
from anomalib.callbacks.visualizer import _VisualizationCallback
from anomalib.engine import Engine
from anomalib.models import AnomalyModule
from anomalib.utils.normalization import NormalizationMethod
from anomalib.utils.types import NORMALIZATION, THRESHOLD
from anomalib.utils.visualization import ImageVisualizer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import Logger
from lightning.pytorch.trainer import Trainer


class _ThresholdCallback_ModifiedV1(_ThresholdCallback):
    def __init__(self, threshold: THRESHOLD = "F1AdaptiveThreshold") -> None:
        super().__init__(threshold)

    def _compute(self, pl_module: AnomalyModule) -> None:

        pl_module.image_threshold.compute()
        pl_module.pixel_threshold.value = pl_module.image_threshold.value

class Engine_ModifiedV1(Engine):
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

        super().__init__(callbacks, normalization, threshold, task, image_metrics, pixel_metrics, logger, default_root_dir, **kwargs)

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
        _callbacks.append(_ThresholdCallback_ModifiedV1(self.threshold))
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


if __name__ == "__main__":
    pass