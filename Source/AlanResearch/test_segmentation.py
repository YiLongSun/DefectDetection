import json
import logging
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from anomalib.data.utils import get_image_filenames, read_image
from anomalib.deploy import TorchInferencer
from anomalib.utils.visualization import ImageVisualizer
from PIL import Image
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def get_parser() -> ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--weights", type=Path,
                        required=True, help="Path to model weights")
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to images to infer.")
    parser.add_argument("--output", type=Path, required=False,
                        help="Path to save the output image.")
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="auto",
        help="Device to use for inference. Defaults to auto.",
        # cuda and gpu are the same but provided for convenience
        choices=["auto", "cpu", "gpu", "cuda"],
    )
    parser.add_argument(
        "--task",
        type=str,
        required=False,
        help="Task type.",
        default="segmentation",
        choices=["classification", "detection", "segmentation"],
    )
    parser.add_argument(
        "--visualization_mode",
        type=str,
        required=False,
        default="full",
        help="Visualization mode.",
        choices=["full", "simple"],
    )
    parser.add_argument(
        "--show",
        action="store_true",
        required=False,
        help="Show the visualized predictions on the screen.",
    )

    return parser


def infer(args: Namespace) -> None:
    """Infer predictions.

    Show/save the output if path is to an image. If the path is a directory, go over each image in the directory.

    Args:
        args (Namespace): The arguments from the command line.
    """
    torch.set_grad_enabled(mode=False)

    # Create the inferencer and visualizer.
    inferencer = TorchInferencer(path=args.weights, device=args.device)
    visualizer = ImageVisualizer(mode=args.visualization_mode, task=args.task)

    filenames = get_image_filenames(path=args.input)
    for filename in filenames:
        image = read_image(filename, as_tensor=True)
        predictions = inferencer.predict(image=image)
        # print(predictions.image.shape)
        # print(predictions.pred_score)
        # print(predictions.pred_label)
        # print(predictions.anomaly_map.shape)
        # print(predictions.pred_mask.shape)
        # print(predictions.pred_boxes)
        # print(predictions.box_labels)

    return predictions


def IoU(pred, target):

    pred_norm = (pred / 255)
    target_norm = (target / 255)

    ious = []
    for cls in range(2):
        pred_inds = pred_norm == cls
        target_inds = target_norm == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            # if there is no ground truth, do not include in evaluation
            ious.append(float("nan"))
        else:
            ious.append(float(intersection) / max(union, 1))
    return ious


def main():

    args = get_parser().parse_args()

    input_folder = args.input  # NG and OK
    ouput_folder = args.output

    os.makedirs(ouput_folder, exist_ok=True)

    detection_gt = []
    detection_pred = []
    segmentation_gt = []
    segmentation_pred = []
    segmentation_iou = []

    for NG_OK in os.listdir(input_folder):

        for img in os.listdir(input_folder/NG_OK/"X"):

            args.input = input_folder/NG_OK/"X"/img
            args.output = ouput_folder/img

            predictions = infer(args=args)

            gt = np.array(Image.open(input_folder/NG_OK/"Y"/img))

            if 255 in gt:
                detection_gt.append(1)
            else:
                detection_gt.append(0)
            if 255 in predictions.pred_mask:
                detection_pred.append(1)
            else:
                detection_pred.append(0)

            segmentation_gt += list(gt.flatten())
            segmentation_pred += list(predictions.pred_mask.flatten())

            iou = IoU(predictions.pred_mask, gt)
            segmentation_iou.append(iou[1])

            fig = plt.figure(figsize=(20, 5))
            fig.suptitle("{}".format(img))
            ax1 = fig.add_subplot(1, 4, 1)
            ax1.imshow(predictions.image)
            ax1.set_title("Image")
            ax1.axis("off")
            ax2 = fig.add_subplot(1, 4, 2)
            ax2.imshow(gt, cmap='gray', vmin=0, vmax=255)
            ax2.set_title("Ground Truth")
            ax2.axis("off")
            ax3 = fig.add_subplot(1, 4, 3)
            ax3.imshow(predictions.anomaly_map)
            ax3.set_title("Anomaly Map")
            ax3.axis("off")
            ax4 = fig.add_subplot(1, 4, 4)
            ax4.imshow(predictions.pred_mask, cmap='gray', vmin=0, vmax=255)
            ax4.set_title("Prediction")
            ax4.axis("off")

            output_path = ouput_folder / "{}".format(img)

            plt.savefig(output_path)
            plt.close()

    total_ious = np.array(segmentation_iou)
    ious = np.nanmean(total_ious)
    i_tn, i_fp, i_fn, i_tp = confusion_matrix(
        detection_gt, detection_pred).ravel()
    p_tn, p_fp, p_fn, p_tp = confusion_matrix(
        segmentation_gt, segmentation_pred).ravel()

    results = {
        "Detection": {
            "TN": int(i_tn),
            "FP": int(i_fp),
            "FN": int(i_fn),
            "TP": int(i_tp)
        },
        "Segmentation": {
            "IoU": float(ious),
            "TN": int(p_tn),
            "FP": int(p_fp),
            "FN": int(p_fn),
            "TP": int(p_tp)
        },
    }

    json.dump(results, open(ouput_folder / "results.json", "w"))


if __name__ == "__main__":
    main()