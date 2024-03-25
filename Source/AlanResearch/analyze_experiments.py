import argparse
import json
import os
from pathlib import Path

import pandas as pd

SEGMENTATINO_METHODS = ["Stfpm", "Cfa", "Cflow", "Csflow", "Dsr", "Draem", "EfficientAd", "Fastflow", "Padim", "Patchcore", "ReverseDistillation", "Uflow"]
CLASSIFICATION_METHODS = ["Dfkde", "Dfm", "Ganomaly"]

def main():

    argparser = argparse.ArgumentParser()

    argparser.add_argument("--exp_root_path", type=str, default="../../Experiments/Anomalib/")
    argparser.add_argument("--category", type=str, default="[11,12,13,14,31,32,33,34]")

    args = argparser.parse_args()

    # Collect results to database
    database = {}

    for method in os.listdir(args.exp_root_path):

        if method not in database.keys():
            database[method] = {}

        for category in eval(args.category):

            product_name = str(category)[0]
            view_name = str(category)[1]
            full_name = "P{}_V{}".format(product_name,view_name)

            if full_name not in database[method].keys():
                database[method][full_name] = {}

            try:

                json_path = Path(args.exp_root_path) / method / full_name / "Test" / "results.json"

                with open(json_path, "r") as f:
                    data = json.load(f)

                detection = data["Detection"]
                recall = detection["TP"] / (detection["TP"] + detection["FN"])
                specificity = detection["TN"] / (detection["TN"] + detection["FP"])

                database[method][full_name]["Recall"] = recall
                database[method][full_name]["Specificity"] = specificity

                if method in SEGMENTATINO_METHODS:
                    segmentation = data["Segmentation"]
                    iou = segmentation["IoU"]
                    database[method][full_name]["IoU"] = iou

            except Exception as e:
                print("{} P{}_V{} Error: {}".format(method, product_name, view_name, e))

    # Output dataframe
    recall_df_segmentation = {}
    specificity_df_segmentation = {}
    iou_df_segmentation = {}

    for method in SEGMENTATINO_METHODS:

        recall_df_segmentation["category"] = []
        specificity_df_segmentation["category"] = []
        iou_df_segmentation["category"] = []

        recall_df_segmentation[method] = []
        specificity_df_segmentation[method] = []
        iou_df_segmentation[method] = []

        for category in eval(args.category):

            full_name = "P{}_V{}".format(str(category)[0],str(category)[1])

            recall_df_segmentation["category"].append(full_name)
            specificity_df_segmentation["category"].append(full_name)
            iou_df_segmentation["category"].append(full_name)

            recall_value = database[method]["P{}_V{}".format(str(category)[0],str(category)[1])]["Recall"]*100
            specificity_value = database[method]["P{}_V{}".format(str(category)[0],str(category)[1])]["Specificity"]*100
            iou_value = database[method]["P{}_V{}".format(str(category)[0],str(category)[1])]["IoU"]

            # format value
            recall_value = "{:.2f}".format(recall_value)
            specificity_value = "{:.2f}".format(specificity_value)
            iou_value = "{:.4f}".format(iou_value)
            recall_df_segmentation[method].append(recall_value)
            specificity_df_segmentation[method].append(specificity_value)
            iou_df_segmentation[method].append(iou_value)

    recall_df_segmentation = pd.DataFrame(recall_df_segmentation)
    specificity_df_segmentation = pd.DataFrame(specificity_df_segmentation)
    iou_df_segmentation = pd.DataFrame(iou_df_segmentation)

    writer = pd.ExcelWriter(Path(args.exp_root_path) / "results.xlsx")
    recall_df_segmentation.to_excel(writer, sheet_name="Recall_Segmentation", index=False)
    specificity_df_segmentation.to_excel(writer, sheet_name="Specificity_Segmentation", index=False)
    iou_df_segmentation.to_excel(writer, sheet_name="IoU_Segmentation", index=False)

    writer.close()

if __name__ == "__main__":
    main()