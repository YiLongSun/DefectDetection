import os
import json
from pathlib import Path


def main():

    EXPERIMENT_PATH = "../../Experiments/Anomalib/"

    for method in os.listdir(EXPERIMENT_PATH):

        try:

            json_path = Path(EXPERIMENT_PATH) / method / \
                "P1_V1" / "Test" / "results.json"

            with open(json_path, "r") as f:
                data = json.load(f)

            print(f"Method: {method}")
            detection = data["Detection"]
            print(f"Detection: {detection}")

        except:
            # print(f"Error with {method}")
            continue


if __name__ == "__main__":
    main()
