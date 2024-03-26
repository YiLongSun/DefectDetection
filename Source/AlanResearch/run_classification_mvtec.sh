source activate anomalib

models=(Dfkde Dfm Ganomaly)
category=("bottle" "cable" "capsule" "carpet" "grid" "hazelnut" "leather" "metal_nut" "pill" "screw" "tile" "toothbrush" "transistor" "wood" "zipper")

target_experiment_path="../../Experiments/Anomalib/"
target_data_root_path="../../Datasets/MVTec/"
target_data_normal_dir="train/good/"

for model in "${models[@]}"; do

    for cat in "${category[@]}"; do

        python train_classification_mvtec.py \
            --model "${model}" \
            --experiment_path "${target_experiment_path}" \
            --data_root_path "${target_data_root_path}${cat}/" \
            --data_normal_dir "${target_data_normal_dir}"

        python test_classification_mvtec.py \
            --weights "${target_experiment_path}${model}/${cat}/latest/weights/torch/model.pt" \
            --input "${target_data_root_path}${cat}/test/" \
            --gt "${target_data_root_path}${cat}/ground_truth/" \
            --output "${target_experiment_path}${model}/${cat}/Test/"

    done

done

conda deactivate