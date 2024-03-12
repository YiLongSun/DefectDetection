source activate anomalib

# MODELS = {
#     "Cfa": Cfa,
#     "Cflow": Cflow,
#     "Csflow": Csflow,
#     "Dfkde": Dfkde,
#     "Dfm": Dfm,
#     "Draem": Draem,
#     "Dsr": Dsr,
#     "EfficientAd": EfficientAd,
#     "Fastflow": Fastflow,
#     "Ganomaly": Ganomaly,
#     "Padim": Padim,
#     "Patchcore": Patchcore,
#     "ReverseDistillation": ReverseDistillation,
#     "Rkde": Rkde,
#     "Stfpm": Stfpm,
#     "Uflow": Uflow,
#     "WinClip": WinClip,
# }

# python train_deltadata_anomalib.py \
#     --model "STFPM" \
#     --experiment_path "../../Experiments/Anomalib/" \
#     --data_root_path "../../Datasets/DeltaDataV1_AnomalibForm/P1_V1/" \
#     --data_normal_dir "trn/OK/X/" \
#     --data_abnormal_dir "val/NG/X/" \
#     --data_mask_dir "val/NG/Y/" \
#     --data_normal_test_dir "val/OK/X/"

# python test_deltadata_anomalib.py \
#     --weights "../../Experiments/Anomalib/Stfpm/P1_V1/latest/weights/torch/model.pt" \
#     --input "../../Datasets/DeltaDataV1_AnomalibForm/P1_V1/tst/" \
#     --output "../../Experiments/Anomalib/Stfpm/P1_V1/Test/"

models=(Cfa Cflow Csflow Dfkde Dfm Draem Dsr EfficientAd Fastflow Ganomaly Padim Patchcore ReverseDistillation Rkde Stfpm Uflow WinClip)
products=(1 3)
views=(1 2 3 4)
target_experiment_path="../../Experiments/Anomalib/"
target_data_root_path="../../Datasets/DeltaDataV1_AnomalibForm/"
target_data_normal_dir="trn/OK/X/"
target_data_abnormal_dir="val/NG/X/"
target_data_mask_dir="val/NG/Y/"
target_data_normal_test_dir="val/OK/X/"

for model in "${models[@]}"; do

    for product in "${products[@]}"; do

        for view in "${views[@]}"; do

            python train_deltadata_anomalib.py \
                --model "${model}" \
                --experiment_path "${target_experiment_path}" \
                --data_root_path "${target_data_root_path}P${product}_V${view}/" \
                --data_normal_dir "${target_data_normal_dir}" \
                --data_abnormal_dir "${target_data_abnormal_dir}" \
                --data_mask_dir "${target_data_mask_dir}" \
                --data_normal_test_dir "${target_data_normal_test_dir}"

            python test_deltadata_anomalib.py \
                --weights "../../Experiments/Anomalib/${model}/P${product}_V${view}/latest/weights/torch/model.pt" \
                --input "../../Datasets/DeltaDataV1_AnomalibForm/P${product}_V${view}/tst/" \
                --output "../../Experiments/Anomalib/${model}/P${product}_V${view}/Test/"

        done

    done

done

conda deactivate
