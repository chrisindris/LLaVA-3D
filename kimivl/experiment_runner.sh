#!/bin/bash

# ======= ./experiment_runner.sh ========
# For reproducibility and organizational purposes, this script is to be used for listing the experiments we want to run.
#source ~/.bashrc
#pyenv activate kimivl

KIMIVL="${PWD%%LLaVA-3D*}/LLaVA-3D/kimivl/"

MODEL="moonshotai/Kimi-VL-A3B-Thinking"
SCENES="/data/SceneUnderstanding/ScanNet/scans/"
EXP_DIR="${KIMIVL}/experiments"
ANNO_DIR="/data/SceneUnderstanding/7792397/ScanQA_format"


# python kimivl_3d_test.py \
#     --question_file ${ANNO_DIR}/SQA_em1-below-35_formatted_LLaVa3d.json \
#     --answer_file ${ANNO_DIR}/SQA_em1-below-35_formatted_LLaVa3d_answers.json \
#     --image_folder ${SCENES} \
#     --export_json ${EXP_DIR}/SQA3D/em1_below_35/SQA_em1-below-35_formatted_LLaVa3d_pred-answers.json \
#     --model_path moonshotai/Kimi-VL-A3B-Thinking \
#     --device cuda \
#     --sample_rate 800

python kimivl_3d_test.py \
    --question_file ${ANNO_DIR}/SQA_650_formatted_LLaVa3d_annotations.json \
    --answer_file ${ANNO_DIR}/SQA_650_formatted_LLaVa3d_answers.json \
    --image_folder ${SCENES} \
    --export_json ${EXP_DIR}/SQA3D/650/SQA_650_formatted_LLaVa3d_pred_answers.json \
    --model_path moonshotai/Kimi-VL-A3B-Thinking \
    --device cuda:0 \
    --sample_rate 200 \
    --num_chunks 5 \
    --chunk_idx 0
