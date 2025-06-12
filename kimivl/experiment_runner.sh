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


python kimivl_3d_test.py \
    --question_file ${ANNO_DIR}/scrap.json \
    --answer_file ${ANNO_DIR}/scrap_answers.json \
    --image_folder ${SCENES} \
    --export_json ${EXP_DIR}/scrap.json \
    --model_path moonshotai/Kimi-VL-A3B-Thinking \
    --device cuda \
    --sample_rate 200