#!/bin/bash

# ======= ./experiment_runner.sh ========
# For reproducibility and organizational purposes, this script is to be used for listing the experiments we want to run.
#source ~/.bashrc
#pyenv activate kimivl

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {0|1|2|3}"
  exit 1
fi

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

case "$1" in
    0)
        echo "On GPU 0"
        python kimivl_3d_test.py \
            --question_file ${ANNO_DIR}/SQA_650_formatted_LLaVa3d_annotations.json \
            --answer_file ${ANNO_DIR}/SQA_650_formatted_LLaVa3d_answers.json \
            --image_folder ${SCENES} \
            --export_json ${EXP_DIR}/SQA3D/650/SQA_650_formatted_LLaVa3d_pred_answers_gpu0.json \
            --model_path moonshotai/Kimi-VL-A3B-Thinking \
            --device cuda:0 \
            --sample_rate 800 \
            --num_chunks 5 \
            --chunk_idx 0
        ;;
    1)
        echo "On GPU 1"
        python kimivl_3d_test.py \
            --question_file ${ANNO_DIR}/SQA_650_formatted_LLaVa3d_annotations.json \
            --answer_file ${ANNO_DIR}/SQA_650_formatted_LLaVa3d_answers.json \
            --image_folder ${SCENES} \
            --export_json ${EXP_DIR}/SQA3D/650/SQA_650_formatted_LLaVa3d_pred_answers_gpu1.json \
            --model_path moonshotai/Kimi-VL-A3B-Thinking \
            --device cuda:1 \
            --sample_rate 800 \
            --num_chunks 5 \
            --chunk_idx 1
        ;;
    2)
        echo "Just testing KimiVL"
        python kimivl_3d_test.py \
            --question_file ${ANNO_DIR}/SQA_em1-below-35_formatted_LLaVa3d.json \
            --answer_file ${ANNO_DIR}/SQA_em1-below-35_formatted_LLaVa3d_answers.json \
            --image_folder ${SCENES} \
            --export_json ${EXP_DIR}/SQA3D/em1_below_35/SQA_em1-below-35_formatted_LLaVa3d_pred-answers_scrap.json \
            --model_path moonshotai/Kimi-VL-A3B-Thinking \
            --device cuda \
            --sample_rate 1000 \
        ;;
    3)
        echo "Google Colab: KimiVL.ipynb"
        SCENES="/content/sqa-3d/ScanNet/scans"
        EXP_DIR="/content/experiments"
        ANNO_DIR="/content/sqa-3d/ScanQA_format"
        EXPORT_JSON="${EXP_DIR}/SQA3D/650/SQA_650_formatted_LLaVa3d_pred_answers_colab1.json"
        mkdir -p $(dirname ${EXPORT_JSON})
        python kimivl_3d_test.py \
            --question_file ${ANNO_DIR}/SQA_650_formatted_LLaVa3d_annotations.json \
            --answer_file ${ANNO_DIR}/SQA_650_formatted_LLaVa3d_answers.json \
            --image_folder ${SCENES} \
            --export_json ${EXPORT_JSON} \
            --model_path moonshotai/Kimi-VL-A3B-Thinking \
            --device cuda:0 \
            --sample_rate 500 \
            --batch_size 8 \
            --num_chunks 5 \
            --chunk_idx 2
        ;;
    4)
        echo "Google Colab: KimiVL.ipynb"
        SCENES="/content/sqa-3d/ScanNet/scans"
        EXP_DIR="/content/experiments"
        ANNO_DIR="/content/sqa-3d/ScanQA_format"
        EXPORT_JSON="${EXP_DIR}/SQA3D/650/SQA_650_formatted_LLaVa3d_pred_answers_colab2.json"
        mkdir -p $(dirname ${EXPORT_JSON})
        python kimivl_3d_test.py \
            --question_file ${ANNO_DIR}/SQA_650_formatted_LLaVa3d_annotations.json \
            --answer_file ${ANNO_DIR}/SQA_650_formatted_LLaVa3d_answers.json \
            --image_folder ${SCENES} \
            --export_json ${EXPORT_JSON} \
            --model_path moonshotai/Kimi-VL-A3B-Thinking \
            --device cuda:0 \
            --sample_rate 500 \
            --batch_size 8 \
            --num_chunks 5 \
            --chunk_idx 3
        ;;
    5)
        echo "Google Colab: KimiVL.ipynb"
        SCENES="/content/sqa-3d/ScanNet/scans"
        EXP_DIR="/content/experiments"
        ANNO_DIR="/content/sqa-3d/ScanQA_format"
        EXPORT_JSON="${EXP_DIR}/SQA3D/650/SQA_650_formatted_LLaVa3d_pred_answers_colab3.json"
        mkdir -p $(dirname ${EXPORT_JSON})
        python kimivl_3d_test.py \
            --question_file ${ANNO_DIR}/SQA_650_formatted_LLaVa3d_annotations.json \
            --answer_file ${ANNO_DIR}/SQA_650_formatted_LLaVa3d_answers.json \
            --image_folder ${SCENES} \
            --export_json ${EXPORT_JSON} \
            --model_path moonshotai/Kimi-VL-A3B-Thinking \
            --device cuda:0 \
            --sample_rate 500 \
            --batch_size 8 \
            --num_chunks 5 \
            --chunk_idx 4
        ;;
    *)
        echo "Error: invalid option '$1'. Use 0 or 1."
        exit 2
        ;;
esac  
