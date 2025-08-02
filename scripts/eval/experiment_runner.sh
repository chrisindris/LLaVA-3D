#!/bin/bash

# ======= ./experiment_runner.sh ========
# For reproducibility and organizational purposes, this script is to be used for listing the experiments we want to run.
# For example, it can be used to facilitate grid search.

#LLAVA_3D="/root/SceneUnderstanding/LLaVA-3D"
#LLAVA_3D="/data/SceneUnderstanding/SU_cursor/LLaVA-3D"
LLAVA_3D="${PWD%%LLaVA-3D*}LLaVA-3D/"

MODEL="/data/SceneUnderstanding/checkpoints/LLaVA-3D-7B/"
SCENES="/data/SceneUnderstanding/ScanNet/scans/"
# EXP_DIR="${LLAVA_3D}/experiments"
# ANNO_DIR="/data/SceneUnderstanding/7792397/ScanQA_format"

# optional model check
#python ./check_model_setup.py --model-path /data/SceneUnderstanding/checkpoints/LLaVA-3D-7B/


## ---------------------------- SQA3D ----------------------------

DATASET="SQA3D"
EXP_DIR="${LLAVA_3D}/experiments/${DATASET}"
ANNO_DIR="/data/SceneUnderstanding/7792397/ScanQA_format"

# # em1_below_35
# ${LLAVA_3D}/scripts/eval/sqa3d.sh \
#     --gpu 0 \
#     --model-path ${MODEL} \
#     --questions ${ANNO_DIR}/SQA_em1-below-35_formatted_LLaVa3d.json \
#     --pred-answers ${EXP_DIR}/SQA3D/em1_below_35/SQA_em1-below-35_formatted_LLaVa3d_pred-answers-rerun.json \
#     --gt-answers ${ANNO_DIR}/SQA_em1-below-35_formatted_LLaVa3d_answers.json \
#     --video-folder ${SCENES} \
#     --chunk-idx 0 \
#     --num-chunks 1 \
#     --outfile ${EXP_DIR}/SQA3D/em1_below_35/SQA_em1-below-35_formatted_LLaVa3d_output-rerun.txt


# #scrap (only one scene)
# ${LLAVA_3D}/scripts/eval/sqa3d.sh \
#     --gpu 0 \
#     --model-path ${MODEL} \
#     --questions ${ANNO_DIR}/scrap.json \
#     --pred-answers ${EXP_DIR}/em1_below_35/scrap.json \
#     --gt-answers ${ANNO_DIR}/scrap_answers.json \
#     --video-folder ${SCENES} \
#     --outfile ${EXP_DIR}/em1_below_35/scrap.txt \
#     --chunk-idx 0 \
#     --num-chunks 1 \
#     --generate echo 'scene0000_00' \
#     --max_new_tokens 8 \
#     --repetition_penalty 1.2 \
#     --no_repeat_ngram_size 3
    
## Scrap (all scenes, with 2 params and modifications)
# ${LLAVA_3D}/scripts/eval/sqa3d.sh \
#     --gpu 0 \
#     --model-path ${MODEL} \
#     --questions ${ANNO_DIR}/scrap.json \
#     --pred-answers ${EXP_DIR}/scrap.json \
#     --gt-answers ${ANNO_DIR}/scrap_answers.json \
#     --video-folder ${SCENES} \
#     --outfile ${EXP_DIR}/SQA3D/em1_below_35/scrap_2params.txt \
#     --chunk-idx 0 \
#     --num-chunks 1 \
#     --generate echo 'scene0000_00' \
#     --frame_selection_mode uniform \
#     --use_paper_decoding_params \
#     --no_prompt_tokenizer_truncation \

# # 1.2: Exact-match normalization
#
# ${LLAVA_3D}/scripts/eval/sqa3d.sh \
#     --gpu 0 \
#     --model-path ${MODEL} \
#     --questions ${ANNO_DIR}/scrap.json \
#     --pred-answers ${EXP_DIR}/scrap.json \
#     --gt-answers ${ANNO_DIR}/scrap_answers.json \
#     --video-folder ${SCENES} \
#     --outfile ${EXP_DIR}/SQA3D/em1_below_35/scrap-1.2_rerunJune6.txt \
#     --chunk-idx 0 \
#     --num-chunks 1 \
#     --generate echo 'scene0000_00' \
#     --use_enhanced_normalization

# 1.3: AI normalization

# ${LLAVA_3D}/scripts/eval/sqa3d.sh \
#     --gpu 0 \
#     --model-path ${MODEL} \
#     --questions ${ANNO_DIR}/scrap.json \
#     --pred-answers ${EXP_DIR}/scrap.json \
#     --gt-answers ${ANNO_DIR}/scrap_answers.json \
#     --video-folder ${SCENES} \
#     --outfile ${EXP_DIR}/SQA3D/em1_below_35/scrap-AI_rerunJune7.txt \
#     --chunk-idx 0 \
#     --num-chunks 1 \
#     --generate echo 'scene0000_00' \
#     --use_openai_evaluation \

# 1.4: AI normalization with enhanced normalization

# ${LLAVA_3D}/scripts/eval/sqa3d.sh \
#     --gpu 0 \
#     --model-path ${MODEL} \
#     --questions ${ANNO_DIR}/scrap.json \
#     --pred-answers ${EXP_DIR}/scrap.json \
#     --gt-answers ${ANNO_DIR}/scrap_answers.json \
#     --video-folder ${SCENES} \
#     --outfile ${EXP_DIR}/SQA3D/em1_below_35/scrap-1.2_AI_rerunJune6.txt \
#     --chunk-idx 0 \
#     --num-chunks 1 \
#     --generate echo 'scene0000_00' \
#     --use_openai_evaluation \
#     --use_enhanced_normalization


## ---------------------------- Beacon-3D ----------------------------

DATASET="Beacon3D"
EXP_DIR="${LLAVA_3D}/experiments/${DATASET}"
ANNO_DIR="/data/SceneUnderstanding/beacon-3d/data/scannet"

${LLAVA_3D}/scripts/eval/beacon3d.sh \
    --gpu 0 \
    --model-path ${MODEL} \
    --questions ${ANNO_DIR}/scannet_qa_scene0025_00.json \
    --pred-answers ${EXP_DIR}/em1_below_35/scrap.json \
    --gt-answers ${ANNO_DIR}/scannet_qa_scene0025_00.json \
    --metadata ${ANNO_DIR}/metadata_scannet_qa_scene0025_00.json \
    --video-folder ${SCENES} \
    --outfile ${EXP_DIR}/em1_below_35/scrap.txt \
    --chunk-idx 0 \
    --num-chunks 1 \
    --max_new_tokens 8 \
    --repetition_penalty 1.2 \
    --no_repeat_ngram_size 3