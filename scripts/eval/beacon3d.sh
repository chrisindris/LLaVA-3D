#!/bin/bash

# ============= ./beacon3d.sh ============
# This script makes running experiments more convenient, in the style of sqa3d.sh.
# For running LLaVA-3D on Beacon-3D, we create llava/eval/model_beacon3d.py, which is a modified version of model_beacon3d.py.
# For evaluation, we can use Beacon-3D's evaluation script: /data/SceneUnderstanding/beacon-3d/evaluate_qa.py
#
# USAGE:
# ./beacon3d.sh --gpu <int> --questions <path> --pred-answers <path> --gt-answers <path> --outfile <path> [--generate <command>] [other arguments...]
# --gpu: the gpu to use (the python scripts currently only support one GPU)
# --questions: the annotations for the scene
# --pred-answers: where the model's predictions will be written to 
# --gt-answers: location of the ground truth file 
# --outfile: where the record of the experiment will be written to 
# --generate: use this if --questions and --gt-answers already exist (i.e. don't generate the files)
#       the <command> argument is what gets used to generate the files; pass it in without quotation marks, something like --generate echo 'scene0566_00'
# [other arguments...] these are arguments that will be passed on to llava/eval/model_beacon3d.py directly; useful for grid search or similar
# For numerous usage examples, see ./experiment_runner.sh
# ===================================

# --- handle arguments ---
declare -A arr
KEY=""
VALUE=""
for ARGUMENT in "$@"; do

        if [[ ${ARGUMENT} == "--use_enhanced_normalization" ]]; then

                USE_ENHANCED_NORMALIZATION="--use_enhanced_normalization"

        elif [[ ${ARGUMENT} == "--use_openai_evaluation" ]]; then

                USE_OPENAI_EVALUATION="--use_openai_evaluation"

        elif [[ ${ARGUMENT} == "--use_paper_decoding_params" ]]; then

                USE_PAPER_DECODING_PARAMS="--use_paper_decoding_params"

        elif [[ ${ARGUMENT} == "--no_prompt_tokenizer_truncation" ]]; then

                NO_PROMPT_TOKENIZER_TRUNCATION="--no_prompt_tokenizer_truncation"

        elif [[ ${ARGUMENT:0:2} == "--" ]]; then

                if [[ $KEY != "" ]]; then
                        arr["$KEY"]="$VALUE"
                        KEY=""
                        VALUE=""
                fi

                KEY=${ARGUMENT:2}
        else
                if [[ $VALUE == "" ]]; then
                        VALUE=$ARGUMENT
                else
                        VALUE="$VALUE $ARGUMENT"
                fi
        fi

done

if [[ $KEY != "" ]]; then
        arr["$KEY"]="$VALUE"
fi




# --- higher-order arguments (not passed directly into the python scripts) 
GPU=${arr['gpu']} && echo "GPU =" $GPU && unset 'arr[gpu]'
QUESTIONS=${arr['questions']} && echo "QUESTIONS =" $QUESTIONS && unset 'arr[questions]'
PRED_ANSWERS=${arr['pred-answers']} && echo "PRED_ANSWERS =" $PRED_ANSWERS && unset 'arr[pred-answers]'
GT_ANSWERS=${arr['gt-answers']} && echo "GT_ANSWERS =" $GT_ANSWERS && unset 'arr[gt-answers]'
OUTFILE=${arr['outfile']} && echo "OUTFILE =" $OUTFILE && unset 'arr[outfile]'
METADATA=${arr['metadata']} && echo "METADATA =" $METADATA && unset 'arr[metadata]'

# Create parent directories for output files if they don't exist
if [[ -n "$PRED_ANSWERS" ]]; then
    mkdir -p "$(dirname "$PRED_ANSWERS")"
fi
if [[ -n "$OUTFILE" ]]; then
    mkdir -p "$(dirname "$OUTFILE")"
fi
GENERATE=${arr['generate']} && echo "GENERATE =" $GENERATE && unset 'arr[generate]'


# --- print arguments ---
echo "arguments:"
for key in "${!arr[@]}"
do
  echo "Key: $key, Value: ${arr[$key]}"
done


# --- for the model:
MODEL_ARGS=""
arr["question-file"]=${QUESTIONS}
arr["answers-file"]=${PRED_ANSWERS}
for i in "${!arr[@]}"; do
        MODEL_ARGS+="--${i} ${arr[${i}]} "
done
if [[ ${USE_PAPER_DECODING_PARAMS} ]]; then
        MODEL_ARGS+=" ${USE_PAPER_DECODING_PARAMS}"
fi
if [[ ${NO_PROMPT_TOKENIZER_TRUNCATION} ]]; then
        MODEL_ARGS+=" ${NO_PROMPT_TOKENIZER_TRUNCATION}"
fi
# --- for sqa3d_evaluator:
EVALUATOR_ARGS=""
arr["pred-json"]=${PRED_ANSWERS}
arr["gt-json"]=${GT_ANSWERS}
ea=(pred-json gt-json num-chunks chunk-idx)
for i in "${ea[@]}"; do
        if [[ ${arr[$i]+_} ]]; then EVALUATOR_ARGS+="--${i} ${arr[${i}]} "; fi
done
if [[ ${USE_ENHANCED_NORMALIZATION} ]]; then
        EVALUATOR_ARGS+=" ${USE_ENHANCED_NORMALIZATION}"
fi
if [[ ${USE_OPENAI_EVALUATION} ]]; then
        EVALUATOR_ARGS+=" ${USE_OPENAI_EVALUATION}"
fi

echo "EVALUATOR_ARGS:"
echo $EVALUATOR_ARGS


# --- main script body ---

export CUDA_VISIBLE_DEVICES="$GPU"

# --- set the embodiedscan file ---
unlink playground/data/annotations/embodiedscan_infos.json
LLAVA_3D="${PWD%%LLaVA-3D*}LLaVA-3D/"
ln -s "${LLAVA_3D}/playground/data/annotations/embodiedscan_infos_full_formatted_cluster.json" playground/data/annotations/embodiedscan_infos.json

# if [[ ${GENERATE} ]]; then
#         echo generating new annotation files...
#         pushd /data/SceneUnderstanding/7792397/ScanQA_format/ # NOTE: hardcoding this should be fine since this is where the various original JSONs live.
#         echo python ../../scripts/generate_SQA3D_LLaVA-3D_annotations.py "<($GENERATE)" ./SQA_train.formatted.json ./SQA_test.formatted.json ./SQA_val.formatted.json ${QUESTIONS}
#         python ../../scripts/generate_SQA3D_LLaVA-3D_annotations.py <($GENERATE) ./SQA_train.formatted.json ./SQA_test.formatted.json ./SQA_val.formatted.json ${QUESTIONS}
#         echo python ../../scripts/generate_SQA3D_LLaVA-3D_gt_answers.py "<($GENERATE)" ./SQA_train.formatted.json ./SQA_test.formatted.json ./SQA_val.formatted.json ${GT_ANSWERS}
#         python ../../scripts/generate_SQA3D_LLaVA-3D_gt_answers.py <($GENERATE) ./SQA_train.formatted.json ./SQA_test.formatted.json ./SQA_val.formatted.json ${GT_ANSWERS}
#         popd
# fi

echo "Commands to run:"
echo python llava/eval/model_beacon3d.py ${MODEL_ARGS} | tee ${OUTFILE}
echo python /data/SceneUnderstanding/beacon-3d/evaluate_qa.py --infer ${PRED_ANSWERS} --data ${GT_ANSWERS} --metadata ${METADATA} --prompt "/data/SceneUnderstanding/beacon-3d/data/system_prompt.json"

python llava/eval/model_beacon3d.py ${MODEL_ARGS}
python /data/SceneUnderstanding/beacon-3d/evaluate_qa.py --infer ${PRED_ANSWERS} --data ${GT_ANSWERS} --metadata ${METADATA} --prompt "/data/SceneUnderstanding/beacon-3d/data/system_prompt.json"