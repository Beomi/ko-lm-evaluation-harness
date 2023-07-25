#!/bin/bash

export TOKENIZERS_PARALLELISM=false

RESULT_DIR='results/korean_origin_bench'
TASKS='klue_nli,klue_sts,klue_ynat,kohatespeech,kohatespeech_apeach,kohatespeech_gen_bias,korunsmile,nsmc,pawsx_ko'
#MODELS=("3.8b" "5.8b")
MODELS=("12.8b")
FEWSHOTS=(10)

GPU_NO=6
for CURRENT_MODEL in "${MODELS[@]}"; do
  for num_fewshot in "${FEWSHOTS[@]}"; do
    screen -dmS "screen_gpu_${GPU_NO}_model_${CURRENT_MODEL}_fewshot_${num_fewshot}" \
    bash ./task_runner.sh $GPU_NO $CURRENT_MODEL $num_fewshot $TASKS $RESULT_DIR
    let "GPU_NO=GPU_NO+1"
  done
done


