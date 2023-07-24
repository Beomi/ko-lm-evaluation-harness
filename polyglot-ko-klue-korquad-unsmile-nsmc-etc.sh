#!/bin/bash

export TOKENIZERS_PARALLELISM=false

RESULT_DIR='results/korean_origin_bench'
TASKS='klue_nli,klue_sts,klue_ynat,kohatespeech,kohatespeech_apeach,kohatespeech_gen_bias,korunsmile,nsmc,pawsx_ko'
GPU_NO=0

MODELS=("3.8b" "5.8b")

for CURRENT_MODEL in "${MODELS[@]}"
do
  CURRENT_TRAINED_TOKENS="polyglot-ko-$CURRENT_MODEL"
  echo "mkdir -p $RESULT_DIR/$CURRENT_TRAINED_TOKENS"
  mkdir -p $RESULT_DIR/$CURRENT_TRAINED_TOKENS

  MODEL="EleutherAI/polyglot-ko-$CURRENT_MODEL"

  for num_fewshot in 0 5 10
  do
    python main.py \
    --model gpt2 \
    --model_args pretrained=$MODEL \
    --tasks $TASKS \
    --num_fewshot $num_fewshot \
    --device cuda:$GPU_NO \
    --output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/${num_fewshot}_shot.json
  done
done
