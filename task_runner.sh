#!/bin/bash

GPU_NO=$1
CURRENT_MODEL=$2
num_fewshot=$3
TASKS=$4
RESULT_DIR=$5

CURRENT_TRAINED_TOKENS="polyglot-ko-$CURRENT_MODEL"
echo "mkdir -p $RESULT_DIR/$CURRENT_TRAINED_TOKENS"
mkdir -p $RESULT_DIR/$CURRENT_TRAINED_TOKENS

MODEL="EleutherAI/polyglot-ko-$CURRENT_MODEL"

python main.py \
--model gpt2 \
--model_args pretrained=$MODEL \
--tasks $TASKS \
--num_fewshot $num_fewshot \
--device cuda:$GPU_NO \
--batch_size 2 \
--no_cache \
--output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/${num_fewshot}_shot.json \
2>&1 | tee "logs/gpu_${GPU_NO}_model_${CURRENT_MODEL}_fewshot_${num_fewshot}.log"

