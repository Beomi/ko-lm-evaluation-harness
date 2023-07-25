export TOKENIZERS_PARALLELISM=false

RESULT_DIR='results/kobest_bench'
TASKS='kobest_hellaswag,kobest_copa,kobest_boolq,kobest_sentineg'
GPU_NO=3

CURRENT_TRAINED_TOKENS=65b
MODEL=/home/beomi/coding-ssd2t/EasyLM/llama-2-ko-7b

echo "mkdir -p $RESULT_DIR/$CURRENT_TRAINED_TOKENS"
mkdir -p $RESULT_DIR/$CURRENT_TRAINED_TOKENS

python main.py \
--model gpt2 \
--model_args pretrained=$MODEL \
--tasks $TASKS \
--num_fewshot 0 \
--device cuda:$GPU_NO \
--no_cache \
--output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/0_shot.json

python main.py \
--model gpt2 \
--model_args pretrained=$MODEL \
--tasks $TASKS \
--num_fewshot 5 \
--no_cache \
--device cuda:$GPU_NO \
--output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/5_shot.json

python main.py \
--model gpt2 \
--model_args pretrained=$MODEL \
--tasks $TASKS \
--num_fewshot 10 \
--no_cache \
--device cuda:$GPU_NO \
--output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/10_shot.json

python main.py \
--model gpt2 \
--model_args pretrained=$MODEL \
--tasks $TASKS \
--num_fewshot 50 \
--no_cache \
--device cuda:$GPU_NO \
--output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/50_shot.json

