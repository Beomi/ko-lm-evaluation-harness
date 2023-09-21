export TOKENIZERS_PARALLELISM=false
# export CUDA_VISIBLE_DEVICES='0,1'

RESULT_DIR='results/kobest_bench'
TASKS='kobest_hellaswag,kobest_copa,kobest_boolq,kobest_sentineg'

CURRENT_TRAINED_TOKENS=20b
MODEL=beomi/llama-2-ko-70b

echo "mkdir -p $RESULT_DIR/$MODEL/$CURRENT_TRAINED_TOKENS"
mkdir -p $RESULT_DIR/$MODEL/$CURRENT_TRAINED_TOKENS

# python main.py \
# --model gpt2 \
# --model_args pretrained=$MODEL \
# --tasks $TASKS \
# --num_fewshot 0 \
# --no_cache \
# --batch_size 8 \
# --output_path $RESULT_DIR/$MODEL/$CURRENT_TRAINED_TOKENS/0_shot.json

# python main.py \
# --model gpt2 \
# --model_args pretrained=$MODEL \
# --tasks $TASKS \
# --num_fewshot 5 \
# --no_cache \
# --batch_size 4 \
# --output_path $RESULT_DIR/$MODEL/$CURRENT_TRAINED_TOKENS/5_shot.json

python main.py \
--model gpt2 \
--model_args pretrained=$MODEL \
--tasks $TASKS \
--num_fewshot 10 \
--no_cache \
--batch_size 1 \
--output_path $RESULT_DIR/$MODEL/$CURRENT_TRAINED_TOKENS/10_shot.json

python main.py \
--model gpt2 \
--model_args pretrained=$MODEL \
--tasks $TASKS \
--num_fewshot 50 \
--no_cache \
--batch_size 1 \
--output_path $RESULT_DIR/$MODEL/$CURRENT_TRAINED_TOKENS/50_shot.json
