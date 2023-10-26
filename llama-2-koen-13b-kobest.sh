export TOKENIZERS_PARALLELISM=false

RESULT_DIR='results/kobest_bench'
TASKS='kobest_hellaswag,kobest_copa,kobest_boolq,kobest_sentineg'
GPU_NO=0

CURRENT_TRAINED_TOKENS=60b
MODEL=/home/jovyan/beomi/llama2-koen-13b #/home/beomi/coding-ssd2t/EasyLM/llama-2-ko-7b

MODEL_PATH=$(echo $MODEL | awk -F/ '{print $(NF-1) "/" $NF}')

echo "mkdir -p $RESULT_DIR/$MODEL_PATH/$CURRENT_TRAINED_TOKENS"
mkdir -p $RESULT_DIR/$MODEL_PATH/$CURRENT_TRAINED_TOKENS

python main.py \
--model hf-causal-experimental \
--model_args pretrained=$MODEL,use_accelerate=true \
--tasks $TASKS \
--num_fewshot 0 \
--no_cache \
--batch_size 32 \
--output_path $RESULT_DIR/$MODEL/$CURRENT_TRAINED_TOKENS/0_shot.json

python main.py \
--model hf-causal-experimental \
--model_args pretrained=$MODEL,use_accelerate=true \
--tasks $TASKS \
--num_fewshot 5 \
--no_cache \
--batch_size 16 \
--output_path $RESULT_DIR/$MODEL/$CURRENT_TRAINED_TOKENS/5_shot.json

python main.py \
--model hf-causal-experimental \
--model_args pretrained=$MODEL,use_accelerate=true \
--tasks $TASKS \
--num_fewshot 10 \
--no_cache \
--batch_size 8 \
--output_path $RESULT_DIR/$MODEL/$CURRENT_TRAINED_TOKENS/10_shot.json

python main.py \
--model hf-causal-experimental \
--model_args pretrained=$MODEL,use_accelerate=true \
--tasks $TASKS \
--num_fewshot 50 \
--no_cache \
--batch_size 4 \
--output_path $RESULT_DIR/$MODEL/$CURRENT_TRAINED_TOKENS/50_shot.json

