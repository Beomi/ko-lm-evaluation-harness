export TOKENIZERS_PARALLELISM=false

RESULT_DIR='results/korean_origin_bench'
# TASKS='klue_mrc klue_nli klue_sts klue_ynat ko_en_translation en_ko_translation kohatespeech kohatespeech_apeach kohatespeech_gen_bias korquad korunsmile nsmc'
TASKS='pawsx_ko'
GPU_NO=3

CURRENT_TRAINED_TOKENS=20b

for TASK in $TASKS
do
    echo "Running task: $TASK"
    python main.py \
    --model gpt2 \
    --model_args pretrained=../EasyLM/llama-2-ko-7b \
    --tasks $TASK \
    --num_fewshot 0 \
    --device cuda:$GPU_NO
    if [ $? -ne 0 ]; then
        echo "Task $TASK failed."
    else
        echo "Task $TASK completed successfully."
    fi
done
