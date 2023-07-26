export TOKENIZERS_PARALLELISM=false

RESULT_DIR='results/korean_origin_bench'
TASKS='klue_nli,klue_sts,klue_ynat,kohatespeech,kohatespeech_apeach,kohatespeech_gen_bias,korunsmile,nsmc,pawsx_ko'
GPU_NO=3

CURRENT_TRAINED_TOKENS=40b

echo "mkdir -p $RESULT_DIR/$CURRENT_TRAINED_TOKENS"
mkdir -p $RESULT_DIR/$CURRENT_TRAINED_TOKENS

MODEL='beomi/llama-2-ko-7b'

python main.py \
--model gpt2 \
--model_args pretrained=$MODEL \
--tasks $TASKS \
--num_fewshot 0 \
--device cuda:$GPU_NO \
--output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/00_shot.json

python main.py \
--model gpt2 \
--model_args pretrained=$MODEL \
--tasks $TASKS \
--num_fewshot 5 \
--device cuda:$GPU_NO \
--output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/05_shot.json

python main.py \
--model gpt2 \
--model_args pretrained=$MODEL \
--tasks $TASKS \
--num_fewshot 10 \
--device cuda:$GPU_NO \
--output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/10_shot.json

#python main.py \
#--model gpt2 \
#--model_args pretrained=$MODEL \
#--tasks $TASKS \
#--num_fewshot 50 \
#--device cuda:$GPU_NO \
#--output_path $RESULT_DIR/$CURRENT_TRAINED_TOKENS/50_shot.json
