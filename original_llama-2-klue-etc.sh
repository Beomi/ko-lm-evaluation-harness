export TOKENIZERS_PARALLELISM=false
python main.py \
--model gpt2 \
--model_args pretrained=meta-llama/Llama-2-7b-hf \
--tasks kobest_hellaswag,kobest_copa,kobest_boolq,kobest_sentineg \
--num_fewshot 0 \
--device cuda:2 \
--output_path results/0b_0_shot.json

python main.py \
--model gpt2 \
--model_args pretrained=meta-llama/Llama-2-7b-hf \
--tasks kobest_hellaswag,kobest_copa,kobest_boolq,kobest_sentineg \
--num_fewshot 5 \
--device cuda:2 \
--output_path results/0b_5_shot.json

python main.py \
--model gpt2 \
--model_args pretrained=meta-llama/Llama-2-7b-hf \
--tasks kobest_hellaswag,kobest_copa,kobest_boolq,kobest_sentineg \
--num_fewshot 10 \
--device cuda:2 \
--output_path results/0b_10_shot.json

python main.py \
--model gpt2 \
--model_args pretrained=meta-llama/Llama-2-7b-hf \
--tasks kobest_hellaswag,kobest_copa,kobest_boolq,kobest_sentineg \
--num_fewshot 50 \
--device cuda:2 \
--output_path results/0b_50_shot.json


# python main.py \
# --model gpt2 \
# --model_args pretrained=meta-llama/Llama-2-7b-hf \
# --tasks nsmc \
# --num_fewshot 0 \
# --device cuda:0 \
# --output_path results/0b_nsmc_0_shot.json

# python main.py \
# --model gpt2 \
# --model_args pretrained=meta-llama/Llama-2-7b-hf \
# --tasks nsmc \
# --num_fewshot 5 \
# --device cuda:0 \
# --output_path results/0b_nsmc_5_shot.json

# python main.py \
# --model gpt2 \
# --model_args pretrained=meta-llama/Llama-2-7b-hf \
# --tasks nsmc \
# --num_fewshot 10 \
# --device cuda:0 \
# --output_path results/0b_nsmc_10_shot.json

# python main.py \
# --model gpt2 \
# --model_args pretrained=meta-llama/Llama-2-7b-hf \
# --tasks nsmc \
# --num_fewshot 50 \
# --device cuda:0 \
# --output_path results/0b_nsmc_50_shot.json
