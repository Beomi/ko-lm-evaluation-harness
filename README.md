# Ko LM Eval Harnesss

한국어 공개 데이터셋으로 평가하는 언어모델 점수!

## 사용법

```bash
git clone https://github.com/Beomi/ko-lm-evaluation-harness
cd ko-lm-evaluation-harness
pip install -r requirements.txt
./run_all.sh 모델이름 'GPU번호들'
# ex) ./run_all.sh beomi/llama-2-ko-7b '0,1' # 이렇게 하면 GPU0,1번에 'beomi/llama-2-ko-7b' 모델을 쪼개서 올리고 평가한다.
```

위 `run_all.sh` 실행 후, ko-lm-evaluation-harness 메인 폴더에서 [결과물한눈에보기.ipynb](https://github.com/Beomi/ko-lm-evaluation-harness/blob/main/%EA%B2%B0%EA%B3%BC%EB%AC%BC%ED%95%9C%EB%88%88%EC%97%90%EB%B3%B4%EA%B8%B0.ipynb) 실행

## 예시 결과(markdown)

- [beomi/llama-2-koen-13b](https://huggingface.co/beomi/llama-2-koen-13b) 테스트 결과

```
| Task                  |   0-shot |   5-shot |   10-shot |   50-shot |
|:----------------------|---------:|---------:|----------:|----------:|
| kobest_boolq          | 0.398848 | 0.703795 |  0.752612 |  0.7578   |
| kobest_copa           | 0.776785 | 0.812796 |  0.818724 |  0.853953 |
| kobest_hellaswag      | 0.499922 | 0.512659 |  0.503365 |  0.524664 |
| kobest_sentineg       | 0.586955 | 0.974811 |  0.982367 |  0.987405 |
| kohatespeech          | 0.278224 | 0.378693 |  0.370702 |  0.509343 |
| kohatespeech_apeach   | 0.337667 | 0.556898 |  0.581788 |  0.667511 |
| kohatespeech_gen_bias | 0.248404 | 0.484745 |  0.473659 |  0.461714 |
| korunsmile            | 0.327145 | 0.329163 |  0.347889 |  0.395522 |
| nsmc                  | 0.6442   | 0.87702  |  0.89982  |  0.90984  |
| pawsx_ko              | 0.5355   | 0.5455   |  0.5435   |  0.5255   |
```

## 지원되는 테스트

**현재 동작 가능 ✅**

- kobest_hellaswag
- kobest_copa
- kobest_boolq
- kobest_sentineg
- kohatespeech
- kohatespeech_apeach
- kohatespeech_gen_bias
- korunsmile
- nsmc
- pawsx_ko
- klue_nli
- klue_sts
- klue_ynat

**디버그 중 🧐**

- korquad
- ko_en_translation
- en_ko_translation
- klue_mrc

## 지원되는 모델

- Hugging Face Transformers에서 `AutoModelForCausalLM.from_pretrained`로 로딩 가능한 모든 모델
  - ex) `beomi/llama-2-ko-7b`, `beomi/KoAlpaca-Polyglot-12.8B`,`EleutherAI/polyglot-ko-12.8b` 등

> 아래는 원본 Eval Harness 레포 README

---

# Language Model Evaluation Harness

## Overview

This project provides a unified framework to test generative language models on a large number of different evaluation tasks.

Features:

- 200+ tasks implemented. See the [task-table](./docs/task_table.md) for a complete list.
- Support for models loaded via [transformers](https://github.com/huggingface/transformers/), [GPT-NeoX](https://github.com/EleutherAI/gpt-neox), and [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/), with a flexible tokenization-agnostic interface.
- Support for commercial APIs including [OpenAI](https://openai.com), [goose.ai](https://goose.ai), and [TextSynth](https://textsynth.com/).
- Support for evaluation on adapters (e.g. LoRa) supported in [HuggingFace's PEFT library](https://github.com/huggingface/peft).
- Evaluating with publicly available prompts ensures reproducibility and comparability between papers.
- Task versioning to ensure reproducibility when tasks are updated.

## Install

To install `lm-eval` from the github repository main branch, run:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

To install additional multilingual tokenization and text segmentation packages, you must install the package with the `multilingual` extra:

```bash
pip install -e ".[multilingual]"
```

## Basic Usage

> **Note**: When reporting results from eval harness, please include the task versions (shown in `results["versions"]`) for reproducibility. This allows bug fixes to tasks while also ensuring that previously reported scores are reproducible. See the [Task Versioning](#task-versioning) section for more info.

### Hugging Face `transformers`

To evaluate a model hosted on the [HuggingFace Hub](https://huggingface.co/models) (e.g. GPT-J-6B) on `hellaswag` you can use the following command:


```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0
```

Additional arguments can be provided to the model constructor using the `--model_args` flag. Most notably, this supports the common practice of using the `revisions` feature on the Hub to store partially trained checkpoints:

```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000 \
    --tasks lambada_openai,hellaswag \
    --device cuda:0
```

To evaluate models that are loaded via `AutoSeq2SeqLM` in Huggingface, you instead use `hf-seq2seq`. *To evaluate (causal) models across multiple GPUs, use `--model hf-causal-experimental`*

> **Warning**: Choosing the wrong model may result in erroneous outputs despite not erroring.

### Commercial APIs

Our library also supports language models served via the OpenAI API:

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python main.py \
    --model gpt3 \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag
```

While this functionality is only officially maintained for the official OpenAI API, it tends to also work for other hosting services that use the same API such as [goose.ai](goose.ai) with minor modification. We also have an implementation for the [TextSynth](https://textsynth.com/index.html) API, using `--model textsynth`.

To verify the data integrity of the tasks you're performing in addition to running the tasks themselves, you can use the `--check_integrity` flag:

```bash
python main.py \
    --model gpt3 \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag \
    --check_integrity
```

### Other Frameworks

A number of other libraries contain scripts for calling the eval harness through their library. These include [GPT-NeoX](https://github.com/EleutherAI/gpt-neox/blob/main/eval_tasks/eval_adapter.py), [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/blob/main/examples/MoE/readme_evalharness.md), and [mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/eval_harness.py).

💡 **Tip**: You can inspect what the LM inputs look like by running the following command:

```bash
python write_out.py \
    --tasks all_tasks \
    --num_fewshot 5 \
    --num_examples 10 \
    --output_base_path /path/to/output/folder
```

This will write out one text file for each task.

## Advanced Usage

For models loaded with the HuggingFace  `transformers` library, any arguments provided via `--model_args` get passed to the relevant constructor directly. This means that anything you can do with `AutoModel` can be done with our library. For example, you can pass a local path via `pretrained=` or use models finetuned with [PEFT](https://github.com/huggingface/peft) by taking the call you would run to evaluate the base model and add `,peft=PATH` to the `model_args` argument:
```bash
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=EleutherAI/gpt-j-6b,peft=nomic-ai/gpt4all-j-lora \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --device cuda:0
```


We support wildcards in task names, for example you can run all of the machine-translated lambada tasks via `--task lambada_openai_mt_*`.

We currently only support one prompt per task, which we strive to make the "standard" as defined by the benchmark's authors. If you would like to study how varying prompts causes changes in the evaluation score, check out the [BigScience fork](https://github.com/bigscience-workshop/lm-evaluation-harness) of this repo. We are currently working on upstreaming this capability to `main`.

## Implementing new tasks

To implement a new task in the eval harness, see [this guide](./docs/task_guide.md).

## Task Versioning

To help improve reproducibility, all tasks have a `VERSION` field. When run from the command line, this is reported in a column in the table, or in the "version" field in the evaluator return dict. The purpose of the version is so that if the task definition changes (i.e to fix a bug), then we can know exactly which metrics were computed using the old buggy implementation to avoid unfair comparisons. To enforce this, there are unit tests that make sure the behavior of all tests remains the same as when they were first implemented. Task versions start at 0, and each time a breaking change is made, the version is incremented by one.

When reporting eval harness results, please also report the version of each task. This can be done either with a separate column in the table, or by reporting the task name with the version appended as such: taskname-v0.

## Test Set Decontamination

To address concerns about train / test contamination, we provide utilities for comparing results on a benchmark using only the data points nto found in the model training set. Unfortunately, outside of models trained on the Pile and C4, its very rare that people who train models disclose the contents of the training data. However this utility can be useful to evaluate models you have trained on private data, provided you are willing to pre-compute the necessary indices. We provide computed indices for 13-gram exact match deduplication against the Pile, and plan to add additional precomputed dataset indices in the future (including C4 and min-hash LSH deduplication).

For details on text decontamination, see the [decontamination guide](./docs/decontamination.md).

Note that the directory provided to the `--decontamination_ngrams_path` argument should contain the ngram files and info.json. See the above guide for ngram generation for the pile, this could be adapted for other training sets.

```bash
python main.py \
    --model gpt2 \
    --tasks sciq \
    --decontamination_ngrams_path path/containing/training/set/ngrams \
    --device cuda:0
```

## Cite as

```
@software{eval-harness,
  author       = {Gao, Leo and
                  Tow, Jonathan and
                  Biderman, Stella and
                  Black, Sid and
                  DiPofi, Anthony and
                  Foster, Charles and
                  Golding, Laurence and
                  Hsu, Jeffrey and
                  McDonell, Kyle and
                  Muennighoff, Niklas and
                  Phang, Jason and
                  Reynolds, Laria and
                  Tang, Eric and
                  Thite, Anish and
                  Wang, Ben and
                  Wang, Kevin and
                  Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```
