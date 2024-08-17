<div align="center">

moe-recipes
===========================
<h4>User-friendly tool for seamless continual pre-training of Mixture of Expert Models</h4>

<img src="images/moe-recipes-logo.webp" alt="moe-recipes" width="300px">

<div align="left">

moe-recipes is a tool designed to make the continual pre-training of Large Language Models (LLMs) with Mixture of Experts (MoE) architecture easy and efficient. With an intuitive interface and flexible configuration options, researchers and developers can effortlessly manage training on any MoE model or dataset. The tool supports distributed training on large GPU clusters using DeepSpeed as its backend and offers extensive customization, enabling users to leverage cutting-edge techniques with ease.

What sets moe-recipes apart is its seamless integration with Hugging Face Transformers, allowing you to continue pre-training or perform instruction tuning on MoE models with minimal changes. This means there’s no need to convert checkpoints or deal with complex workflows—just focus on refining your model.

| Feature                         | moe-recipes | llm-recipes |
|---------------------------------|-------------|---------------|
| **MoE Support**                 | ✅          | ❌            |
| **Dense LLM Support**           | ❌          | ✅            |
| **Continual Pre-Training**      | ✅          | ✅            |
| **Multi-Node Support**          | ✅          | ✅            |

# Table of Contents

- [Installation](#installation)
  - [Multi-node Support](#multi-node-support)
  - [FlashAttention](#flashattention)
- [Usage](#usage)
  - [MoE Instruction Tuning](#moe-instruction-tuning)
  - [MoE Continual Pre-Training](#moe-continual-pre-training)
- [Checkpoint formats](#checkpoint-formats)
  - [DeepSpeed format to Hugging Face format](#deepspeed-format-to-hugging-face-format)
- [Inference](#inference)
- [Training Speed and Scalability](#training-speed-and-scalability)
- [Projects Using moe-recipes](#projects-using-moe-recipes)
- [Citation](#citation)

## Installation

This package has been tested with Python 3.10 and 3.11. The recommended environment is with CUDA Toolkit 12.1.

To install the required packages, simply run:

```bash
pip install -r requirements.txt
```
> Note: The requirements.txt assumes that CUDA Toolkit 12.1 is installed on your system.

### Multi-node Support

For multi-node support, ensure you have the following dependencies installed:

```bash
module load openmpi/4.x.x

pip install mpi4py
```

### FlashAttention

For GPU-accelerated FlashAttention, follow these steps:

```bash
pip install ninja packaging wheel
pip install flash-attn --no-build-isolation
```

## Usage

### MoE Instruction Tuning

we experimentally support instruction tuning for MoE models.
we don't fully test the instruction tuning, so please be careful when using it.

#### 1. **Data Preparation**

Prepare your data in the below format and save it as a JSONL file:

```jsonl
{
  "input": [
    {
      "role": "user",
      "content": "What is the weather like today?"
    }
  ],
  "output": {
    "role": "assistant",
    "content": "The weather is sunny with a high of 25 degrees."
  }
}
```

#### 2. **Change Dataset Class**

Please modify the `Dataset` class in `src/llama_recipes/utils/instruction_tuning.py` to adjust to the model's expected format.
But, almost all the models have chat templates, so you may not need to change the `Dataset` class.

#### 3. **Indexing**

To load dataset efficiently, create an index file using the following command:

```bash
python tools/pre-process/index_dataset.py \
  --data-file-path <path-to-jsonl-file>
```

After indexing, `.index_cache` directory will be created in the same directory as the JSONL file.

#### 4. **Training**

We does not provide a script for instruction tuning, but we are planning to provide it in the future.

### MoE Continual Pre-Training

#### 1. **Data Preparation**

Prepare your data in the below format and save it as a JSONL file:

```jsonl
{
  "text": "What is the weather like today?\nThe weather is sunny with a high of 25 degrees."
}
```

#### 2. **Tokenize Data**

Tokenize your data using the tokenizer provided by the model you are using.
For example, to tokenize data for Qwen-2-57A, run the following command:

```bash
DATASET_DIR=/pat/to/datasets/
OUTPUT_DIR=/path/datasets/

mkdir -p $OUTPUT_DIR

python megatron_lm/tools/preprocess_data.py \
  --input ${DATASET_DIR}/wiki-base.jsonl \
  --output-prefix ${OUTPUT_DIR}/ja_wiki \
  --tokenizer-type Qwen2Tokenizer \
  --tokenizer-model /path/to/hf-checkpoints/Qwen2-57B-A14B/tokenizer.json \
  --append-eod \
  --workers 64
```

#### 3. **Training**

We support Mixtral, Qwen-2-MoE, deepseek-moe.
If you want to continually pre-train or instruction tune other models, you should modify `src/llama_recipes/get_models.py` and `src/llama_recipes/get_model_decoder_layer.py`.

We provide example scripts for continual pre-training for Mixtral-8x7B in `scripts/tsubame/Mixtral-8x7B-VE/mixtral-8x7b.sh`.
You can modify the script to suit your needs.

## Checkpoint formats

### DeepSpeed format to Hugging Face format

You can convert DeepSpeed checkpoints to Hugging Face format in two stages: first, convert the checkpoint to PyTorch format, and then convert the PyTorch checkpoint to Hugging Face format.

#### 1. **Convert DeepSpeed checkpoint to PyTorch format**

```bash
ITERATION=2000
FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

CHECK_POINT_DIR=/path/Mixtral-8x7b/${FORMATTED_ITERATION}

python tools/checkpoint-convert/zero_to_fp32.py \
  --checkpoint-dir $CHECK_POINT_DIR \
  --output-file $CHECK_POINT_DIR/model.pt \
  --debug
```

#### 2. **Convert PyTorch checkpoint to Hugging Face format**

```bash
  ITERATION=2000
  FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

  CHECK_POINT_PATH=/path/to/checkpoints/Mixtral-8x7b/${FORMATTED_ITERATION}/model.pt
  OUTPUT_PATH=/path/to/Mixtral-8x7b/${FORMATTED_ITERATION}

  echo "convert ${CHECK_POINT_PATH} to ${OUTPUT_PATH}"

  mkdir -p $OUTPUT_PATH

  BASE_MODEL_CHECKPOINT=/path/to/Mixtral-8x7B-v0.1

  python tools/checkpoint-convert/convert_ckpt.py \
    --model $BASE_MODEL_CHECKPOINT \
    --ckpt $CHECK_POINT_PATH \
    --out $OUTPUT_PATH \
    --sequence-length 8192
```

## Inference

After checkpoint conversion, you can use the Hugging Face Transformers library to load the converted checkpoint and perform inference.

The following is an example of how to do inference using the converted checkpoint (huggingface format):

```bash
python tools/inference/inference-mixtral.py \
  --model-path /path/to/converted/iter_0004000 \
  --tokenizer-path /path/to/tokenizer/path \
  --prompt "Tokyo is the capital of"
```

## Training Speed and Scalability

We are currently working on improving the training speed and scalability of moe-recipes.
We will update this section with more information soon.

## Projects Using moe-recipes

Below are some of the projects where we have directly used moe-recipes:

- [Building a Large Japanese Web Corpus for Large Language Models](https://arxiv.org/abs/2404.17733)

## Citation

we are current submitting the paper to SC24 workshop, and the citation will be updated soon.

```bibtex
@software{fujii_moe-recipes_2024,
author = {Kazuki Fujii and Taishi Nakamura and Rio Yokota},
month = {March},
title = {{moe-recipes}},
url = {https://github.com/rioyokotalab/moe-recipes},
version = {1.0.0},
year = {2024}
}
```
