#!/bin/sh
#$ -cwd
#$ -l cpu_160=1
#$ -l h_rt=5:00:00
#$ -o outputs/tokenize/$JOB_ID.log
#$ -e outputs/tokenize/$JOB_ID.log
#$ -p -5

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.2/2.20.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

source .env/bin/activate

# dataset
DATASET_DIR=/gs/bs/tga-NII-LLM/datasets/raw/pretrain/swallow-meta-tag
OUTPUT_DIR=/gs/bs/tga-NII-LLM/datasets/binarized/swallow-meta-tag/DeepseekTokenizer

mkdir -p $OUTPUT_DIR

# tokenize
python megatron_lm/tools/preprocess_data.py \
  --input ${DATASET_DIR}/wiki-base.jsonl \
  --output-prefix ${OUTPUT_DIR}/ja_wiki \
  --tokenizer-type DeepseekTokenizer \
  --tokenizer-model /gs/bs/tga-NII-LLM/hf-checkpoints/deepseek-moe-16b-base/tokenizer.json \
  --append-eod \
  --workers 64
