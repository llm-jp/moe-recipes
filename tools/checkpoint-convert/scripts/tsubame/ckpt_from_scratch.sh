#!/bin/sh
#$ -cwd
#$ -l node_h=1
#$ -l h_rt=2:00:00
#$ -o outputs/convert/from_scratch_Mixtral-8x1.56B/$JOB_ID
#$ -e outputs/convert/from_scratch_Mixtral-8x1.56B/$JOB_ID
#$ -p -5

set -e
# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.2/2.20.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

# CUTLASS
CUTLASS_HOME=/gs/fs/tga-NII-LLM/modules/apps/cutlass/cutlass/build
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUTLASS_HOME}/lib

# swich virtual env
source venv/bin/activate
# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=4
NODE_TYPE="h100"

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r hostname _ rest; do
  echo "${hostname} slots=${NUM_GPU_PER_NODE}"
done <"$PE_HOSTFILE" >"$HOSTFILE_NAME"

START_ITERATION=6000
END_ITERATION=42000

BASE_MODEL_CHECKPOINT=/gs/bs/tgh-NII-LLM/checkpoints/from_scratch_Mixtral-8x1.56B

for (( ITERATION=$START_ITERATION; ITERATION<=$END_ITERATION; ITERATION+=6000 )); do
    FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

    CHECK_POINT_PATH=/gs/bs/tgh-NII-LLM/zero3_to_fp32/from_scratch_Mixtral-8x1.56B_zero3/lr_2e-4-minlr_2e-5_warmup_2000_seq_4096/${FORMATTED_ITERATION}/model.pt
    OUTPUT_PATH=/gs/bs/tgh-NII-LLM/fp32_to_hf/from_scratch_Mixtral-8x1.56B_zero3/lr_2e-4-minlr_2e-5_warmup_2000_seq_4096/${FORMATTED_ITERATION}
    mkdir -p $OUTPUT_PATH
    python tools/checkpoint-convert/convert_ckpt.py \
        --model $BASE_MODEL_CHECKPOINT \
        --ckpt $CHECK_POINT_PATH \
        --out $OUTPUT_PATH \
        --sequence-length 4096
    cp -r /gs/fs/tga-bayes-crest/taishi/workspace/llm-jp-tokenizer-v2.2-50k-add-bos/* $OUTPUT_PATH
    echo "Conversion completed for iteration $ITERATION"
done

echo "All conversions completed."
