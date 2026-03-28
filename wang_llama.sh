#!/bin/bash

# --- 基础配置 ---
METHOD=STE              # STE HTGE Uniform Normal Laplace
TASK=MultiRC               # SST2 RTE CB BoolQ WSC WIC MultiRC  
STEPS=0
IR=1e-4
DELTA=0.05              # Uniform Normal Laplace
T=0.5                   # HTGE
USE_SUM=False
MODEL=Llama-2-7b
BATCH_SIZE=1

WBITS=3
ABITS=16
MAX_LENGTH=2048

if [ "$WBITS" -eq 2 ]; then
    RESUME="./pre_quantized_models/Llama-2-7b-w2a16.pth"
elif [ "$WBITS" -eq 3 ]; then
    RESUME="./pre_quantized_models/Llama-2-7b-w3a16g.pth"
elif [ "$WBITS" -eq 4 ]; then
    RESUME="./pre_quantized_models/Llama-2-7b-w4a16.pth"
else
    echo "Error: WBITS=$WBITS not supported"
    exit 1
fi

# 根据 METHOD 决定路径后缀
if [ "$METHOD" == "STE" ]; then
    # STE 方法不需要 DELTA 和 T 参数
    DIR_SUFFIX="-USE_SUM-$USE_SUM-BATCH_SIZE-$BATCH_SIZE"
else
    # HTGE, Uniform, Normal, Laplace 需要 DELTA 和 T 参数
    DIR_SUFFIX="-DELTA-$DELTA-T-$T-USE_SUM-$USE_SUM-BATCH_SIZE-$BATCH_SIZE"
fi


# Zero-Shot-Q
SAVE_DIR="./log1/$MODEL-w${WBITS}a${ABITS}/Zero-Shot-Q/$TASK-MAX_LENGTH-$MAX_LENGTH-STEPS-$STEPS-IR-$IR$DIR_SUFFIX"

# 构建完整路径
# SAVE_DIR="./log1/$MODEL-w${WBITS}a${ABITS}/$METHOD/$TASK/MAX_LENGTH-$MAX_LENGTH-STEPS-$STEPS-IR-$IR$DIR_SUFFIX"

# --- 执行训练 ---  -m debugpy --listen 6001 --wait-for-client
CUDA_VISIBLE_DEVICES=1 python train_main.py \
  --model "meta-llama/$MODEL-hf" \
  --epochs 0 \
  --q_output_dir "$SAVE_DIR" \
  --wbits "$WBITS" \
  --abits "$ABITS" \
  --lwc --let \
  --resume "$RESUME" \
  --train \
  --train_as_classification True \
  --task_name "$TASK" \
  --trainer "$METHOD" \
  --max_steps "$STEPS" \
  --learning_rate "$IR" \
  --output_dir "$SAVE_DIR" \
  --delta "$DELTA" \
  --use_sum "$USE_SUM" \
  --t "$T" \
  --max_length "$MAX_LENGTH" \
  --train_batch_size "$BATCH_SIZE"