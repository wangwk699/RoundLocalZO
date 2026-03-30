#!/bin/bash

# --- 基础配置 ---
METHOD=STE              # STE HTGE Uniform Normal Laplace
TASK=SST2               # SST2 RTE CB BoolQ WSC WIC   
STEPS=5000
IR=1e-5
DELTA=0.285              # Uniform Normal Laplace
T=0.5                   # HTGE
USE_SUM=False
MODEL=opt-1.3b
BATCH_SIZE=1

WBITS=3
ABITS=16
MAX_LENGTH=2048

# 根据 WBITS 选择 checkpoint
if [ "$WBITS" -eq 2 ]; then
    RESUME="./pre_quantized_models/$MODEL-w2a16g128.pth"
    GROUP_SIZE=128
elif [ "$WBITS" -eq 3 ]; then
    RESUME="./pre_quantized_models/$MODEL-w3a16.pth"
elif [ "$WBITS" -eq 4 ]; then
    RESUME="./pre_quantized_models/$MODEL-w4a16.pth"
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


# 构建完整路径
# Zero-Shot-Q
SAVE_DIR="./log1/$MODEL-w${WBITS}a${ABITS}/Zero-Shot-Q-weight2/$TASK-MAX_LENGTH-$MAX_LENGTH-STEPS-$STEPS-IR-$IR$DIR_SUFFIX"

# 注意：保留了原脚本中 "GROUP_NUM-$GROUP_NUM" 的字符串格式
# SAVE_DIR="./log1/$MODEL-w${WBITS}a${ABITS}/$METHOD-$GROUP_NUM-MAX_LENGTH-$MAX_LENGTH-$TASK-STEPS-$STEPS-IR-$IR$DIR_SUFFIX"

# --- 执行训练 ---  -m debugpy --listen 6001 --wait-for-client
CUDA_VISIBLE_DEVICES=0 python train_main.py \
  --model "facebook/$MODEL" \
  --epochs 0 \
  --q_output_dir "$SAVE_DIR" \
  --wbits "$WBITS" \
  --abits "$ABITS" \
  --lwc \
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
  --train_batch_size "$BATCH_SIZE" \
#   --group_size "$GROUP_SIZE"