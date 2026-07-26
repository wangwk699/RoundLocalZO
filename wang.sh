#!/bin/bash

# --- 基础配置 ---
METHOD="Uniform"              # STE HTGE Uniform Normal PWL MAD DSQ
TASK="SST2"                  # SST2 RTE WIC SQuAD
STEPS=10
IR=5e-6
IR_scheduler="constant_with_warmup"  # "constant_with_warmup"  "constant"
Warmup_ratio=0.03

USE_SUM=False
DSQ_ALPHA=0.2
MODEL=Qwen3-8B
BATCH_SIZE=2
LOGGING_STEPS=$((1000 / BATCH_SIZE))

WBITS=4
ABITS=16
MAX_LENGTH=2048   # 2048

RESUME="./pre_quantized_models/Qwen3-8B-w4a16.pth"


# 根据 METHOD 决定路径后缀
if [ "$METHOD" == "STE" ]; then
    # STE 方法不需要 DELTA 和 T 参数,但必须得有，传参才能不报错
    T=16
    DELTA=0.285        
    DIR_SUFFIX="USE_SUM-$USE_SUM-BATCH_SIZE-$BATCH_SIZE"
elif [ "$METHOD" == "HTGE" ]; then
    T=16
    DELTA=0.285
    DIR_SUFFIX="T-$T-USE_SUM-$USE_SUM-BATCH_SIZE-$BATCH_SIZE"
elif [ "$METHOD" == "Uniform" ]; then
    T=16    
    DELTA=0.285
    DIR_SUFFIX="DELTA-$DELTA-USE_SUM-$USE_SUM-BATCH_SIZE-$BATCH_SIZE"
elif [ "$METHOD" == "Normal" ]; then
    T=16    
    DELTA=0.15
    DIR_SUFFIX="DELTA-$DELTA-USE_SUM-$USE_SUM-BATCH_SIZE-$BATCH_SIZE"
elif [ "$METHOD" == "PWL" ] || [ "$METHOD" == "MAD" ]; then
    T=16
    DELTA=0.285
    DIR_SUFFIX="BATCH_SIZE-$BATCH_SIZE"
elif [ "$METHOD" == "DSQ" ]; then
    T=16
    DELTA=0.285
    DIR_SUFFIX="DSQ_ALPHA-$DSQ_ALPHA-BATCH_SIZE-$BATCH_SIZE"
else
    echo "Error: METHOD=$METHOD not supported"
    exit 1
fi

if [ "$TASK" == "SQuAD" ]; then
    TRAIN_AS_CLASSIFICATION=False
else
    TRAIN_AS_CLASSIFICATION=True
fi

# 构建完整路径
SAVE_DIR="./logs/log0/$MODEL-w${WBITS}a${ABITS}/$METHOD/$TASK/$MAX_LENGTH-STEPS-$STEPS-IR-$IR-$IR_scheduler-$Warmup_ratio-$DIR_SUFFIX"

# --- 执行训练 ---
CUDA_VISIBLE_DEVICES=2 python -m debugpy --listen 6001 --wait-for-client train_main.py \
    --model "Qwen/$MODEL" \
    --epochs 0 \
    --q_output_dir "$SAVE_DIR" \
    --wbits "$WBITS" \
    --abits "$ABITS" \
    --lwc \
    --resume "$RESUME" \
    --train \
    --train_as_classification "$TRAIN_AS_CLASSIFICATION" \
    --task_name "$TASK" \
    --trainer "$METHOD" \
    --max_steps "$STEPS" \
    --learning_rate "$IR" \
    --lr_scheduler_type "$IR_scheduler" \
    --warmup_ratio "$Warmup_ratio" \
    --output_dir "$SAVE_DIR" \
    --delta "$DELTA" \
    --use_sum "$USE_SUM" \
    --t "$T" \
    --dsq_alpha "$DSQ_ALPHA" \
    --max_length "$MAX_LENGTH" \
    --train_batch_size "$BATCH_SIZE" \
    --logging_steps "$LOGGING_STEPS" \
    --save_strategy "no" \
    --save_total_limit 0 \
    --save_steps 999999 \
    --evaluation_strategy "no" \