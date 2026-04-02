#!/bin/bash

# --- 基础配置 ---
METHOD="Normal"              # STE HTGE Uniform Normal Laplace
TASK="MultiRC" # 6 个任务
STEPS=5000
USE_SUM=False
MODEL=Llama-2-7b
BATCH_SIZE=1

WBITS=3
ABITS=16
MAX_LENGTH=512

# --- 关联数组：任务 → 学习率 ---
declare -A TASK_LR
TASK_LR["SST2"]="1e-7"
TASK_LR["RTE"]="1e-7"
TASK_LR["CB"]="5e-6"
TASK_LR["BoolQ"]="5e-7"
TASK_LR["WSC"]="1e-8"
TASK_LR["WIC"]="5e-7"
TASK_LR["MultiRC"]="1e-7"

# --- 设置 Resume 路径 ---
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

# 获取当前任务对应的学习率
IR="${TASK_LR[$TASK]}"

# 根据 METHOD 决定路径后缀
if [ "$METHOD" == "STE" ]; then
    T=16
    DELTA=0.285        
    DIR_SUFFIX="-USE_SUM-$USE_SUM-BATCH_SIZE-$BATCH_SIZE"
elif [ "$METHOD" == "HTGE" ]; then
    T=16
    DELTA=0.285
    DIR_SUFFIX="-T-$T-USE_SUM-$USE_SUM-BATCH_SIZE-$BATCH_SIZE"
elif [ "$METHOD" == "Uniform" ]; then
    T=16    
    DELTA=0.285
    DIR_SUFFIX="-DELTA-$DELTA-USE_SUM-$USE_SUM-BATCH_SIZE-$BATCH_SIZE"
elif [ "$METHOD" == "Normal" ]; then
    T=16    
    DELTA=0.15
    DIR_SUFFIX="-DELTA-$DELTA-USE_SUM-$USE_SUM-BATCH_SIZE-$BATCH_SIZE"
else
    echo "Error: METHOD=$METHOD not supported"
    exit 1
fi

# 构建完整路径
SAVE_DIR="./log3/$MODEL-w${WBITS}a${ABITS}/$METHOD/$TASK/MAX_LENGTH-$MAX_LENGTH-STEPS-$STEPS-IR-$IR$DIR_SUFFIX"

# --- 执行训练 ---
CUDA_VISIBLE_DEVICES=0 python train_main.py \
    --model "meta-llama/$MODEL-hf" \
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
    --save_strategy "no" \
    --save_total_limit 0 \
    --save_steps 999999 \
    --evaluation_strategy "no" 
        
        

    



#     # === 添加以下参数禁用 checkpoint 保存 ===
#   --save_strategy "no" \           # 禁用训练中间 checkpoint
#   --save_total_limit 0 \            # 不保留任何 checkpoint
#   --save_steps 999999 \             # 设置极大值避免触发
#   --evaluation_strategy "no" \      # 禁用评估时保存
#   --group_size "$GROUP_SIZE"