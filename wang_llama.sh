#!/bin/bash

# --- 基础配置 ---
METHODS=("HTGE" "Uniform" "Normal")     # STE HTGE Uniform Normal
TASKS=("RTE" "CB")                               # SST2 RTE CB BoolQ WSC WIC MultiRC  
STEPS=5000
IR=1e-7
USE_SUM=False
MODEL=Llama-2-7b
BATCH_SIZE=1

WBITS=3
ABITS=16
MAX_LENGTH=2048

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

# --- For 循环遍历不同 METHOD ---
for METHOD in "${METHODS[@]}"; do
    echo "=========================================="
    echo "Running Method: $METHOD"
    echo "=========================================="
    
    # ✅ 根据 METHOD 决定路径后缀（移到循环内部）
    if [ "$METHOD" == "STE" ]; then
        # STE 方法不需要 DELTA 和 T 参数,但必须得有，传参才能不报错
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

    # ✅ 构建完整路径（移到循环内部）
    SAVE_DIR="./log3/$MODEL-w${WBITS}a${ABITS}/$METHOD/$TASK/MAX_LENGTH-$MAX_LENGTH-STEPS-$STEPS-IR-$IR$DIR_SUFFIX"

    # --- 执行训练 ---
    CUDA_VISIBLE_DEVICES=4 python train_main.py \
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
    --train_batch_size "$BATCH_SIZE"
    
    # --- 检查训练是否成功 ---
    if [ $? -eq 0 ]; then
        echo "✅ Method $METHOD completed successfully!"
    else
        echo "❌ Method $METHOD failed!"
        # 可选：失败时是否继续
        # exit 1
    fi
    
    echo ""
done

echo "=========================================="
echo "All methods completed!"
echo "=========================================="


# Zero-Shot-Q
# SAVE_DIR="./log1/$MODEL-w${WBITS}a${ABITS}/Zero-Shot-Q-weight2/$TASK-MAX_LENGTH-$MAX_LENGTH-STEPS-$STEPS-IR-$IR$DIR_SUFFIX"