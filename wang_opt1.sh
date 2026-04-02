#!/bin/bash

# --- 基础配置 ---
METHODS=("STE" "HTGE" "Uniform" "Normal")                        # 遍历不同的梯度估计方法 "STE" "HTGE" "Uniform" "Normal"
TASK=SST2                                     # SST2 RTE CB BoolQ WSC WIC MultiRC
STEPS=5000
IRS=("1e-8" "5e-8" "1e-7" "5e-7" "1e-6")      # 遍历不同的学习率  ("1e-8" "5e-8" "1e-7" "5e-7" "1e-6")
USE_SUM=False
MODEL=opt-1.3b
BATCH_SIZE=1
WBITS=4
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

# --- 外层循环：遍历不同的 METHOD ---
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

    # --- 内层循环：遍历不同的学习率 IR ---
    for IR in "${IRS[@]}"; do
        echo "------------------------------------------"
        echo "Running with Learning Rate: $IR"
        echo "------------------------------------------"
        
        # 构建完整路径（包含 IR 信息）
        SAVE_DIR="./log5/$MODEL-w${WBITS}a${ABITS}/$METHOD/$TASK/MAX_LENGTH-$MAX_LENGTH-STEPS-$STEPS-IR-$IR$DIR_SUFFIX"
        
        # --- 执行训练 ---
        CUDA_VISIBLE_DEVICES=1 python train_main.py \
            --model "facebook/$MODEL" \
            --epochs 0 \
            --q_output_dir "$SAVE_DIR" \
            --wbits "$WBITS" \
            --abits "$ABITS" \
            --lwc --let\
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
        
        # --- 检查训练是否成功 ---
        if [ $? -eq 0 ]; then
            echo "✅ Method $METHOD with IR $IR completed successfully!"
        else
            echo "❌ Method $METHOD with IR $IR failed!"
            # 可选：失败时是否继续
            # exit 1
        fi
        
        echo ""
    done
    
    echo "=========================================="
    echo "Method $METHOD all learning rates completed!"
    echo "=========================================="
    echo ""
done

echo "=========================================="
echo "All methods and learning rates completed!"
echo "=========================================="