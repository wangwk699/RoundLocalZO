#!/bin/bash

# --- 基础配置 ---
METHODS=("HTGE" "Uniform" "Normal")              # STE HTGE Uniform Normal Laplace
TASKS=("RTE" "CB") # 6 个任务
STEPS=5000
USE_SUM=False
MODEL=Llama-2-7b
BATCH_SIZE=1

WBITS=3
ABITS=16
MAX_LENGTH=2048

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

# --- GPU 内存清空函数 ---
clear_gpu_memory() {
    echo "🧹 Clearing GPU memory..."
    # 方法 1: 等待 Python 进程结束（自动释放）
    sleep 2
    
    # 方法 2: 强制清空 GPU 缓存（可选，谨慎使用）
    # nvidia-smi --gpu-reset -i 5 2>/dev/null || true
    
    # 方法 3: 查找并杀死占用 GPU 的进程（谨慎使用）
    # fuser -v /dev/nvidia* 2>/dev/null | awk '{print $2}' | xargs kill -9 2>/dev/null || true
    
    # 显示 GPU 使用情况
    echo "📊 GPU Memory Usage:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits,noheader -i 5 2>/dev/null || echo "N/A"
}

# --- 外层循环：遍历 TASKS ---
for TASK in "${TASKS[@]}"; do
    echo "=========================================="
    echo "Running Task: $TASK"
    echo "=========================================="
    
    # 获取当前任务对应的学习率
    IR="${TASK_LR[$TASK]}"
    echo "Learning Rate for $TASK: $IR"
    
    # --- 内层循环：遍历 METHODS ---
    for METHOD in "${METHODS[@]}"; do
        echo "------------------------------------------"
        echo "Running Method: $METHOD (Task: $TASK)"
        echo "------------------------------------------"
        
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
        CUDA_VISIBLE_DEVICES=5 python train_main.py \
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
        --evaluation_strategy "no" \
        
        # --- 检查训练是否成功 ---
        if [ $? -eq 0 ]; then
            echo "✅ Method $METHOD on Task $TASK with IR $IR completed successfully!"
        else
            echo "❌ Method $METHOD on Task $TASK with IR $IR failed!"
            # 可选：失败时是否继续
            # exit 1
        fi

        # --- 任务间等待和内存清空 ---
        echo ""
        echo "⏳ Waiting 10 seconds to cool down GPU..."
        sleep 10
        
        # 清空 GPU 内存
        clear_gpu_memory        
        
        echo ""
    done
    
    echo "=========================================="
    echo "Task $TASK completed!"
    echo "=========================================="
    echo ""
done

echo "=========================================="
echo "All tasks and methods completed!"
echo "=========================================="

#     # === 添加以下参数禁用 checkpoint 保存 ===
#   --save_strategy "no" \           # 禁用训练中间 checkpoint
#   --save_total_limit 0 \            # 不保留任何 checkpoint
#   --save_steps 999999 \             # 设置极大值避免触发
#   --evaluation_strategy "no" \      # 禁用评估时保存
#   --group_size "$GROUP_SIZE"