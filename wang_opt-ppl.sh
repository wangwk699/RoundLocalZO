#!/bin/bash

# ============ WikiText2 PPL 微调训练配置 ============
MODEL=facebook/opt-1.3b
METHOD=STE                    # 可选项: STE | HTGE | Uniform | Normal | Laplace
TASK=wikitext2                # 语言建模数据集
STEPS=5000
GROUP_NUM=1
IR=1e-5
DELTA=0.21                    # Uniform/Normal 的扰动尺度 (根据论文 Table 2 最优值)
T=0.5                          # HTGE 的温度参数
USE_SUM=False

# ============ 训练 + 评估参数 ============
EVAL_PPL=True                 # 训练后启用困惑度评估
CALIB_DATASET=wikitext2       # 校准/评估数据集
CACHE_DIR=./cache             # 数据集缓存目录

# ============ 运行命令 ============
CUDA_VISIBLE_DEVICES=7 python train_main.py \
  --model "$MODEL" \
  --epochs 0 \
  --q_output_dir ./log/opt-1.3b-w4a16 \
  --wbits 4 --lwc --let \
  --abits 16 \
  --resume ./pre_quantized_models/opt-1.3b-w4a16.pth \
  --train True \                  # 【关键】启用微调训练
  --train_as_classification False \   # 【关键】WikiText2 是语言建模任务，不是分类
  --task_name "$TASK" \             # 【关键】指定 wikitext2 数据集
  --trainer "$METHOD" \
  --max_steps "$STEPS" \
  --learning_rate "$IR" \
  --output_dir "./log/opt-1.3b-$METHOD-$GROUP_NUM-$TASK-$STEPS-$IR-$DELTA-$T-$USE_SUM" \
  --delta "$DELTA" \
  --use_sum "$USE_SUM" \
  --t "$T" \
  --max_length 2048 \
  --train_batch_size 1 \
  --eval_ppl "$EVAL_PPL" \          # 【关键】训练后评估困惑度
  --calib_dataset "$CALIB_DATASET" \
  --cache_dir "$CACHE_DIR"