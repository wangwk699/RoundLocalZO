# SST2 CB WSC WIC
# STE HTGE Uniform Normal Laplace
METHOD=STE
TASK=SST2                  
STEPS=5000
GROUP_NUM=1
IR=1e-5
DELTA=0.05 # Uniform Normal Laplace
T=0.5 # HTGE
USE_SUM=False

CUDA_VISIBLE_DEVICES=7 python train_main.py \
  --model facebook/opt-1.3b \
  --epochs 0 \
  --q_output_dir ./log/opt-1.3b-w4a16 \
  --wbits 4 --lwc --let \
  --abits 16 \
  --resume ./pre_quantized_models/opt-1.3b-w4a16.pth \
  --train \
  --train_as_classification True \
  --task_name "$TASK" \
  --trainer "$METHOD" \
  --max_steps "$STEPS" \
  --learning_rate "$IR" \
  --output_dir "./log/opt-1.3b-$METHOD-$GROUP_NUM-$TASK-$STEPS-$IR-$DELTA-$T-$USE_SUM" \
  --delta "$DELTA" \
  --use_sum $USE_SUM \
  --t "$T" \
  --max_length 2048 \
  --train_batch_size 1