# SST2 CB WSC WIC
# STE HTGE Uniform Normal Laplace
METHOD=Uniform
TASK=ReCoRD                  
STEPS=5000
GROUP_NUM=1
IR=1e-10
DELTA=0.285 # Uniform Normal Laplace
T=16 # HTGE
MODEL=Llama-2-7b
USE_SUM=False

CUDA_VISIBLE_DEVICES=7 python train_main.py \
  --model meta-llama/Llama-2-7b-hf \
  --epochs 0 \
  --q_output_dir ./log/Llama-2-7b-w4a16 \
  --wbits 4 --lwc --let \
  --abits 16 \
  --resume ./pre_quantized_models/Llama-2-7b-w4a16.pth \
  --train \
  --train_as_classification True \
  --task_name "$TASK" \
  --trainer "$METHOD" \
  --max_steps "$STEPS" \
  --learning_rate "$IR" \
  --output_dir "./log/$MODEL/$METHOD-$GROUP_NUM-$TASK-$STEPS-$IR-$DELTA-$T-$USE_SUM" \
  --delta "$DELTA" \
  --use_sum $USE_SUM \
  --t "$T" \
  --max_length 32 \
  --train_batch_size 1