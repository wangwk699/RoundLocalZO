# CUDA_VISIBLE_DEVICES=3 python -m debugpy --listen 5766 --wait-for-client chents_main.py \
# --model facebook/opt-2.7b  \
# --epochs 0 --output_dir ./log/opt-2.7b-w4a16 \
# --wbits 4  --lwc --let \
# --abits 16 \
# --tasks cb \
# --resume ./pre_quantized_models/opt-2.7b-w4a16.pth \
# --train \
#--save_dir ./pre_quantized_models/opt-2.7b-w4a16 \
#--train \
#--resume ./pre_quantized_models/opt-2.7b-w4a16 \
#sst2 boolq rte cb
#--abits 16
#--save_dir ./pre_quantized_models/opt-1.3b-w4a16 \
#-m debugpy --listen 5678 --wait-for-client 


# METHOD=qazo 
# TASK=MultiRC                  
# STEPS=5000
# GROUP_NUM=2
# IR=1e-7

# CUDA_VISIBLE_DEVICES=1 python train_main.py \
# --model facebook/opt-6.7b  \
# --epochs 0 --q_output_dir ./log/opt-6.7b-w4a16 \
# --wbits 4  --lwc --let \
# --abits 16 \
# --resume ./pre_quantized_models/opt-6.7b-w4a16.pth \
# --train \
# --train_as_classification True \
# --task_name $TASK \
# --trainer $METHOD \
# --max_steps $STEPS \
# --learning_rate $IR \
# --output_dir ./log/$METHOD-$GROUP_NUM-$TASK-$STEPS-$IR-6.7 \
# #Copa
# #--save_dir ./pre_quantized_models/opt-2.7b-w4a16 \
# #--train \
# #--resume ./pre_quantized_models/opt-2.7b-w4a16 \
# #sst2 boolq rte cb
# #--abits 16
# #--save_dir ./pre_quantized_models/opt-1.3b-w4a16 \
# #-m debugpy --listen 5678 --wait-for-client 
# #--tasks cb \


# METHOD=localzo
# TASK=MultiRC                  
# STEPS=5000
# GROUP_NUM=2
# IR=1e-7

# CUDA_VISIBLE_DEVICES=7 python -m debugpy --listen 2026 --wait-for-client train_main.py \
#   --model facebook/opt-1.3b \
#   --epochs 0 \
#   --q_output_dir ./log/opt-1.3b-w4a16 \
#   --wbits 4 --lwc --let \
#   --abits 16 \
#   --resume ./pre_quantized_models/opt-1.3b-w4a16.pth \
#   --train \
#   --train_as_classification True \
#   --task_name "$TASK" \
#   --trainer "$METHOD" \
#   --max_steps "$STEPS" \
#   --learning_rate "$IR" \
#   --output_dir "./log/$METHOD-$GROUP_NUM-$TASK-$STEPS-$IR-1.3" \
#   --_lambda 3 \
#   --delta 0.1 \
#   --sample_size 5 
#Copa
#--save_dir ./pre_quantized_models/opt-2.7b-w4a16 \
#--train \
#--resume ./pre_quantized_models/opt-2.7b-w4a16 \
#sst2 boolq rte cb
#--abits 16
#--save_dir ./pre_quantized_models/opt-1.3b-w4a16 \
#-m debugpy --listen 5678 --wait-for-client 
#--tasks cb \

# # SST2 MultiRC STE localzo
# METHOD=localzo
# TASK=SST2                  
# STEPS=5000
# GROUP_NUM=2
# IR=1e-7

# CUDA_VISIBLE_DEVICES=7 python -m debugpy --listen 2026 --wait-for-client train_main.py \
#   --model facebook/opt-1.3b \
#   --epochs 0 \
#   --q_output_dir ./log/opt-1.3b-w4a16 \
#   --wbits 4 --lwc --let \
#   --abits 16 \
#   --resume ./pre_quantized_models/opt-1.3b-w4a16.pth \
#   --train \
#   --train_as_classification True \
#   --task_name "$TASK" \
#   --trainer "$METHOD" \
#   --max_steps "$STEPS" \
#   --learning_rate "$IR" \
#   --output_dir "./log/$METHOD-$GROUP_NUM-$TASK-$STEPS-$IR-1.3" \
#   --_lambda 1.732 \
#   --delta 0.01 \
#   --sample_size 10 

# # SST2 MultiRC STE localzo
# # STE HTGE Uniform Normal Laplace
# METHOD=STE
# TASK=MultiRC                  
# STEPS=5000
# GROUP_NUM=1
# IR=1e-5
# DELTA=0.285
# LAMBDA=3

# export DEBUG_ROUNDZO=1
# export BREAK_ROUNDZO=1

# CUDA_VISIBLE_DEVICES=4 python train_main.py \
#   --model facebook/opt-1.3b \
#   --epochs 0 \
#   --q_output_dir ./log/opt-1.3b-w4a16 \
#   --wbits 4 --lwc --let \
#   --abits 16 \
#   --resume ./pre_quantized_models/opt-1.3b-w4a16.pth \
#   --train \
#   --train_as_classification True \
#   --task_name "$TASK" \
#   --trainer "$METHOD" \
#   --max_steps "$STEPS" \
#   --learning_rate "$IR" \
#   --output_dir "./log/$METHOD-$GROUP_NUM-$TASK-$STEPS-$IR-$DELTA-$LAMBDA-1.3" \
#   --_lambda "$LAMBDA" \
#   --delta "$DELTA" \
#   --sample_size 5


# # SST2 CB WSC WIC
# # STE HTGE Uniform Normal Laplace
# METHOD=Normal
# TASK=BoolQ                  
# STEPS=500
# GROUP_NUM=1
# IR=1e-7
# DELTA=0.285 # Uniform Normal Laplace
# T=0.5 # HTGE
# USE_SUM=True

# CUDA_VISIBLE_DEVICES=0 python train_main.py \
#   --model facebook/opt-1.3b \
#   --epochs 0 \
#   --q_output_dir ./log/opt-1.3b-w4a16 \
#   --wbits 4 --lwc --let \
#   --abits 16 \
#   --resume ./pre_quantized_models/opt-1.3b-w4a16.pth \
#   --train \
#   --train_as_classification True \
#   --task_name "$TASK" \
#   --trainer "$METHOD" \
#   --max_steps "$STEPS" \
#   --learning_rate "$IR" \
#   --output_dir "./log/$METHOD-$GROUP_NUM-$TASK-$STEPS-$IR-$DELTA-$T-$USE_SUM" \
#   --delta "$DELTA" \
#   --use_sum $USE_SUM \
#   --t "$T" \
#   --max_length 1024 \
#   --train_batch_size 1




# SST2 CB WSC WIC
# STE HTGE Uniform Normal Laplace
METHOD=Normal
TASK=BoolQ                  
STEPS=500
GROUP_NUM=1
IR=1e-7
DELTA=0.285 # Uniform Normal Laplace
T=0.5 # HTGE
USE_SUM=False

CUDA_VISIBLE_DEVICES=0 python train_main.py \
  --model facebook/opt-6.7b \
  --epochs 0 \
  --q_output_dir ./log/opt-6.7b-w4a16 \
  --wbits 4 --lwc --let \
  --abits 16 \
  --resume ./pre_quantized_models/opt-6.7b-w4a16.pth \
  --train \
  --train_as_classification True \
  --task_name "$TASK" \
  --trainer "$METHOD" \
  --max_steps "$STEPS" \
  --learning_rate "$IR" \
  --output_dir "./log/$METHOD-$GROUP_NUM-$TASK-$STEPS-$IR-$DELTA-$T-$USE_SUM" \
  --delta "$DELTA" \
  --use_sum $USE_SUM \
  --t "$T" \
  --max_length 2048 \
  --train_batch_size 2