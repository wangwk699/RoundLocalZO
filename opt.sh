METHOD=qazo 
TASK=MultiRC                  
STEPS=5000
GROUP_NUM=2
IR=1e-7

CUDA_VISIBLE_DEVICES=1 python -m debugpy --listen 2026 --wait-for-client train_main.py \
--model facebook/opt-6.7b  \
--epochs 0 --q_output_dir ./log/opt-6.7b-w4a16 \
--wbits 4  --lwc --let \
--abits 16 \
--resume ./pre_quantized_models/opt-6.7b-w4a16.pth \
--train \
--train_as_classification True \
--task_name $TASK \
--trainer $METHOD \
--max_steps $STEPS \
--learning_rate $IR \
--output_dir ./log/$METHOD-$GROUP_NUM-$TASK-$STEPS-$IR-6.7 \
#Copa
#--save_dir ./pre_quantized_models/opt-2.7b-w4a16 \
#--train \
#--resume ./pre_quantized_models/opt-2.7b-w4a16 \
#sst2 boolq rte cb
#--abits 16
#--save_dir ./pre_quantized_models/opt-1.3b-w4a16 \
#-m debugpy --listen 5678 --wait-for-client 
#--tasks cb \
