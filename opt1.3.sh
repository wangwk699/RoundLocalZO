CUDA_VISIBLE_DEVICES=3 python -m debugpy --listen 5766 --wait-for-client chents_main.py \
--model facebook/opt-2.7b  \
--epochs 0 --output_dir ./log/opt-2.7b-w4a16 \
--wbits 4  --lwc --let \
--abits 16 \
--tasks cb \
--resume ./pre_quantized_models/opt-2.7b-w4a16.pth \
--train \
#--save_dir ./pre_quantized_models/opt-2.7b-w4a16 \
#--train \
#--resume ./pre_quantized_models/opt-2.7b-w4a16 \
#sst2 boolq rte cb
#--abits 16
#--save_dir ./pre_quantized_models/opt-1.3b-w4a16 \
#-m debugpy --listen 5678 --wait-for-client 