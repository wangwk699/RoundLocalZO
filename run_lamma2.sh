# W4A4
CUDA_VISIBLE_DEVICES=0 python main.py \
--model /home/user/Workspace/Models/Llama-2-7b-hf  \
--epochs 20 --output_dir ./log/llama-7b-w4a4 \
--wbits 8 --abits 8 --lwc --let \
--save_dir ./pre_quantized_models/Llama-2-7B-w8a8 \
--tasks piqa,winogrande