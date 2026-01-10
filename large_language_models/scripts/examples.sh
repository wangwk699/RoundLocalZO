# MeZO
#CUDA_VISIBLE_DEVICES=0 MODEL=facebook/opt-6.7b TASK=SST2 MODE=ft LR=1e-7 EPS=1e-3 bash scripts/mezo.sh

# GPTQ Quantization
#CUDA_VISIBLE_DEVICES=0 python quantization.py --model_path facebook/opt-1.3b --quant_mode gptq --quant_path quantized

# QZO
CUDA_VISIBLE_DEVICES=0 TASK=SST2 MODE=ft LR=1e-7 EPS=1e-3 bash scripts/qzo.sh

#MODEL=quantized/opt-1.3b-gptq-b4-g128