"""
测试双重量化问题
"""
import torch
from quantize.quantizer import UniformAffineQuantizer

# 模拟权重
weight = torch.randn(128, 256).cuda()

# 创建量化器
quantizer = UniformAffineQuantizer(
    n_bits=4,
    per_channel_axes=[0],
    symmetric=True,
    dynamic_method="per_channel"
)

print("原始权重范围:", weight.min().item(), weight.max().item())

# 第一次量化（模拟smooth_and_quant_inplace）
weight_quantized_once = quantizer(weight)
print("第一次量化后范围:", weight_quantized_once.min().item(), weight_quantized_once.max().item())
print("第一次量化后权重:", weight_quantized_once[0, :5])

# 第二次量化（模拟推理时weight_quant=True）
weight_quantized_twice = quantizer(weight_quantized_once)
print("第二次量化后范围:", weight_quantized_twice.min().item(), weight_quantized_twice.max().item())
print("第二次量化后权重:", weight_quantized_twice[0, :5])

# 检查差异
diff = (weight_quantized_twice - weight_quantized_once).abs().max()
print(f"\n双重量化导致的最大差异: {diff.item()}")
print("如果差异很大，说明双重量化会严重损失精度！")
