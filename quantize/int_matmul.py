import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer


class QuantMatMul(nn.Module):
    def __init__(
        self,
        x1_quant_params: dict = {},  # 输入 x1 的量化参数
        x2_quant_params: dict = {},  # 输入 x2 的量化参数
        disable_act_quant=False,     # 是否禁用激活量化的开关
        matmul_func=torch.bmm,       # 矩阵乘法函数，默认为批量矩阵乘法
    ):
        super().__init__()
        # de-activate the quantized forward default  禁用默认的量化前向传播
        self.use_act_quant = False
        # initialize quantizer
        self.i_cluster_counts = None
        self.x1_quantizer = UniformAffineQuantizer(**x1_quant_params)
        self.x2_quantizer = UniformAffineQuantizer(**x2_quant_params)
        self.matmul_func = matmul_func  # 矩阵乘法函数

        self.disable_act_quant = disable_act_quant


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def quant_x1(self, x1):
        if self.use_act_quant:
            x1 = self.x1_quantizer(x1)
        return x1

    def quant_x2(self, x2):
        if self.use_act_quant:
            x2 = self.x2_quantizer(x2)
        return x2

    def forward(self, x1, x2):
        out = self.matmul_func(x1, x2)
        return out


class QuantMatMulZO(nn.Module):
    def __init__(
        self,
        x1_quant_params: dict = {},
        x2_quant_params: dict = {},
        disable_act_quant=False,
        matmul_func=torch.bmm,
    ):
        super().__init__()
        # de-activate the quantized forward default
        self.use_act_quant = False
        # initialize quantizer
        self.i_cluster_counts = None
        self.x1_quantizer = UniformAffineQuantizerZO(**x1_quant_params)
        self.x2_quantizer = UniformAffineQuantizerZO(**x2_quant_params)
        self.matmul_func = matmul_func

        self.disable_act_quant = disable_act_quant


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def quant_x1(self, x1):
        if self.use_act_quant:
            x1 = self.x1_quantizer(x1)
        return x1

    def quant_x2(self, x2):
        if self.use_act_quant:
            x2 = self.x2_quantizer(x2)
        return x2

    def forward(self, x1, x2):
        out = self.matmul_func(x1, x2)
        return out
