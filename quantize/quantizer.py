import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import tqdm
import numpy as np
import pdb
import math

CLIPMIN = 1e-5




def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

class roundSTE(torch.autograd.Function):
    """
    Straight-Through Estimator for rounding operation.
    Forward: round(x)
    Backward: gradient = 1 (identity)
    """
    @staticmethod
    def forward(ctx, x):
        # 前向传播：执行四舍五入
        return torch.round(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：梯度直接通过，乘以1
        return grad_output

'''
class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_cluster",
        group_size=None,
        shape=None,
        lwc=False,
        disable_zero_point=False,
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        self.disable_zero_point = disable_zero_point
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = lwc
        
        init_value = 4.             # inti value of learnable weight clipping
        if lwc:
            if group_size:
                dim1 = int(shape[0]*math.ceil(shape[1]/group_size))
                self.deficiency = shape[-1]%group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric   # support for mlc-llm symmetric quantization
            else:
                dim1 = shape[0]
            self.upbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
        self.sigmoid = nn.Sigmoid()

        self.enable = True
        self.group_size = group_size

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1

    def fake_quant(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
            x = torch.cat((x,pad_zeros),dim=1)
        
        if self.group_size:
            assert len(x.shape)==2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:,:-self.deficiency]
        return x_dequant
    

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits-1).round_().div_(2**self.n_bits-1)

        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            self.per_token_dynamic_calibration(x)
        else:
            raise NotImplementedError()   

        x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
        return x_dequant

    def per_token_dynamic_calibration(self, x):
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1,self.group_size)
            else:
                pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
                x = torch.cat((x,pad_zeros),dim=1)
                x = x.reshape(-1,self.group_size)
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax =  x.amax(reduce_shape, keepdim=True)
        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor)*xmax
            xmin = self.sigmoid(self.lowbound_factor)*xmin
        if self.symmetric:
            abs_max = torch.max(xmax.abs(),xmin.abs())
            scale = abs_max / (2**(self.n_bits-1)-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (2**(self.n_bits-1)-1)*torch.ones_like(self.scale)
        else:
            range = xmax - xmin
            scale = range / (2**self.n_bits-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = -(xmin) / (self.scale)
        if self.disable_zero_point:
            self.round_zero_point = None
        else:
            self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
        
    def register_scales_and_zeros(self):
        self.register_buffer('scales', self.scale)
        self.register_buffer('zeros', self.round_zero_point)
        del self.scale
        del self.round_zero_point
'''

class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_cluster",
        group_size=None,
        shape=None,
        lwc=False,
        disable_zero_point=True,
        delta=None,  # 零阶梯度估计的扰动幅度
        _lambda=None,
        sample_size=None
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        self.disable_zero_point = disable_zero_point
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None
        self.descale = None
        # self.dezero = None
        # self.group_zero = 1
        # self.group_scale = 1

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = lwc
        
        init_value = 4.             # inti value of learnable weight clipping
        if lwc:
            if group_size:
                dim1 = int(shape[0]*math.ceil(shape[1]/group_size))
                self.deficiency = shape[-1]%group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric   # support for mlc-llm symmetric quantization
            else:
                dim1 = shape[0]
            self.upbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
        self.sigmoid = nn.Sigmoid()

        self.enable = True
        self.group_size = group_size

        # ldx_add:
        self.delta = delta  # 零阶梯度估计的扰动幅度
        self._lambda = _lambda
        self.sample_size = sample_size

        if self.delta is not None and self._lambda is not None and self.sample_size is not None:
            self.round_module = RoundZOModule(delta, _lambda, sample_size)
            # self._round_func = RoundZO.apply(delta, _lambda, sample_size)
        else:
            self.round_module = RoundSTE()
            # self._round_func = round_ste
            self._round_func = roundSTE.apply
        
        # if hasattr(self, 'descale') :
        #     nn.init.zeros_(self.descale)  # 初始化为0，表示无偏移

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1

    def fake_quant(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
            x = torch.cat((x, pad_zeros),dim=1)
        if self.descale is not None:
            # 使用对数空间或更稳定的表示
            # log_scale = torch.log(scale.clamp(min=CLIPMIN))
            # log_descale = self.descale  # 学习log尺度偏移
            # eff_scale = torch.exp(log_scale + log_descale)
            eff_scale = (scale + self.descale)
            # 同时优化内外的scale
            # scale = 
        else:
            eff_scale = scale

        # with torch.no_grad():
        #     eff_scale_detach = eff_scale.clamp(min=CLIPMIN, max=1e4)
        # eff_scale = eff_scale_detach
        eff_scale = eff_scale.clamp(min=1e-2, max=1e4)

        # eff_scale = scale
        if self.group_size:
            assert len(x.shape)==2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        # if self.descale is not None and self.group_scale:
        #     assert len(x.shape)==2, "only support linear layer now"
        #     dim1, dim2 = x.shape
        #     x = x.reshape(-1, self.group_scale)
        # x_int = round_ste(x / scale)
        #x_int = round_ste(x / eff_scale)
        # ldx_add:
        # if self.delta is not None and self._lambda is not None and self.sample_size is not None:
        #     x_int = self._round(x / scale, self.delta, self._lambda, self.sample_size) # 先用scale去做。
        # else:
        #     x_int = self._round(x / scale)
        # x_int = self.round_module.forward(x / eff_scale)
        # x_int = self._round_func(x / eff_scale)
        # x_int = self._round_func(x * (1.0 / eff_scale.clamp(min=CLIPMIN)))
        # x_int = self._round_func(x * (1.0 / eff_scale.clamp(min=CLIPMIN)))
        # x_int = self._round_func(x * (1.0 / scale))
        x_int = self.round_module.forward(x * (1.0 / eff_scale))
        # x_int = self.round_module.forward(x * (1.0 / scale))
        # x_int = self._round_func(x / scale)
        # x_int = round_ste(x / eff_scale)

        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        #add
        if self.descale is not None:
            # x_dequant = x_dequant.view(x_dequant.shape[0],-1,self.group_scale)
            # x_dequant = x_dequant.mul(scale.unsqueeze(-1).repeat(1,1,self.group_scale)+self.descale)
            x_dequant = x_dequant.mul(eff_scale)
            # x_dequant = x_dequant.mul(scale)
            # x_dequant = x_dequant.view(x_dequant.shape[0],-1)
        else:
            x_dequant = x_dequant.mul(scale)
        #add
        # if self.dezero is not None:
        #     x_dequant = x_dequant.view(x_dequant.shape[0],-1,self.group_zero)
        #     x_dequant = x_dequant.add(self.dezero)
        #     x_dequant = x_dequant.view(x_dequant.shape[0],-1)
        # if self.group_size or (self.group_scale and self.descale is not None):
        #     x_dequant = x_dequant.reshape(dim1, dim2)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:,:-self.deficiency]
        return x_dequant
    

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits-1).round_().div_(2**self.n_bits-1)

        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            with torch.no_grad():
                x_detached = x.detach()
                self.per_token_dynamic_calibration(x_detached)
            # print(f"动态调整量化参数...")
            # x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
            # x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
        else:
            raise NotImplementedError()  
            # x_dequant = self.fake_quant(x, self.scales, self.zeros)
            # pass
            # # print(f"不动态调整量化参数...")
            # x_dequant = self.fake_quant(x, self.scales, self.zeros)
            # pass 
        # x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
        x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
        # x_dequant = self.fake_quant(x, self.scales, self.zeros)
        # x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
        # ldx:add:
        # x_dequant = self.fake_quant(x, self.scale, self.zeros)
        
        return x_dequant

    def per_token_dynamic_calibration(self, x):
        with torch.no_grad():
            if self.group_size:
                if self.deficiency == 0:
                    x = x.reshape(-1,self.group_size)
                else:
                    pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
                    x = torch.cat((x,pad_zeros),dim=1)
                    x = x.reshape(-1,self.group_size)
            # if self.descale is not None and self.group_scale:
            #     if self.deficiency == 0:
            #         x = x.reshape(-1,self.group_scale)
            #     else:
            #         pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
            #         x = torch.cat((x,pad_zeros),dim=1)
            #         x = x.reshape(-1,self.group_scale)
            reduce_shape = [-1]
            # 这里的x.amin传梯度吗？
            xmin = x.amin(reduce_shape, keepdim=True)
            xmax =  x.amax(reduce_shape, keepdim=True)
            if self.lwc:
                xmax = self.sigmoid(self.upbound_factor)*xmax
                xmin = self.sigmoid(self.lowbound_factor)*xmin
            if self.symmetric:
                abs_max = torch.max(xmax.abs(),xmin.abs())
                scale = abs_max / (2**(self.n_bits-1)-1)
                self.scale = scale.clamp(min=CLIPMIN, max=1e4)
                zero_point = (2**(self.n_bits-1)-1)*torch.ones_like(self.scale)
            else:
                range = xmax - xmin
                scale = range / (2**self.n_bits-1)
                self.scale = scale.clamp(min=CLIPMIN, max=1e4)
                zero_point = -(xmin) / (self.scale)
            if self.disable_zero_point:
                self.round_zero_point = None
            else:
                self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
        
    def register_scales_and_zeros(self):
        self.register_buffer('scales', self.scale)
        self.register_buffer('zeros', self.round_zero_point)
        #add
        # print(f"{self.scale}")
        # print(f"self.scale is nan: {torch.isnan(self.scale).any()}")
        descale = torch.zeros_like(self.scale)
        # print(f"self.descale{descale}")
        self.descale = nn.Parameter(descale)
        # self.descale = nn.Parameter(descale.unsqueeze(-1).repeat(1,1,self.group_scale))
        # dezero = torch.zeros_like(self.round_zero_point)
        # self.dezero = nn.Parameter(dezero.unsqueeze(-1).repeat(1,1,self.group_zero))
        del self.scale
        del self.round_zero_point

# class RoundZO(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, delta, _lambda, sample_size):
#         out = torch.round(x)
#         ctx.save_for_backward(x)
#         ctx.delta = delta
#         ctx._lambda = _lambda
#         ctx.sample_size = sample_size
#         return out
#     @staticmethod
#     def backward(ctx, grad_output):
#         (x,) = ctx.saved_tensors
#         delta = ctx.delta
#         original_shape = x.shape
#         x_flat = x.view(-1)
#         sample_size = ctx.sample_size
#         _lambda = ctx._lambda
#         grad_input = torch.zeros_like(x_flat)
#         for i in range(x_flat.numel()):
#             # u = x_flat[i].item()
#             u = x_flat[i]
#             k = torch.round(u - 0.5)
#             b = k + 0.5
#             v = u - b
#             z_samples = []
#             for _ in range(sample_size):
#                 z = (2 * _lambda) * torch.rand(1) - _lambda
#                 z = z.item()
#                 # ldx_add:必要条件？不满足条件怎么办？
#                 while abs(z) >= (1/(2*delta)):
#                     # z = torch.randn(1).item()
#                     z = (2 * _lambda) * torch.rand(1) - _lambda
#                     z = z.item()
#                 z_samples.append(z)
#             grad_est = 0
#             # ldx_add:梯度估计的正负,公式中乘以z了。
#             for z in z_samples:
#                 abs_z = abs(z)
#                 abs_v = abs(v)
#                 if abs_z < abs_v / delta:
#                     grad_contrib = 0
#                 elif abs_z < 1/(2*delta) and abs_z >= (abs_v/delta):
#                     grad_contrib = (abs_z/(2*delta)) 
#                 else:
#                     grad_contrib = 0
#                 grad_est += grad_contrib
#             grad_est /= sample_size
#             grad_input[i] = grad_est
#         grad_input = grad_input.view(original_shape) * grad_output
#         return grad_input, None, None, None
    

# 向量化加速版本
class RoundZO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, delta, _lambda, sample_size):
        out = torch.round(x)
        ctx.save_for_backward(x)
        ctx.delta = delta
        ctx._lambda = _lambda
        ctx.sample_size = sample_size
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # (x,) = ctx.saved_tensors
        # delta = ctx.delta
        # _lambda = ctx._lambda
        # sample_size = ctx.sample_size
        # # print(f"使用了我们的梯度反传")
        
        # original_shape = x.shape
        # x_flat = x.view(-1)
        # num_elements = x_flat.numel()
        # u = x_flat
        # k = torch.round(u - 0.5)
        # b = k + 0.5
        # v = u - b
        # z_samples_uniform = (2 * _lambda) * torch.rand(num_elements, sample_size, device=x.device) - _lambda
        # mask_invalid = torch.abs(z_samples_uniform) >= (1/(2*delta))
        # total_invalid = mask_invalid.sum().item()
        # while total_invalid > 0:
        #     new_samples = (2 * _lambda) * torch.rand(num_elements, sample_size, device=x.device) - _lambda
        #     new_mask_invalid = torch.abs(new_samples) >= (1/(2*delta))
        #     replace_mask = mask_invalid & (~new_mask_invalid)
        #     # z_samples_uniform = torch.where(replace_mask.unsqueeze(1).expand_as(z_samples_uniform) if z_samples_uniform.dim() > 1 else replace_mask, 
        #     #                               new_samples, z_samples_uniform)
        #     z_samples_uniform = torch.where(replace_mask, new_samples, z_samples_uniform)
        #     mask_invalid = torch.abs(z_samples_uniform) >= (1/(2*delta))
        #     total_invalid = mask_invalid.sum().item()
        # z_samples = z_samples_uniform
        # abs_z = torch.abs(z_samples) 
        # abs_v = torch.abs(v).unsqueeze(1) 
        # # cond1 = abs_z < (abs_v / delta)
        # cond2 = (abs_z < 1/(2*delta)) & (abs_z >= (abs_v / delta))
        # grad_contrib = torch.zeros_like(z_samples)
        # grad_contrib = torch.where(cond2, abs_z/(2*delta), grad_contrib)
        
        # grad_est = grad_contrib.mean(dim=1)  
        # grad_input = grad_est.view(original_shape) * grad_output
        
        # return grad_input, None, None, None
        """
        反向传播：使用显式表达式计算代理梯度
        """
        # import pdb; pdb.set_trace()
        # 简单的调试标记 - 在这里添加VS Code条件断点
        import os
        if os.getenv('DEBUG_ROUNDZO') == '1':
            _debug_flag = True  # 在这里添加断点，条件：os.getenv('DEBUG_ROUNDZO') == '1'
        
        (x,) = ctx.saved_tensors
        delta = ctx.delta
        _lambda = ctx._lambda
        # 按_lambda=3计算
        
        # 计算v(u) = u - b(u)，其中b(u)是最近的半整数点
        # b(u) = round(u - 0.5) + 0.5
        b = torch.round(x - 0.5) + 0.5
        v = x - b  # v ∈ [-0.5, 0.5]
        
        # 计算|v|
        abs_v = torch.abs(v)
        
        # 计算显式代理梯度表达式
        # 根据推导，当2|z|δ < 1时有效，即|z| < 1/(2δ)
        # 代理梯度 = (1/(2λ)) * ∫_{|v|/δ}^{1/(2δ)} (z/(2δ)) dz * 2
        #         = (1/(4λδ)) * [(1/(2δ))² - (|v|/δ)²]
        #         = 1/(16λδ³) - |v|²/(4λδ³)
        
        # 首先计算M = min(λ, 1/(2δ))
        M = _lambda
        
        # 计算条件：|v| < δ * M = min(δλ, 1/2)
        # 这个条件确保积分下限小于上限
        condition = v < (delta * math.sqrt(M))
        
        # 初始化梯度为0
        grad_input = torch.zeros_like(x)
        
        # 当条件满足时，计算显式梯度
        if condition.any():
            # 计算显式表达式
            v_squared_over_delta_squared = (abs_v * abs_v) / (delta * delta)
            
            # 代理梯度表达式
            explicit_grad = (1/(4 * math.sqrt(_lambda) * delta)) * (M - v_squared_over_delta_squared)
            
            # 应用条件
            grad_input = torch.where(condition, explicit_grad, grad_input)
        
        # 返回梯度，乘以上游梯度
        return grad_input * grad_output, None, None, None



class RoundSTE(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return round_ste(x)
    
    def extra_repr(self):
        return "round_ste"

class RoundZOModule(nn.Module):
    def __init__(self, delta, _lambda, sample_size):
        super().__init__()
        self.delta = delta
        self._lambda = _lambda
        self.sample_size = sample_size
        
    def forward(self, x):
        return RoundZO.apply(x, self.delta, self._lambda, self.sample_size)
    
    def extra_repr(self):
        return f"RoundZO(delta={self.delta}, lambda={self._lambda}, sample_size={self.sample_size})"
    


# class UniformAffineQuantizerZO(nn.Module):
#     def __init__(
#         self,
#         n_bits: int = 8,
#         symmetric: bool = False,
#         per_channel_axes=[],
#         metric="minmax",
#         dynamic=False,
#         dynamic_method="per_cluster",
#         group_size=None,
#         shape=None,
#         lwc=False,
#         disable_zero_point=False,
#         delta=0.1,  # 零阶梯度估计的扰动幅度
#         _lambda=3,
#         sample_size=5
#     ):
#         super().__init__()
#         self.symmetric = symmetric
#         self.disable_zero_point = disable_zero_point
#         assert 2 <= n_bits <= 16, "bitwidth not supported"
#         self.n_bits = n_bits
#         if self.disable_zero_point:
#             self.qmin = -(2 ** (n_bits - 1))
#             self.qmax = 2 ** (n_bits - 1) - 1
#         else:
#             self.qmin = 0
#             self.qmax = 2 ** (n_bits) - 1
#         self.per_channel_axes = per_channel_axes
#         self.metric = metric
#         self.cluster_counts = None
#         self.cluster_dim = None
#         self.scale = None
#         self.zero_point = None
#         self.round_zero_point = None
#         self.descale = None
#         self.dezero = None
#         self.group_zero = 2
#         self.group_scale = 2
#         self.cached_xmin = None
#         self.cached_xmax = None
#         self.dynamic = dynamic
#         self.dynamic_method = dynamic_method
#         self.deficiency = 0
#         self.lwc = lwc
#         init_value = 4.             # inti value of learnable weight clipping
#         if lwc:
#             if group_size:
#                 dim1 = int(shape[0]*math.ceil(shape[1]/group_size))
#                 self.deficiency = shape[-1]%group_size
#                 if self.deficiency > 0:
#                     self.deficiency = group_size - self.deficiency
#                     assert self.symmetric   # support for mlc-llm symmetric quantization
#             else:
#                 dim1 = shape[0]
#             self.upbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
#             self.lowbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
#         self.sigmoid = nn.Sigmoid()
#         self.enable = True
#         self.group_size = group_size

#         self.delta = delta  # 零阶梯度估计的扰动幅度
#         self._lambda = _lambda
#         self.sample_size = sample_size
#         # 使用我们的零阶梯度估计round函数
#         self.round_zo = RoundZO.apply

#     def change_n_bits(self, n_bits):
#         self.n_bits = n_bits
#         if self.disable_zero_point:
#             self.qmin = -(2 ** (n_bits - 1))
#             self.qmax = 2 ** (n_bits - 1) - 1
#         else:
#             self.qmin = 0
#             self.qmax = 2 ** (n_bits) - 1
    
#     def fake_quant(self, x, scale, round_zero_point):
#         """
#         伪量化函数，使用零阶梯度估计的round
#         """
#         if self.deficiency > 0:
#             pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
#             x = torch.cat((x,pad_zeros),dim=1)
#         if self.descale is not None:
#             eff_scale = (scale + self.descale)
#         else:
#             eff_scale = scale
            
#         if self.group_size:
#             assert len(x.shape)==2, "only support linear layer now"
#             dim1, dim2 = x.shape
#             x = x.reshape(-1, self.group_size)
        
#         # 使用零阶梯度估计的round函数替代round_ste
#         # x_int = self.round_zo(x / eff_scale, self.delta)
#         x_int = self.round_zo(x / scale, self.delta, self._lambda, self.sample_size) # 先用scale去做。
        
#         if round_zero_point is not None:
#             x_int = x_int.add(round_zero_point)
#         x_int = x_int.clamp(self.qmin, self.qmax)
#         x_dequant = x_int
#         if round_zero_point is not None:
#             x_dequant = x_dequant.sub(round_zero_point)
        
#         # 添加反量化调整
#         if self.descale is not None:
#             x_dequant = x_dequant.view(x_dequant.shape[0],-1,self.group_scale)
#             x_dequant = x_dequant.mul(scale.unsqueeze(-1).repeat(1,1,self.group_scale)+self.descale)
#             x_dequant = x_dequant.view(x_dequant.shape[0],-1)
#         else:
#             x_dequant = x_dequant.mul(scale)
        
#         if self.dezero is not None:
#             x_dequant = x_dequant.view(x_dequant.shape[0],-1,self.group_zero)
#             x_dequant = x_dequant.add(self.dezero)
#             x_dequant = x_dequant.view(x_dequant.shape[0],-1)
        
#         if self.group_size:
#             x_dequant = x_dequant.reshape(dim1, dim2)
        
#         if self.deficiency > 0:
#             x_dequant = x_dequant[:,:-self.deficiency]
        
#         return x_dequant
    
#     def forward(self, x: torch.Tensor):
#         if self.n_bits >= 16 or not self.enable:
#             return x
#         if self.metric == "fix0to1":
#             return x.mul_(2**self.n_bits-1).round_().div_(2**self.n_bits-1)

#         if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
#             self.per_token_dynamic_calibration(x)
#         else:
#             raise NotImplementedError()   
        
#         # x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
#         # ldx:add:
#         x_dequant = self.fake_quant(x, self.scales, self.zeros)
#         return x_dequant

#     def per_token_dynamic_calibration(self, x):
#         if self.group_size:
#             if self.deficiency == 0:
#                 x = x.reshape(-1,self.group_size)
#             else:
#                 pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
#                 x = torch.cat((x,pad_zeros),dim=1)
#                 x = x.reshape(-1,self.group_size)
#         reduce_shape = [-1]
#         xmin = x.amin(reduce_shape, keepdim=True)
#         xmax =  x.amax(reduce_shape, keepdim=True)
#         if self.lwc:
#             xmax = self.sigmoid(self.upbound_factor)*xmax
#             xmin = self.sigmoid(self.lowbound_factor)*xmin
#         if self.symmetric:
#             abs_max = torch.max(xmax.abs(),xmin.abs())
#             scale = abs_max / (2**(self.n_bits-1)-1)
#             self.scale = scale.clamp(min=CLIPMIN, max=1e4)  
#             zero_point = (2**(self.n_bits-1)-1)*torch.ones_like(self.scale)
#         else:
#             range = xmax - xmin
#             scale = range / (2**self.n_bits-1)
#             self.scale = scale.clamp(min=CLIPMIN, max=1e4)
#             zero_point = -(xmin) / (self.scale)
#         if self.disable_zero_point:
#             self.round_zero_point = None
#         else:
#             self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
    
#     def register_scales_and_zeros(self):
#         self.register_buffer('scales', self.scale)
#         self.register_buffer('zeros', self.round_zero_point)
#         descale = torch.zeros_like(self.scale)
#         self.descale = nn.Parameter(descale.unsqueeze(-1).repeat(1,1,self.group_scale))
#         dezero = torch.zeros_like(self.round_zero_point)
#         self.dezero = nn.Parameter(dezero.unsqueeze(-1).repeat(1,1,self.group_zero))
#         del self.scale
#         del self.round_zero_point




# class UniformAffineQuantizer(nn.Module):
#     def __init__(
#         self,
#         n_bits: int = 8,
#         symmetric: bool = False,
#         per_channel_axes=[],
#         metric="minmax",
#         dynamic=False,
#         dynamic_method="per_cluster",
#         group_size=None,
#         shape=None,
#         lwc=False,
#         disable_zero_point=True,
#         delta=None,  # 零阶梯度估计的扰动幅度
#         _lambda=None,
#         sample_size=None
#     ):
#         """
#         support cluster quantize
#         dynamic_method support per_token and per_cluster
#         """
#         super().__init__()
#         self.symmetric = symmetric
#         self.disable_zero_point = disable_zero_point
#         assert 2 <= n_bits <= 16, "bitwidth not supported"
#         self.n_bits = n_bits
#         if self.disable_zero_point:
#             self.qmin = -(2 ** (n_bits - 1))
#             self.qmax = 2 ** (n_bits - 1) - 1
#         else:
#             self.qmin = 0
#             self.qmax = 2 ** (n_bits) - 1
#         self.per_channel_axes = per_channel_axes
#         self.metric = metric
#         self.cluster_counts = None
#         self.cluster_dim = None

#         self.scale = None
#         self.zero_point = None
#         self.round_zero_point = None
#         self.descale = None
#         # self.dezero = None
#         # self.group_zero = 1
#         # self.group_scale = 1

#         self.cached_xmin = None
#         self.cached_xmax = None
#         self.dynamic = dynamic
#         self.dynamic_method = dynamic_method
#         self.deficiency = 0
#         self.lwc = lwc
        
#         init_value = 4.             # inti value of learnable weight clipping
#         if lwc:
#             if group_size:
#                 dim1 = int(shape[0]*math.ceil(shape[1]/group_size))
#                 self.deficiency = shape[-1]%group_size
#                 if self.deficiency > 0:
#                     self.deficiency = group_size - self.deficiency
#                     assert self.symmetric   # support for mlc-llm symmetric quantization
#             else:
#                 dim1 = shape[0]
#             self.upbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
#             self.lowbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
#         self.sigmoid = nn.Sigmoid()

#         self.enable = True
#         self.group_size = group_size

#         # ldx_add:
#         self.delta = delta  # 零阶梯度估计的扰动幅度
#         self._lambda = _lambda
#         self.sample_size = sample_size

#         if self.delta is not None and self._lambda is not None and self.sample_size is not None:
#             self.round_module = RoundZOModule(delta, _lambda, sample_size)
#             self._round_func = RoundZO.apply(delta, _lambda, sample_size)
#         else:
#             self.round_module = RoundSTE()
#             self._round_func = round_ste


#     def change_n_bits(self, n_bits):
#         self.n_bits = n_bits
#         if self.disable_zero_point:
#             self.qmin = -(2 ** (n_bits - 1))
#             self.qmax = 2 ** (n_bits - 1) - 1
#         else:
#             self.qmin = 0
#             self.qmax = 2 ** (n_bits) - 1

#     def fake_quant(self, x, scale, round_zero_point):
#         # 保存原始形状
#         original_shape = x.shape
        
#         # 1. 先处理padding
#         if self.deficiency > 0:
#             pad_zeros = torch.zeros((x.shape[0], self.deficiency), 
#                                 dtype=x.dtype, device=x.device)
#             x = torch.cat((x, pad_zeros), dim=1)
        
#         # 2. 处理分组
#         if self.group_size:
#             assert len(x.shape) == 2, "only support linear layer now"
#             dim1, dim2 = x.shape
#             x_grouped = x.reshape(-1, self.group_size)
#         else:
#             x_grouped = x
        
#         # 3. 确保scale数值稳定
#         if scale is None:
#             scale = self.scales if hasattr(self, 'scales') else torch.ones_like(x_grouped)
        
#         # 4. 处理descale（如果存在）
#         if self.descale is not None:
#             # 使用更稳定的log空间表示
#             scale_safe = scale.clamp(min=CLIPMIN, max=1e4)
#             log_scale = torch.log(scale_safe)
            
#             # 限制descale的范围避免爆炸
#             log_descale = torch.clamp(self.descale, -10.0, 10.0)
#             eff_scale = torch.exp(log_scale + log_descale)
#         else:
#             eff_scale = scale.clamp(min=CLIPMIN, max=1e4)
        
#         # 5. 安全的逆运算
#         inv_eff_scale = 1.0 / eff_scale
#         # 防止inf
#         inv_eff_scale = torch.where(
#             torch.isfinite(inv_eff_scale),
#             inv_eff_scale,
#             torch.zeros_like(inv_eff_scale)
#         )
        
#         # 6. 量化（使用稳定的round）
#         x_int = self.round_module.forward(x_grouped * inv_eff_scale)
        
#         # 7. 处理zero_point
#         if round_zero_point is not None:
#             round_zero_point_safe = torch.clamp(round_zero_point, self.qmin, self.qmax)
#             x_int = x_int + round_zero_point_safe
        
#         # 8. 截断到量化范围
#         x_int = torch.clamp(x_int, self.qmin, self.qmax)
        
#         # 9. 反量化
#         x_dequant = x_int
#         if round_zero_point is not None:
#             x_dequant = x_dequant - round_zero_point_safe
        
#         x_dequant = x_dequant * eff_scale
        
#         # 10. 恢复形状
#         if self.group_size:
#             x_dequant = x_dequant.reshape(dim1, dim2)
        
#         # 11. 移除padding
#         if self.deficiency > 0:
#             x_dequant = x_dequant[:, :-self.deficiency]
        
#         # 12. 形状检查
#         assert x_dequant.shape == original_shape, \
#             f"Shape mismatch: {x_dequant.shape} vs {original_shape}"
        
#         return x_dequant
    

#     def forward(self, x: torch.Tensor):
#         if self.n_bits >= 16 or not self.enable:
#             return x
        
#         if self.metric == "fix0to1":
#             return x.mul_(2**self.n_bits-1).round_().div_(2**self.n_bits-1)
        
#         # 动态计算量化参数（完全不参与梯度）
#         with torch.no_grad():
#             if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
#                 x_detached = x.detach()
#                 self.per_token_dynamic_calibration(x_detached)
#             else:
#                 raise NotImplementedError()
        
#         # 分离动态计算的参数，只传递值，不传递梯度
#         if hasattr(self, 'scale') and self.scale is not None:
#             scale_to_use = self.scale.detach().clone()
#         elif hasattr(self, 'scales') and self.scales is not None:
#             scale_to_use = self.scales.detach().clone()
#         else:
#             scale_to_use = None
        
#         if hasattr(self, 'round_zero_point') and self.round_zero_point is not None:
#             zp_to_use = self.round_zero_point.detach().clone()
#         elif hasattr(self, 'zeros') and self.zeros is not None:
#             zp_to_use = self.zeros.detach().clone()
#         else:
#             zp_to_use = None
        
#         # 使用分离的参数进行fake_quant，但保留对可学习参数（如descale）的梯度
#         x_dequant = self.fake_quant(x, scale_to_use, zp_to_use)
        
#         return x_dequant

#     def per_token_dynamic_calibration(self, x):
#         with torch.no_grad():
#             if self.group_size:
#                 if self.deficiency == 0:
#                     x = x.reshape(-1,self.group_size)
#                 else:
#                     pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
#                     x = torch.cat((x,pad_zeros),dim=1)
#                     x = x.reshape(-1,self.group_size)
#             # if self.descale is not None and self.group_scale:
#             #     if self.deficiency == 0:
#             #         x = x.reshape(-1,self.group_scale)
#             #     else:
#             #         pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
#             #         x = torch.cat((x,pad_zeros),dim=1)
#             #         x = x.reshape(-1,self.group_scale)
#             reduce_shape = [-1]
#             xmin = x.amin(reduce_shape, keepdim=True)
#             xmax =  x.amax(reduce_shape, keepdim=True)
#             if self.lwc:
#                 xmax = self.sigmoid(self.upbound_factor)*xmax
#                 xmin = self.sigmoid(self.lowbound_factor)*xmin
#             if self.symmetric:
#                 abs_max = torch.max(xmax.abs(),xmin.abs())
#                 scale = abs_max / (2**(self.n_bits-1)-1)
#                 self.scale = scale.clamp(min=CLIPMIN, max=1e4)
#                 zero_point = (2**(self.n_bits-1)-1)*torch.ones_like(self.scale)
#             else:
#                 range = xmax - xmin
#                 scale = range / (2**self.n_bits-1)
#                 self.scale = scale.clamp(min=CLIPMIN, max=1e4)
#                 zero_point = -(xmin) / (self.scale)
#             if self.disable_zero_point:
#                 self.round_zero_point = None
#             else:
#                 self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
        
#     def register_scales_and_zeros(self):
#         self.register_buffer('scales', self.scale)
#         self.register_buffer('zeros', self.round_zero_point)
#         #add
#         # print(f"{self.scale}")
#         # print(f"self.scale is nan: {torch.isnan(self.scale).any()}")
#         descale = torch.zeros_like(self.scale)
#         # print(f"self.descale{descale}")
#         self.descale = nn.Parameter(descale)
#         # self.descale = nn.Parameter(descale.unsqueeze(-1).repeat(1,1,self.group_scale))
#         # dezero = torch.zeros_like(self.round_zero_point)
#         # self.dezero = nn.Parameter(dezero.unsqueeze(-1).repeat(1,1,self.group_zero))
#         del self.scale
#         del self.round_zero_point