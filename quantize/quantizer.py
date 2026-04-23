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
        t=None,
        method=None,
        use_sum=None
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
        self.method = method
        self.t = t
        self.use_sum = use_sum

        if self.delta is not None and self.method == "Uniform":
            self.round_module = UniformModule(delta, use_sum)
            # self._round_func = Uniform.apply(delta)
        elif self.delta is not None and self.method == "Normal":
            self.round_module = NormalModule(delta, use_sum)
            # self._round_func = Normal.apply(delta)
        elif self.delta is not None and self.method == "Laplace":
            self.round_module = LaplaceModule(delta, use_sum)
            # self._round_func = Laplace.apply(delta)
        elif self.t is not None and self.method == "HTGE":
            self.round_module = HTGEModule(t)
            # self._round_func = HTGE.apply(t)
        else:
            self.round_module = RoundSTE()
            # self._round_func = round_ste
            # self._round_func = roundSTE.apply


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
            # 量化统计量应基于当前张量值计算，但 LWC 的可学习裁剪参数
            # 仍然需要从量化损失接收梯度，因此这里只切断到输入 x 的梯度，
            # 不再把整个标定过程包进 no_grad。
            self.per_token_dynamic_calibration(x.detach())
        else:
            raise NotImplementedError()  
            # x_dequant = self.fake_quant(x, self.scales, self.zeros)
            # pass
            # # print(f"不动态调整量化参数...")
            # x_dequant = self.fake_quant(x, self.scales, self.zeros)
            # pass 
        # x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
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
        # if self.descale is not None and self.group_scale:
        #     if self.deficiency == 0:
        #         x = x.reshape(-1,self.group_scale)
        #     else:
        #         pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
        #         x = torch.cat((x,pad_zeros),dim=1)
        #         x = x.reshape(-1,self.group_scale)
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax =  x.amax(reduce_shape, keepdim=True)
        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor) * xmax
            xmin = self.sigmoid(self.lowbound_factor) * xmin
        if self.symmetric:
            abs_max = torch.max(xmax.abs(),xmin.abs())
            scale = abs_max / (2**(self.n_bits-1)-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (2**(self.n_bits-1)-1) * torch.ones_like(self.scale)
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

# 向量化加速版本
class Uniform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, delta, use_sum):
        out = torch.round(x)
        ctx.save_for_backward(x)
        ctx.delta = delta
        ctx.use_sum = use_sum
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
        (x, ) = ctx.saved_tensors
        delta = ctx.delta
        use_sum = ctx.use_sum
        # # 保存原始形状以便后续恢复
        # original_shape = x.shape
        # x_flat = x.view(-1)  # 展平为一维向量以便处理
        if use_sum == False:
            _lambda = 3.0
            # 按_lambda=3计算
            # print(f"Uniform")
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
        else:
            # 保存原始形状以便后续恢复
            original_shape = x.shape
            x_flat = x.view(-1)  # 展平为一维向量以便处理
            # 超参数设置：根据均匀分布[-√3, √3]，λ=√3
            # print(f"use_sum")
            lambda_val = math.sqrt(3)  # 对应公式中的C = √3
            
            # 计算阈值：|u - b_k| < √3δ，即搜索半径
            search_radius = lambda_val * delta
            
            # 核心思想：为每个u找到所有可能的跳变点b_k = k + 0.5
            # 由于b_k是周期性的（间隔为1），我们可以通过以下方式高效搜索：
            # 1. 找到u附近的第一个跳变点
            # 2. 基于搜索半径向两边扩展
            
            # 方法1: 直接计算所有可能满足条件的k值范围
            # 对于每个u，k的范围是：floor(u - search_radius - 0.5) 到 ceil(u + search_radius - 0.5)
            # 但这种方法需要为每个u创建不同长度的列表，难以向量化
            
            # 方法2: 使用固定数量的候选k值（基于最大可能范围）
            # 由于search_radius通常较小（δ较小），我们可以设置一个合理的最大搜索数量
            
            # 计算最大可能包含的跳变点数量
            # 跳变点间隔为1，所以搜索范围内最多有 ceil(2 * search_radius) + 1 个点
            # max_k_count = int(math.ceil(2 * search_radius)) + 1
            max_k_count = 2 * (int(math.ceil(search_radius))) + 1
            
            # 为每个u准备候选k值矩阵，形状为 (num_elements, max_k_count)
            # 首先找到每个u最近的跳变点对应的k0
            # b_k = k + 0.5，最近的b_k是 round(u - 0.5)
            k0 = torch.round(x_flat - 0.5)  # 每个u对应的中心k值
            
            # 创建偏移量，从中心向两边扩展
            # 例如，如果max_k_count=5，则offsets = [-2, -1, 0, 1, 2]
            half_range = max_k_count // 2
            offsets = torch.arange(-half_range, half_range + 1, 
                                  device=x.device, dtype=torch.float32)
            
            # 为每个u生成候选k值：k0扩展为二维矩阵
            # 形状: (num_elements, max_k_count)
            k_candidates = k0.unsqueeze(1) + offsets
            
            # 计算对应的跳变点b_k = k + 0.5
            b_candidates = k_candidates + 0.5
            
            # 计算每个u与每个候选b_k的距离
            # 扩展x_flat以便广播计算
            x_expanded = x_flat.unsqueeze(1)  # 形状: (num_elements, 1)
            distances = torch.abs(x_expanded - b_candidates)  # 形状: (num_elements, max_k_count)
            
            # 创建掩码：哪些候选跳变点满足 |u - b_k| < √3δ
            mask = distances < search_radius  # 形状: (num_elements, max_k_count)
            
            # 初始化梯度贡献张量
            grad_contrib = torch.zeros_like(distances)
            
            # 对于满足条件的跳变点，计算其贡献
            # 公式: 对于每个满足条件的k，贡献为 3 - (u-b_k)²/δ²
            # 注意：这里使用了条件 "∃k, s.t. |u-b_k| < √3δ"，否则为0
            with torch.no_grad():
                # 计算平方距离
                squared_distances = (x_expanded - b_candidates) ** 2
                # 计算每个候选点的贡献
                candidate_contrib = 3 - squared_distances / (delta ** 2)
                # 只保留满足条件的贡献
                grad_contrib = torch.where(mask, candidate_contrib, torch.zeros_like(candidate_contrib))
            
            # 对每个u的所有跳变点贡献求和
            # 形状: (num_elements,)
            sum_contrib = grad_contrib.sum(dim=1)
            
            # 应用公式中的系数: 1/(4√3δ)
            coeff = 1.0 / (4.0 * lambda_val * delta)
            grad_est = coeff * sum_contrib
            
            # 恢复原始形状
            grad_input = grad_est.view(original_shape)
        
        # 返回梯度，乘以上游梯度
        return grad_input * grad_output, None, None



class RoundSTE(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return round_ste(x)
    
    def extra_repr(self):
        return "round_ste"

class UniformModule(nn.Module):
    def __init__(self, delta, use_sum):
        super().__init__()
        self.delta = delta
        self.use_sum = use_sum
        
    def forward(self, x):
        return Uniform.apply(x, self.delta, self.use_sum)
    
    def extra_repr(self):
        return f"Uniform(delta={self.delta} use_sum={self.use_sum})"
    

class NormalModule(nn.Module):
    def __init__(self, delta, use_sum):
        super().__init__()
        self.delta = delta
        self.use_sum = use_sum
        
    def forward(self, x):
        return Normal.apply(x, self.delta, self.use_sum)
    
    def extra_repr(self):
        return f"Normal(delta={self.delta} use_sum={self.use_sum})"
    


class LaplaceModule(nn.Module):
    def __init__(self, delta, use_sum):
        super().__init__()
        self.delta = delta
        self.use_sum = use_sum
        
    def forward(self, x):
        return Laplace.apply(x, self.delta, self.use_sum)
    
    def extra_repr(self):
        return f"Laplace(delta={self.delta} use_sum={self.use_sum})"
    

class HTGEModule(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t = t
        
    def forward(self, x):
        return HTGE.apply(x, self.t)
    
    def extra_repr(self):
        return f"HTGE(delta={self.t})"
    

# 向量化加速版本
class Normal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, delta, use_sum):
        out = torch.round(x)
        ctx.save_for_backward(x)
        ctx.delta = delta
        ctx.use_sum = use_sum
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：使用正态分布近似计算代理梯度
        
        公式对应：
        给定公式：I = 1 / (2Φ(1/(2δ)) - 1) * 1/(δ√(2π)) * [exp(-(u - s(u))²/(2δ²)) - exp(-1/(8δ²))]
        
        其中：
        - Φ是标准正态分布的累积分布函数(CDF)
        - δ是正态分布的标准差
        - s(u) = round(u - 0.5) + 0.5，即最近的半整数点
        - u是输入值x
        
        这个公式是在条件2|z|δ ≤ 1下推导得到的，其中C = 1/(2δ)
        
        推导过程：
        1. 令C = 1/(2δ)
        2. 计算Φ(C)，即标准正态分布在C处的累积概率
        3. 计算归一化因子：1/(2Φ(C) - 1)
        4. 计算高斯核部分：1/(δ√(2π)) * exp(-(x - s(x))²/(2δ²))
        5. 减去截断项：- 1/(δ√(2π)) * exp(-C²/2)
           = - 1/(δ√(2π)) * exp(-1/(8δ²))
        
        最终梯度 = 归一化因子 * (高斯核 - 截断项)
        
        Args:
            grad_output: 上游梯度
        Returns:
            grad_input: 输入x的梯度
            None: delta的梯度（不计算）
        """
        # 从上下文中获取保存的张量和参数
        (x, ) = ctx.saved_tensors
        delta = ctx.delta
        use_sum = ctx.use_sum
        if use_sum == False:
            # print(f"normal")
            
            # 步骤1: 计算s(u) = round(u - 0.5) + 0.5，即最近的半整数点
            # 这是公式中的s(u)函数，用于找到最近的半整数点
            s_u = torch.round(x - 0.5) + 0.5
            
            # 步骤2: 计算C = 1/(2δ)
            # 这是公式中的截断点
            # C = 1.0 / (2.0 * delta)
            # 这个位置可能得改。
            C = torch.tensor(1.0 / (2.0 * delta), device=x.device, dtype=x.dtype)
            
            # 步骤3: 计算Φ(C)，即标准正态分布在C处的累积分布函数值
            # 使用误差函数erf计算标准正态分布CDF: Φ(x) = 0.5 * [1 + erf(x/√2)]
            # Phi_C = 0.5 * (1.0 + torch.erf(C / math.sqrt(2.0)))
            # Phi_C = 0.5 * (1.0 + torch.erf(C / torch.sqrt(torch.tensor(2.0, device=C.device, dtype=C.dtype))))
            Phi_C = 0.5 * (1.0 + torch.erf(C / torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype))))

            # 步骤4: 计算归一化因子: 1/(2Φ(C) - 1)
            # 这是公式中的归一化项，确保概率密度函数在截断后仍然归一化
            normalization_factor = 1.0 / (2.0 * Phi_C - 1.0)
            
            # 步骤5: 计算高斯核部分: exp(-(x - s(x))²/(2δ²))
            # 这是公式中的exp(-(u - s(u))²/(2δ²))项
            gaussian_kernel = torch.exp(-(x - s_u) ** 2 / (2.0 * delta ** 2))
            
            # 步骤6: 计算截断项: exp(-C²/2) = exp(-1/(8δ²))
            # 这是公式中的exp(-C²/2)项，其中C²/2 = 1/(8δ²)
            truncation_term = torch.exp(-C ** 2 / 2.0)  # 等价于exp(-1/(8δ²))
            
            # 步骤7: 计算完整梯度表达式
            # 根据公式: I = normalization_factor * 1/(δ√(2π)) * (gaussian_kernel - truncation_term)
            # 其中1/(δ√(2π))是正态分布的归一化常数
            normalizing_constant = 1.0 / (delta * math.sqrt(2.0 * math.pi))
            
            # 计算最终梯度
            grad_input = normalization_factor * normalizing_constant * (gaussian_kernel - truncation_term)
            
            # 将上游梯度乘以本地梯度
            # 这是链式法则的应用
            grad_input = grad_input * grad_output
        else:
            # print("usesum")
                        # 求和版本（新逻辑）
            # 保存原始形状以便后续恢复
            original_shape = x.shape
            x_flat = x.view(-1)  # 展平为一维向量以便处理
            
            # 固定参数：C = 3
            C = 3.0
            
            # 计算搜索半径：|u - b_k| < δC
            search_radius = C * delta
            
            # 计算最大可能包含的跳变点数量
            # 跳变点间隔为1，所以搜索范围内最多有 ceil(2 * search_radius) + 1 个点
            # max_k_count = int(math.ceil(2 * search_radius)) + 1
            max_k_count = 2 * (int(math.ceil(2 * search_radius))) + 1
            # 确保至少为1
            max_k_count = max(max_k_count, 1)
            
            # 为每个u准备候选k值矩阵，形状为 (num_elements, max_k_count)
            # 首先找到每个u最近的跳变点对应的k0
            # b_k = k + 0.5，最近的b_k是 round(u - 0.5)
            k0 = torch.round(x_flat - 0.5)  # 每个u对应的中心k值
            
            # 创建偏移量，从中心向两边扩展
            half_range = max_k_count // 2
            offsets = torch.arange(-half_range, half_range + 1, 
                                  device=x.device, dtype=torch.float32)
            
            # 为每个u生成候选k值：k0扩展为二维矩阵
            # 形状: (num_elements, max_k_count)
            k_candidates = k0.unsqueeze(1) + offsets
            
            # 计算对应的跳变点b_k = k + 0.5
            b_candidates = k_candidates + 0.5
            
            # 计算每个u与每个候选b_k的距离
            x_expanded = x_flat.unsqueeze(1)  # 形状: (num_elements, 1)
            distances = torch.abs(x_expanded - b_candidates)  # 形状: (num_elements, max_k_count)
            
            # 创建掩码：哪些候选跳变点满足 |u - b_k| < δC
            mask = distances < search_radius  # 形状: (num_elements, max_k_count)
            
            # 初始化梯度贡献张量
            grad_contrib = torch.zeros_like(distances)
            
            # 计算归一化因子相关项
            # 需要计算Φ(C)，C=3
            # Phi_C_value = 0.5 * (1.0 + torch.erf(torch.tensor(C / math.sqrt(2.0), device=x.device, dtype=x.dtype)))
            Phi_C_value = 0.5 * (1.0 + torch.erf(torch.tensor(C / math.sqrt(2.0), device=x.device, dtype=x.dtype)))
            normalization_factor = 1.0 / (2.0 * Phi_C_value - 1.0)
            
            # 计算正常数项：1/(δ√(2π))
            # normalizing_constant = 1.0 / (delta * math.sqrt(2.0 * math.pi))
            normalizing_constant = 1.0 / (torch.tensor(delta * math.sqrt(2.0 * math.pi), device=x.device, dtype=x.dtype))
            
            # 计算截断项：exp(-C²/2)
            # truncation_term = torch.exp(-C ** 2 / 2.0)
            truncation_term = torch.exp(-torch.tensor(C ** 2 / 2.0, device=x.device, dtype=x.dtype))
            
            # 对于满足条件的跳变点，计算其贡献
            # 公式: 对于每个满足条件的k，贡献为 exp(-(u-b_k)²/(2δ²)) - exp(-C²/2)
            with torch.no_grad():
                # 计算平方距离
                squared_distances = (x_expanded - b_candidates) ** 2
                # 计算高斯核部分
                gaussian_kernels = torch.exp(-squared_distances / (2.0 * delta ** 2))
                # 计算每个候选点的贡献
                candidate_contrib = gaussian_kernels - truncation_term
                # 只保留满足条件的贡献
                grad_contrib = torch.where(mask, candidate_contrib, torch.zeros_like(candidate_contrib))
            
            # 对每个u的所有跳变点贡献求和
            sum_contrib = grad_contrib.sum(dim=1)  # 形状: (num_elements,)
            
            # 应用公式中的系数: normalization_factor * normalizing_constant
            coeff = normalization_factor * normalizing_constant
            grad_est = coeff * sum_contrib
            
            # 恢复原始形状
            grad_input = grad_est.view(original_shape)
            
            # 应用上游梯度
            grad_input = grad_input * grad_output
        
        # 返回梯度
        # 第一个返回值是x的梯度，第二个是delta的梯度（这里不计算，返回None）
        return grad_input, None, None
    


# 向量化加速版本
class Laplace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, delta, use_sum):
        out = torch.round(x)
        ctx.save_for_backward(x)
        ctx.delta = delta
        ctx.use_sum = use_sum
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：使用拉普拉斯分布近似计算代理梯度
        
        公式对应：
        给定公式：I = 1/[2δ(1 - e^{-1/(√2δ)})] * [(a + 1/√2)e^{-√2a} - (1/(2δ) + 1/√2)e^{-1/(√2δ)}]
        
        其中：
        - δ是拉普拉斯分布的尺度参数
        - a = |u - s(u)|/δ，是归一化的距离
        - s(u) = round(u - 0.5) + 0.5，即最近的半整数点
        - u是输入值x
        
        这个公式是在条件2|z|δ/2 ≤ 1下推导得到的，其中C = 1/(2δ)
        
        推导过程：
        1. 计算s(u) = round(u - 0.5) + 0.5
        2. 计算归一化距离a = |u - s(u)|/δ
        3. 计算分母：2δ(1 - e^{-1/(√2δ)})
        4. 计算分子第一项：(a + 1/√2)e^{-√2a}
        5. 计算分子第二项：(1/(2δ) + 1/√2)e^{-1/(√2δ)}
        6. 计算完整梯度：I = 分母倒数 × (分子第一项 - 分子第二项)
        
        注意：拉普拉斯分布的概率密度函数为 f(x) = (1/(2δ)) * e^{-|x|/δ}
        
        Args:
            grad_output: 上游梯度
        Returns:
            grad_input: 输入x的梯度
            None: delta的梯度（不计算）
        """
        # 从上下文中获取保存的张量和参数
        (x, ) = ctx.saved_tensors
        delta = ctx.delta
        use_sum = ctx.use_sum
        if use_sum == False:
            # print(f"laplace")

            # if not isinstance(delta, torch.Tensor):
            #     delta_tensor = torch.tensor(delta, device=x.device, dtype=x.dtype)
            # else:
            #     delta_tensor = delta
            
            # 步骤1: 计算s(u) = round(u - 0.5) + 0.5，即最近的半整数点
            # 这是公式中的s(u)函数，用于找到最近的半整数点
            s_u = torch.round(x - 0.5) + 0.5
            
            # 步骤2: 计算归一化距离a = |u - s(u)|/δ
            # 这是公式中的a，表示输入点到最近半整数点的归一化距离
            a = torch.abs(x - s_u) / delta
            
            # 步骤3: 计算分母部分：2δ(1 - e^{-1/(√2δ)})
            # 这是公式中的归一化常数分母
            # sqrt_2 = math.sqrt(2.0)
            sqrt_2 = torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype))
            denominator = 2.0 * delta * (1.0 - torch.exp(-1.0 / (sqrt_2 * delta)))
            
            # 步骤4: 计算分子第一项：(a + 1/√2)e^{-√2a}
            # 这是公式中的(a + 1/√2)e^{-√2a}项
            numerator_part1 = (a + 1.0 / sqrt_2) * torch.exp(-sqrt_2 * a)
            
            # 步骤5: 计算分子第二项：(1/(2δ) + 1/√2)e^{-1/(√2δ)}
            # 这是公式中的(1/(2δ) + 1/√2)e^{-1/(√2δ)}项
            # 注意：这一项与a无关，是常数项
            constant_term = (1.0 / (2.0 * delta) + 1.0 / sqrt_2) * torch.exp(-1.0 / (sqrt_2 * delta))
            
            # 步骤6: 计算完整梯度表达式
            # 根据公式：I = 1/分母 * (分子第一项 - 分子第二项)
            grad_input = (numerator_part1 - constant_term) / denominator
            
            # 将上游梯度乘以本地梯度
            # 这是链式法则的应用
            grad_input = grad_input * grad_output
        else:
            pass
        
        # 返回梯度
        # 第一个返回值是x的梯度，第二个是delta的梯度（这里不计算，返回None）
        return grad_input, None, None



# 向量化加速版本
class HTGE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, t):
        out = torch.round(x)
        ctx.save_for_backward(x)
        ctx.t = t
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：使用HTGE近似梯度公式计算代理梯度
        
        公式对应：
        根据公式(24): H(x) = (a+b)/2 + (1/2)*tanh(t*(x - (a+b)/2))
        根据公式(25): ∂round(x)/∂x ≈ ∂H(x)/∂x = 1/2 * (1 - tanh^2(t*(x - (a+b)/2)))
        
        推导过程：
        1. 令 u = t*(x - (a+b)/2)
        2. 则 H(x) = (a+b)/2 + (1/2)*tanh(u)
        3. ∂H/∂x = (1/2) * ∂tanh(u)/∂x
        4. ∂tanh(u)/∂x = (1 - tanh^2(u)) * ∂u/∂x
        5. ∂u/∂x = t
        6. 所以 ∂H/∂x = (1/2) * (1 - tanh^2(u)) * t
        7. 因此梯度公式为: t/2 * (1 - tanh^2(t*(x - (a+b)/2)))
        
        Args:
            grad_output: 上游梯度
        Returns:
            grad_input: 输入x的梯度
            None: t的梯度（不计算）
        """
        # 从上下文中获取保存的张量和参数
        (x, ) = ctx.saved_tensors
        t = ctx.t
        # print(f"HTGE")
        
        # 计算a = floor(x), b = ceil(x)
        # 使用torch.floor和torch.ceil获取向下和向上取整值
        a = torch.floor(x)  # 公式中的floor(x)
        b = torch.ceil(x)   # 公式中的ceil(x)
        
        # 计算中间点 mid = (a+b)/2
        # 这是公式(24)中的(a+b)/2项
        mid = (a + b) / 2.0
        
        # 计算u = t*(x - (a+b)/2)
        # 这是公式中tanh的参数
        u = t * (x - mid)
        
        # 计算tanh(u)和tanh^2(u)
        # 使用torch.tanh计算双曲正切函数
        tanh_u = torch.tanh(u)  # tanh(t*(x - (a+b)/2))
        tanh_u_squared = tanh_u * tanh_u  # tanh^2(t*(x - (a+b)/2))
        
        # 根据公式(25)计算代理梯度
        # 完整梯度公式: (t/2) * (1 - tanh^2(t*(x - (a+b)/2)))
        # 注意：原始公式(25)可能缺少t因子，根据推导应该包含t
        grad_input = (t / 2.0) * (1.0 - tanh_u_squared)
        
        # 将上游梯度乘以本地梯度
        # 这是链式法则的应用
        grad_input = grad_input * grad_output
        
        # 返回梯度
        # 第一个返回值是x的梯度，第二个是t的梯度（这里不计算，返回None）
        return grad_input, None