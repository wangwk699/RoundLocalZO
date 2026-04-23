import torch
from torch import nn
from typing import Optional, Tuple, List
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
import torch.nn.functional as F
from quantize.omni_norm import OmniLayerNorm
from collections import OrderedDict
import pdb
from models.models_utils import truncate_number
from models.transformation import *



# Lucifer Li: 量化版的注意力层(vOpt)
class QuantOPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        org_module: nn.Module,  # 原始注意力模块
        embed_dim: int,  # 嵌入维度, 即词向量的维度大小
        num_heads: int,  # (多)注意力头的数量
        dropout: float = 0.0,  # Dropout率
        is_decoder: bool = False,  # 是否为解码器模块
        bias: bool = True,  # 是否使用偏置项
        args=None,  # 包含量化参数的配置对象
        disable_act_quant=False,  # 是否禁用激活值量化
        layer_idx: Optional[int] = None,  # 层索引，用于访问缓存
    ):
        super().__init__()
        # Lucifer Li: 基础属性设置
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.layer_idx = layer_idx

        # Lucifer Li: 参数校验, 确保嵌入维度能被头数整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )

        # Lucifer Li: `scaling`是注意力计算中的缩放因子, 用于防止内积过大导致Softmax梯度消失
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # Lucifer Li: 使用量化版本的线性层创建了注意力机制的核心投影层(Key, Value, Query和输出投影)
        # input is quantized by LayerNorm, set disable_input_quant=True
        self.k_proj = QuantLinear(
            org_module.k_proj,
            args.weight_quant_params,
            args.act_quant_params,
        )
        self.v_proj = QuantLinear(
            org_module.v_proj,
            args.weight_quant_params,
            args.act_quant_params,
        )
        self.q_proj = QuantLinear(
            org_module.q_proj,
            args.weight_quant_params,
            args.act_quant_params,
        )
        self.out_proj = QuantLinear(
            org_module.out_proj, args.weight_quant_params, args.act_quant_params
        )
        # Lucifer Li: 量化了注意力计算中的两个关键矩阵乘法:
        # 1. 处理Q和K的矩阵乘法, 计算注意力权重;
        self.qkt_matmul = QuantMatMul(
            args.q_quant_params, args.k_quant_params, matmul_func=torch.bmm
        )
        # 2. 处理注意力权重和V的矩阵乘法;
        self.pv_matmul = QuantMatMul(
            args.p_quant_params, args.v_quant_params, matmul_func=torch.bmm
        )

        self.use_weight_quant = False
        self.use_act_quant = False

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[object] = None,  # DynamicCache对象
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[object]]:
        """Input shape: Batch x Time x Channel"""
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()
        
        # ✅ DEBUG: 打印关键信息
        # print(f"[QuantOPTAttention] hidden_states.shape: {hidden_states.shape}")
        # print(f"[QuantOPTAttention] tgt_len: {tgt_len}")
        # print(f"[QuantOPTAttention] layer_idx: {self.layer_idx}")
        # if attention_mask is not None:
            # print(f"[QuantOPTAttention] attention_mask.shape: {attention_mask.shape}")
            # print(f"[QuantOPTAttention] attention_mask.dtype: {attention_mask.dtype}")
        # else:
            # print(f"[QuantOPTAttention] attention_mask is None")

        # 从 DynamicCache 中获取缓存的 key 和 value
        past_key_value = None
        if past_key_values is not None and self.layer_idx is not None:
            layer_cache = past_key_values.layers[self.layer_idx]
            if hasattr(layer_cache, 'keys') and layer_cache.keys is not None:
                past_key_value = (layer_cache.keys, layer_cache.values)
                # print(f"[QuantOPTAttention] Retrieved from cache: keys.shape={layer_cache.keys.shape}, values.shape={layer_cache.values.shape}")
            else:
                # print(f"[QuantOPTAttention] Cache layer {self.layer_idx} is empty or not initialized")
                pass
        else:
            # print(f"[QuantOPTAttention] past_key_values is None or layer_idx is None")
            pass

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = self.qkt_matmul.quant_x1(query_states)

        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self.k_proj(key_value_states)
            key_states = self.qkt_matmul.quant_x2(key_states)
            key_states = self._shape(key_states, -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            # bsz, seq_len, self.num_heads, self.head_dim -> bsz, self.num_heads, seq_len, self.head_dim
            key_states = self.k_proj(hidden_states)
            key_states = self.qkt_matmul.quant_x2(key_states)
            key_states = self._shape(key_states, -1, bsz)

            value_states = self.v_proj(hidden_states)
            value_states = self.pv_matmul.quant_x2(value_states)
            value_states = self._shape(value_states, -1, bsz)
            
            # Lucifer Li: 将缓存的KV和新计算的KV转换为统一格式后再拼接
            # 缓存格式: [bsz*num_heads, seq_len, head_dim] (3维)
            # 新计算格式: [bsz, num_heads, seq_len, head_dim] (4维)
            # 需要先将新计算的KV转换为3维格式
            proj_shape = (bsz * self.num_heads, -1, self.head_dim)
            key_states = key_states.view(*proj_shape)  # [bsz*num_heads, 1, head_dim]
            value_states = value_states.view(*proj_shape)  # [bsz*num_heads, 1, head_dim]
            
            # 在seq_len维度(dim=1)拼接
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
        else:
            # self_attention
            key_states = self.k_proj(hidden_states)
            key_states = self.qkt_matmul.quant_x2(key_states)
            key_states = self._shape(key_states, -1, bsz)

            value_states = self.v_proj(hidden_states)
            value_states = self.pv_matmul.quant_x2(value_states)
            value_states = self._shape(value_states, -1, bsz)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        # Lucifer Li: key_states 和 value_states 在有缓存的情况下已经view过了
        # 需要检查是否需要再次view
        if key_states.dim() == 4:
            # 无缓存情况，需要从 [bsz, num_heads, seq_len, head_dim] 转换为 [bsz*num_heads, seq_len, head_dim]
            key_states = key_states.view(*proj_shape)
            value_states = value_states.view(*proj_shape)
        # 有缓存情况下，已经在上面view过了，这里不需要再次处理

        src_len = key_states.size(1)

        # ✅ DEBUG: 打印 src_len
        # print(f"[QuantOPTAttention] src_len (after KV cache): {src_len}")
        # print(f"[QuantOPTAttention] Expected attention_mask size: {(bsz, 1, tgt_len, src_len)}")

        attn_weights = self.qkt_matmul(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                print(f"\n{'='*60}")
                print(f"❌ ATTENTION MASK SIZE MISMATCH!")
                print(f"Expected: {(bsz, 1, tgt_len, src_len)}")
                print(f"Got: {attention_mask.size()}")
                print(f"{'='*60}\n")
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
            attn_probs_reshaped = attn_weights_reshaped
        else:
            attn_probs_reshaped = None

        # attention shape bsz * self.num_heads, tgt_len, src_len
        attn_weights = self.pv_matmul.quant_x1(attn_weights)
        attn_output = self.pv_matmul(attn_weights, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        # 更新 DynamicCache
        if self.is_decoder and past_key_values is not None and self.layer_idx is not None:
            # Lucifer Li: 直接覆盖缓存，而不是使用update方法（避免重复拼接）
            # 因为key_states已经是拼接后的完整序列，直接设置即可
            layer_cache = past_key_values.layers[self.layer_idx]
            layer_cache.keys = key_states
            layer_cache.values = value_states
            # print(f"[QuantOPTAttention] Updated cache for layer {self.layer_idx}: keys={key_states.shape}, values={value_states.shape}")
        # Lucifer Li: 只在异常情况下打印警告
        elif self.is_decoder and (past_key_values is None or layer_idx is None):
            # print(f"[QuantOPTAttention] WARNING: is_decoder={self.is_decoder}, past_key_values={'None' if past_key_values is None else 'provided'}, layer_idx={self.layer_idx}")
            pass

        return attn_output, attn_probs_reshaped, past_key_values

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)






  
# Lucifer Li: 量化版的Transformer解码器层(vOpt)
class QuantOPTDecoderLayer(nn.Module):
    def __init__(
        self,
        config,  # 模型配置
        ori_layer,  # 原始层对象
        args,  # 量化参数
        layer_idx: Optional[int] = None,  # 层索引
    ):
        super().__init__()
        # Lucifer Li: 层索引，用于访问缓存
        self.layer_idx = layer_idx
        # Lucifer Li: 嵌入维度, 通常为768, 1024, ...
        self.embed_dim = config.hidden_size
        # Lucifer Li: 将原始的注意力模块替换为量化版的注意力模块
        self.self_attn = QuantOPTAttention(
            org_module=ori_layer.self_attn,
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=config.enable_bias,
            args=args,
            layer_idx=layer_idx,  # 传递 layer_idx
        )
        # Lucifer Li: 记录是否采用Pre-Layer Normalization结构
        self.do_layer_norm_before = config.do_layer_norm_before
        # Lucifer Li: 设置Dropout率, 防止过拟合
        self.dropout = config.dropout
        # Lucifer Li: 将原始LayerNorm替换为支持量化的OmniLayerNorm
        self.self_attn_layer_norm = OmniLayerNorm(
            ori_layer.self_attn_layer_norm
        )
        # Lucifer Li: 将前馈网络的第一个全连接层替换为量化线性层
        self.fc1 = QuantLinear(
            ori_layer.fc1,
            weight_quant_params=args.weight_quant_params,
            act_quant_params=args.act_quant_params,
        )
        # Lucifer Li: 将前馈网络的第二个全连接层替换为量化线性层
        self.fc2 = QuantLinear(
            ori_layer.fc2,
            weight_quant_params=args.weight_quant_params,
            act_quant_params=args.act_quant_params,
        )
        # Lucifer Li: 将最终LayerNorm替换为支持量化的OmniLayerNorm
        self.final_layer_norm = OmniLayerNorm(
            ori_layer.final_layer_norm
        )
        # Lucifer Li: 保留原始模型权重的数据类型
        self.type = ori_layer.fc1.weight.dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs
    ):
        """
        Args:
            hidden_states (`torch.Int8Tensor`): the output of previous layer's layernorm in INT8
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states (deprecated)
            **kwargs: additional keyword arguments, including 'past_key_values' (DynamicCache object)
        """
        # 从 kwargs 中获取 DynamicCache 格式的 past_key_values
        past_key_values = kwargs.get('past_key_values', None)

        '''
        # Debug: 打印缓存信息
        if past_key_values is not None:
            print(f"[QuantOPTDecoderLayer] layer_idx={self.layer_idx}, past_key_values type: {type(past_key_values)}")
            if hasattr(past_key_values, 'layers'):
                print(f"[QuantOPTDecoderLayer] Number of layers in cache: {len(past_key_values.layers)}")
        else:
            print(f"[QuantOPTDecoderLayer] past_key_values is None")

        print(f"[output_attentions] {output_attentions}")
        # ✅ DEBUG: 打印 QuantOPTDecoderLayer 的输入
        print(f"\n{'='*60}")
        print(f"[QuantOPTDecoderLayer] Forward called")
        print(f"[QuantOPTDecoderLayer] hidden_states.shape: {hidden_states.shape}")
        if attention_mask is not None:
            print(f"[QuantOPTDecoderLayer] attention_mask.shape: {attention_mask.shape}")
            print(f"[QuantOPTDecoderLayer] attention_mask.dtype: {attention_mask.dtype}")
        else:
            print(f"[QuantOPTDecoderLayer] attention_mask is None")
        print(f"[QuantOPTDecoderLayer] use_cache: {use_cache}")
        print(f"[QuantOPTDecoderLayer] layer_idx: {self.layer_idx}")
        print(f"{'='*60}\n")
        '''

        # Self Attention
        # Lucifer Li: 保存了前一层输出的`hidden_states`到`residual`变量, 方便后续进行残差连接, 帮助缓解深层网络中的梯度消失问题
        residual = hidden_states
        # Lucifer Li: 如果模型采用Pre-LayerNorm结构, 先对输入进行层归一化
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        # hidden_states = self.self_attn_layer_norm(hidden_states.float()).to(self.type)

        # Lucifer Li: 调用量化版自注意力模块, 返回注意力输出, 注意力权重, 当前KV缓存(用于生成任务)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,  # 传递 DynamicCache 对象
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        # Lucifer Li: 禁用dropout, p=0.0, 实际dropout不起任何作用
        hidden_states = nn.functional.dropout(hidden_states, p=0.0, training=False)

        # Lucifer Li: 将注意力输出与原始输入相加, 实现残差连接
        hidden_states = residual + hidden_states

        # Lucifer Li: 模型采用Post-LayerNorm结构, 此时进行层归一化
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Lucifer Li: 前馈网络准备(保存张量形状以便后续恢复, 展平为二维张量, 再次保存残差连接用的输入)
        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # residual.add_(hidden_states.to(residual.dtype))
        # Lucifer Li: Pre-LayerNorm结构
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        # hidden_states = self.final_layer_norm(hidden_states.float()).to(self.type)

        
        # Lucifer Li: 前馈网络计算
        hidden_states = self.fc1(hidden_states)  # 量化版全连接层1
        hidden_states = F.relu(hidden_states)  # ReLU激活
        hidden_states = self.fc2(hidden_states)  # 量化版全连接层2
        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # Lucifer Li: 残差连接, 恢复原始张量形状
        hidden_states = (residual + hidden_states).view(hidden_states_shape)
        # residual.add_(hidden_states.to(residual.dtype))
        # Lucifer Li: Post-LayerNorm结构
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        # Lucifer Li: 根据参数决定是否返回, 处理后的隐藏状态(必选), 注意力权重(可选), KV缓存(可选)
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)  # present_key_value 现在是 DynamicCache 对象
            # ✅ DEBUG: 验证 present_key_value 是否被返回
            # print(f"[QuantOPTDecoderLayer] Returning present_key_value (DynamicCache)")
            # print(f"[QuantOPTDecoderLayer] outputs length: {len(outputs)}")
        else:
            # print(f"[QuantOPTDecoderLayer] WARNING: use_cache={use_cache}, NOT returning KV cache!")
            pass

        return outputs

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        names = []
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                names.append(name)
                m.set_quant_state(weight_quant, act_quant)

    @torch.no_grad()
    def smooth_and_quant_inplace(self):
        # return
        if self.let:
            for name, module in self.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
            smooth_ln_fcs_inplace(self.self_attn_layer_norm,[self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                                    self.qkv_smooth_scale,self.qkv_smooth_shift)
            smooth_ln_fcs_inplace(self.final_layer_norm,[self.fc1],
                                    self.fc1_smooth_scale,self.fc1_smooth_shift)
            smooth_fc_fc_inplace(self.self_attn.v_proj,self.self_attn.out_proj,
                                self.out_smooth_scale, self.out_smooth_shift)
            smooth_q_k_inplace(self.self_attn.q_proj, self.self_attn.k_proj,
                                self.qkt_smooth_scale)
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight = module.weight_quantizer(module.weight)
                module.use_temporary_parameter=False

    def clear_temp_variable(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                del module.temp_weight
                del module.temp_bias

    def smooth_and_quant_temporary(self):
        if self.let:
            with torch.no_grad():
                for name, module in self.named_parameters():
                    if "smooth_scale" in name:
                        module.data = truncate_number(module)
            smooth_ln_fcs_temporary(self.self_attn_layer_norm,[self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                                    self.qkv_smooth_scale,self.qkv_smooth_shift)
            smooth_ln_fcs_temporary(self.final_layer_norm,[self.fc1],
                                    self.fc1_smooth_scale,self.fc1_smooth_shift)
            smooth_fc_fc_temporary(self.self_attn.v_proj,self.self_attn.out_proj,
                                self.out_smooth_scale, self.out_smooth_shift)
            smooth_q_k_temporary(self.self_attn.q_proj, self.self_attn.k_proj,
                                self.qkt_smooth_scale)
            self.fc2.temp_weight = self.fc2.weight
        else:
            for name, module in self.named_modules():
                if isinstance(module, QuantLinear):
                    module.temp_weight = module.weight
        # quant
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                if hasattr(module, "temp_weight"):
                    module.temp_weight = module.weight_quantizer(module.temp_weight)
                else:
                    module.temp_weight = module.weight_quantizer(module.weight)
                if not hasattr(module, "temp_bias"):
                    module.temp_bias = module.bias
                module.use_temporary_parameter=True

    def let_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params)  

    def lwc_parameters(self):
        params = []
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1:
                params.append(m)
        return iter(params)  

    def omni_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1 or n.find(template) > -1:
                params.append(m)
        return iter(params)  
    
    def omni_state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_parameters():
            if name.find('smooth') > -1 or name.find('bound_factor') > -1:
                destination[prefix + name] = param if keep_vars else param.detach()
        return destination

    def register_scales_and_zeros(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_scales_and_zeros()
