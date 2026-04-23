import torch
from torch import nn
from typing import Optional, Tuple, List
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
import torch.nn.functional as F
from quantize.omni_norm import OmniLlamaRMSNorm
from collections import OrderedDict
import math
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding,apply_rotary_pos_emb,LlamaRMSNorm,repeat_kv
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.activations import ACT2FN
import pdb
import copy
from models.transformation import *




class QuantLlamaMLP(nn.Module):
    def __init__(
        self,
        org_module: nn.Module,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        args=None,
    ):
        super().__init__()
        # self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        # self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_proj = QuantLinear(org_module.gate_proj,
                                           args.weight_quant_params,
                                           args.act_quant_params)
        self.down_proj = QuantLinear(org_module.down_proj,
                                           args.weight_quant_params,
                                           args.act_quant_params)
        self.up_proj = QuantLinear(org_module.up_proj,
                                           args.weight_quant_params,
                                           args.act_quant_params)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class QuantLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        org_module: nn.Module,
        config: LlamaConfig,
        args=None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.layer_idx = layer_idx

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Lucifer Li: 版本问题, 直接创建`rotary_emb`而不是从`org_module`中复制, 需要传入`config`
        self.rotary_emb = LlamaRotaryEmbedding(config)

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
        self.o_proj = QuantLinear(
            org_module.o_proj, args.weight_quant_params, args.act_quant_params
        )
        self.qkt_matmul = QuantMatMul(
            args.q_quant_params, args.k_quant_params, matmul_func=torch.matmul
        )
        self.pv_matmul = QuantMatMul(
            args.p_quant_params, args.v_quant_params, matmul_func=torch.matmul
        )

        self.use_weight_quant = False
        self.use_act_quant = False

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[object] = None,
        output_attentions: bool = False,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[object]]:
        bsz, q_len, _ = hidden_states.size()

        # ==================== DEBUG PRINT ====================
        # print(f"\n{'='*60}")
        # print(f"[QuantLlamaAttention] layer_idx: {self.layer_idx}")
        # print(f"[QuantLlamaAttention] hidden_states.shape: {hidden_states.shape}")
        # print(f"[QuantLlamaAttention] q_len: {q_len}")
        # print(f"[QuantLlamaAttention] past_key_values type: {type(past_key_values)}")
        # print(f"[QuantLlamaAttention] use_cache: {use_cache}")
        # print(f"[QuantLlamaAttention] output_attentions: {output_attentions}")
        # if attention_mask is not None:
        #     print(f"[QuantLlamaAttention] attention_mask.shape: {attention_mask.shape}")
        #     print(f"[QuantLlamaAttention] attention_mask.dtype: {attention_mask.dtype}")
        # else:
        #     print(f"[QuantLlamaAttention] attention_mask: None")
        # print(f"{'='*60}")
        # ====================================================

        # Lucifer Li: 从 DynamicCache 中获取缓存的 key 和 value（缓存中是未扩展的 states）
        past_key_value = None
        if past_key_values is not None and self.layer_idx is not None:
            layer_cache = past_key_values.layers[self.layer_idx]
            if hasattr(layer_cache, 'keys') and layer_cache.keys is not None:
                past_key_value = (layer_cache.keys, layer_cache.values)
                # print(f"[QuantLlamaAttention] Retrieved cache: keys.shape={layer_cache.keys.shape}, values.shape={layer_cache.values.shape}")

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        # ==================== DEBUG PRINT ====================
        # print(f"[QuantLlamaAttention] kv_seq_len (after cache): {kv_seq_len}")
        # print(f"[QuantLlamaAttention] Expected attention_mask size: {(bsz, 1, q_len, kv_seq_len)}")
        # ====================================================
        
        # Lucifer Li: transformers 4.57.3 版本中，LlamaRotaryEmbedding.forward() 不再接受 seq_len 参数
        # 只需要传递 value_states 和 position_ids
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Lucifer Li: 【关键修复】先更新缓存（存入未扩展的 states），再拼接，最后 repeat_kv
        # 1. 拼接缓存（缓存中存储的是未扩展的 states: [bsz, num_kv_heads, seq_len, head_dim]）
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # 2. 更新 DynamicCache（存入未扩展的 states，避免重复 repeat_kv）
        if use_cache and past_key_values is not None and self.layer_idx is not None:
            layer_cache = past_key_values.layers[self.layer_idx]
            layer_cache.keys = key_states   # 未扩展: [bsz, num_kv_heads, kv_seq_len, head_dim]
            layer_cache.values = value_states  # 未扩展: [bsz, num_kv_heads, kv_seq_len, head_dim]

        # 3. 统一在拼接后进行 repeat_kv（确保缓存部分和新部分只扩展一次）
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states = self.qkt_matmul.quant_x2(key_states)
        attn_weights = self.qkt_matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            # Lucifer Li: 【兼容性修复】transformers 新版本不再自动扩展 attention_mask
            # 需要手动检查并扩展 attention_mask 以匹配 kv_seq_len
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                expected_size = (bsz, 1, q_len, kv_seq_len)
                actual_size = attention_mask.size()
                
                # 检查是否是 attention_mask 需要扩展的情况
                # 通常发生在生成阶段：attention_mask 是 (bsz, 1, 1, 1) 但需要 (bsz, 1, 1, kv_seq_len)
                if actual_size[2] == q_len and actual_size[3] < kv_seq_len:
                    # 扩展 attention_mask：填充左侧（缓存部分）为 0（不屏蔽）
                    # attention_mask 格式：0 表示不屏蔽，负无穷表示屏蔽
                    pad_len = kv_seq_len - actual_size[3]
                    # 在左侧填充 0（不屏蔽缓存部分）
                    attention_mask = torch.nn.functional.pad(
                        attention_mask, 
                        (pad_len, 0, 0, 0),  # (left, right, top, bottom) 在最后一个维度左侧填充
                        value=0
                    )
                    # print(f"[QuantLlamaAttention] Auto-expanded attention_mask: {actual_size} -> {attention_mask.size()}")
                else:
                    # 其他不匹配情况，报错
                    print(f"\n{'!'*60}")
                    print(f"❌ ATTENTION MASK SIZE MISMATCH!")
                    print(f"Expected: {expected_size}")
                    print(f"Got: {actual_size}")
                    print(f"{'!'*60}\n")
                    raise ValueError(
                        f"Attention mask should be of size {expected_size}, but is {actual_size}"
                    )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.pv_matmul.quant_x1(attn_weights)
        value_states = self.pv_matmul.quant_x2(value_states)
        attn_output = self.pv_matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        # ==================== DEBUG PRINT ====================
        # print(f"[QuantLlamaAttention] Returning: attn_output.shape={attn_output.shape}")
        # print(f"[QuantLlamaAttention] Return type: tuple with 3 elements")
        # print(f"{'='*60}\n")
        # ====================================================

        return attn_output, attn_weights, past_key_values
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)
                


class QuantLlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        ori_layer,
        args,
        layer_idx: Optional[int] = None
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = QuantLlamaAttention(
            org_module=ori_layer.self_attn,
            config=config,
            args=args,
            layer_idx=layer_idx,
            )
        self.mlp = QuantLlamaMLP(
            org_module=ori_layer.mlp,
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            args=args,
        )
        self.input_layernorm = OmniLlamaRMSNorm(ori_layer.input_layernorm,eps=ori_layer.input_layernorm.variance_epsilon)
        self.post_attention_layernorm = OmniLlamaRMSNorm(ori_layer.post_attention_layernorm,eps=ori_layer.post_attention_layernorm.variance_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[object] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        **kwargs
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_values (`object`, *optional*): DynamicCache object for cached past key and value projection states
            **kwargs: additional keyword arguments
        """
        # ==================== DEBUG PRINT ====================
        # print(f"\n{'*'*60}")
        # print(f"[QuantLlamaDecoderLayer] layer_idx: {self.layer_idx}")
        # print(f"[QuantLlamaDecoderLayer] hidden_states type: {type(hidden_states)}")
        # if isinstance(hidden_states, torch.Tensor):
        #     print(f"[QuantLlamaDecoderLayer] hidden_states.shape: {hidden_states.shape}")
        # print(f"[QuantLlamaDecoderLayer] past_key_values type: {type(past_key_values)}")
        # print(f"[QuantLlamaDecoderLayer] use_cache: {use_cache}")
        # print(f"[QuantLlamaDecoderLayer] output_attentions: {output_attentions}")
        # print(f"[QuantLlamaDecoderLayer] kwargs keys: {list(kwargs.keys())}")
        # print(f"{'*'*60}")
        # ====================================================

        # Lucifer Li: 安全检查, 确保`hidden_states`是`tensor`而不是`tuple`, 某些`transformers`版本可能会传递`tuple`形式的`hidden_states`
        if isinstance(hidden_states, tuple):
            # print(f"[QuantLlamaDecoderLayer] WARNING: hidden_states is tuple, extracting first element")
            hidden_states = hidden_states[0]

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)


        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # ==================== DEBUG PRINT ====================
        # print(f"\n{'*'*60}")
        # print(f"[QuantLlamaDecoderLayer] Return outputs length: {len(outputs)}")
        # print(f"[QuantLlamaDecoderLayer] outputs[0] type: {type(outputs[0])}")
        # if isinstance(outputs[0], torch.Tensor):
        #     print(f"[QuantLlamaDecoderLayer] outputs[0].shape: {outputs[0].shape}")
        #     print(f"[QuantLlamaDecoderLayer] outputs[0].dtype: {outputs[0].dtype}")
        # print(f"[QuantLlamaDecoderLayer] past_key_values: {past_key_values}")
        # print(f"[QuantLlamaDecoderLayer] Returning: {'tensor' if past_key_values is not None else 'tuple'}")
        # print(f"{'*'*60}\n")
        # ====================================================

        # Lucifer Li: 【智能返回策略 - 版本兼容修复】
        # 
        # 场景1：量化训练（omniquant.py）
        #   - past_key_values = None（不使用缓存）
        #   - 调用方式：qlayer(...)[0] 需要索引
        #   - 返回 tuple 以支持 [0] 索引
        # 
        # 场景2：推理（transformers generate）
        #   - past_key_values = DynamicCache 对象
        #   - transformers 新版本会把返回值直接传给下一层
        #   - 返回 tensor 避免嵌套 tuple 问题
        # 
        # KV 缓存通过 past_key_values 参数引用传递，内部修改自动生效
        if past_key_values is not None:
            # 推理模式：返回 tensor
            return outputs[0]
        else:
            # 量化训练模式：返回 tuple
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
      
    def smooth_and_quant_temporary(self):
        if self.let:
            with torch.no_grad():
                for name, module in self.named_parameters():
                    if "smooth_scale" in name:
                        module.data = truncate_number(module)
            smooth_ln_fcs_temporary(self.input_layernorm,[self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                                    self.qkv_smooth_scale,self.qkv_smooth_shift)
            smooth_ln_fcs_temporary(self.post_attention_layernorm,[self.mlp.up_proj,self.mlp.gate_proj],
                                    self.fc1_smooth_scale,self.fc1_smooth_shift)
            smooth_fc_fc_temporary(self.self_attn.v_proj,self.self_attn.o_proj,
                                self.out_smooth_scale, self.out_smooth_shift)
            smooth_q_k_temporary(self.self_attn.q_proj, self.self_attn.k_proj,
                                self.qkt_smooth_scale)
            self.mlp.down_proj.temp_weight = self.mlp.down_proj.weight
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

    def clear_temp_variable(self):
       for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                del module.temp_weight
                del module.temp_bias

    @torch.no_grad()
    def smooth_and_quant_inplace(self):
        if self.let:
            for name, module in self.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
            smooth_ln_fcs_inplace(self.input_layernorm,[self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                                    self.qkv_smooth_scale,self.qkv_smooth_shift)
            smooth_ln_fcs_inplace(self.post_attention_layernorm,[self.mlp.up_proj,self.mlp.gate_proj],
                                    self.fc1_smooth_scale,self.fc1_smooth_shift)
            smooth_fc_fc_inplace(self.self_attn.v_proj,self.self_attn.o_proj,
                                self.out_smooth_scale, self.out_smooth_shift)
            smooth_q_k_inplace(self.self_attn.q_proj, self.self_attn.k_proj,
                                self.qkt_smooth_scale)
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight = module.weight_quantizer(module.weight)
                module.use_temporary_parameter=False

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
