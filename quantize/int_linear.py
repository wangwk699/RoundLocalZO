import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer






class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=False,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        # ldx:add:
        # self.register_parameter('orig_weight', nn.Parameter(org_module.weight.clone()))
        # if org_module.bias is not None:
        #     self.register_parameter('orig_bias', nn.Parameter(org_module.bias.clone()))
        # else:
        #     self.orig_bias = None


        self.register_buffer('weight',org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        # ldx_add:
        # self.register_parameter('orig_weight', nn.Parameter(org_module.weight.clone()))
        # if org_module.bias is not None:
        #     self.register_parameter('orig_bias', nn.Parameter(org_module.bias.clone()))
        # else:
        #     self.orig_bias = None
        # self.register_buffer('quant_weight', org_module.weight.clone())
        # if org_module.bias is not None:
        #     self.register_buffer('quant_bias', org_module.bias.clone())
        # else:
        #     self.quant_bias = None


        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default  禁用量化前向传播的默认设置
        self.use_weight_quant = False
        self.use_act_quant = False

        # ldx:add:
        # self.is_quant_training = True
        # self.is_finetune_training = False

        # initialize quantizer 
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params, shape=org_module.weight.shape)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False

    
    
    def forward(self, input: torch.Tensor):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
            # ldx_add:
            # if self.is_quant_training and not self.is_finetune_training:
            #     weight = self.weight_quantizer(self.weight)
            #     bias = self.bias
            # elif self.is_finetune_training and not self.is_quant_training:
            #     # print(f"使用了我们像优化的权重")
            #     weight = self.weight_quantizer(self.orig_weight)
            #     bias = self.orig_bias
            # else:
            #     raise NotImplementedError()

        else:

            weight = self.weight
            bias = self.bias
            # ldx_add：使用原始权重
            # weight = self.orig_weight
            # bias = self.orig_bias


        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input)
        
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)


        return out
    # ldx:add
    # def update_quantized_weights(self):
    #     """将当前原始权重量化并存储到量化权重buffer中"""
    #     with torch.no_grad():
    #         if self.use_weight_quant:
    #             self.quant_weight.data = self.weight_quantizer(self.orig_weight)
    #             if self.orig_bias is not None:
    #                 self.quant_bias.data = self.orig_bias.clone()
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    # def set_training_phase(self, phase: str):
    #     if phase == 'quant':
    #         self.is_quant_training = True
    #         self.is_finetune_training = False
    #     elif phase == 'finetune':
    #         self.is_quant_training = False
    #         self.is_finetune_training = True
    #     elif phase == 'eval':
    #         self.is_quant_training = False
    #         self.is_finetune_training = False
    #     else:
    #         raise NotImplementedError()



# class QuantLinearZO(nn.Module):
#     """
#     Quantized Module that can perform quantized convolution or normal convolution.
#     To activate quantization, please use set_quant_state function.
#     """
#     def __init__(
#         self,
#         org_module: nn.Linear,
#         weight_quant_params: dict = {},
#         act_quant_params: dict = {},
#         disable_input_quant=False,
#     ):
#         super().__init__()
#         self.fwd_kwargs = dict()
#         self.fwd_func = F.linear
#         self.register_buffer('weight',org_module.weight)
#         if org_module.bias is not None:
#             self.register_buffer('bias',org_module.bias)
#         else:
#             self.bias = None
#         self.in_features = org_module.in_features
#         self.out_features = org_module.out_features
#         # de-activate the quantized forward default
#         self.use_weight_quant = False
#         self.use_act_quant = False
#         # initialize quantizer
#         self.weight_quantizer = UniformAffineQuantizerZO(**weight_quant_params,shape=org_module.weight.shape)
#         if not disable_input_quant:
#             self.act_quantizer = UniformAffineQuantizerZO(**act_quant_params)
#         else:
#             self.act_quantizer = None

#         self.disable_input_quant = disable_input_quant
#         self.use_temporary_parameter = False

    
    
#     def forward(self, input: torch.Tensor):
#         if self.use_temporary_parameter:
#             weight = self.temp_weight
#             bias = self.temp_bias
#         elif self.use_weight_quant:
#             weight = self.weight_quantizer(self.weight)
#             bias = self.bias
#         else:
#             weight = self.weight
#             bias = self.bias

#         if self.use_act_quant and not self.disable_input_quant:
#             input = self.act_quantizer(input)
        
#         out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)


#         return out

#     def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
#         self.use_weight_quant = weight_quant
#         self.use_act_quant = act_quant

