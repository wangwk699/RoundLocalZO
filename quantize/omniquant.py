import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
from quantize.utils import let_parameters, lwc_parameters, get_omni_parameters,\
                            omni_state_dict, register_scales_and_zeros,smooth_and_quant_temporary,\
                            smooth_and_quant_inplace,clear_temp_variable,set_quant_state
try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     

def omniquant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    # Lucifer Li: args.net > `opt-125m`, 以opt为例
    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        # Lucifer Li: 因果语言模型中解码器部分的层级堆叠
        layers = model.model.decoder.layers
        # Lucifer Li: 加载到设备中(CPU, GPU, ...)
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        # Lucifer Li: 自定义解码器层 > QuantOPTDecoderLayer
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    elif 'mixtral' in args.net.lower():
        is_llama = True   # same to llama except ffn
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    

    layers[0] = layers[0].to(dev)
    # Lucifer Li: 根据args.deactive_amp和args.epochs确定训练精度和训练上下文
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float  # FP32
        traincast = nullcontext  # 禁用自动混合精度
    else:
        dtype = torch.float16  # FP16
        traincast = torch.cuda.amp.autocast  # 启用AMP
    # Lucifer Li: 创建全零张量作为量化校准的输入容器, 形状为(样本数, 序列长度, 隐藏层维度)
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {"i": 0}
    # Lucifer Li: 数据捕获工具, 在不修改原有模型结构的前提下, 拦截并保存流经模型第一层的输入数据
    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            # 保留inp, attention_mask和position_ids数据后中断
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    # Lucifer Li: 将第一层替换为数据捕获工具包装下的第一层, 方便数据捕捉
    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    # Lucifer Li: 用实际数据执行一次前向传播, 专门触发Catcher来收集量化校准所需的输入样本, 最终收集args.nsamples个样本
    with torch.no_grad():
        for batch in dataloader:  # (trainloader)
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower() or "mixtral" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        # Lucifer Li: 加载到CPU中
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    # 清理缓存分配器中当前未使用的显存
    torch.cuda.empty_cache()

    
    # Lucifer Li: 为后续并行运行全精度模型和量化模型做准备
    # same input of first layer for fp model and quant model
    # Lucifer Li: 将`quant_inps`送入量化模型进行前向传播; 将`fp_inps`送入全精度模型进行前向传播
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None
    
    attention_mask = cache["attention_mask"]

    # Lucifer Li: 将捕获的单个样本的注意力掩码, 扩展为适用于整个批次的掩码
    if attention_mask is not None:
        # Lucifer Li: 为批量化的校准或训练准备格式正确的注意力掩码
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    # Lucifer Li: 指定损失函数为均方误差损失
    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        # Lucifer Li: 以opt为例, 无`position_ids`
        position_ids = None



    # Lucifer Li: 是否从中断中恢复
    if args.resume:
        omni_parameters = torch.load(args.resume)
    else:
        omni_parameters = {}

    
    # Lucifer Li: 循环遍历层并执行量化
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        if "mixtral" in args.net.lower():  
            # for mixtral, we only leverage lwc, which can be achieve by simply replace Linear with QuantLinear
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module,torch.nn.Linear) and not "gate" in name:       # do not quantize gate
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)    
        else:
            # Lucifer Li: 以opt模型为例, DecoderLayer==QuantOPTDecoderLayer, 传入参数: 模型配置; 原始层对象; 量化参数; 层索引;
            qlayer = DecoderLayer(lm.model.config, layer, args, layer_idx=i)
        # Lucifer Li: 将量化层`qlayer`加载到设备中
        qlayer = qlayer.to(dev)



        # Lucifer Li: 为后续的量化训练或校准建立一个标准, 方便衡量量化带来的精度损失, 并保存结果到原有变量`fp_inps`和`fp_inps_2`中
        # obtain output of full-precision model
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
                        if args.aug_loss:
                            fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
        # init smooth parameters
        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        # Lucifer Li: 决定是否在该量化层中启用可学习等效变换, 这是一种通过可学习的缩放和偏移参数来补偿量化误差的高级技术
        qlayer.let = args.let
        use_shift = True
        # Lucifer Li: 以opt为例
        if is_llama:
            use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
        # Lucifer Li: 目的是让量化后的模型性能尽可能接近原始的全精度模型
        if args.let:
            # Lucifer Li: 为注意力机制中的QK矩阵乘法结果注册一个可学习的通道级缩放因子, 初始值设为1, 形状与查询投影层的输出特征数相同
            # init channel-wise scaling and shift
            qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    # Lucifer Li: 以opt为例, pairs: {'q_proj': 'qkv', 'out_proj': 'out', 'fc1': 'fc1'}
                    for key in pairs.keys():
                        if key in name:
                            # Lucifer Li: 从预计算的激活值尺度字典中获取当前层对应的尺度数据, 并转移到指定设备, 同时避免除零错误
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            # Lucifer Li: 计算权重张量沿指定维度(dim=0，通常是输出通道维度)的绝对值的最大值, 这反映了每个通道权重的动态范围
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            # Lucifer Li: 计算平滑缩放因子, 通过一个平衡超参数`args.alpha`, 将激活值的统计信息`act`和权重的统计信息`weight`融合, 生成一个更优的缩放因子`scale`
                            # Lucifer Li: 默认`args.alpha`为0.5, 此时相当于在激活和权重之间平衡量化难度
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5, max=1e4)
                            # Lucifer Li: 以opt为例, 如果启用了偏移, 则从预计算的激活值偏移字典中获取偏移量; 否则, 初始化为零
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))

        # Lucifer Li: 是否从中断中恢复
        if args.resume:
            qlayer.load_state_dict(omni_parameters[i], strict=False)
        

        # Lucifer Li: 若执行微调, 则启动训练; 若不执行, 直接使用预校准的量化参数
        if args.epochs > 0:
            with torch.no_grad():
                # Lucifer Li: 在开始训练前, 将量化层暂时转换为FP32模式, 这是为了与自动混合精度训练更好地兼容, 确保数值稳定性
                qlayer.float()      # required for AMP training
            # Lucifer Li: 创建优化器, 专门用于更新量化相关的参数(let_parameters, lwc_parameters), 还配置了权重衰减来防止过拟合
            # create optimizer
            optimizer = torch.optim.AdamW(
                [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr}, {"params":lwc_parameters(qlayer),"lr":args.lwc_lr}],weight_decay=args.wd)
            # Lucifer Li: 损失缩放器, 用于混合精度训练
            loss_scaler = utils.NativeScalerWithGradNormCount()
            
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):    
                    index = j * args.batch_size
                    # obtain output of quantization model
                    with traincast():
                        # Lucifer Li: 执行量化
                        smooth_and_quant_temporary(qlayer, args, is_llama)
                        # Lucifer Li: 执行前向传播, 获取量化模型在当前批次输入下的输出
                        quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                        # Lucifer Li: 计算量化模型输出与全精度模型输出之间的均方误差
                        loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)
                        # Lucifer Li: 如果启用增强损失, 则添加额外的损失项
                        if args.aug_loss:
                            loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)
                    # Lucifer Li: 检查损失值是否有效
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    # Lucifer Li: 使用`utils.NativeScalerWithGradNormCount`执行梯度缩放和反向传播, 计算梯度并返回梯度范数
                    # .step(), .update()
                    norm = loss_scaler(loss, optimizer,parameters= get_omni_parameters(qlayer, use_shift)).cpu()
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
            # Lucifer Li: 释放内存
            clear_temp_variable(qlayer)
            del optimizer
        # Lucifer Li: 将训练后的qlayer的参数和计算转换为半精度浮点数格式, dtype=torch.float16
        qlayer.half()
        # Lucifer Li: SmoothQuant技术
        # real smooth and quantization
        smooth_and_quant_inplace(qlayer, args, is_llama)
        if args.epochs>0:
            # update input of quantization model
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                with traincast():
                    for j in range(args.nsamples):
                        # Lucifer Li: 使用刚刚完成训练的量化层对校准数据重新进行前向传播, 更新后的`quant_inps`将作为下一层量化校准的输入
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
            # Lucifer Li: 将量化层中计算好的scale和zero_point等关键参数注册为模型的持久化参数或缓存起来, 为后续的模型保存或真实量化推理做准备
            register_scales_and_zeros(qlayer)
            # Lucifer Li: 将处理好的量化层移动回CPU, 替换原始模型的对应层
            layers[i] = qlayer.to("cpu")
            # Lucifer Li: 提取当前量化层的状态字典, 应包含训练好的量化参数, 并存储到全局参数字典中
            omni_parameters[i] = omni_state_dict(qlayer)
            # Lucifer Li: 将包含所有已处理层量化参数的字典保存到硬盘, 防止程序中断导致进度丢失, 也便于后续直接加载这些参数进行推理
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))
        else:
            # Lucifer Li: 校准完成后, 直接注册量化参数, 然后将量化层替换回原始模型并移动回CPU
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
        # Lucifer Li: 执行真实量化
        if args.real_quant:
            assert args.wbits in [2,3,4] and args.abits >= 16   # only support weight-only quantization
            # Lucifer Li: 获取当前量化层中所有需要被量化的线性层的名称和对象
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                # Lucifer Li: 根据目标比特数和硬件支持, 创建一个真正的量化线性层对象, 该层将直接执行低精度整数计算
                if args.wbits == 3:
                    q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                else:
                    q_linear = qlinear_triton.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                # Lucifer Li: 将原始的全精度权重, 缩放因子和零点真正放入量化层中, 此过程会根据量化算法将权重转换为低比特整数格式
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                # Lucifer Li: 新创建的真实量化层`q_linear`替换掉原有层中对应的线性层
                add_new_module(name, qlayer, q_linear)
                print(f"pack quantized {name} finished")
                del module        
        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

