import os
import random
import torch
import utils
import time
import sys

import numpy as np
import torch.nn as nn

from typing import Optional
from models.LMClass import LMClass
from datautils import get_loaders
from quantize.omniquant import omniquant
from tqdm import tqdm
from pathlib import Path
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
from tasks import get_task
from dataclasses import dataclass
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from transformers.trainer_utils import get_last_checkpoint
from torch.utils.data import Dataset
from utils import (
    encode_prompt,
    count_time,
    DataCollatorWithPaddingAndNesting,
    NondiffCollator,
    Prediction
)
from metrics import calculate_metric
from trainer import QZOTrainer, QAZOTrainer, RoundZOTrainer

torch.backends.cudnn.benchmark = True


@dataclass
class OurArguments(TrainingArguments):
    task_name: str = "SQuAD"
    model_name: str = "facebook/opt-6.7b"

    # Number of examples
    num_train: int = 1000
    num_dev: int = 100
    num_eval: int = 1000

    # Number of training sets (set to None if not specified)
    num_train_sets: int = 0

    # Model loading
    load_float16: bool = True  # --load_float16
    load_bfloat16: bool = False  # 没有提到使用 bfloat16
    load_int8: bool = False  # 没有提到使用 int8
    max_length: int = 2048  # OPT 模型常用的最大长度

    # Calibration
    sfc: bool = False  # 默认值 False，脚本未提到需要 SFC 校准
    icl_sfc: bool = False  # 默认值 False，脚本未提到

    # Training
    trainer: str = "qzo"  # "qzo"  # --trainer qzo
    only_train_option: bool = True  # 设置为 True，表示只训练输入的选项部分
    train_as_classification: bool = True

    # MeZO
    zo_eps: float = 1e-3  # --zo_eps 1e-3

    # QZO Added: Training arguments
    quant_method: str = "omni"  # --quant_method omniquant
    should_save: bool = True  # 默认值为 True，保存模型权重
    clip_zo_grad: bool = True  # --clip_zo_grad
    train_unquantized: bool = False  # 脚本中未提到训练时使用未量化参数
    max_steps: int = 5000  # --max_steps 20000
    learning_rate: float = 1e-7  # --learning_rate 1e-5
    
    # Lucifer Li: 生成参数
    sampling: bool = False
    temperature: float = 1.0
    num_beams: int = 1
    top_p: float = 0.95
    top_k: int = None
    max_new_tokens: int = 50
    repetition_penalty: float = 1.2
    eos_token: str = "\n"

    # Saving
    save_model: bool = False  # 默认值 False，除非需要保存模型
    no_eval: bool = False  # 不跳过评估
    tag: str = (
        "qazo-ft-20000-16-1e-5-1e-3-0"  # tag= qzo-$MODE-$STEPS-$BS-$LR-$EPS-$SEED
    )
    save_total_limit: int = 1  # 设置保存总限制为 1，避免保存过多检查点

    # Linear probing
    linear_probing: bool = False  # 默认值 False
    lp_early_stopping: bool = False  # 默认值 False
    head_tuning: bool = False  # 默认值 False

    # Untie emb/lm_head weights
    untie_emb: bool = False  # 默认值 False

    # Display
    verbose: bool = False  # 是否显示详细输出

    # Non-diff objective
    """Lucifer Li: -注意事项- 对于SQuAD任务, non_diff应设置为True"""
    non_diff: bool = False  # 目前只支持 SQuAD 的 F1

    # Auto saving when interrupted
    save_on_interrupt: bool = False  # 默认 False，不会在中断时保存

    # Additional parameters from the script
    train_set_seed: int = 42  # SEED 设置为 train_set_seed (默认值 42)
    result_file: str = None  # 如果没有指定，默认为 None
    logging_steps: int = 10  # 脚本中设置了 --logging_steps 10
    evaluation_strategy: str = "steps"  # 脚本中设置了 --evaluation_strategy steps
    save_strategy: str = "steps"  # 脚本中设置了 --save_strategy steps
    lr_scheduler_type: str = "constant"  # 脚本中设置了 --lr_scheduler_type linear
    output_dir = "./log/wic-2.7"  # 输出目录

    # 其他参数
    model: str = "facebook/opt-2.7b"  # model name or path
    quant_seed: int = 42  # random seed
    cache_dir: str = "./cache"  # cache dir of dataset
    q_output_dir: str = "../log/"  # direction of logging file
    save_dir: Optional[str] = None  # direction for saving fake quantization model
    resume: Optional[str] = None  # resume training
    real_quant: bool = False  # real quantization for memory reduction

    # 量化相关
    calib_dataset: str = "wikitext2"  # calibration dataset
    nsamples: int = 128  # Number of calibration data samples
    batch_size: int = 1  # batch size for calibration
    wbits: int = 4  # weight bits
    abits: int = 16  # activation bits
    group_size: Optional[int] = None  # group size for quantization
    alpha: float = 0.5  # alpha parameter
    let_lr: float = 5e-3  # learnable equivalent transformation learning rate
    lwc_lr: float = 1e-2  # learnable weight clipping learning rate
    wd: float = 0.0  # weight decay
    epochs: int = 10  # number of epochs
    let: bool = False  # activate learnable equivalent transformation
    lwc: bool = False  # activate learnable weight clipping
    aug_loss: bool = False  # calculate additional loss with same input
    symmetric: bool = False  # symmetric quantization
    disable_zero_point: bool = False  # quantization without zero_point
    a_dynamic_method: str = "per_token"  # activation dynamic method
    w_dynamic_method: str = "per_channel"  # weight dynamic method

    # 评估相关
    tasks: str = ""  # evaluation tasks
    eval_ppl: bool = False  # evaluate perplexity
    num_fewshot: int = 0  # number of few-shot examples

    # 其他
    limit: int = -1  # limit number of samples
    multigpu: bool = False  # map model to multiple gpus at eval
    deactive_amp: bool = False  # deactivate AMP when 8<=bits<16
    attn_implementation: str = "eager"  # attention implementation
    net: Optional[str] = None  # network choice
    act_scales: Optional[str] = None  # activation scales file
    act_shifts: Optional[str] = None  # activation shifts file
    train: bool = False  # enable training

    delta: float = 0.1
    t: float = 0.5
    use_sum: bool = False
    train_batch_size: int = 4


def parse_args():
    # parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_quant(model, wq=True, aq=False):
    """
    设置模型的量化状态
    Args:
        model: 模型
        wq: 权重量化标志
        aq: 激活值量化标志（对于伪量化，推理时应为False）
    """
    for m in model.modules():
        if isinstance(m, QuantLinear):
            m.set_quant_state(weight_quant=wq, act_quant=aq)
        elif isinstance(m, QuantMatMul):
            m.set_quant_state(act_quant=aq)


class Framework:
    def __init__(self, args, task, model, tokenizer):
        self.args = args
        self.task = task
        self.model = model
        self.tokenizer = tokenizer

    def train(self, train_samples, eval_samples):
        # Set tokenizer to left padding (so that all the options are right aligned)
        self.tokenizer.padding_side = "left"

        class HFDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        def _convert(samples):
            data = []
            for sample in samples:
                # Lucifer Li: 编码所有候选答案, 获取每个答案的token IDs和对应的选项长度（用于后续掩码）
                # Lucifer Li: 在这种传入方式下, 实际上`encoded_candidates`和`option_lens`中均只包含一个元素
                encoded_candidates, option_lens = encode_prompt(
                    self.task,
                    self.task.get_template(),
                    [],  # train_samples
                    sample,  # eval_sample
                    self.tokenizer,
                    self.args.max_length,
                    generation=self.task.generation,
                    generation_with_gold=True,
                    max_new_tokens=self.args.max_new_tokens,
                )

                # 确定正确答案在候选答案列表中的索引
                if self.task.generation:
                    # 生成任务：正确答案总是第一个候选（索引0）
                    correct_candidate_id = 0
                elif isinstance(sample.correct_candidate, list):
                    # 正确答案是列表：取第一个元素在候选列表中的索引
                    correct_candidate_id = sample.candidates.index(
                        sample.correct_candidate[0]
                    )
                else:
                    # 正确答案是单个值：直接查找其在候选列表中的索引
                    correct_candidate_id = sample.candidates.index(
                        sample.correct_candidate
                    )

                # Lucifer Li: 非可微目标处理: 从正确答案编码中移除答案部分，只保留问题上下文
                # Lucifer Li: 这用于需要单独计算指标(如F1)而不依赖可微损失的情况
                if self.args.non_diff:
                    encoded_candidates[correct_candidate_id] = encoded_candidates[
                        correct_candidate_id
                    ][:-option_lens[correct_candidate_id]]

                # 分类训练模式：将问题转换为多分类任务
                if self.args.train_as_classification:
                    # 为每个候选答案创建一个数据项，标签是正确答案的索引
                    # 模型需要从所有候选中选择正确答案（类似多项选择题）
                    sample_data = []
                    for _i in range(len(encoded_candidates)):
                        sample_data.append(
                            {
                                "input_ids": encoded_candidates[
                                    _i
                                ],  # 候选答案的完整编码
                                "labels": correct_candidate_id,  # 正确答案索引作为分类标签
                                "option_len": option_lens[
                                    _i
                                ],  # 当前选项的长度（用于掩码）
                                "num_options": len(sample.candidates),  # 总选项数量
                            }
                        )
                    data.append(sample_data)
                # 仅训练选项模式：专注于训练模型生成正确答案的能力
                elif self.args.only_train_option:
                    if self.args.non_diff:
                        """
                        Lucifer Li: -潜在问题-
                        非可微目标模式
                        `encoded_candidates`是直接由`tokenizer.encode`出的结果, 除`input_ids`外还可能包含`attention_mask`等其他字段
                        这种赋值可能过于笼统
                        """
                        data.append(
                            {
                                "input_ids": encoded_candidates[
                                    correct_candidate_id
                                ],  # 正确答案的输入编码
                                "labels": encoded_candidates[
                                    correct_candidate_id
                                ],  # 标签与输入相同
                                "option_len": option_lens[
                                    correct_candidate_id
                                ],  # 答案部分长度
                                "gold": sample.correct_candidate,  # 原始正确答案文本
                            }
                        )
                    else:
                        # 标准训练：使用教师强制，让模型学习生成正确答案
                        data.append(
                            {
                                "input_ids": encoded_candidates[
                                    correct_candidate_id
                                ],  # 正确答案的输入编码
                                "labels": encoded_candidates[
                                    correct_candidate_id
                                ],  # 标签与输入相同（自回归训练）
                                "option_len": option_lens[
                                    correct_candidate_id
                                ],  # 答案部分长度
                            }
                        )
                # 标准语言模型训练模式：最简单的序列到序列训练
                else:
                    data.append(
                        {
                            "input_ids": encoded_candidates[
                                correct_candidate_id
                            ],  # 输入序列
                            "labels": encoded_candidates[
                                correct_candidate_id
                            ],  # 目标序列（与输入相同）
                        }
                    )

            # Lucifer Li: 字典列表, 每个元素对应一个样本, 包含"input_ids", "labels"等字段
            return data

        # Lucifer Li: 计时处理训练样本的编码
        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(_convert(train_samples))
            eval_dataset = HFDataset(_convert(eval_samples))

        # Lucifer Li: 选择合适的数据整理器, 在训练过程中将多个样本批量整理成模型可处理的格式
        # Lucifer Li: args.non_diff为True时, 仅支持SQuAD任务(F1)
        if self.args.non_diff:
            collator = NondiffCollator  # 非差分模式, 用于不可微分的处理方式
        else:
            collator = DataCollatorForTokenClassification  # 标准模式

        if self.args.trainer == "qzo" and self.args.quant_method == "omni":
            trainer = QZOTrainer(
                model=self.model,
                args=self.args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                # 对于SQuAD任务, data_collator === NondiffCollator, 会把数据包装为字典格式
                data_collator=(
                    DataCollatorWithPaddingAndNesting(
                        self.tokenizer, pad_to_multiple_of=8
                    )
                    if self.args.train_as_classification
                    else collator(self.tokenizer, pad_to_multiple_of=8)
                ),
            )
        elif self.args.trainer == "qazo" and self.args.quant_method == "omni":
            trainer = QAZOTrainer(
                model=self.model,
                args=self.args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                # 对于SQuAD任务, data_collator === NondiffCollator, 会把数据包装为字典格式
                data_collator=(
                    DataCollatorWithPaddingAndNesting(
                        self.tokenizer, pad_to_multiple_of=8
                    )
                    if self.args.train_as_classification
                    else collator(self.tokenizer, pad_to_multiple_of=8)
                ),
            )
        elif (self.args.trainer == 'STE' or self.args.trainer == 'HTGE' or self.args.trainer == 'Uniform' or self.args.trainer == 'Normal' or self.args.trainer == 'Laplace') and self.args.quant_method != '':
            assert self.args.quant_method in ['gptq', 'omni','aqlm']            
            from transformers import Trainer
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    pass
            for name, param in self.model.named_parameters():
                param.requires_grad = False
            for name, param in self.model.named_parameters():
                if any(keyword in name.lower() for keyword in ['descale']):
                    param.requires_grad = True
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    pass
            
            if torch.cuda.device_count() > 1:
                from accelerate import infer_auto_device_map, dispatch_model
                block_class_name = self.model.model.layers[0].__class__.__name__
                device_map = infer_auto_device_map(
                    self.model, 
                    max_memory={i: "13GiB" for i in range(torch.cuda.device_count())}, 
                    no_split_module_classes=[block_class_name]
                )
                self.model = dispatch_model(self.model, device_map=device_map, skip_keys="past_key_values")
                
            trainer = RoundZOTrainer(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8)
            )
            trainable_params = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    num_params = param.numel()
                    trainable_params += num_params
            print(f"trainable parameters:{trainable_params}")
        else:
            raise NotImplementedError

        # Lucifer Li: 开始训练
        if self.args.max_steps > 0:
            trainer.train(resume_from_checkpoint=None)

        self.model = trainer.model

    def forward(self, input_ids, option_len=None, generation=False):
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        if generation:
            args = self.args
            # print(f"\n{'#'*60}")
            # print(f"[Framework.forward] Calling model.generate()")
            # print(f"[Framework.forward] input_ids.shape: {input_ids.shape}")
            # print(f"[Framework.forward] input_ids length: {input_ids.size(1)}")
            # print(f"[Framework.forward] max_new_tokens: {min(args.max_new_tokens, args.max_length - input_ids.size(1))}")
            # print(f"{'#'*60}\n")
            # Lucifer Li: 使用贪婪解码或beam search，避免sampling和beam search冲突
            outputs = self.model.generate(
                input_ids,
                do_sample=args.sampling,  # 使用贪婪解码或beam search
                temperature=args.temperature,
                num_beams=args.num_beams,  # 使用参数中的beam数量
                top_p=args.top_p,  # beam search时不使用nucleus sampling
                top_k=args.top_k,  # beam search时不使用top-k
                repetition_penalty=args.repetition_penalty,  # 使用参数中的重复惩罚
                max_new_tokens=min(
                    args.max_new_tokens, args.max_length - input_ids.size(1)
                ),
                num_return_sequences=1,
                eos_token_id=[
                    self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1],
                    self.tokenizer.eos_token_id,
                ],
                pad_token_id=self.tokenizer.pad_token_id,
            )
            output_text = self.tokenizer.decode(
                outputs[0][input_ids.size(1) :], skip_special_tokens=True
            ).strip()
            return output_text

    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        verbose = verbose or self.args.verbose
        if verbose:
            print("========= Example =========")
            print(f"Candidate: {eval_sample.candidates}")
            print(f"Correct candidate: {eval_sample.correct_candidate}")

        encoded_candidates, option_lens = encode_prompt(
            self.task,
            self.task.get_template(),
            train_samples,
            eval_sample,
            self.tokenizer,
            max_length=self.args.max_length,
            generation=self.task.generation,
            max_new_tokens=self.args.max_new_tokens,
        )

        # print(f"encoded_candidates: {encoded_candidates}")

        outputs = []
        if self.task.generation:
            output_text = self.forward(encoded_candidates[0], generation=True)
            if verbose:
                print("=== Prompt ===")
                print(self.tokenizer.decode(encoded_candidates[0]))
                print(f"Output: {output_text}")
            return Prediction(
                correct_candidate=eval_sample.correct_candidate,
                predicted_candidate=output_text,
            )

    def evaluate(
        self, train_samples, eval_samples, one_train_set_per_eval_sample=False
    ):
        if one_train_set_per_eval_sample:
            print(
                f"There are {len(eval_samples)} validation samples and one train set per eval sample"
            )
        else:
            print(
                f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples"
            )

        predictions = []
        for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
            predictions.append(
                self.one_step_pred(
                    (
                        train_samples[eval_id]
                        if one_train_set_per_eval_sample
                        else train_samples[0]
                    ),
                    eval_sample,
                    verbose=(eval_id < 10),
                )
            )

        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        return metrics


def main():
    start_time = time.time()  # 记录开始时间 
    args = parse_args()
    set_seed(args.quant_seed)

    # check
    if args.epochs > 0:
        assert args.lwc or args

    # init logger
    if args.q_output_dir:
        Path(args.q_output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    q_output_dir = Path(args.q_output_dir)
    logger = utils.create_logger(q_output_dir)
    logger.info(args)

    if args.net is None:
        args.net = args.model.split("/")[-1]
    args.model_family = args.net.split("-")[0]

    lm = LMClass(args)
    lm.seqlen = 2048
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc": args.lwc,
        "disable_zero_point": args.disable_zero_point,
    }
    args.act_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }

    if args.trainer == 'HTGE':
        quant_param_dicts = [
            args.weight_quant_params,
            args.act_quant_params,
            args.q_quant_params,
            args.k_quant_params,
            args.v_quant_params,
            args.p_quant_params
        ]
        for param_dict in quant_param_dicts:
            param_dict['t'] = args.t
            param_dict['method'] = args.trainer
            # param_dict['use_sum'] = args.use_sum

    if args.trainer == 'Uniform' or args.trainer == 'Normal' or args.trainer == 'Laplace':
        quant_param_dicts = [
            args.weight_quant_params,
            args.act_quant_params,
            args.q_quant_params,
            args.k_quant_params,
            args.v_quant_params,
            args.p_quant_params
        ]
        for param_dict in quant_param_dicts:
            param_dict['delta'] = args.delta
            param_dict['method'] = args.trainer
            param_dict['use_sum'] = args.use_sum

    if args.act_scales is None:
        args.act_scales = f"./act_scales/{args.net}.pt"
    if args.act_shifts is None:
        args.act_shifts = f"./act_shifts/{args.net}.pt"

    if args.wbits < 16 or args.abits < 16:
        logger.info("=== start quantization ===")
        tick = time.time()
        cache_dataloader = f"{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache"
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            logger.info(f"load calibration from {cache_dataloader}")
        else:
            dataloader, _ = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=args.quant_seed,
                model=args.model,
                seqlen=lm.seqlen,
            )
            torch.save(dataloader, cache_dataloader)
        act_scales = None
        act_shifts = None
        if args.let:
            print("load act scales and shifts from ", args.act_scales, args.act_shifts)
            act_scales = torch.load(args.act_scales, weights_only=False)
            act_shifts = torch.load(args.act_shifts, weights_only=False)
        omniquant(
            lm,
            args,
            dataloader,
            act_scales,
            act_shifts,
            logger,
        )
        logger.info(f"===== Quant Time =====")
        logger.info(time.time() - tick)
    else:
        print("=" * 100)
        print("Passing quantization process!")
        print("=" * 100)

    # if "opt" in args.model.lower():
    #     lm.tokenizer.bos_token_id = 0
    if "llama" in args.model.lower():
        lm.tokenizer.pad_token_id = 0

    set_seed(42)
    task = get_task(args.task_name)
    train_sets = task.sample_train_sets(
        num_train=args.num_train,
        num_dev=args.num_dev,
        num_eval=args.num_eval,
        num_train_sets=args.num_train_sets,  # 0
        seed=args.train_set_seed,  # 42
    )

    # Lucifer Li: 根据是否使用真实量化来决定量化状态
    # 如果使用真实量化(real_quant=True)，需要在推理时动态量化权重
    # 如果使用伪量化(real_quant=False)，权重在训练后已被永久量化，不需要再次量化
    # 激活值量化：由于args.abits=16（未量化），推理时应禁用激活值量化(aq=False)
    if args.real_quant:
        # 真实量化：推理时动态量化权重，禁用激活值量化
        enable_quant(lm.model, wq=True, aq=False)
    else:
        # 伪量化：权重已被fake quantize，禁用所有量化（直接使用已量化的权重）
        enable_quant(lm.model, wq=False, aq=False)

    framework = Framework(args, task, lm.model, lm.tokenizer)

    if args.train:
        # Lucifer Li: 其中每个train_samples都是一个训练子集, 每个子集中的元素都是Sample对象, 包含id, data, correct_candidate, candidates
        for train_set_id, train_samples in enumerate(train_sets):
            train_set_seed = (
                train_set_id if args.train_set_seed is None else args.train_set_seed
            )

            # Lucifer Li: args.num_eval为评估样本数量, 默认存在且设置为1000
            if args.num_eval is not None:
                eval_samples = task.sample_subset(
                    data_split="valid", seed=train_set_seed, num=args.num_eval
                )
            else:
                # eval_samples > Dataset.samples["valid"]
                eval_samples = task.valid_samples

            # Lucifer Li: args.num_dev为验证样本数量, 默认存在且设置为100
            if args.num_dev is not None:
                # Lucifer Li: 划分数据集, 从train_samples中提取最后args.num_dev个样本作为验证样本
                dev_samples = train_samples[-args.num_dev:]
                train_samples = train_samples[:-args.num_dev]
            else:
                dev_samples = None

            # ====================================================================================================
            # Lucifer Li: train_samples, dev_samples, eval_samples均为Sample列表, 默认数量分别为1000, 100, 1000
            # ====================================================================================================

            # Lucifer Li: Framework.train() -> (QZO)Trainer.train() -> _inner_training_loop()
            framework.train(
                train_samples,
                dev_samples if dev_samples is not None else eval_samples,
            )
            # Lucifer Li: 训练完成后, 获取训练后的模型
            lm.model = framework.model
    else:
        print("=" * 100)
        print("Passing training process!")
        print("=" * 100)

    if args.num_eval is not None:
        eval_samples = task.sample_subset(
            data_split="valid", seed=0, num=args.num_eval
        )
    else:
        eval_samples = task.valid_samples

    metrics = framework.evaluate(
        train_sets, eval_samples, one_train_set_per_eval_sample=False
    )
    logger.info(metrics)

    # evaluate(lm, args, logger)

    # 【新增】记录执行时间并输出到日志
    end_time = time.time()
    execution_time = end_time - start_time
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = execution_time % 60
    logger.info(f"\n===== Execution Time =====")
    logger.info(f"Total: {execution_time:.2f} seconds")
    logger.info(f"Format: {hours}h {minutes}m {seconds:.2f}s")
    logger.info(f"==========================")

    # ============================================================================
    # ldx:add: 绘制 Loss 和 Grad_Norm 曲线图
    # ============================================================================
    def plot_training_metrics(output_dir):
        """读取 loss_history.json 并绘制训练指标曲线"""
        
        loss_history_path = os.path.join(output_dir, "loss_history.json")
        
        if not os.path.exists(loss_history_path):
            logger.warning(f"Loss history file not found: {loss_history_path}")
            return
        
        # 读取 JSON 文件
        with open(loss_history_path, 'r') as f:
            history_data = json.load(f)
        
        # 提取 train 部分的数据
        train_data = history_data.get("train", [])
        
        if not train_data:
            logger.warning("No training data found in loss_history.json")
            return
        
        # 提取 global_step, loss, grad_norm
        steps = [item["global_step"] for item in train_data]
        losses = [item["loss"] for item in train_data]
        grad_norms = [item["grad_norm"] for item in train_data]
        
        # 创建画布，两个子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # === 子图 1: Loss 曲线 ===
        ax1.plot(steps, losses, 'b-', linewidth=1.5, label='Loss')
        ax1.set_xlabel('Global Step', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss vs Global Step', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_axisbelow(True)
        
        # 添加 Loss 统计信息
        loss_min = min(losses)
        loss_max = max(losses)
        loss_avg = sum(losses) / len(losses)
        loss_final = losses[-1]
        ax1.text(0.02, 0.98, 
                f'Min: {loss_min:.4f}\nMax: {loss_max:.4f}\nAvg: {loss_avg:.4f}\nFinal: {loss_final:.4f}',
                transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # === 子图 2: Grad_Norm 曲线 ===
        ax2.plot(steps, grad_norms, 'r-', linewidth=1.5, label='Grad Norm')
        ax2.set_xlabel('Global Step', fontsize=12)
        ax2.set_ylabel('Gradient Norm', fontsize=12)
        ax2.set_title('Gradient Norm vs Global Step', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_axisbelow(True)
        
        # 添加 Grad_Norm 统计信息
        grad_min = min(grad_norms)
        grad_max = max(grad_norms)
        grad_avg = sum(grad_norms) / len(grad_norms)
        grad_final = grad_norms[-1]
        ax2.text(0.02, 0.98, 
                f'Min: {grad_min:.4f}\nMax: {grad_max:.4f}\nAvg: {grad_avg:.4f}\nFinal: {grad_final:.4f}',
                transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(output_dir, "training_metrics.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Training metrics plot saved to: {save_path}")
        
        # 显示图片（可选，服务器环境建议注释掉）
        # plt.show()
        
        plt.close()

    # 在 main() 函数末尾调用
    if args.train and not args.no_eval:
        plot_training_metrics(args.output_dir)

if __name__ == "__main__":
    print(sys.argv)
    main()
