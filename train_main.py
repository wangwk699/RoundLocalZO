import os
import sys
import random
import numpy as np
from models.LMClass import LMClass
import torch
import time
from datautils import get_loaders
from lm_eval import evaluator
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from quantize.omniquant import omniquant
from tqdm import tqdm
import utils
from pathlib import Path
from categories import subcategories, categories

from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from quantize.int_linear import QuantLinear
from quantize.quantizer import UniformAffineQuantizer

import pdb
import json
import matplotlib.pyplot as plt

import argparse
from tasks import get_task
from dataclasses import dataclass
from transformers import HfArgumentParser, TrainingArguments, DataCollatorForTokenClassification
from utils import *
from trainer import ZOTrainer, QZOTrainer, QAZOTrainer, RoundZOTrainer
from torch.utils.data import Dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from metrics import calculate_metric
from accelerate import Accelerator
from types import SimpleNamespace
torch.backends.cudnn.benchmark = True


net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    "llama-7b",
    "llama-13b",
    "llama-30b",
    "llama-65b",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "Llama-2-7b-chat",
    "Llama-2-13b-chat",
    "llava-llama-2-13b-chat-lightning-preview",
    "falcon-180b",
    "falcon-7b",
    "mixtral-8x7b"
    "Qwen3-8B"
    "Qwen2-7B"
]

#chents_train
@dataclass
class OurArguments(TrainingArguments):
    # dataset and sampling strategy
    task_name: str = "WIC"  # 根据你的任务，可以替换为实际任务名 (如 CB, Copa, ReCoRD, 等)[SST2,BoolQ，RTE,CB]

    # Number of examples
    num_train: int = 1000  # TRAIN=1000
    num_dev: int = 100  # DEV=100 (对于任务如 Copa，训练集小于1000样本时设置为100)
    num_eval: int = 1000  # EVAL=1000

    # Number of training sets (set to None if not specified)
    num_train_sets: int = None  # 默认值 None，根据脚本逻辑，未设置时为 None

    # Model loading
    model_name: str = "facebook/opt-2.7b"  # 根据你的脚本 MODEL=facebook/opt-1.3b
    load_float16: bool = True  # --load_float16
    load_bfloat16: bool = False  # 没有提到使用 bfloat16
    load_int8: bool = False  # 没有提到使用 int8
    max_length: int = 2048  # OPT 模型常用的最大长度

    # Calibration
    sfc: bool = False  # 默认值 False，脚本未提到需要 SFC 校准
    icl_sfc: bool = False  # 默认值 False，脚本未提到

    # Training
    trainer: str = "qazo"#"qzo"  # --trainer qzo
    only_train_option: bool = True  # 设置为 True，表示只训练输入的选项部分
    train_as_classification: bool = True  
    
    # MeZO
    zo_eps: float = 1e-3  # --zo_eps 1e-3

    # QZO Added: Training arguments
    quant_method: str = 'omni'  # --quant_method omniquant
    should_save: bool = True  # 默认值为 True，保存模型权重
    clip_zo_grad: bool = True  # --clip_zo_grad
    train_unquantized: bool = False  # 脚本中未提到训练时使用未量化参数
    max_steps: int = 5000 # --max_steps 20000
    learning_rate: float = 1e-7  # --learning_rate 1e-5
    # Generation
    sampling: bool = False  # 脚本中未提到使用采样
    temperature: float = 1.0  # 默认温度值
    num_beams: int = 1  # 默认设置为 1
    top_k: int = None  # 默认情况下未指定
    top_p: float = 0.95  # 默认 top-p
    max_new_tokens: int = 50  # 默认最大生成新 token 数量
    eos_token: str = "\n"  # 默认结束标志为换行符

    # Saving
    save_model: bool = False  # 默认值 False，除非需要保存模型
    no_eval: bool = False  # 不跳过评估
    tag: str = "qazo-ft-20000-16-1e-5-1e-3-0"  # tag= qzo-$MODE-$STEPS-$BS-$LR-$EPS-$SEED
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
    non_diff: bool = False  # 目前只支持 SQuAD 的 F1

    # Auto saving when interrupted
    save_on_interrupt: bool = False  # 默认 False，不会在中断时保存

    # Additional parameters from the script
    train_set_seed: int = 42  # SEED 设置为 train_set_seed (默认值 0)
    result_file: str = None  # 如果没有指定，默认为 None
    logging_steps: int = 125  # 脚本中设置了 --logging_steps 10   logging_steps = 1000 // BATCH_SIZE
    evaluation_strategy: str = "steps"  # 脚本中设置了 --evaluation_strategy steps
    save_strategy: str = "steps"  # 脚本中设置了 --save_strategy steps
    lr_scheduler_type: str = "constant"  # 脚本中设置了 --lr_scheduler_type linear
    output_dir = "./log/wic-2.7"  # 输出目录

    # 其他参数
    model: str = "facebook/opt-2.7b"  # model name or path
    quant_seed : int = 42  # random seed
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
    let_lr: float = 0  # learnable equivalent transformation learning rate
    lwc_lr: float = 1e-2  # learnable weight clipping learning rate
    wd: float = 0.0  # weight decay
    epochs: int = 10  # number of epochs
    let: bool = False  # activate learnable equivalent transformation
    lwc: bool = False  # activate learnable weight clipping
    aug_loss: bool = False  # calculate additional loss with same input
    # symmetric: bool = False  # symmetric quantization
    symmetric: bool = True  # symmetric quantization
    # disable_zero_point: bool = False  # quantization without zero_point
    disable_zero_point: bool = True  # quantization without zero_point
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

    # ldx:add
    delta: float = 0.1
    t: float = 0.5
    use_sum: bool = False
    train_batch_size: int = 4
    

# ldx:add:
# 在 quantize/utils.py 中添加

def set_quant_layers_training_phase(model, phase: str):
    for module in model.modules():
        if hasattr(module, 'set_training_phase'):
            module.set_training_phase(phase)

def parse_args():
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print(args)
    return args

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def result_file_tag(args):
    """
    Get the result file tag
    """
    save_model_name = args.model_name.split("/")[-1]
    sfc_tag = "-sfc" if args.sfc else ""
    icl_sfc_tag = "-icl_sfc" if args.icl_sfc else ""
    sample_eval_tag = "-sampleeval%d" % args.num_eval if args.num_eval is not None else ""
    sample_train_tag = "-ntrain%d" % args.num_train if args.num_train > 0 else ""
    sample_dev_tag = "-ndev%d" % args.num_dev if args.num_dev is not None else ""
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    return f"{args.task_name}-{save_model_name}" + sfc_tag + icl_sfc_tag + sample_eval_tag + sample_train_tag + sample_dev_tag + customized_tag

class Framework:
    def __init__(self, args, task, model,tokenizer):
        self.args = args
        self.task = task
        self.model, self.tokenizer = model,tokenizer
    def train(self, train_samples, eval_samples):
        """
        Training function
        """
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
            """
            Convert samples to HF-compatible dataset
            """
            data = []
            for sample in samples:
                encoded_candidates, option_lens = encode_prompt(
                    self.task, self.task.get_template(), [], sample, self.tokenizer, 
                    max_length=self.args.max_length, generation=self.task.generation, generation_with_gold=True, 
                    max_new_tokens=self.args.max_new_tokens
                )
                if self.task.generation:
                    correct_candidate_id = 0
                elif isinstance(sample.correct_candidate, list):
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
                else:
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate)
                
                if self.args.non_diff:
                    # For non-differentiable objective, there is no teacher forcing thus the 
                    # current answer part is removed
                    encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][:-option_lens[correct_candidate_id]]

                if self.args.train_as_classification:
                    # For classification, we provide the label as the correct candidate id
                    data.append([{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id, "option_len": option_lens[_i], "num_options": len(sample.candidates)} for _i in range(len(encoded_candidates))])
                elif self.args.only_train_option:
                    # Otherwise, it is just LM-style teacher forcing
                    if self.args.non_diff:
                        # For non-differentiable objective, we need to provide the gold answer to calculate F1/acc
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id], "gold": sample.correct_candidate})
                    else:
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id]})
                else:
                    data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id]})
            return data

        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(_convert(train_samples))
            eval_dataset = HFDataset(_convert(eval_samples))
        
        # 如果使用了 --only_train_option 且没有非可微目标函数，我们会对前向函数进行封装
        if self.args.only_train_option and not self.args.non_diff:
            # If --only_train_option and not with a non-differentiable objective, we wrap the forward function
            self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))

        if self.args.non_diff:
            collator = NondiffCollator
        else:
            collator = DataCollatorForTokenClassification

        if self.args.trainer == 'zo': # MeZO
            trainer = ZOTrainer(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8)
            )
        elif self.args.trainer == 'regular': # Fine-tune (this is not used in our experiments)
            from transformers import Trainer
            trainer = Trainer(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8)
            )
        # QZO Added: set QZOTrainer
        elif self.args.trainer == 'qzo' and self.args.quant_method != '': 
            assert self.args.quant_method in ['gptq', 'omni','aqlm'] # supported methods
            trainer = QZOTrainer(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8)
            )
        elif self.args.trainer == 'qazo' and self.args.quant_method != '': 
            assert self.args.quant_method in ['gptq', 'omni','aqlm'] # supported methods
            trainer = QAZOTrainer(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8)
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
            
    
            from accelerate import infer_auto_device_map, dispatch_model
            block_class_name = self.model.model.layers[0].__class__.__name__
            device_map = infer_auto_device_map(self.model, max_memory={i: "15GiB" for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
            self.model = dispatch_model(self.model, device_map=device_map, skip_keys='past_key_values')
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
            raise NotImplementedError()

        if self.args.save_on_interrupt:
            trainer.add_callback(SIGUSR1Callback())

        # Resume training from a last checkpoint
        last_checkpoint = None
        from transformers.trainer_utils import get_last_checkpoint
        if os.path.isdir(self.args.output_dir) and not self.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
        if last_checkpoint is not None and self.args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        if self.args.resume_from_checkpoint is not None:
            last_checkpoint = self.args.resume_from_checkpoint
        #add
        if self.args.max_steps > 0:
            trainer.train(resume_from_checkpoint=last_checkpoint) 
        else:
            accelerator = Accelerator()
            self.model, _ = accelerator.prepare(self.model, torch.optim.AdamW(self.model.parameters()))

        # Explicitly save the model
        if self.args.save_model:
            logger.warn("Save model..")
            trainer.save_model()
        
        # ldx:add:record loss
        loss_history = trainer.get_loss_history()
        loss_df = trainer.get_loss_history(as_dataframe=True)
        trainer.plot_loss_curve(f"{self.args.output_dir}/loss_curve.png", show=True)
        # trainer.plot_learning_rate(f"{self.args.output_dir}/lr_curve.png", show=True)
        trainer.print_loss_summary()
        trainer.save_loss_history()
        
        # FSDP compatibility
        self.model = trainer.model 
        
        # Reset the forward function for evaluation
        if self.args.only_train_option and not self.args.non_diff:
            if type(self.model) == FSDP:
                logger.info("This is an FSDP model now. Be careful when assigning back the original forward function")
                self.model._fsdp_wrapped_module.forward = self.model._fsdp_wrapped_module.original_forward
            else:
                self.model.forward = self.model.original_forward
    def forward(self, input_ids, option_len=None, generation=False):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        if generation:
            args = self.args
            # Autoregressive generation
            outputs = self.model.generate(
                input_ids, do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)), 
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1], self.tokenizer.eos_token_id],
            )
            # For generation, directly return the text output
            output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.model.eval()
                logits = self.model(input_ids=input_ids).logits
            labels = input_ids[0, 1:]
            logits = logits[0, :-1] 
            log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[torch.arange(len(labels)).to(log_probs.device), labels.to(log_probs.device)]
            #selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()
            # Only return the option (candidate) part
            return selected_log_probs[-option_len:]
    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        """
        Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
        """

        verbose = verbose or self.args.verbose
        if verbose:
            logger.info("========= Example =========")
            logger.info(f"Candidate: {eval_sample.candidates}")
            logger.info(f"Correct candidate: {eval_sample.correct_candidate}")


        # Encode (add prompt and tokenize) the sample; if multiple-choice/classification, encode all candidates (options)
        encoded_candidates, option_lens = encode_prompt(
            self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length, 
            generation=self.task.generation, max_new_tokens=self.args.max_new_tokens
        )

        # Calibration
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(self.task, self.task.get_template(), 
                train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length,
                sfc=self.args.sfc, icl_sfc=self.args.icl_sfc, generation=self.task.generation, 
                max_new_tokens=self.args.max_new_tokens
            )

        outputs = []
        if self.task.generation:
            # For generation tasks, return the autoregressively-generated text
            output_text = self.forward(encoded_candidates[0], generation=True)
            if verbose:
                logger.info("=== Prompt ===")
                logger.info(self.tokenizer.decode(encoded_candidates[0]))
                logger.info(f"Output: {output_text}") 
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            # For classification/multiple-choice, calculate the probabilities of all candidates
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])
                if verbose:
                    if candidate_id == 0:
                        logger.info("=== Candidate %d ===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate))
                    else:
                        logger.info("=== Candidate %d (without context)===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate).split(self.task.train_sep)[-1])
                    logger.info(f"Log probabilities of the option tokens: {selected_log_probs}")

                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.forward(sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id])
                    if verbose:
                        logger.info("=== Candidate %d (without context) SFC ===" % candidate_id)
                        logger.info(self.tokenizer.decode(sfc_encoded_candidates[candidate_id]).split(self.task.train_sep)[-1])
                        logger.info(f"Log probabilities of the option tokens: {sfc_selected_log_probs}")

                outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None})

            if self.args.sfc or self.args.icl_sfc:
                # Calibrated probabilities (surface form competition; https://arxiv.org/pdf/2104.08315.pdf)
                # log p(candidate | input) = log p_lm(candidate | input) - log p_lm(candidate | sfc prompt)
                scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
            else:
                # (Default) length-normalized log probabilities
                # log p(candidate | input) = log p_lm(candidate | input) / |candidate #tokens|
                scores = [x['log_probs'].mean().item() for x in outputs]


            if verbose:
                logger.info(f"Prediction scores: {scores}")

            if isinstance(eval_sample.correct_candidate, list):
                # For some datasets there are multiple correct answers
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))


    def evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False):
        """
        Evaluate function. If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
        """
        if one_train_set_per_eval_sample:
            logger.info(f"There are {len(eval_samples)} validation samples and one train set per eval sample")
        else:
            logger.info(f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples")

        # Prediction loop
        predictions = []  
        for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
            predictions.append(
                self.one_step_pred(train_samples[eval_id] if one_train_set_per_eval_sample else train_samples, eval_sample, verbose=(eval_id < 3))
            )

        # Calculate metrics 
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        return metrics

def main():
    start_time = time.time()  # 记录开始时间    
    args = parse_args()
    random.seed(args.quant_seed)
    np.random.seed(args.quant_seed)
    torch.manual_seed(args.quant_seed)
    torch.cuda.manual_seed(args.quant_seed)

    # check
    if args.epochs > 0:
        assert args.lwc or args.let
        
    if (args.wbits<16 and args.wbits>=8) or (args.abits<16 and args.abits>=8):
        args.deactive_amp = True

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
    
    # load model
    if args.net is None:
        args.net = args.model.split('/')[-1]
    # assert args.net in net_choices
    args.model_family = args.net.split('-')[0]

    # 1. 加载原始模型
    lm = LMClass(args)
    lm.seqlen = 2048
    lm.model.eval()
    # 冻结参数
    # for param in lm.model.parameters():
    #     param.requires_grad = False
    for name, param in lm.model.named_parameters():
        # print(f"{name}")
        param.requires_grad = False

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc":args.lwc,
        "disable_zero_point": args.disable_zero_point
    }
    args.act_quant_params = {
        "n_bits":  args.abits,
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



    if args.multigpu:
        gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
        lm._device = f"cuda:{gpu_id}"
        logger.info(f"set quantization in gpu {gpu_id}")

    # act scales and shifts
    if args.act_scales is None:
        args.act_scales = f'./act_scales/{args.net}.pt'
    if args.act_shifts is None:
        args.act_shifts = f'./act_shifts/{args.net}.pt'

    # quantization
    if args.wbits < 16 or args.abits <16:
        logger.info("=== start quantization ===")
        tick = time.time()     
        # load calibration dataset
        cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache'
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
            #act_scales = torch.load(args.act_scales)
            #act_shifts = torch.load(args.act_shifts)
            print("load act scales and shifts from ", args.act_scales, args.act_shifts)
            act_scales = torch.load(args.act_scales,weights_only=False)
            act_shifts = torch.load(args.act_shifts,weights_only=False)
        # 2. 量化校准（引入 LWC/LET 参数）
        omniquant(
            lm,
            args,
            dataloader,
            act_scales,
            act_shifts,
            logger,
        )
        logger.info(time.time() - tick)
        # for name, param in lm.model.named_parameters():
        #     # print(f"{name}")
        #     param.requires_grad = False

    # 3. 清理辅助参数 + 保存（本代码段）
    if args.save_dir:
        # delete omni parameters
        for name, module in lm.model.named_modules():
            if isinstance(module, QuantLinear):
                del module.weight_quantizer.lowbound_factor
                del module.weight_quantizer.upbound_factor
            if isinstance(module,QuantLlamaDecoderLayer) or isinstance(module,QuantOPTDecoderLayer):
                if args.let:
                    del module.qkv_smooth_scale
                    del module.qkv_smooth_shift
                    del module.out_smooth_scale
                    del module.out_smooth_shift
                    del module.fc1_smooth_scale
                    del module.fc1_smooth_shift           
        lm.model.save_pretrained(args.save_dir)  
        lm.tokenizer.save_pretrained(args.save_dir) 
    #chents_add
    if 'opt' in args.model.lower():
        lm.tokenizer.bos_token_id = 0
    # ldx:add for llama
    if "llama" in args.model.lower():
        lm.tokenizer.pad_token_id = 0 # technically <unk>
    #chents_train
    #train_args = parse_args()
    set_seed(args.seed)
    task = get_task(args.task_name)
    train_sets = task.sample_train_sets(num_train=args.num_train, num_dev=args.num_dev, num_eval=args.num_eval, num_train_sets=args.num_train_sets, seed=args.train_set_seed)
    #evaluate(lm, args,logger)
    def enable_quant(model, wq=True):
        for m in model.modules():
            if isinstance(m, QuantLinear):
                m.set_quant_state(weight_quant=wq)
    enable_quant(lm.model, wq=True)  
    # lm.model = 
    framework = Framework(args, task, lm.model, lm.tokenizer)
    if args.train_set_seed is not None or args.num_train_sets is not None:
        # Eval samples share one (or multiple) training set(s)
        for train_set_id, train_samples in enumerate(train_sets):
            train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

            # Sample eval samples
            if args.num_eval is not None:
                eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
            else:
                eval_samples = task.valid_samples

            if args.trainer != "none":
                if args.num_dev is not None:
                    # Dev samples
                    dev_samples = train_samples[-args.num_dev:] 
                    train_samples = train_samples[:-args.num_dev]
                else:
                    dev_samples = None

                # Training  # 3. 可选：RoundLocalZO 微调（进一步优化舍入策略）
                if args.train:
                    framework.train(train_samples, dev_samples if dev_samples is not None else eval_samples)
                lm.model = framework.model  # Update the model after training
                #evaluate(lm, args,logger)
                #exit(0)
                if not args.no_eval:
                    metrics = framework.evaluate([], eval_samples) # No in-context learning if there is training
                    if dev_samples is not None:
                        dev_metrics = framework.evaluate([], dev_samples) 
                        for m in dev_metrics:
                            metrics["dev_" + m] = dev_metrics[m]
            else:
                assert args.num_dev is None
                # Zero-shot / in-context learning
                metrics = framework.evaluate(train_samples, eval_samples)

            if not args.no_eval:
                logger.info("===== Train set %d =====" % train_set_seed)
                logger.info(metrics)
                if args.local_rank <= 0:
                    write_metrics_to_file(metrics, "result/" + result_file_tag(args) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file)

    else:
        # For each eval sample, there is a training set. no training is allowed  对于每个评估样本，均对应一个训练集。不允许进行任何训练。
        # This is for in-context learning (ICL)
        assert args.trainer == "none"
        if args.num_eval is not None:
            eval_samples = task.sample_subset(data_split="valid", seed=0, num=args.num_eval)
        else:
            eval_samples = task.valid_samples

        metrics = framework.evaluate(train_sets, eval_samples, one_train_set_per_eval_sample=True)
        logger.info(metrics)
        if args.local_rank <= 0:
            write_metrics_to_file(metrics, args.output_dir + result_file_tag(args) + "-onetrainpereval.json" if args.result_file is None else args.result_file)    

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
