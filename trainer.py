"""
Custom trainers for QZO
Following the license Apache 2.0, we add comentary to where the codes are modified
Search 'QZO Added' for details 
"""

# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union

import transformers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import PredictionOutput, EvalLoopOutput

# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations import (
    get_reporting_integration_callbacks,
    hp_params,
)

# isort: on

import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import ModelCard, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model

from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.optimization import Adafactor, get_scheduler
from transformers.processing_utils import ProcessorMixin
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    is_torch_greater_or_equal_than_2_3,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    EvalLoopContainer,
    IterableDatasetShard,
    LabelSmoother,
    LayerWiseDummyOptimizer,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    SaveStrategy,
    TrainerMemoryTracker,
    TrainOutput,
    check_target_module_exists,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    XLA_FSDPV2_MIN_VERSION,
    PushInProgress,
    PushToHubMixin,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_grokadamw_available,
    is_in_notebook,
    is_ipex_available,
    is_liger_kernel_available,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_schedulefree_available,
    is_torch_compile_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_torchao_available,
    logging,
    strtobool,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.quantization_config import QuantizationMethod


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback
    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(
        XLA_FSDPV2_MIN_VERSION
    )
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
else:
    IS_XLA_FSDPV2_POST_2_2 = False

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import (
        smp_forward_backward,
        smp_forward_only,
        smp_gather,
        smp_nested_concat,
    )
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import PeftModel

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.state import AcceleratorState
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        DistributedType,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration


def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False


def _get_fsdp_ckpt_kwargs():
    # TODO: @AjayP13, @younesbelkada replace this check with version check at the next `accelerate` release
    if is_accelerate_available() and "adapter_only" in list(
        inspect.signature(save_fsdp_model).parameters
    ):
        return {"adapter_only": True}
    else:
        return {}


def safe_globals():
    # Starting from version 2.4 PyTorch introduces a check for the objects loaded
    # with torch.load(weights_only=True). Starting from 2.6 weights_only=True becomes
    # a default and requires allowlisting of objects being loaded.
    # See: https://github.com/pytorch/pytorch/pull/137602
    # See: https://pytorch.org/docs/stable/notes/serialization.html#torch.serialization.add_safe_globals
    # See: https://github.com/huggingface/accelerate/pull/3036
    if version.parse(torch.__version__).release < version.parse("2.6").release:
        return contextlib.nullcontext()

    np_core = (
        np._core if version.parse(np.__version__) >= version.parse("2.0.0") else np.core
    )
    allowlist = [np_core.multiarray._reconstruct, np.ndarray, np.dtype]
    # numpy >1.25 defines numpy.dtypes.UInt32DType, but below works for
    # all versions of numpy
    allowlist += [type(np.dtype(np.uint32))]

    return torch.serialization.safe_globals(allowlist)


if TYPE_CHECKING:
    import optuna

    if is_datasets_available():
        import datasets

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


from transformers import Trainer
from transformers.trainer_callback import ProgressCallback
from metrics import f1


class QZOTrainer(Trainer):
    # Lucifer Li: 核心方法, 覆写Trainer的`_inner_training_loop`方法
    def _inner_training_loop(
        self,
        batch_size=None,
        args=None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        # Lucifer Li: 该部分是QZO训练框架中针对GPTQ量化模型的一个补丁, 自动为GPTQ量化模型设置默认的标签名称, 避免在评估过程中因标签名不匹配而报错
        # QZO Added: set default labels names for GPTQModel class to avoid bugs in evaluation
        if args.trainer == "qzo" and args.quant_method in ["gptq", "omni"]:
            if "opt" in args.model_name.lower() or "llama" in args.model_name.lower():
                self.label_names = ["labels"]
                # TODO we need a more convenient way to automatically find label_names of a specific model architecture
            else:
                raise NotImplementedError(
                    "Please set label_names by yourself for a custom model"
                )
                # You can find the label_name by the following codes:
                # from transformers.utils.generic import find_labels
                # find_labels(model.__class__)

        self.accelerator.free_memory()  # 显式释放内存中不再使用的缓存和临时变量
        self._train_batch_size = batch_size  # 将当前批处理大小赋值给类的内部属性
        # Lucifer Li: 自动批次大小调整, 根据内存情况动态调整训练批次大小, 但`args`中不存在`auto_find_batch_size`, 所以不自动调整
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = (
                        self._train_batch_size // max(1, self.args.n_gpu)
                    )
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(
            f"Currently training with a batch size of: {self._train_batch_size}"
        )

        # Lucifer Li: 调用框架的`get_train_dataloader`方法获取训练数据迭代器, 返回一个可迭代对象, 用于在训练循环中获取批次数据
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        # Lucifer Li: 如果是TPU训练环境, 使用`tpu_spmd_dataloader`对数据加载器进行额外包装, 确保数据正确分发给各个TPU核心
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs, 训练总轮数
        # number of training steps per epoch: num_update_steps_per_epoch, 每轮训练的更新步数
        # total number of training steps to execute: max_steps, 总训练步数
        # 总训练批次大小 = 单设备批次大小 * 梯度累积步数 * 分布式进程数
        total_train_batch_size = (
            self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        )

        # Lucifer Li: 计算所有相关的训练控制变量
        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)  # 获取数据加载器的总批次数
            num_update_steps_per_epoch = (
                len_dataloader // args.gradient_accumulation_steps
            )  # 计算每轮的实际参数更新步数
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)  # 获取总样本数量
            if args.max_steps > 0:
                max_steps = args.max_steps  # 直接使用用户设置的`max_steps`
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )  # 根据`max_steps`反向计算需要的训练轮数
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = (
                    args.max_steps * total_train_batch_size
                )  # 总训练样本数 = `max_steps` * 总训练批次大小
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps)
                        * args.gradient_accumulation_steps
                    )  # 计算总处理的token数量，用于性能监控
            else:
                # Lucifer Li: 另一种计算方式
                max_steps = math.ceil(
                    args.num_train_epochs * num_update_steps_per_epoch
                )
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = (
                    self.num_examples(train_dataloader) * args.num_train_epochs
                )
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader) * args.num_train_epochs
                    )
        elif (
            args.max_steps > 0
        ):  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = (
                    self.num_tokens(train_dataloader, args.max_steps)
                    * args.gradient_accumulation_steps
                )
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        # Lucifer Li: 在训练过程中检测和监控浮点数计算异常
        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        # Lucifer Li: SageMaker模型并行; FSDP XLA训练(TPU上的全分片数据并行); 普通FSDP训练(全分片数据并行), 需要特殊的优化器初始化时机, 标记为需要延迟创建
        delay_optimizer_creation = (
            is_sagemaker_mp_enabled()
            or self.is_fsdp_xla_enabled
            or self.is_fsdp_enabled
        )

        # Lucifer Li: 重置学习率调度器
        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        # Lucifer Li: 如果启用了DeepSpeed, 调用专用的初始化函数初始化优化器和学习率调度器
        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps
            )

        # Lucifer Li: 常规训练环境创建优化器和学习率调度器
        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb
                for cb in self.callback_handler.callbacks + [self.control]
                if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = (
            trial is not None
        )  # 如果trial参数不为None, 说明正在进行超参数优化搜索
        self.state.train_batch_size = (
            self._train_batch_size
        )  # 将当前训练批次大小保存到状态中

        # Lucifer Li: 处理三个关键训练控制参数, 日志记录频率, 模型评估频率和模型保存频率
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Lucifer Li: 梯度检查点, 一种用时间换空间的内存优化技术
        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs
            )

        # Lucifer Li: 模型包装和加速器准备, 处理不同训练环境下的模型预处理和优化器延迟创建
        model = self._wrap_model(self.model_wrapped)

        # Lucifer Li: 如果包装后的模型仍然是原始模型, 则需要使用accelerator.prepare进行包装
        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        # Lucifer Li: 递归解包模型; 在自动寻找批次大小时, 需要访问原始模型属性, FSDP包装可能阻碍此过程
        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # Lucifer Li: 如果使用了FSDP+QLoRA, 需要更新FSDP插件配置
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(
                num_training_steps=max_steps
            )  # 延迟创建优化器和学习率调度器

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()  # 设置模型模式为训练模式
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(
                        self.model
                    )  # 使用NVIDIA Apex混合精度, 则只准备模型
                else:
                    model, self.optimizer = self.accelerator.prepare(
                        self.model, self.optimizer
                    )  # 同时准备模型和优化器
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        # Lucifer Li: 针对内存高效优化器(LOMO/ADALOMO)的特殊处理
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        # Lucifer Li: 全分片数据并行训练, 确保`self.model`和`self.model_wrapped`都指向包装后的模型
        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model  # 后续代码中`model`变量代表外部包装后的模型

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped,
                    resume_from_checkpoint,
                    load_module_strict=not _is_peft_model(self.model),
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Lucifer Li: 至此, 训练前的准备已全部完成,
        # 包括: 环境检测与配置, 数据加载器准备, 训练控制参数计算, 模型包装与准备, 优化器与调度器初始化, 内存与性能优化, 训练状态管理和检查点与恢复

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}"
        )
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(
                f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}"
            )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(
            f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Lucifer Li: 断点续训
        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch
                )
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}"
            )
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Lucifer Li: 引用同步, 试验追踪, 状态固化和进程标识
        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = (
                trial.assignments
                if self.hp_search_backend == HPSearchBackend.SIGOPT
                else trial
            )
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = (
            self.state.global_step
        )  # 记录上一次日志记录的全局步数, 用于计算日志间隔
        model.zero_grad()  # 清除模型中的所有梯度缓存
        grad_norm: Optional[float] = None  # 初始化梯度范数变量
        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control
        )  # 通知所有注册的回调函数训练即将开始

        # Lucifer Li: 如果设置了`eval_on_start`为True, 在训练开始前先进行一次评估
        """
        Lucifer Li: -潜在风险-
        `args`中并不包含`eval_on_start`参数
        """
        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        # Lucifer Li: 核心训练循环, 遍历所有训练轮次, 支持从断点续训
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)  # 分布式训练中设置epoch以打乱数据

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None  # 重置记忆状态

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )  # 计算当前epoch的步数
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control
            )

            # Lucifer Li: 断点续训处理开始
            if (
                epoch == epochs_trained
                and resume_from_checkpoint is not None
                and steps_trained_in_current_epoch == 0
            ):
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(
                    epoch_dataloader, steps_trained_in_current_epoch
                )
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True
            # Lucifer Li: 断点续训处理结束

            # --------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------

            # Lucifer Li: 核心批次处理循环, 遍历批次
            step = -1
            epoch_iterator = iter(
                epoch_dataloader
            )  # 将数据加载器转换为迭代器, 便于逐个获取批次
            # Lucifer Li: 处理不完整批次
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = (
                num_examples % args.gradient_accumulation_steps
            )  # 每`梯度累积步数`次更新一次参数
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = (
                steps_in_epoch // args.gradient_accumulation_steps + 1
            )  # 计算当前epoch需要的总更新次数

            # Lucifer Li: 遍历所有更新次数
            for _ in range(total_updates):
                update_step += 1
                num_batches = (
                    args.gradient_accumulation_steps
                    if update_step != (total_updates - 1)
                    else remainder
                )  # 确定当前块的大小
                batch_samples, num_items_in_batch = self.get_batch_samples(
                    epoch_iterator, num_batches, device=self.args.device
                )  # 从`epoch_iterator`中获取`num_batches`个批次
                # Lucifer Li: 内层循环处理每个批次
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    # 判断是否需要执行同步步骤, 即参数更新点(达到梯度累积步数或到达epoch的最后一步)
                    do_sync_step = (
                        step + 1
                    ) % args.gradient_accumulation_steps == 0 or (
                        step + 1
                    ) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    if not do_sync_step:
                        self.accelerator.gradient_state._set_sync_gradients(
                            False
                        )  # 手动控制梯度同步
                    else:
                        self.accelerator.gradient_state._set_sync_gradients(
                            True
                        )  # 手动控制梯度同步

                    # Lucifer Li: 统计已处理的token数量, 用于计算吞吐量
                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(
                            self.model, "main_input_name", "input_ids"
                        )  # 自动检测模型的主要输入字段, 默认input_ids
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(
                                input_tokens, device=self.args.device, dtype=torch.int64
                            )
                            self.state.num_input_tokens_seen += (
                                self.accelerator.gather(input_tokens).sum().cpu().item()
                            )

                    # Lucifer Li: 断点续训, 随机数状态同步
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Lucifer Li: 断点续训, 跳过已训练的批次
                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    # Lucifer Li: 每次参数更新时触发回调
                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(
                            args, self.state, self.control
                        )

                    # Lucifer Li: 控制分布式训练中的梯度同步行为
                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                        and self.accelerator.distributed_type
                        != DistributedType.DEEPSPEED
                        else contextlib.nullcontext
                    )

                    # --------------------------------------------------------------------------------
                    # --------------------------------------------------------------------------------
                    # --------------------------------------------------------------------------------
                    # --------------------------------------------------------------------------------

                    # Lucifer Li: `qzo`模式使用零阶优化方法计算损失和梯度估计, 其他模式则使用标准训练步骤(如一阶优化)
                    # QZO Added:
                    # compute per-step loss & estimated gradient with ZO forward passes
                    if args.trainer == "qzo":
                        tr_loss_step = self.zo_step(
                            model, inputs
                        )  # 返回的损失是正向扰动后的损失值
                    else:
                        with context():
                            tr_loss_step = self.training_step(
                                model, inputs, num_items_in_batch
                            )

                    # Lucifer Li: 检查损失值是否异常
                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        """
                        Lucifer Li: -潜在风险-
                        如果损失异常, 使用历史平均损失替代, 从而避免训练崩溃
                        """
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (
                            1 + self.state.global_step - self._globalstep_last_logged
                        )
                    else:
                        # Lucifer Li: 确保损失值在正确设备上
                        if tr_loss.device != tr_loss_step.device:
                            tr_loss_step = tr_loss_step.to(tr_loss.device)
                        # Lucifer Li: 累积将当前步骤的损失
                        tr_loss = tr_loss + tr_loss_step

                    # Lucifer Li: 统计浮点运算次数, 用于计算效率
                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:  # 执行参数更新
                        # QZO Added:
                        # Update model weights with estimated gradients
                        if args.trainer == "qzo":
                            verbose = (self.state.global_step + 1) % 1 == 0
                            self.zo_update(model, print_d=verbose, tr_loss=tr_loss)
                            # the following lines ensure that training progress bar can work smoothly
                            self.state.global_step += 1
                            self.state.epoch = (
                                epoch + (step + 1 + steps_skipped) / steps_in_epoch
                            )
                            self.control = self.callback_handler.on_step_end(
                                args, self.state, self.control
                            )
                            self._maybe_log_save_evaluate(
                                tr_loss,
                                grad_norm,
                                model,
                                trial,
                                epoch,
                                ignore_keys_for_eval,
                                start_time,
                            )
                        # Lucifer Li: 标准模式参数更新步骤
                        else:
                            # Since we perform prefetching, we need to manually set sync_gradients to True
                            self.accelerator.gradient_state._set_sync_gradients(
                                True
                            )  # 强制开启梯度同步

                            # Lucifer Li: 梯度裁剪, 防止梯度爆炸或消失
                            # Gradient clipping
                            if (
                                args.max_grad_norm is not None
                                and args.max_grad_norm > 0
                            ):
                                # Lucifer Li: 多种梯度裁剪实现
                                # deepspeed does its own clipping
                                if is_sagemaker_mp_enabled() and args.fp16:
                                    _grad_norm = self.optimizer.clip_master_grads(
                                        args.max_grad_norm
                                    )
                                elif self.use_apex:
                                    # Revert to normal clipping otherwise, handling Apex or full precision
                                    _grad_norm = nn.utils.clip_grad_norm_(
                                        amp.master_params(self.optimizer),
                                        args.max_grad_norm,
                                    )
                                else:
                                    _grad_norm = self.accelerator.clip_grad_norm_(
                                        model.parameters(),
                                        args.max_grad_norm,
                                    )

                                if (
                                    is_accelerate_available()
                                    and self.accelerator.distributed_type
                                    == DistributedType.DEEPSPEED
                                ):
                                    grad_norm = model.get_global_grad_norm()
                                    # In some cases the grad norm may not return a float
                                    if hasattr(grad_norm, "item"):
                                        grad_norm = grad_norm.item()
                                else:
                                    grad_norm = _grad_norm

                            # Lucifer Li: 优化器步骤
                            self.control = self.callback_handler.on_pre_optimizer_step(
                                args, self.state, self.control
                            )
                            self.optimizer.step()
                            self.control = self.callback_handler.on_optimizer_step(
                                args, self.state, self.control
                            )

                            # Lucifer Li: 学习率调度
                            optimizer_was_run = (
                                not self.accelerator.optimizer_step_was_skipped
                            )
                            if optimizer_was_run:
                                # Delay optimizer scheduling until metrics are generated
                                if not isinstance(
                                    self.lr_scheduler,
                                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                                ):
                                    self.lr_scheduler.step()

                            # Lucifer Li: 清理和状态更新
                            model.zero_grad()
                            self.state.global_step += 1
                            self.state.epoch = (
                                epoch + (step + 1 + steps_skipped) / steps_in_epoch
                            )
                            self.control = self.callback_handler.on_step_end(
                                args, self.state, self.control
                            )
                            self._maybe_log_save_evaluate(
                                tr_loss,
                                grad_norm,
                                model,
                                trial,
                                epoch,
                                ignore_keys_for_eval,
                                start_time,
                            )
                    else:  # 非参数更新时, 触发子步骤结束回调
                        self.control = self.callback_handler.on_substep_end(
                            args, self.state, self.control
                        )

                    # Lucifer Li: 检查是否需要提前停止训练
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if (
                        self.control.should_epoch_stop
                        or self.control.should_training_stop
                    ):
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break

            # Lucifer Li: 空数据检测
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(
                args, self.state, self.control
            )
            # 日志记录、模型保存和评估
            self._maybe_log_save_evaluate(
                tr_loss,
                grad_norm,
                model,
                trial,
                epoch,
                ignore_keys_for_eval,
                start_time,
            )

            # TPU调试指标输出
            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )

            if self.control.should_training_stop:  # 训练终止检查
                break

        # --------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
        )

        # Lucifer Li: 按需加载最佳训练模型
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Lucifer Li: 针对分布式训练, 确保所有训练完成并完成同步
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous(
                    "load_best_model_at_end"
                )  # TPU的同步原语, 执行TPU环境同步
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()  # PyTorch分布式训练中的屏障同步
            elif is_sagemaker_mp_enabled():
                smp.barrier()  # Amazon SageMaker模型并行框架的同步

            self._load_best_model()  # 从记录的最佳检查点路径加载模型权重

        # add remaining tr_loss
        self._total_loss_scalar += (
            tr_loss.item()
        )  # 将最后一个训练步骤的损失值标量加到总损失中, _total_loss_scalar -> Python浮点数, tr_loss -> Tensor
        effective_global_step = max(
            self.state.global_step, 0.001
        )  # Avoid ZeroDivisionError, 立即停止的训练`global_step`可能是0, 因此设置保护值
        train_loss = self._total_loss_scalar / effective_global_step  # 平均训练损失计算

        # Lucifer Li: 训练框架的内置函数, 计算各种速度相关指标
        metrics = speed_metrics(
            "train",  # 指标前缀
            start_time,  # 训练开始时间
            num_samples=num_train_samples,  # 总训练样本数
            num_steps=self.state.max_steps,  # 总训练步数
            num_tokens=num_train_tokens,  # 总处理的token数
        )  # 返回的指标通常包含: train_runtime, train_samples_per_second, train_steps_per_second, train_tokens_per_second, etc.
        self.store_flos()  # 将当前FLOPS计数保存到训练状态
        # Lucifer Li: 指标字典`metrics`中新增`total_flos`和`train_loss`
        metrics["total_flos"] = (
            self.state.total_flos
        )  # 在整个训练过程中累计的浮点运算总数
        metrics["train_loss"] = train_loss  # 平均训练损失

        self.is_in_train = False  # 训练状态标记
        self._memory_tracker.stop_and_update_metrics(metrics)  # 停止内存使用监控
        self.log(metrics)  # 将所有指标记录到日志
        run_dir = self._get_output_dir(trial)  # 获取当前训练运行的输出目录路径
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=False, output_dir=run_dir
        )  # 对输出目录中的所有检查点进行排序, 返回按训练步数排序的检查点路径列表

        # Lucifer Li: 删除除最佳检查点外的所有其他检查点
        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if (
            self.args.should_save
            and self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
        ):
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(
                        f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
                    )
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )  # 训练结束, 触发回调

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            # NEFTune, 一种训练时在前向传播中添加噪声的技术
            # 训练结束后移除噪声注入钩子, 恢复原始模型行为, 确保推理时使用干净的模型
            self._deactivate_neftune(self.model)

        # Lucifer Li: 返回训练结果, 包含总训练步数, 平均训练损失及各类指标字典
        return TrainOutput(self.state.global_step, train_loss, metrics)

    ############## QZO ##############
    # Lucifer Li: 调用流程: zo_step(zo_perturb_parameters -> zo_forward(zo_forward_nondiff)) -> zo_update
    # Lucifer Li: `random_seed`为随机种子, 控制随机方向z的生成; `scaling_factor`为扰动缩放因子, 控制扰动的大小和方向
    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input:
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(
            random_seed if random_seed is not None else self.zo_random_seed
        )

        if self.args.train_unquantized:  # 可选, 是否也训练未量化的参数
            for name, param in self.fp16_to_optimize["regular"]:
                # 标准正态分布采样
                z = torch.normal(
                    mean=0,
                    std=1,
                    size=param.data.size(),
                    device=param.data.device,
                    dtype=param.data.dtype,
                )
                param.data = (
                    param.data + scaling_factor * z * self.args.zo_eps
                )  # self.args.zo_eps === 1e-3

        for param in self.fp16_to_optimize["scales"]:  # 必选, 训练量化缩放因子参数
            z = torch.normal(
                mean=0,
                std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
            param.data = (
                param.data + scaling_factor * z * self.args.zo_eps
            )  # self.args.zo_eps === 1e-3

    # Lucifer Li: 零阶优化中前向传播计算损失的核心方法
    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()  # 设置模型为评估模式, 确保前向传播的确定性
        if self.args.non_diff:  # 指示是否处理不可微分的目标函数
            # Non-differentiable objective (may require autoregressive generation)
            return self.zo_forward_nondiff(model, inputs)

        with torch.inference_mode():  # 更激进的优化, 完全禁用梯度计算和追踪
            inputs = self._prepare_inputs(
                inputs
            )  # 准备输入数据, 如将输入数据移动到正确设备, 确保数据格式正确
            with self.compute_loss_context_manager():  # 创建损失计算的上下文管理器
                loss = self.compute_loss(
                    model, inputs
                )  # 计算模型损失的通用方法, 返回损失张量, 根据任务类型有不同的实现
            if (
                self.args.n_gpu > 1
            ):  # 在多GPU训练中, 每个GPU计算部分损失, 对所有GPU的损失求平均
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return (
            loss.detach()
        )  # 返回去梯度的损失, `detach`表示从计算图中分离损失张量, 返回Tensor数值

    # Lucifer Li: 仅适配`SQuAD`任务, 处理不可微目标的零阶前向传播方法
    def zo_forward_nondiff(self, model, inputs):
        """
        Get (no gradient) non-diffiable loss from the model.
        """
        model.eval()  # 设置模型为评估模式, 确保前向传播的确定性
        # Lucifer Li: 目前只支持SQuAD任务
        assert (
            self.args.task_name == "SQuAD"
        ), "Non differentiable objective only supports SQuAD for now."

        with torch.inference_mode():  # 完全禁用梯度计算和追踪
            inputs = self._prepare_inputs(inputs)  # 准备输入
            args = self.args

            # Lucifer Li: 可能需要输入`inputs["attention_mask"]`、`inputs["labels"]`和`inputs["gold"]`
            outputs = self.model.generate(
                inputs["input_ids"],
                do_sample=args.sampling,  # 使用贪婪解码或beam search
                temperature=args.temperature,
                num_beams=args.num_beams,  # 使用参数中的beam数量
                top_p=args.top_p,  # beam search时不使用nucleus sampling
                top_k=args.top_k,  # beam search时不使用top-k
                repetition_penalty=args.repetition_penalty,  # 使用参数中的重复惩罚
                max_new_tokens=min(
                    args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)
                ),
                num_return_sequences=1,
                eos_token_id=[
                    self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1],
                    self.tokenizer.eos_token_id,
                ],
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # Lucifer Li: 解码生成的文本
            output_text = []
            for i in range(len(outputs)):  # 遍历每个生成的序列
                # outputs[i][inputs["input_ids"].size(1):] -> 去掉输入部分, 只保留模型生成的部分
                # skip_special_tokens=True -> 跳过特殊token
                # self.tokenizer.decode(...) -> 解码为文本
                # strip() -> 移除文本首尾的空格
                output_text.append(
                    self.tokenizer.decode(
                        outputs[i][inputs["input_ids"].size(1) :],
                        skip_special_tokens=True,
                    ).strip()
                )
            # Lucifer Li: 为每个样本计算独立的F1分数(生成与真实之间的差距), 得到一个F1分数列表
            f1s = [
                f1(output_text[i], inputs["gold"][i]) for i in range(len(output_text))
            ]

        return -torch.tensor(
            np.mean(f1s), dtype=torch.float32
        )  # 最小化loss <-> 最大化F1分数

    # Lucifer Li: 零阶优化中梯度估计的核心方法
    def zo_step(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.args

        self.fp16_to_optimize = {"scales": [], "regular": []}  # param  # (name, param)
        if args.quant_method == "omni":
            from quantize.int_linear import QuantLinear

            # Lucifer Li: 遍历找到模型中的所有量化线性层, 收集反量化缩放因子(待优化参数, 通过优化缩放因子来提升模型精度)
            for name, module in model.named_modules():
                if isinstance(module, QuantLinear):
                    self.fp16_to_optimize["scales"].append(
                        module.weight_quantizer.descale
                    )
        else:
            raise NotImplementedError

        # Lucifer Li: 确保试验可以复现, 设定随机种子
        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # Lucifer Li: 零阶优化中计算梯度估计的核心部分, 实现了对称差分的梯度估计方法
        # First function evaluation
        self.zo_perturb_parameters(
            scaling_factor=1
        )  # 将模型参数向正方向扰动, 数学上表示+ez
        loss1 = self.zo_forward(model, inputs)  # 计算损失1

        # Second function evaluation
        self.zo_perturb_parameters(
            scaling_factor=-2
        )  # 将模型参数向负方向扰动, 数学上表示-2ez
        loss2 = self.zo_forward(model, inputs)  # 计算损失2

        # Lucifer Li: 中心差分公式计算零阶梯度, 比前向差分或后向差分有更高的精度
        self.projected_grad = (
            (loss1 - loss2) / (2 * self.args.zo_eps)
        ).item()  # 转换为Python标量

        # Lucifer Li: 禁止梯度累积, 零阶优化中每个梯度估计都需要两次前向传播, 参数会被扰动, 而梯度累积需要保持参数不变并累积多个批次的梯度(冲突)
        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        # Lucifer Li: 由于`zo_perturb_parameters`执行的是原地操作, 因此需要在每次梯度估计后将参数恢复到原始状态
        # Reset model back to its parameters at start of step
        self.zo_perturb_parameters(scaling_factor=1)

        # Lucifer Li: 保存损失值1, 可用于后续计算
        self.zo_loss1 = loss1

        return loss1

    # Lucifer Li: 零阶优化中参数更新的核心方法
    def zo_update(self, model, print_d=False, tr_loss=0):
        """
        Update the parameters with the estimated gradients.
        """
        args = self.args

        # Lucifer Li: 重置随机种子, 确保生成相同的随机方向z, 必须与`zo_perturb_parameters`中使用的随机方向完全一致
        # Reset the random seed for sampling zs
        torch.manual_seed(self.zo_random_seed)

        # Lucifer Li: 方向导数裁剪, 防止梯度爆炸, 避免参数更新步长过大
        # QZO Added: Directional Derivative Clipping
        if args.clip_zo_grad:  # args.clip_zo_grad === True
            # clip projected_grad, [-100, 100]
            self.projected_grad = min(max(-100, self.projected_grad), 100)

        if self.args.train_unquantized:  # 可选, 更新未量化参数
            for name, param in self.fp16_to_optimize["regular"]:
                z = torch.normal(
                    mean=0,
                    std=1,
                    size=param.data.size(),
                    device=param.data.device,
                    dtype=param.data.dtype,
                )

                # Lucifer Li: 一般来说, 只有权重参数需要正则化防止过拟合
                if (
                    "bias" not in name
                    and "layer_norm" not in name
                    and "layernorm" not in name
                ):
                    # Lucifer Li: 应用权重衰减的参数更新公式, 进而可以继续扩展为带动量的更新
                    # update = self.projected_grad * z + args.weight_decay * param.data -> 权重衰减
                    # param.data = param.data - lr * update
                    param.data = param.data - self._get_learning_rate() * (
                        self.projected_grad * z + args.weight_decay * param.data
                    )
                else:
                    # 偏置和层归一化参数不应用权重衰减
                    param.data = param.data - self._get_learning_rate() * (
                        self.projected_grad * z
                    )

        # Lucifer Li: 必选, 更新量化缩放因子参数
        for param in self.fp16_to_optimize["scales"]:
            z = torch.normal(
                mean=0,
                std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
            # Lucifer Li: 根据梯度方向更新缩放因子, 执行裁剪是因为: 必须是正数, 避免除零错误, 缩放因子接近0会导致量化失效
            param.data = torch.clamp(
                param.data - self._get_learning_rate() * (self.projected_grad * z),
                min=1e-7,
            )

        self.lr_scheduler.step()  # 更新学习率调度器

    ############## Misc overload functions ##############
    def _set_signature_columns_if_needed(self):
        """
        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(
                set(["label", "label_ids"] + self.label_names)
            )
            self._signature_columns += ["gold"]

    # QZO Added: Overload _save function to support GPTQModel checkpoints
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (
            (PreTrainedModel,)
            if not is_peft_available()
            else (PreTrainedModel, PeftModel)
        )
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`

        # to support GPTQModel
        if self.args.trainer == "qzo" and self.args.quant_method == "gptq":
            if isinstance(self.model.model, supported_classes):
                self.model.model.save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    safe_serialization=self.args.save_safetensors,
                )
            else:
                raise NotImplementedError

        elif not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    safe_serialization=self.args.save_safetensors,
                )
            else:
                logger.info(
                    "Trainer.model is not a `PreTrainedModel`, only saving its state dict."
                )
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict,
                        os.path.join(output_dir, SAFE_WEIGHTS_NAME),
                        metadata={"format": "pt"},
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        else:
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
            )

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


class QAZOTrainer(Trainer):
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        # QZO Added: set default labels names for GPTQModel class to avoid bugs in evaluation
        if args.trainer == 'qazo' and args.quant_method in ['gptq','omni']:
            if 'opt' in args.model_name.lower() or 'llama' in args.model_name.lower():
                self.label_names = ['labels']
                # TODO we need a more convenient way to automatically find label_names of a specific model architecture
            else:
                raise NotImplementedError("Please set label_names by yourself for a custom model")
                # You can find the label_name by the following codes:
                # from transformers.utils.generic import find_labels
                # find_labels(model.__class__)

        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = num_examples % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches,device=self.args.device)
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    if not do_sync_step:
                        self.accelerator.gradient_state._set_sync_gradients(False)
                    else:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += (
                                self.accelerator.gather(input_tokens).sum().cpu().item()
                            )
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                        and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                        else contextlib.nullcontext
                    )
                    # QZO Added: 
                    # compute per-step loss & estimated gradient with ZO forward passes
                    if args.trainer == 'qazo':
                        tr_loss_step = self.zo_step(model, inputs)

                    else:
                        with context():
                            tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        # Lucifer Li: 确保损失值在正确设备上
                        if tr_loss.device != tr_loss_step.device:
                            tr_loss_step = tr_loss_step.to(tr_loss.device)
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # QZO Added:
                        # Update model weights with estimated gradients
                        if args.trainer == 'qazo':
                            # verbose = (self.state.global_step + 1) % 1 == 0
                            # self.zo_update(model, print_d=verbose, tr_loss=tr_loss)
                            # the following lines ensure that training progress bar can work smoothly
                            self.state.global_step += 1
                            self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                            self._maybe_log_save_evaluate(
                                tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time
                            )
                        else:
                            # Since we perform prefetching, we need to manually set sync_gradients to True
                            self.accelerator.gradient_state._set_sync_gradients(True)

                            # Gradient clipping
                            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                                # deepspeed does its own clipping

                                if is_sagemaker_mp_enabled() and args.fp16:
                                    _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                                elif self.use_apex:
                                    # Revert to normal clipping otherwise, handling Apex or full precision
                                    _grad_norm = nn.utils.clip_grad_norm_(
                                        amp.master_params(self.optimizer),
                                        args.max_grad_norm,
                                    )
                                else:
                                    _grad_norm = self.accelerator.clip_grad_norm_(
                                        model.parameters(),
                                        args.max_grad_norm,
                                    )

                                if (
                                    is_accelerate_available()
                                    and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                                ):
                                    grad_norm = model.get_global_grad_norm()
                                    # In some cases the grad norm may not return a float
                                    if hasattr(grad_norm, "item"):
                                        grad_norm = grad_norm.item()
                                else:
                                    grad_norm = _grad_norm

                            self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                            self.optimizer.step()

                            self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                            optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                            if optimizer_was_run:
                                # Delay optimizer scheduling until metrics are generated
                                if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                    self.lr_scheduler.step()

                            model.zero_grad()
                            self.state.global_step += 1
                            self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                            self._maybe_log_save_evaluate(
                                tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time
                            )
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    ############## QAZO ##############
    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1, target="scales"):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        if self.args.train_unquantized:
            for name, param in self.fp16_to_optimize['regular']:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                param.data = param.data + scaling_factor * z * self.args.zo_eps
        if target in ("scales"):
            for param in self.fp16_to_optimize['scales']:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                param.data = param.data + scaling_factor * z * self.args.zo_eps
        if target in ("zeros"):
            for param in self.fp16_to_optimize['zeros']:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                param.data = param.data + scaling_factor * z * self.args.zo_eps

    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        if self.args.non_diff:
            # Non-differentiable objective (may require autoregressive generation)
            return self.zo_forward_nondiff(model, inputs)

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()

    def zo_forward_nondiff(self, model, inputs):
        """
        Get (no gradient) non-diffiable loss from the model.
        """
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args

            outputs = self.model.generate(
                inputs["input_ids"], 
                do_sample=args.sampling,
                temperature=args.temperature,
                num_beams=args.num_beams,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)),
                num_return_sequences=1,
                eos_token_id=[
                    self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1],
                    self.tokenizer.eos_token_id
                ],
                pad_token_id=self.tokenizer.pad_token_id,
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]
        
        return -torch.tensor(np.mean(f1s), dtype=torch.float32)

    def zo_step_target(self, model, inputs, target: str):
        # 为该 target 采样独立种子
        seed = np.random.randint(1000000000)
        self.zo_random_seed_dict[target] = seed

        self.zo_random_seed = seed  # 复用你原本函数里用的字段

        # f(theta + eps z)
        self.zo_perturb_parameters(random_seed=seed, scaling_factor=1, target=target)
        loss1 = self.zo_forward(model, inputs)

        # f(theta - eps z)
        self.zo_perturb_parameters(random_seed=seed, scaling_factor=-2, target=target)
        loss2 = self.zo_forward(model, inputs)

        g = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        self.projected_grad_dict[target] = g

        # reset 回 theta
        self.zo_perturb_parameters(random_seed=seed, scaling_factor=1, target=target)

        return loss1

    def zo_step(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.args
 
        self.fp16_to_optimize = {
            'scales': [], # param
            'regular': [], # (name, param)
            'zeros': [],
        }

        if args.quant_method == 'omni': 
            from quantize.int_linear import QuantLinear

            for name, module in model.named_modules():
                if isinstance(module, QuantLinear):
                    self.fp16_to_optimize['scales'].append(module.weight_quantizer.descale)
                    self.fp16_to_optimize['zeros'].append(module.weight_quantizer.dezero)
        else:
            raise NotImplementedError
        
        self.zo_random_seed_dict = {}
        self.projected_grad_dict = {}
        assert self.args.gradient_accumulation_steps == 1
        
        loss_scales = self.zo_step_target(model, inputs, target="scales")
        self.zo_update_target("scales")
        loss_zeros = self.zo_step_target(model, inputs, target="zeros")
        self.zo_update_target("zeros")
        # Sample the random seed for sampling z
        # Reset model back to its parameters at start of step
        self.lr_scheduler.step()
        return 0.5 * (loss_scales + loss_zeros)

    def zo_update_target(self, target: str, print_d=False):
        args = self.args

        # 取该 target 对应的 seed / projected_grad
        seed = self.zo_random_seed_dict[target]
        g = self.projected_grad_dict[target]

        torch.manual_seed(seed)

        if args.clip_zo_grad:
            g = min(max(-100, g), 100)

        lr = self._get_learning_rate()

        if target == "scales":
            for param in self.fp16_to_optimize['scales']:
                z = torch.normal(mean=0, std=1, size=param.data.size(),
                                device=param.data.device, dtype=param.data.dtype)
                param.data = torch.clamp(param.data - lr * (g * z), min=1e-7)

        elif target == "zeros":
            for param in self.fp16_to_optimize['zeros']:
                z = torch.normal(mean=0, std=1, size=param.data.size(),
                                device=param.data.device, dtype=param.data.dtype)
                # zeros 是否要 clamp 取决于你定义；先给一个温和的 clamp，避免极端数值
                param.data = torch.clamp(param.data - lr * (g * z), min=-1e4, max=1e4)
        else:
            raise ValueError(f"Unknown target: {target}")

    def zo_update(self, model, print_d=False, tr_loss=0):
        """
        Update the parameters with the estimated gradients.
        """
        args = self.args

        # Reset the random seed for sampling zs
        torch.manual_seed(self.zo_random_seed)     

        # QZO Added: Directional Derivative Clipping
        if args.clip_zo_grad:
            # clip projected_grad
            self.projected_grad = min(max(-100, self.projected_grad), 100)

        if self.args.train_unquantized:
            for name, param in self.fp16_to_optimize['regular']:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    param.data = param.data - self._get_learning_rate() * (self.projected_grad * z + args.weight_decay * param.data)
                else:
                    param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)

        for param in self.fp16_to_optimize['scales']: 
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = torch.clamp(param.data - self._get_learning_rate() * (self.projected_grad * z), min=1e-7)
       
        self.lr_scheduler.step()

    ############## Misc overload functions ##############    
    def _set_signature_columns_if_needed(self):
        """
        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    # QZO Added: Overload _save function to support GPTQModel checkpoints
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`

        # to support GPTQModel
        if self.args.trainer =='qazo' and self.args.quant_method == 'gptq': 
            if isinstance(self.model.model, supported_classes):
                self.model.model.save_pretrained( 
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                raise NotImplementedError
            
        elif not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
            
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

class RoundZOTrainer(Trainer):
    """
    扩展的 Trainer 类，添加 loss 记录功能，方便绘制训练曲线
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 初始化 loss 记录容器
        self.train_loss_history = []  # 记录每个优化步骤的 loss
        self.eval_loss_history = []   # 记录每次评估的 loss
        self.current_step_losses = []  # 临时存储当前梯度累积步骤的 loss
        
        # 训练状态记录
        self.current_epoch = 0
        self.current_step = 0
        
        # 是否已经加载过 loss 历史（用于恢复训练）
        self._loss_history_loaded = False
        
        # 添加回调函数（如果需要的话）
        # from transformers import TrainerCallback
        # self.add_callback(LossHistoryCallback(self))
    
    def training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor], 
                      num_items_in_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        重写训练步骤，记录每个 mini-batch 的 loss
        """
        # 调用父类的训练步骤
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # 记录当前 mini-batch 的 loss（在主进程中）
        if self.is_world_process_zero():
            self.current_step_losses.append(loss.detach().clone().cpu())
        
        return loss
    
    def _inner_training_loop(self, *args, **kwargs):
        """
        重写内部训练循环，在关键位置记录 loss
        """
        # 调用父类的训练循环
        result = super()._inner_training_loop(*args, **kwargs)
        
        # 训练结束后保存 loss 历史
        if self.is_world_process_zero():
            self.save_loss_history()
        
        return result
    
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, 
                                 ignore_keys_for_eval, start_time, learning_rate=None):
        """
        重写日志记录和保存评估方法，在记录日志时也记录 loss
        """
        # 保存当前步骤的 loss（如果是在梯度累积步骤结束时）
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if self.is_world_process_zero() and hasattr(self, 'current_step_losses') and self.current_step_losses:
                # 计算当前优化步骤的平均 loss
                avg_step_loss = torch.stack(self.current_step_losses).mean().item()
                
                self.train_loss_history.append({
                    'global_step': self.state.global_step,
                    'loss': avg_step_loss,
                    'epoch': epoch + (self.state.global_step / self.state.max_steps) if self.state.max_steps > 0 else epoch,
                    'learning_rate': learning_rate if learning_rate is not None else self._get_learning_rate(),
                    'timestamp': time.time(),
                    'grad_norm': grad_norm.item() if grad_norm is not None and torch.is_tensor(grad_norm) else grad_norm
                })
                
                # 清空当前步骤的 loss 记录
                self.current_step_losses = []
        
        # 调用父类的方法
        return super()._maybe_log_save_evaluate(
            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate
        )
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        重写评估方法，记录评估 loss
        """
        # 调用父类的评估方法
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # 记录评估 loss（在主进程中）
        if self.is_world_process_zero():
            eval_loss_key = f"{metric_key_prefix}_loss"
            if eval_loss_key in metrics:
                self.eval_loss_history.append({
                    'global_step': self.state.global_step,
                    'loss': metrics[eval_loss_key],
                    'metric_key_prefix': metric_key_prefix,
                    'timestamp': time.time(),
                    'metrics': {k: v for k, v in metrics.items() if k != eval_loss_key}
                })
        
        return metrics
    
    def _save_checkpoint(self, model, trial):
        """
        重写保存检查点方法，同时保存 loss 历史
        """
        # 调用父类方法保存检查点
        checkpoint_folder = super()._save_checkpoint(model, trial)
        
        # 在主进程中保存 loss 历史
        if self.is_world_process_zero():
            self.save_loss_history()
        
        return checkpoint_folder
    
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """
        重写加载检查点方法，同时加载 loss 历史
        """
        # 调用父类方法加载模型
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        
        # 加载 loss 历史（仅加载一次）
        if not self._loss_history_loaded and self.is_world_process_zero():
            self.load_loss_history(resume_from_checkpoint)
            self._loss_history_loaded = True
    
    # ==================== 新增方法 ====================
    
    def save_loss_history(self, output_dir: Optional[str] = None):
        """
        保存 loss 历史到文件
        """
        if not self.is_world_process_zero():
            return
            
        if output_dir is None:
            output_dir = self.args.output_dir
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        loss_history = {
            'train': self.train_loss_history,
            'eval': self.eval_loss_history,
            'metadata': {
                'total_steps': self.state.global_step,
                # 'total_epochs': int(self.state.epoch) if hasattr(self.state, 'epoch') else 0,
                'total_epochs': int(self.state.epoch) if hasattr(self.state, 'epoch') and self.state.epoch is not None else 0,
                'save_time': time.time(),
                'output_dir': self.args.output_dir,
                'model_name': self.model.__class__.__name__ if hasattr(self.model, '__class__') else 'Unknown'
            }
        }
        
        # 保存为 JSON 文件
        with open(os.path.join(output_dir, 'loss_history.json'), 'w') as f:
            json.dump(loss_history, f, indent=2, default=self._json_serializer)
        
        # 也可以保存为 CSV 格式（可选）
        self._save_loss_history_csv(output_dir)
    
    def load_loss_history(self, checkpoint_dir: str):
        """
        从检查点加载 loss 历史
        """
        import os
        
        loss_history_file = os.path.join(checkpoint_dir, 'loss_history.json')
        if os.path.exists(loss_history_file):
            with open(loss_history_file, 'r') as f:
                loss_history = json.load(f)
                self.train_loss_history = loss_history.get('train', [])
                self.eval_loss_history = loss_history.get('eval', [])
                print(f"Loaded loss history from {loss_history_file}")
                print(f"  Train records: {len(self.train_loss_history)}")
                print(f"  Eval records: {len(self.eval_loss_history)}")
        else:
            warnings.warn(f"No loss history found at {loss_history_file}")
    
    def get_loss_history(self, as_dataframe: bool = False):
        """
        获取 loss 历史数据
        
        Args:
            as_dataframe: 是否返回 pandas DataFrame（需要安装 pandas）
        """
        if as_dataframe:
            try:
                import pandas as pd
                
                train_df = pd.DataFrame(self.train_loss_history) if self.train_loss_history else pd.DataFrame()
                eval_df = pd.DataFrame(self.eval_loss_history) if self.eval_loss_history else pd.DataFrame()
                
                return {
                    'train': train_df,
                    'eval': eval_df
                }
            except ImportError:
                warnings.warn("pandas is not installed. Install with 'pip install pandas' to use DataFrame output.")
                return self.get_loss_history(as_dataframe=False)
        
        return {
            'train': self.train_loss_history,
            'eval': self.eval_loss_history
        }
    
    def plot_loss_curve(self, output_path: Optional[str] = None, show: bool = True, 
                       figsize=(12, 6), dpi=100):
        """
        绘制 loss 曲线图
        
        Args:
            output_path: 保存图片的路径
            show: 是否显示图片
            figsize: 图片大小
            dpi: 图片分辨率
        """
        try:
            import matplotlib.pyplot as plt
            # import numpy as np
            
            if not self.train_loss_history:
                warnings.warn("No training loss history to plot.")
                return
            
            # 创建图形
            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
            fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
            
            # 准备训练数据
            if self.train_loss_history:
                train_steps = [entry['global_step'] for entry in self.train_loss_history]
                train_losses = [entry['loss'] for entry in self.train_loss_history]
                
                # 左图：训练 loss
                ax1.plot(train_steps, train_losses, 'b-', alpha=0.7, linewidth=1.5, label='Training Loss')
                ax1.set_xlabel('Global Step')
                ax1.set_ylabel('Loss')
                ax1.set_title('Training Loss Curve')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 如果 loss 变化很大，使用对数刻度
                # if max(train_losses) / min(train_losses) > 100 and min(train_losses) > 0:
                #     ax1.set_yscale('log')
            
            # # 准备评估数据
            # if self.eval_loss_history:
            #     eval_steps = [entry['global_step'] for entry in self.eval_loss_history]
            #     eval_losses = [entry['loss'] for entry in self.eval_loss_history]
                
            #     # 右图：评估 loss
            #     ax2.scatter(eval_steps, eval_losses, color='red', s=50, label='Evaluation Loss', zorder=5)
            #     if len(eval_steps) > 1:
            #         # 连接评估点（按时间顺序）
            #         sorted_indices = np.argsort(eval_steps)
            #         ax2.plot(np.array(eval_steps)[sorted_indices], 
            #                 np.array(eval_losses)[sorted_indices], 
            #                 'r--', alpha=0.5, linewidth=1)
                
            #     ax2.set_xlabel('Global Step')
            #     ax2.set_ylabel('Loss')
            #     ax2.set_title('Evaluation Loss')
            #     ax2.legend()
            #     ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
                print(f"Loss curve saved to {output_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
                
        except ImportError:
            warnings.warn("matplotlib is not installed. Install with 'pip install matplotlib' to plot loss curves.")
    
    def plot_learning_rate(self, output_path: Optional[str] = None, show: bool = True):
        """
        绘制学习率变化曲线
        """
        try:
            import matplotlib.pyplot as plt
            
            # 提取学习率数据
            lr_data = []
            for entry in self.train_loss_history:
                if 'learning_rate' in entry:
                    lr_data.append({
                        'global_step': entry['global_step'],
                        'learning_rate': entry['learning_rate']
                    })
            
            if not lr_data:
                warnings.warn("No learning rate data found in loss history.")
                return
            
            steps = [d['global_step'] for d in lr_data]
            lrs = [d['learning_rate'] for d in lr_data]
            
            plt.figure(figsize=(10, 6))
            plt.plot(steps, lrs, 'g-', linewidth=2, label='Learning Rate')
            plt.xlabel('Global Step')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 使用对数刻度如果学习率变化很大
            if max(lrs) / min(lrs) > 100 and min(lrs) > 0:
                plt.yscale('log')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            if show:
                plt.show()
            else:
                plt.close()
                
        except ImportError:
            warnings.warn("matplotlib is not installed. Install with 'pip install matplotlib' to plot learning rate curve.")
    
    def print_loss_summary(self):
        """
        打印 loss 统计摘要
        """
        if not self.train_loss_history:
            print("No training loss history available.")
            return
        
        print("=" * 60)
        print("LOSS HISTORY SUMMARY")
        print("=" * 60)
        
        # 训练 loss 统计
        train_losses = [entry['loss'] for entry in self.train_loss_history]
        print(f"Training Loss Records: {len(train_losses)}")
        print(f"  Final Loss: {train_losses[-1]:.6f}")
        print(f"  Min Loss:   {min(train_losses):.6f}")
        print(f"  Max Loss:   {max(train_losses):.6f}")
        print(f"  Avg Loss:   {sum(train_losses)/len(train_losses):.6f}")
        
        # 评估 loss 统计
        if self.eval_loss_history:
            eval_losses = [entry['loss'] for entry in self.eval_loss_history]
            print(f"\nEvaluation Loss Records: {len(eval_losses)}")
            print(f"  Final Loss: {eval_losses[-1]:.6f}")
            print(f"  Min Loss:   {min(eval_losses):.6f}")
            print(f"  Max Loss:   {max(eval_losses):.6f}")
            print(f"  Avg Loss:   {sum(eval_losses)/len(eval_losses):.6f}")
        
        print("=" * 60)
    
    def _save_loss_history_csv(self, output_dir: str):
        """
        将 loss 历史保存为 CSV 格式（可选）
        """
        try:
            import pandas as pd
            import os
            
            # 保存训练 loss
            if self.train_loss_history:
                train_df = pd.DataFrame(self.train_loss_history)
                train_csv_path = os.path.join(output_dir, 'train_loss_history.csv')
                train_df.to_csv(train_csv_path, index=False)
            
            # 保存评估 loss
            if self.eval_loss_history:
                eval_df = pd.DataFrame(self.eval_loss_history)
                eval_csv_path = os.path.join(output_dir, 'eval_loss_history.csv')
                eval_df.to_csv(eval_csv_path, index=False)
                
        except ImportError:
            pass  # pandas 未安装，跳过 CSV 保存
    
    def _json_serializer(self, obj):
        """
        JSON 序列化辅助函数
        """
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return str(obj)


# ==================== 可选：回调函数版本 ====================

class LossHistoryCallback:
    """
    使用回调函数的方式记录 loss 历史
    可以作为 Trainer 的回调函数使用
    """
    def __init__(self, trainer):
        self.trainer = trainer
    
    def on_log(self, args, state, control, logs, **kwargs):
        """在日志记录时触发"""
        if 'loss' in logs and state.global_step > 0:
            # 这里可以根据需要添加额外的记录逻辑
            pass
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """在评估完成后触发"""
        # 这里可以根据需要添加额外的记录逻辑
        pass