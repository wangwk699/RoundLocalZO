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

from transformers.integrations import (
    get_reporting_integration_callbacks,
    hp_params,
)

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

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
else:
    IS_XLA_FSDPV2_POST_2_2 = False


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
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
    if is_accelerate_available() and "adapter_only" in list(inspect.signature(save_fsdp_model).parameters):
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

    np_core = np._core if version.parse(np.__version__) >= version.parse("2.0.0") else np.core
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

# Definition of custom trainers starts from here

"""
QZO Added:
MeZO Trainer
Coding Reference:
https://github.com/princeton-nlp/MeZO/blob/main/large_models/trainer.py
"""

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