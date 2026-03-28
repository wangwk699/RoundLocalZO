import transformers
import torch
from .models_utils import BaseLM, find_layers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
import pdb


class LMClass(BaseLM):
    def __init__(self, args):

        super().__init__()

        self.args = args
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = args.model
        self.batch_size_per_gpu = args.batch_size

        # 1. 加载配置与 Tokenizer
        self.model_config = args.model
        config = AutoConfig.from_pretrained(
            args.model, attn_implementation=args.attn_implementation
        )
        # config = AutoConfig.from_pretrained(
        #     args.model
        # )

        # 2. 处理 pad_token（关键！不同模型策略不同）
        #self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False,legacy=False)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token   # OPT/Llama 常用
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                print("Setting pad_token is `eos_token`")
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # 回退方案
                print("Adding new pad_token is `[PAD]`")

        # 3. 加载模型（关键设计：先加载到 CPU + float16）
        # self.model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=config.torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, 
            config=config, 
            device_map='cpu',           # ← 避免初始化时占满显存
            dtype=torch.float16   # ← 节省内存，后续量化会进一步压缩  # torch_dtype=torch.float16
        )

        # 4. 初始化元信息
        self.seqlen = self.model.config.max_position_embeddings
        self.model.eval()  # ← 默认评估模式，训练时由 Trainer 切换
        self.vocab_size = self.tokenizer.vocab_size
        print("vocab size: ", self.vocab_size)

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        print("max_gen_toks fn")
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_encode_batch(self, strings):
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

    def tok_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():

            return self.model(inps)["logits"]

    def model_batched_set(self, inps):
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(
                self._model_call(batch), dim=-1
            ).cpu()  # [batch, padding_length, vocab]
            dataset_logits.append(multi_logits)
        return dataset_logits

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )