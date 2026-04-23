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
        # Lucifer Li: 可用GPU则使用GPU, 否则使用CPU
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Lucifer Li: `facebook/opt-125m`, ...
        self.model_name = args.model
        # Lucifer Li: args.batch_size > default:1
        self.batch_size_per_gpu = args.batch_size

        # Lucifer Li: `facebook/opt-125m`, ...
        self.model_config = args.model
        # Lucifer Li: `config`告诉程序"要构建一个什么样的模型", 包括模型的结构细节和采用的算法 (如注意力实现方式)
        # Lucifer Li: args.attn_implementation > `eager`; `sdpa`; `flash_attention_2`;
        config = AutoConfig.from_pretrained(
            args.model, attn_implementation=args.attn_implementation
        )

        # Lucifer Li: 分词器设置, 初始化一个与预训练模型完全匹配的分词器, 分词器负责将原始文本转换成模型能够理解的数字序列(`input_ids`, `attention_masks`)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False,legacy=False)
        # self.model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=config.torch_dtype)
        # Lucifer Li: 模型设置, 加载一个预训练的因果语言模型, 并指定了模型加载的配置 (AutoConfig)
        self.model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=torch.float16)
        # Lucifer Li: 模型能够处理的最大序列长度
        self.seqlen = self.model.config.max_position_embeddings
        self.model.eval()
        # Lucifer Li: 分词器的词汇量大小
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
