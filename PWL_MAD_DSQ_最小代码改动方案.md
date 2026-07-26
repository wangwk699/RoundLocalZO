# RoundLocalZO：PWL、MAD、DSQ-style 最小代码改动方案

## 0. 文档用途

本文档交给部署在服务器、没有本次对话背景的 Codex 执行。

服务器 Codex 的任务仅限于：

1. 以仓库 `https://github.com/wangwk699/RoundLocalZO` 的 `dev2` 分支为代码基线；
2. 在现有 W4A16 downstream fine-tuning 流程中新增 `PWL`、`MAD`、`DSQ` 三种 surrogate-gradient 选项；
3. 保持 hard quantization forward、模型、训练流程和现有四种方法不变；
4. 完成静态检查和小型解析单元测试；
5. 不启动 Qwen3-8B 训练，不运行 `wang.sh`，不下载模型或数据。

本方案在本地核对的 `dev2` 提交为：

```text
9baa557697f5b7d7dc73eda569b2de50c45f7722
```

该提交的关键文件为：

```text
wang.sh
train_main.py
quantize/quantizer.py
```

如果服务器上的 `dev2` 已经更新，必须先检查相关代码是否仍具有本文描述的结构，再做等价的最小适配。不得通过 `git reset --hard`、覆盖文件或丢弃服务器已有修改来强行匹配上述提交。

---

## 1. 已确认的实验语义

### 1.1 实验配置

- 模型：`Qwen/Qwen3-8B`

- 量化：W4A16

- 新 baseline：`PWL`、`MAD`、`DSQ`

- 已有方法必须保持：`STE`、`HTGE`、`Uniform`、`Normal`

- downstream tasks：

  - `SST2`
  - `RTE`
  - `WIC`
  - `SQuAD`

- DSQ 使用固定：

  ```text
  dsq_alpha = 0.2
  beta = log(2 / dsq_alpha - 1) = log(9) ≈ 2.197224577
  ```

- DSQ 不学习 `alpha`，不增加正则项，不做 annealing。

- 三个新增方法均保持 hard forward，仅替换 backward surrogate。

- 后续训练由用户自行运行；服务器 Codex 不得启动实验。

### 1.2 不允许改变的内容

- 不改变当前 `round -> clamp -> dequantize` 的 forward 数值。
- 不改变 Qwen3-8B checkpoint 路径。
- 不改变 W4A16、LWC、学习率、scheduler、batch size、训练步数、seed、最大长度、GPU 选择或 debugpy 调用方式。
- 不修改现有 `STE`、`HTGE`、`Uniform`、`Normal` 的公式或调用路径。
- 不把 DSQ 实现成 soft forward。
- 不学习 DSQ 的 `alpha`。
- 不给 MAD 增加衰减指数、epsilon 或可调系数。
- 不把 PWL 直接别名为 STE。
- 不修改 `large_language_models/quantize/quantizer.py`。根目录的 `train_main.py` 导入的是根目录 `quantize/quantizer.py`。
- 不创建实验循环，不运行 12 个实验。
- 不改依赖，不安装包。

---

## 2. 开始修改前的强制检查

在仓库根目录运行：

```bash
git branch --show-current
git rev-parse HEAD
git status --short
sed -n '1,120p' wang.sh
grep -n "class UniformAffineQuantizer" quantize/quantizer.py
grep -n "x_int = self.round_module" quantize/quantizer.py
grep -n "args.trainer" train_main.py
```

预期：

- 当前分支为 `dev2`；
- `wang.sh` 通过 `--trainer "$METHOD"` 调用 `train_main.py`；
- `UniformAffineQuantizer.fake_quant()` 当前先调用 `round_module`，随后调用原生 `clamp`；
- `train_main.py` 当前只接受 `STE/HTGE/Uniform/Normal/Laplace` 等已有 trainer 名称。

如果工作树已有修改：

- 保留服务器已有修改；
- 先阅读 diff；
- 将本方案以最小增量合并进去；
- 若已有修改与本方案修改同一区域且语义不清楚，停止并报告用户，不得覆盖。

---

## 3. 为什么新增方法必须接管完整的 `round + clamp`

当前 `quantize/quantizer.py` 的核心顺序是：

```python
u = x / eff_scale
x_int = self.round_module(u)
x_int = x_int.clamp(self.qmin, self.qmax)
```

PyTorch 原生 `clamp` 在饱和范围外的 backward 为零。

如果只新增一个 MAD rounding module，那么范围外的上游梯度在到达 MAD backward 之前已经被原生 clamp 清零，MAD 的幅值衰减永远不会生效。

因此，`PWL`、`MAD`、`DSQ` 三种新增方法必须各自使用一个 custom autograd Function，在同一个节点中完成：

```text
forward:  round(u) -> clamp(qmin, qmax)
backward: method-specific surrogate gradient
```

对三个新增方法，`fake_quant()` 后面不得再次经过一个参与 autograd 的原生 clamp。否则 MAD 会再次退化为范围外零梯度。

现有方法继续沿用当前路径，不得改变：

```text
existing round surrogate -> native clamp
```

---

## 4. 三种新增 backward 的精确定义

设归一化量化输入为：

```math
u = x / s_q
```

W4 且 `disable_zero_point=True` 时：

```math
q_{\min}=-8,\qquad q_{\max}=7.
```

三个新增方法的 hard forward 完全相同：

```math
q_{\mathrm{hard}}(u)
=
\operatorname{clamp}
\left(
\operatorname{round}(u),
q_{\min},
q_{\max}
\right).
```

范围判断统一使用归一化连续输入：

```math
u\in[q_{\min},q_{\max}].
```

不要改成半整数扩展区间 `[-8.5, 7.5]`。

### 4.1 PWL

```math
g_{\mathrm{PWL}}(u)
=
\begin{cases}
1,&q_{\min}\le u\le q_{\max},\\
0,&\text{otherwise}.
\end{cases}
```

PWL 没有方法特有超参数。

### 4.2 MAD

使用适配非对称 two's-complement 范围的表达式：

```math
g_{\mathrm{MAD}}(u)
=
\begin{cases}
q_{\min}/u,&u<q_{\min},\\
1,&q_{\min}\le u\le q_{\max},\\
q_{\max}/u,&u>q_{\max}.
\end{cases}
```

左右范围外的比值均为正数。

MAD 没有方法特有超参数。实现除法时不得在 `u=0` 上生成实际选中的 `Inf/NaN`；只在对应的范围外 mask 上执行除法，或使用安全分母。

### 4.3 DSQ-style hard-forward/soft-backward

定义当前 unit interval 内的量化边界：

```math
b(u)=\lfloor u\rfloor+\frac12,
\qquad
d(u)=u-b(u).
```

固定：

```math
\alpha=0.2,
\qquad
\beta=\log\left(\frac{2}{\alpha}-1\right)=\log 9.
```

DSQ backward：

```math
g_{\mathrm{DSQ}}(u)
=
\begin{cases}
\dfrac{\beta}{2\tanh(\beta/2)}
\operatorname{sech}^{2}\left(\beta d(u)\right),
&q_{\min}\le u\le q_{\max},\\
0,&\text{otherwise}.
\end{cases}
```

数值实现建议使用：

```python
tanh_z = torch.tanh(beta * (u - boundary))
local_grad = beta / (2.0 * math.tanh(beta / 2.0))
local_grad = local_grad * (1.0 - tanh_z.square())
```

不要使用现有 HTGE 中的 `floor/ceil` midpoint 写法。DSQ 必须使用：

```python
boundary = torch.floor(u) + 0.5
```

这样整数点属于相邻 soft interval 的端点，而不是梯度峰值。

---

## 5. 只允许修改的文件

```text
quantize/quantizer.py
train_main.py
wang.sh
```

不要修改其他代码文件。

---

## 6. 修改 `quantize/quantizer.py`

### 6.1 给 quantizer 增加 DSQ 参数

在 `UniformAffineQuantizer.__init__()` 参数中增加：

```python
dsq_alpha: float = 0.2,
```

保存：

```python
self.dsq_alpha = dsq_alpha
```

不要复用 `train_main.py` 中已有的 `alpha` 字段。已有 `alpha` 属于原量化流程，和 DSQ similarity factor 不是同一个配置。

### 6.2 新增三个完整 hard-quant custom Functions

在 `quantize/quantizer.py` 中新增三个 `torch.autograd.Function`，建议命名：

```python
PWLQuantize
MADQuantize
DSQQuantize
```

它们的 forward 必须直接返回：

```python
torch.round(x).clamp(qmin, qmax)
```

#### PWL 参考结构

```python
class PWLQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, qmin, qmax):
        ctx.save_for_backward(x)
        ctx.qmin = qmin
        ctx.qmax = qmax
        return torch.round(x).clamp(qmin, qmax)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        mask = (x >= ctx.qmin) & (x <= ctx.qmax)
        local_grad = mask.to(dtype=x.dtype)
        return grad_output * local_grad, None, None
```

#### MAD 参考结构

```python
class MADQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, qmin, qmax):
        ctx.save_for_backward(x)
        ctx.qmin = qmin
        ctx.qmax = qmax
        return torch.round(x).clamp(qmin, qmax)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        qmin = ctx.qmin
        qmax = ctx.qmax

        local_grad = torch.ones_like(x)
        lower = x < qmin
        upper = x > qmax

        # 只在真实范围外元素上做除法，避免对 x=0 计算有效除法。
        if lower.any():
            local_grad[lower] = qmin / x[lower]
        if upper.any():
            local_grad[upper] = qmax / x[upper]

        return grad_output * local_grad, None, None
```

如果为了兼容编译/向量化而不希望使用布尔索引，可以使用 `torch.where` 和安全分母，但结果必须与上述公式严格一致。

#### DSQ 参考结构

```python
class DSQQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, qmin, qmax, dsq_alpha):
        if not 0.0 < dsq_alpha < 0.5:
            raise ValueError(
                f"dsq_alpha must be in (0, 0.5), got {dsq_alpha}"
            )
        ctx.save_for_backward(x)
        ctx.qmin = qmin
        ctx.qmax = qmax
        ctx.dsq_alpha = dsq_alpha
        return torch.round(x).clamp(qmin, qmax)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.dsq_alpha
        beta = math.log(2.0 / alpha - 1.0)

        boundary = torch.floor(x) + 0.5
        z = beta * (x - boundary)
        tanh_z = torch.tanh(z)
        normalization = beta / (2.0 * math.tanh(beta / 2.0))
        local_grad = normalization * (1.0 - tanh_z.square())

        mask = (x >= ctx.qmin) & (x <= ctx.qmax)
        local_grad = local_grad * mask.to(dtype=x.dtype)

        return grad_output * local_grad, None, None, None
```

### 6.3 新增三个轻量 Module wrapper

沿用现有 `UniformModule`、`NormalModule`、`HTGEModule` 的风格，新增：

```python
PWLModule
MADModule
DSQModule
```

建议接口：

```python
class PWLModule(nn.Module):
    def __init__(self, qmin, qmax):
        super().__init__()
        self.qmin = qmin
        self.qmax = qmax

    def forward(self, x):
        return PWLQuantize.apply(x, self.qmin, self.qmax)


class MADModule(nn.Module):
    def __init__(self, qmin, qmax):
        super().__init__()
        self.qmin = qmin
        self.qmax = qmax

    def forward(self, x):
        return MADQuantize.apply(x, self.qmin, self.qmax)


class DSQModule(nn.Module):
    def __init__(self, qmin, qmax, dsq_alpha):
        super().__init__()
        self.qmin = qmin
        self.qmax = qmax
        self.dsq_alpha = dsq_alpha

    def forward(self, x):
        return DSQQuantize.apply(
            x, self.qmin, self.qmax, self.dsq_alpha
        )
```

为三个 wrapper 增加简洁的 `extra_repr()`，方便日志确认实际方法、范围及 `dsq_alpha`。

### 6.4 在 quantizer 初始化中选择新方法

在 `UniformAffineQuantizer.__init__()` 中增加一个布尔标记，例如：

```python
self.round_module_handles_clamp = False
```

新增方法选择：

```python
if self.method == "PWL":
    self.round_module = PWLModule(self.qmin, self.qmax)
    self.round_module_handles_clamp = True
elif self.method == "MAD":
    self.round_module = MADModule(self.qmin, self.qmax)
    self.round_module_handles_clamp = True
elif self.method == "DSQ":
    self.round_module = DSQModule(
        self.qmin, self.qmax, self.dsq_alpha
    )
    self.round_module_handles_clamp = True
elif ...:
    # 原有 Uniform/Normal/Laplace/HTGE/STE 分支保持原样
```

新分支应放在依赖 `delta` 或 `t` 的现有判断之前，因为 PWL/MAD 不需要这些参数。

### 6.5 修改 `fake_quant()`，避免第二次 native clamp

保留当前 scale、group reshape、dequantize、LWC 等逻辑，只调整 round/clamp 的最小局部。

目标结构：

```python
normalized_x = x * (1.0 / eff_scale)
x_int = self.round_module(normalized_x)

if self.round_module_handles_clamp:
    if round_zero_point is not None:
        raise NotImplementedError(
            f"{self.method} full round+clamp surrogate currently "
            "supports disable_zero_point=True only"
        )
    # x_int 已经在 custom Function forward 中完成 round + clamp。
    # 此处绝对不能再次调用参与 autograd 的 native clamp。
else:
    if round_zero_point is not None:
        x_int = x_int.add(round_zero_point)
    x_int = x_int.clamp(self.qmin, self.qmax)
```

后面的 dequantize 逻辑保持原样。

为什么要显式拒绝 zero-point：

- 本次实验是 W4、`symmetric=True`、`disable_zero_point=True`；
- 当前确认的 PWL/MAD/DSQ 公式基于无 zero-point 的归一化区间；
- 静默支持其他配置可能让 full `round + clamp` wrapper 的范围语义错误。

不得为了“通用化”而擅自推导 asymmetric zero-point 版本。

---

## 7. 修改 `train_main.py`

### 7.1 新增命令行参数

在 `OurArguments` 中、现有 `delta` 和 `t` 附近增加：

```python
dsq_alpha: float = 0.2
```

不要修改已有：

```python
alpha: float = 0.5
```

两者用途不同。

### 7.2 扩展 trainer 白名单

当前 `Framework.train()` 的白名单需要加入：

```text
PWL
MAD
DSQ
```

建议将冗长的 `or` 判断最小改成：

```python
if (
    self.args.trainer
    in [
        "STE",
        "HTGE",
        "Uniform",
        "Normal",
        "Laplace",
        "PWL",
        "MAD",
        "DSQ",
    ]
    and self.args.quant_method != ""
):
```

不要改变其余冻结参数、只训练 `descale`、trainer 构造和多 GPU 逻辑。

### 7.3 将新方法写入 quantizer 参数字典

沿用现有 `HTGE`、`Uniform/Normal/Laplace` 的配置传播方式。

对 `PWL`、`MAD`：

```python
param_dict["method"] = args.trainer
```

对 `DSQ`：

```python
param_dict["method"] = "DSQ"
param_dict["dsq_alpha"] = args.dsq_alpha
```

应用到现有同一组字典：

```python
args.weight_quant_params
args.act_quant_params
args.q_quant_params
args.k_quant_params
args.v_quant_params
args.p_quant_params
```

虽然 W4A16 中只有 weight quantizer 真正执行低比特路径，但应保持和现有方法相同的参数传播风格。`n_bits >= 16` 的 quantizer 会直接返回输入。

不要给 PWL/MAD 添加 `delta` 或 `t` 到 quantizer 参数。

---

## 8. 修改 `wang.sh`

`wang.sh` 继续保持“一次只运行一个方法、一个任务”的调用方式，不增加循环。

### 8.1 扩展方法列表

更新注释，使 `METHOD` 支持：

```text
STE HTGE Uniform Normal PWL MAD DSQ
```

新增：

```bash
DSQ_ALPHA=0.2
```

### 8.2 增加方法分支

现有四个方法分支及其目录命名不得改变。

新增 PWL/MAD 分支。由于脚本末尾仍统一传入 `--delta` 和 `--t`，可以保留与 STE 相同的 dummy 值，但这些值不得进入 PWL/MAD 公式：

```bash
elif [ "$METHOD" == "PWL" ] || [ "$METHOD" == "MAD" ]; then
    T=16
    DELTA=0.285
    DIR_SUFFIX="BATCH_SIZE-$BATCH_SIZE"
```

新增 DSQ 分支：

```bash
elif [ "$METHOD" == "DSQ" ]; then
    T=16
    DELTA=0.285
    DIR_SUFFIX="DSQ_ALPHA-$DSQ_ALPHA-BATCH_SIZE-$BATCH_SIZE"
```

在 Python 调用中新增：

```bash
--dsq_alpha "$DSQ_ALPHA" \
```

### 8.3 正确处理 SQuAD

代码中的 `SQuADDataset` 是 generation task，`sample.candidates` 为 `None`。如果继续传：

```bash
--train_as_classification True
```

`Framework.train()` 的 classification 数据转换会访问 `len(sample.candidates)`，不适用于 SQuAD。

因此在 `wang.sh` 中根据任务设置：

```bash
if [ "$TASK" == "SQuAD" ]; then
    TRAIN_AS_CLASSIFICATION=False
else
    TRAIN_AS_CLASSIFICATION=True
fi
```

将原来的：

```bash
--train_as_classification True \
```

替换为：

```bash
--train_as_classification "$TRAIN_AS_CLASSIFICATION" \
```

任务字符串必须使用仓库中的类名大小写：

```text
SST2
RTE
WIC
SQuAD
```

不要写成 `WiC`、`SQUAD` 或 `SQuADv2`。

### 8.4 不得改动的脚本配置

不要修改当前：

- `STEPS`
- `IR`
- `IR_scheduler`
- `Warmup_ratio`
- `MODEL`
- `BATCH_SIZE`
- `WBITS`
- `ABITS`
- `MAX_LENGTH`
- `RESUME`
- `CUDA_VISIBLE_DEVICES`
- debugpy 参数
- checkpoint/save 策略

用户后续会自行设置方法、任务和正式训练步数。

---

## 9. 强制验证：不得运行训练

### 9.1 语法检查

```bash
python -m py_compile train_main.py quantize/quantizer.py
bash -n wang.sh
```

### 9.2 custom Function 解析单元测试

在仓库根目录执行下面的临时内联测试。不得创建长期训练任务：

```bash
python - <<'PY'
import math
import torch

from quantize.quantizer import PWLQuantize, MADQuantize, DSQQuantize

qmin, qmax = -8, 7
values = torch.tensor(
    [-16.0, -10.0, -8.0, -7.75, -1.0, 0.0, 0.25, 6.75, 7.0, 8.0, 14.0],
    dtype=torch.float64,
)
hard_expected = torch.round(values).clamp(qmin, qmax)


def run(function, *extra):
    x = values.clone().requires_grad_(True)
    y = function.apply(x, qmin, qmax, *extra)
    assert torch.equal(y.detach(), hard_expected), (function.__name__, y, hard_expected)
    y.sum().backward()
    return x.grad


pwl_grad = run(PWLQuantize)
pwl_expected = ((values >= qmin) & (values <= qmax)).to(values.dtype)
torch.testing.assert_close(pwl_grad, pwl_expected)

mad_grad = run(MADQuantize)
mad_expected = torch.ones_like(values)
lower = values < qmin
upper = values > qmax
mad_expected[lower] = qmin / values[lower]
mad_expected[upper] = qmax / values[upper]
torch.testing.assert_close(mad_grad, mad_expected)

alpha = 0.2
dsq_grad = run(DSQQuantize, alpha)
beta = math.log(2.0 / alpha - 1.0)
boundary = torch.floor(values) + 0.5
z = beta * (values - boundary)
dsq_expected = (
    beta
    / (2.0 * math.tanh(beta / 2.0))
    * (1.0 - torch.tanh(z).square())
)
dsq_expected *= ((values >= qmin) & (values <= qmax)).to(values.dtype)
torch.testing.assert_close(dsq_grad, dsq_expected)

assert torch.isfinite(pwl_grad).all()
assert torch.isfinite(mad_grad).all()
assert torch.isfinite(dsq_grad).all()

print("PWL/MAD/DSQ custom-function checks passed")
PY
```

### 9.3 quantizer 集成 forward 检查

再增加一个轻量检查，构造相同输入和相同 quantizer 配置，确认 PWL/MAD/DSQ 的输出逐元素等于现有 STE hard forward。

检查必须覆盖：

- 正常范围；
- 正负饱和值；
- 半整数；
- 整数；
- `n_bits >= 16` 时恒等返回。

如果集成测试发现 forward 不一致，停止并修复，不得用容差掩盖 hard-forward 差异。相同 dtype 下应逐元素相同。

### 9.4 zero-point 防误用检查

确认：

- 本次 W4 weight quantizer 的 `disable_zero_point=True`；
- 对新增 full `round + clamp` 方法，如果 `round_zero_point is not None`，代码会明确抛出 `NotImplementedError`；
- 现有方法的 zero-point 路径完全不受影响。

### 9.5 改动范围检查

```bash
git diff --check
git status --short
git diff -- quantize/quantizer.py train_main.py wang.sh
```

最终只允许三个文件有代码改动：

```text
quantize/quantizer.py
train_main.py
wang.sh
```

---

## 10. 完成标准

服务器 Codex 只有在以下条件全部满足后才能报告完成：

- [ ] `PWL` 可通过 `--trainer PWL` 选择；
- [ ] `MAD` 可通过 `--trainer MAD` 选择；
- [ ] `DSQ` 可通过 `--trainer DSQ --dsq_alpha 0.2` 选择；
- [ ] 三种方法 forward 都是严格的 `round -> clamp`；
- [ ] 新增方法不会再经过第二个 native clamp；
- [ ] PWL backward 与范围 mask 完全一致；
- [ ] MAD 在范围外保留正的幅值衰减，而不是被 clamp 清零；
- [ ] DSQ 使用 `floor(u)+0.5` 和固定 `alpha=0.2`；
- [ ] DSQ 没有 soft forward、learnable alpha 或 annealing；
- [ ] 现有 STE/HTGE/Uniform/Normal 未改变；
- [ ] `wang.sh` 支持 PWL/MAD/DSQ；
- [ ] `wang.sh` 对 SQuAD 自动设置 `train_as_classification=False`；
- [ ] Python 和 Bash 语法检查通过；
- [ ] 解析梯度测试通过；
- [ ] hard-forward 集成检查通过；
- [ ] 没有启动训练、评估或模型下载；
- [ ] 没有修改三个允许文件以外的代码。

---

## 11. 服务器 Codex 的最终报告格式

完成后只需向用户报告：

1. 修改了哪些文件；
2. 每个文件增加了什么；
3. PWL/MAD/DSQ 的实际 backward 公式；
4. hard-forward 等价检查是否通过；
5. 语法和单元测试是否通过；
6. 明确声明没有运行 Qwen3-8B 训练；
7. 如有任何偏离本方案的地方，逐项说明原因。

不得虚构实验分数，不得声称新 baseline 优于任何已有方法。

