# PWL、MAD、DSQ-style 实施报告

## 1. 实际修改的文件

代码修改严格限定为：

- `quantize/quantizer.py`
- `train_main.py`
- `wang.sh`

另外按任务要求新增本报告文件 `PWL_MAD_DSQ_实施报告.md`；它不是代码文件。权威方案文件 `PWL_MAD_DSQ_最小代码改动方案.md` 未修改。

## 2. 每个文件的具体改动

### `quantize/quantizer.py`

- 新增 `PWLQuantize`、`MADQuantize`、`DSQQuantize` 三个 custom autograd Function。
- 三者都在同一个 autograd 节点中执行 hard forward：`round(x) -> clamp(qmin, qmax)`。
- 新增 `PWLModule`、`MADModule`、`DSQModule` wrapper，并通过 `extra_repr()` 输出量化范围；DSQ 额外输出 `dsq_alpha`。
- `UniformAffineQuantizer` 新增独立参数 `dsq_alpha: float = 0.2`，没有复用已有 `alpha`。
- 新增 `round_module_handles_clamp` 标记。PWL、MAD、DSQ 路径由 custom Function 完整接管 round + clamp，之后不再经过 native clamp；STE、HTGE、Uniform、Normal、Laplace 仍走原有 round surrogate + native clamp 路径。
- 对 PWL、MAD、DSQ 的非空 zero-point 明确抛出 `NotImplementedError`；现有方法的 zero-point 路径保持不变。
- scale、LWC、group reshape、dequantize 和 `n_bits >= 16` 路径未改变。

实际 backward 公式如下，其中范围判断使用连续归一化输入 `u in [qmin, qmax]`：

- PWL：范围内为 `1`，范围外为 `0`。
- MAD：`u < qmin` 时为 `qmin / u`；范围内为 `1`；`u > qmax` 时为 `qmax / u`。除法只在真实范围外 mask 上执行。
- DSQ：固定 `alpha = 0.2`，`beta = log(2 / alpha - 1) = log(9)`；使用 `boundary = floor(u) + 0.5`，范围内梯度为 `beta / (2*tanh(beta/2)) * (1 - tanh(beta*(u-boundary))^2)`，范围外为 `0`。

DSQ 是 hard-forward/soft-backward；没有 soft forward、可学习 alpha、annealing 或额外正则项。

### `train_main.py`

- `OurArguments` 新增 `dsq_alpha: float = 0.2`。
- trainer 白名单加入 `PWL`、`MAD`、`DSQ`，保留全部已有方法。
- 将 PWL/MAD 的 `method` 传播到 weight、act、q、k、v、p 六组 quantizer 参数字典。
- 将 DSQ 的 `method="DSQ"` 和 `dsq_alpha` 传播到同一组六个参数字典。
- 未向 PWL/MAD 传播 `delta` 或 `t`，也未改变冻结参数、`descale` 训练、trainer 构造和多 GPU 逻辑。

### `wang.sh`

- 支持的方法列表扩展为 `STE HTGE Uniform Normal PWL MAD DSQ`。
- 支持任务注释更新为 `SST2 RTE WIC SQuAD`。
- 新增固定 `DSQ_ALPHA=0.2` 和 PWL/MAD/DSQ 的单方法分支；仍然一次调用一个方法、一个任务，没有训练循环。
- 新增 SQuAD 判断：SQuAD 使用 `TRAIN_AS_CLASSIFICATION=False`，其他三个任务使用 `True`。
- 向 `train_main.py` 传入 `--dsq_alpha "$DSQ_ALPHA"`。
- 原有模型、checkpoint、W4A16、训练步数、学习率、scheduler、batch size、最大长度、GPU、debugpy、保存策略等配置未改动。

## 3. 检查与单元测试结果

- `python -m py_compile train_main.py quantize/quantizer.py`：通过。
- `bash -n wang.sh`：通过。
- 方案原文的 custom autograd 解析梯度测试：通过。首次使用系统默认 Python 尝试时因该环境未安装 `torch` 而未进入测试代码；随后使用服务器已有 `/home/wangwenkang/miniconda3/envs/RoundZO/bin/python`（PyTorch 2.9.1）原样重跑并通过，未安装或下载任何依赖。
- hard-forward 集成检查：通过。PWL、MAD、DSQ 与现有 STE 的输出逐元素严格相同，覆盖正常范围、正负饱和、半整数和整数。
- `n_bits >= 16` 恒等返回检查：通过。
- MAD 量化器级范围外梯度检查：通过，范围外梯度未被第二个 native clamp 截断。
- zero-point 防误用检查：通过。W4 no-zero-point 配置为 `disable_zero_point=True`、`qmin=-8`、`qmax=7`；三个新增方法对非空 zero-point 均抛出 `NotImplementedError`；现有 STE zero-point 路径仍正常。
- `git diff --check`：通过。
- 修改范围检查：通过。生成本报告前，所有代码 diff 仅涉及上述三个允许文件；方案文件无 diff。

## 4. 未执行的操作

没有运行 `wang.sh`，没有启动 Qwen3-8B 训练或评估，没有下载模型、数据或依赖；没有执行 commit、push、reset、checkout 或删除操作。

## 5. 与方案的偏离

未偏离方案。

实施期间 Codex 的 `apply_patch` 入口因服务器沙箱初始化错误无法工作，因此使用了带唯一锚点校验、全部校验成功后才写回的内存文本替换。该工具层替代没有改变代码范围或方案语义。

## 6. 后续运行方式

1. 进入仓库：`cd /home/wangwenkang/RoundLocalZO2`。
2. 在 `wang.sh` 顶部把 `METHOD` 设置为单个方法：`PWL`、`MAD` 或 `DSQ`。
3. 把 `TASK` 设置为单个任务：`SST2`、`RTE`、`WIC` 或 `SQuAD`。SQuAD 会自动传入 `train_as_classification=False`。
4. 按正式实验要求设置 `STEPS`；其余既有实验配置保持不变。DSQ 保持 `DSQ_ALPHA=0.2`。
5. 每次只运行一个方法与一个任务：`bash wang.sh`。脚本保留现有 debugpy `--wait-for-client`，因此需要按原流程连接调试器后才会继续。
6. 其余方法/任务组合需要手动逐次修改 `METHOD` 和 `TASK` 后分别调用，不要增加循环。
