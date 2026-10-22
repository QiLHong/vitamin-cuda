# 用不到 200 行核心 PyTorch 代码理解分布式推理：DP、TP 与 PP

[中文首页](../../../README.md) · [英文原文](./readme.en.md) · [原有中文文章](./readme_cn.md)

本页逐段翻译原英文文档，代码块与历史输出原样保留；原有 `readme_cn.md` 未修改。
>
> 原文说明：文章由 AI 翻译，代码由作者编写。此中文分支保留原始英文文件与原有中文文章。
>

## 0. 前言

模型越来越大。即使 INT4 量化让权重能够放进单张 GPU，推理仍需为 KV 缓存和激活值分配显存，而这些开销会随 batch 大小和序列长度增长。单卡很容易达到显存上限，这也是使用多卡分布式推理的重要动机。

直接阅读 Megatron-LM、DeepSpeed、vLLM 或 SGLang 源码时，大量工程封装可能遮住核心思路。本文把例子压缩到最基本的组成，直接用 PyTorch 分布式通信原语实现三种典型的推理并行策略。

演示使用：

- 一个简单的两层 `Linear` 模型
- 2 张 GPU
- 不到 200 行核心代码

在此基础上构建并比较：

- 数据并行（DP）
- 张量并行（TP）
- 流水线并行（PP）

完整代码：[vitamin-cuda torch_dist](https://github.com/WingEdge777/vitamin-cuda/tree/main/infer/torch/torch_dist)；本分支对应文件为 [test.py](./test.py)。

原文列出的环境：

- PyTorch >= 2.0（宽泛的原文条件，具体 API 兼容性见下方译注）
- 2 张 NVIDIA GPU
- NCCL 通信后端

原作者建议使用 NVIDIA NGC PyTorch 容器以减少环境配置工作；仍需核对实际脚本调用的 API。

## 1. 分布式初始化与通信基础

### 1.1 初始化进程组

```python
import os
import torch
import torch.distributed as dist

def setup_dist(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=rank)
    torch.cuda.set_device(rank)

def cleanup_dist():
    dist.destroy_process_group()
```

需要理解三个术语：

- `rank`：进程在通信组中的编号，本例为 `0` 或 `1`
- `world_size`：进程总数，本例固定为 `2`
- `NCCL`：NVIDIA 的 GPU 集合通信与点对点通信后端

### 1.2 启动多个进程

```python
import torch.multiprocessing as mp

torch.manual_seed(42)
model = TwoLayerModel(hidden_size)
x = torch.randn(batch_size, hidden_size)

with torch.inference_mode():
    ref = model(x)

mp.spawn(
    run_demo,
    args=(cls, label, world_size, model, x, ref),
    nprocs=world_size,
    join=True,
)
```

本例在父进程的 CPU 上创建模型和输入，再由各子进程移动到自己的 GPU。不要把 `mp.spawn` 理解为多个工作进程直接共享父进程的 CUDA 执行状态。

### 1.3 本例使用的通信原语

| 原语 | 类型 | 用途 |
| --- | --- | --- |
| `all_gather` | 集合通信 | 收集所有 rank 的数据分片 |
| `all_reduce` | 集合通信 | 跨 rank 规约，并让各 rank 得到结果 |
| `isend` / `irecv` | 点对点通信 | 异步发送与接收 |
| `batch_isend_irecv` | 点对点通信 | 一次调用提交一组 P2P 操作 |

```text
all_gather:
Rank 0: [a] ──┐
              ├──> [[a], [b]] on every rank
Rank 1: [b] ──┘

all_reduce(SUM):
Rank 0: 10 ──┐
             ├──> 30 on every rank
Rank 1: 20 ──┘

isend / irecv:
Rank 0               Rank 1
  │                    │
  ├─── isend ─────────>│
  │<──────── irecv ────┤
  │                    │
```

## 2. 演示模型

```python
class TwoLayerModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_size, hidden_size, bias=False),
                nn.Linear(hidden_size, hidden_size, bias=False),
            ]
        )

    def forward(self, x):
        x = torch.relu(self.layers[0](x))
        return self.layers[1](x)
```

模型使用 `ModuleList`，方便后续按层切分进行流水线并行。设置 `bias=False`，使张量并行示例能够集中解释权重分片，不引入额外的偏置处理。

## 3. 数据并行（DP）

### 3.1 核心思路

每张 GPU 持有完整模型副本，把一个 batch 切给不同 rank。各 rank 计算自己的输入分片，最后收集并拼接输出。

```text
        Input [B, H]
             │
       ┌─────┴─────┐
       ▼           ▼
┌──────────┐ ┌──────────┐
│  Rank 0  │ │  Rank 1  │
│ full net │ │ full net │
│ Batch 0  │ │ Batch 1  │
└────┬─────┘ └────┬─────┘
     │            │
     ▼            ▼
 [B/2, H]     [B/2, H]
     │            │
     └─────┬──────┘
           ▼
      all_gather
           │
           ▼
       Output [B, H]
```

### 3.2 实现

```python
class DPModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.gathered = None

    def forward(self, x):
        local_batch = x.shape[0] // self.world_size
        start_idx = self.rank * local_batch
        end_idx = (self.rank + 1) * local_batch
        local_x = x[start_idx:end_idx]
        out = self.model(local_x)
        if self.gathered is None:
            self.gathered = [torch.empty_like(out) for _ in range(self.world_size)]
        dist.all_gather(self.gathered, out)
        return torch.cat(self.gathered, dim=0)
```

实现要点：

- 每个 rank 只计算自己负责的 batch 切片
- 输出收集缓冲区在首次使用时按实际输出形状分配
- `all_gather` 重建完整 batch 的输出

这是本例推理数据并行的工作方式。
训练中的经典 `DistributedDataParallel` 有不同的通信重点：主要通过梯度 `all_reduce` 同步更新所需信息。

## 4. 张量并行（TP）

### 4.1 核心思路

张量并行直接切分权重。

- 列并行线性层：切分输出特征
- 行并行线性层：切分输入特征，再用 `all_reduce` 汇总部分和

```text
Column Parallel Linear:
X [B,H]      W [H,H]
                │
        ┌───────┴───────┐
        ▼               ▼
   W_0 [H/2,H]     W_1 [H/2,H]
        │               │
        ▼               ▼
    X @ W_0^T       X @ W_1^T
     [B,H/2]         [B,H/2]

Row Parallel Linear:
X [B,H] ─split─> X_0 [B,H/2]    X_1 [B,H/2]
                    │              │
                    ▼              ▼
               X_0 @ W_0^T    X_1 @ W_1^T
                    │              │
                    └──────┬───────┘
                           ▼
                    all_reduce(SUM)
                           │
                           ▼
                      Output [B, H]
```

两种切分方式可以自然衔接：

- 列并行层在各 rank 上产生 `[B, H/2]` 输出
- 行并行层恰好以这个分片作为输入
- 两层之间无需交换这些中间分片
- 只需在第二层末尾执行 `all_reduce`

这种结构也常用于 LLM 模块的 QKV 投影与输出投影。

### 4.2 实现

```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        out_per_rank = linear.out_features // world_size
        self.weight = nn.Parameter(
            linear.weight.detach()[
                rank * out_per_rank : (rank + 1) * out_per_rank
            ].clone()
        )

    def forward(self, x):
        return x @ self.weight.t()

class RowParallelLinear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        in_per_rank = linear.in_features // world_size
        self.weight = nn.Parameter(
            linear.weight.detach()[
                :, rank * in_per_rank : (rank + 1) * in_per_rank
            ].clone()
        )

    def forward(self, x):
        out = x @ self.weight.t()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

class TPModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fc1 = ColumnParallelLinear(model.layers[0])
        self.fc2 = RowParallelLinear(model.layers[1])

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
```

本节需要抓住的组合关系是：
**列并行与行并行是常见的 TP 配对方式。**

## 5. 流水线并行（PP）

### 5.1 核心思路

流水线并行按层切分模型，让激活值依次流过不同 rank。

```text
Time T0:
┌──────────┐    ┌──────────┐
│  Rank 0  │    │  Rank 1  │
│ Layer 0  │    │  idle    │
│ compute  │    │          │
└────┬─────┘    └──────────┘
     │ isend
     ▼
 activation

Time T1:
┌──────────┐    ┌──────────┐
│  Rank 0  │    │  Rank 1  │
│  idle    │    │ Layer 1  │
│          │    │ compute  │
└──────────┘    └────┬─────┘
     ▲               │ isend
     │               ▼
     └── irecv   final output
```

这种顺序依赖会带来流水线气泡，即部分阶段等待而空闲。

rank 0 计算时，rank 1 等待输入。
随后 rank 1 计算，rank 0 又等待结果。
在本例只有两个阶段、没有微批次重叠时，空闲时间很明显；这解释了为什么这种小型演示中的 PP 延迟可能差于 TP。

实际系统通常借助微批次减少气泡：

- 训练侧：例如 `1F1B`（一次前向、一次反向）调度
- 推理侧：分块预填充或连续派发前向微批次

共同目标是让各阶段持续有任务可处理。

### 5.2 实现

```python
class PPModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        self.rank = rank
        self.world_size = world_size
        self.is_first = rank == 0
        self.is_last = rank == world_size - 1

        layers_per_rank = len(model.layers) // world_size
        start_layer = rank * layers_per_rank
        end_layer = (rank + 1) * layers_per_rank
        self.layers = nn.ModuleList(model.layers[start_layer:end_layer])

        self.start_idx = start_layer
        self.total_layers = len(model.layers)
        self.buf = None

    def _send(self, tensor, dst):
        reqs = dist.batch_isend_irecv(
            [dist.P2POp(dist.isend, tensor.contiguous(), dst)]
        )
        for req in reqs:
            req.wait()

    def _recv(self, src):
        reqs = dist.batch_isend_irecv([dist.P2POp(dist.irecv, self.buf, src)])
        for req in reqs:
            req.wait()
        return self.buf

    def forward(self, x):
        if self.buf is None:
            self.buf = torch.empty_like(x)

        if self.is_first:
            out = x
        else:
            out = self._recv(self.rank - 1)

        for i, layer in enumerate(self.layers):
            out = layer(out)
            if self.start_idx + i < self.total_layers - 1:
                out = torch.relu(out)

        if self.is_first:
            self._send(out, self.rank + 1)
            return self._recv(self.world_size - 1)
        elif self.is_last:
            self._send(out, 0)
            return None
        else:
            self._send(out, self.rank + 1)
            return None
```

### 5.3 为什么使用 `batch_isend_irecv`？

原文给出以下两个理由；有关 API 支持情况的概括需要结合译注理解：

1. 原文认为，NCCL 的阻塞 `send` / `recv` 路径与一些使用者的预期不同。
2. 原文采用 `batch_isend_irecv` 作为 NCCL P2P 接口，便于扩展到需要同时提交多次收发的流水线调度，协调操作顺序。

虽然本例每次只提交一个操作，这层包装仍保留了批量接口的形式。

> 译注：[PyTorch 分布式文档](https://docs.pytorch.org/docs/stable/distributed.html)的支持表列出 NCCL GPU 的 `send` / `recv` 支持，因此不能把原文概括为“NCCL 只支持异步接口”。`batch_isend_irecv` 也不会自动消除所有死锁；对端收发顺序、设备选择与参与进程仍需匹配。本示例在返回的 Work 上立即 `wait()`，没有实现微批次流水重叠。

## 6. 基准测试

在本目录运行：

```bash
python test.py
```

原始历史输出（不是本次运行结果）：

```text
==================================================
Demo 0: Single GPU
==================================================
[Single GPU] Output match reference: True, Avg time: 10.009 ms

==================================================
Demo 1: DP
==================================================
[DP] Output match reference: True, Avg time: 5.123 ms

==================================================
Demo 2: TP
==================================================
[TP] Output match reference: True, Avg time: 5.748 ms

==================================================
Demo 3: PP
==================================================
[PP] Output match reference: True, Avg time: 11.032 ms
```

具体数值随 GPU 与软件环境变化。原文对这组历史结果的解读是：

- DP 通过切分 batch 分担计算
- TP 通过切分计算和参数存储分担工作
- 本例 PP 受明显气泡影响，且没有微批次重叠

## 7. DP、TP、PP 对照

| 维度 | DP | TP | PP |
| --- | --- | --- | --- |
| 切分对象 | 数据 | 参数 | 层 |
| 每卡模型内容 | 完整副本 | 部分参数 | 部分层 |
| 模型存储开销（原文概括） | 较高 | 较低 | 较低 |
| 本例主要通信 | `all_gather` | `all_reduce` | `isend` / `irecv` |
| 通信时机 | 每次推理收集输出 | 每个 TP 规约点 | 阶段交接时 |
| 原文概括的适用情况 | 模型较小、batch 较大 | 隐藏层较宽、延迟要求较高 | 模型很深 |
| 延迟特征（依具体实现） | 通常较好 | 通常较好 | 易受流水线气泡影响 |

原文给出的经验选择：

- 模型能单卡容纳且需要吞吐时考虑 DP
- 单层过宽、单卡难以容纳时考虑 TP
- 模型过深而需要按层分配时考虑 PP

实际 LLM 服务通常组合使用这些策略：

- 用 `TP + PP` 分布超大模型
- 用 `DP` 提高请求吞吐量

在预填充阶段，TP 可用于降低首词元延迟，效果仍取决于计算规模与通信成本。
在解码阶段，合适的 TP 头部切分也可把 KV 缓存分摊到多卡，这可能与切分计算同样重要；具体分布取决于注意力结构和框架实现。

## 8. 延伸阅读

本演示限定为单机两卡，以便清楚展示概念。生产环境可结合 `torchrun`、NCCL 环境配置和网络条件扩展到多节点；InfiniBand、RoCE 等支持 RDMA 的网络常用于降低跨节点通信成本，但集群部署不在本文范围内。

理解通信模式后，还可以继续研究张量在单张 GPU 上的执行方式，以及如何用自定义核函数优化计算。

vLLM、SGLang 等推理框架也会把优化后的自定义核函数封装为 PyTorch 算子，与分布式调度配合。

相关手写 CUDA 核函数与底层实验可继续阅读作者的 GitHub 项目 [vitamin-cuda](https://github.com/WingEdge777/vitamin-cuda)。

## 本分支补充：按实际脚本理解范围

`test.py` 固定 `world_size=2`、两层同宽线性层与固定 batch。DP 的整数切分需要 batch 可被进程数整除；TP 的特征维也需要相应整除。PP 的接收缓冲与输入形状相同，不能直接套到层宽变化的任意模型。PyTorch 的 `weight` 存储顺序是 `[out_features, in_features]`：列并行在这块张量上切第 0 维，行并行切第 1 维。

原环境说明 `PyTorch >=2.0` 并不是对每个后续版本的兼容保证。源码使用 `init_process_group(..., device_id=rank)`，请核对安装版本对该参数及整数设备编号的支持。参考 [PyTorch 进程组初始化文档](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)。

`benchmark` 的 `allclose` 结果只被打印，不会在 False 时断言失败；应主动查看输出，不能以进程退出成功代替正确性验证。历史性能表是此演示的结果，不是各种真实 LLM、网络拓扑与 batch 下的固定排序。本次未执行 CUDA、多 GPU/NCCL 正确性或性能测试。
