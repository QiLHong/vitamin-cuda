# 中文导读：在单机两卡上比较 DP、TP、PP；仅补注释，不调整通信顺序或原基准设置。
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


# 单机示例让 rank 同时作为 GPU 编号；MASTER_ADDR/PORT 供各子进程会合。
# device_id 参数的形式需由所用 PyTorch 版本支持，不能只凭 README 的宽泛版本下限判断。
def setup_dist(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=rank)
    torch.cuda.set_device(rank)


def cleanup_dist():
    dist.destroy_process_group()


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


# 数据并行：每个进程保留完整模型，仅按 batch 维切分输入，再收集所有输出。
# 当前整数除法要求 batch 可被 world_size 整除；缓存的 gathered 还假定后续输出形状不变。
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


# 按输出特征切分。PyTorch 权重实际存为 [out_features, in_features]，因此这里切第 0 维。
# “列并行”来自矩阵乘法约定，不表示在这个转置存储的 weight 上切列。
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


# 按输入特征切分 weight 的第 1 维；各 rank 产生部分和，all_reduce 求和恢复完整输出。
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


# 第一层输出分片恰好是第二层需要的输入分片，ReLU 在本地分片上执行。
class TPModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fc1 = ColumnParallelLinear(model.layers[0])
        self.fc2 = RowParallelLinear(model.layers[1])

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# 流水线并行：按层分工；当前演示固定两层、两卡，没有微批次重叠调度。
# 层数须能按进程数分配，buf 的 shape 固定等于输入；不能直接套到任意层宽或单进程。
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
        # 这里立即等待 Work 完成；异步 API 的名称不代表本示例已经重叠计算与通信。
        for req in reqs:
            req.wait()

    def _recv(self, src):
        reqs = dist.batch_isend_irecv([dist.P2POp(dist.irecv, self.buf, src)])
        # 这里立即等待 Work 完成；异步 API 的名称不代表本示例已经重叠计算与通信。
        for req in reqs:
            req.wait()
        return self.buf

    def forward(self, x):
        if self.buf is None:
            self.buf = torch.empty_like(x)

        # 1. 接收阶段：首个 rank 直接使用输入 x；其余 rank 使用上一阶段发送的激活值。
        if self.is_first:
            out = x
        else:
            out = self._recv(self.rank - 1)

        # 2. 计算本进程负责的层，仅在全模型的非末层后添加 ReLU。
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if self.start_idx + i < self.total_layers - 1:
                out = torch.relu(out)

        # 3. 发送阶段：首 rank 等待最终输出，末 rank 把结果发回 0，中间 rank 发往下一阶段。
        if self.is_first:
            self._send(out, self.rank + 1)
            return self._recv(self.world_size - 1)
        elif self.is_last:
            self._send(out, 0)
            return None
        else:
            self._send(out, self.rank + 1)
            return None


@torch.inference_mode()
def benchmark(fn, *args, ref=None, warmup=3, repeat=10):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    start = time.perf_counter()
    for _ in range(repeat):
        out = fn(*args)
    torch.cuda.synchronize()
    avg_time = (time.perf_counter() - start) / repeat * 1000
    # 原脚本只打印 allclose 结果，不会在 False 时触发断言。
    if ref is not None:
        print(
            f"Output match reference: {torch.allclose(out, ref.cuda(), rtol=1e-4, atol=1e-4)}, Avg time: {avg_time:.3f} ms"
        )
    return out, avg_time


def run_demo(rank, cls, label, world_size, model, x, ref):
    setup_dist(rank, world_size)
    dist_model = cls(model).cuda()
    device_x = x.cuda()
    if rank == 0:
        print(f"[{label}]", end=" ")
        benchmark(dist_model, device_x, ref=ref)
    else:
        benchmark(dist_model, device_x)
    cleanup_dist()


if __name__ == "__main__":
    world_size = 2
    hidden_size = 8192
    batch_size = 1024
    # 原示例使用较大 batch，以观察 DP/TP 的潜在加速；工作量小时通信开销可能占主导。

    torch.manual_seed(42)
    model = TwoLayerModel(hidden_size)
    x = torch.randn(batch_size, hidden_size)
    with torch.inference_mode():
        ref = model(x)

    demos = [("Single GPU", None), ("DP", DPModel), ("TP", TPModel), ("PP", PPModel)]
    for i, (label, cls) in enumerate(demos):
        print(f"\n{'='*50}\nDemo {i}: {label}\n{'='*50}")
        if cls is None:
            model.cuda()
            x_cuda = x.cuda()
            print(f"[{label}]", end=" ")
            benchmark(model, x_cuda, ref=ref)
            model.cpu()
        else:
            # 子进程从 CPU 模型和输入开始，在 setup_dist 后移动到各自 GPU。
            mp.spawn(
                run_demo,
                args=(cls, label, world_size, model, x, ref),
                nprocs=world_size,
                join=True,
            )
