// 中文导读：这些辅助类型负责“按一组字节搬运数据”，不是计算新的数值。
// 初学者先读 FLOAT4/HALF2，再读 pack128，最后回到 elementwise.cu 看用法。
// 本分支只增加说明，保留原始宏、成员名、类型和布局。
#pragma once
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector_types.h>

// reinterpret_cast 改变的是访问内存的类型视图，不会把 half 的数值转换为 float。
// 例如 FLOAT4(a[idx]) 从 &a[idx] 开始访问连续 16 字节；表达式也可在赋值号左边使用。
// 必须由调用者保证：地址满足目标类型对齐、整组字节有效、张量类型与指针类型一致。
// float4/uint2/half2 等各有自己的对齐要求；“张量连续”并不等价于“任意切片都对齐”。
// 这类低层类型重解释依赖目标编译环境；不要把它当作可无条件移植的通用 C++ 写法。
#define UINT2(value) (reinterpret_cast<uint2 *>(&(value))[0])
#define FLOAT2(value) (reinterpret_cast<float2 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
// 128 bit = 16 byte。这里借用 float4 作为搬运视图，并不要求原数据是四个 FP32。
// 宏名表达作者意图；最终是否生成单条宽访存指令，应查看编译后的 PTX/SASS。
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// #f 是预处理器“字符串化”：把函数标识符 f 变成 Python 可见的函数名。
// &f 是 C++ 函数地址；m 来自各 .cu 文件末尾的 PYBIND11_MODULE。
#define torch_pybinding_func(f) m.def(#f, &f, #f)

// 本项目使用 NVIDIA CUDA 的 32 线程 warp；warp 是理解 shuffle/归约的基本单位。
const int WARP_SIZE = 32;

// union 的成员共享同一段存储，而不是把各成员大小累加。
// alignas(16) 要求对象起始地址按 16 字节对齐；本类型装下 16 字节数据。
// 常见流程：用 f4 一次装载原始位模式，再按四组 h2 计算，最后用 f4 写回。
// 这种 union 类型双关是本项目的 CUDA 低层实现习惯；阅读时区分“位模式”与“数值转换”。
union alignas(16) pack128 {
    float4 f4;
    float f[4];
    int4 i4;
    int8_t i8[16];
    half2 h2[4]; // 四对 FP16，共八个 half；每对可用 __hadd2 等指令计算。
    half h[8];
    __nv_bfloat162 bf2[4]; // BF16 的双元素视图；它与 FP16 的编码不同。
    __nv_bfloat16 bf[8];
};

// 64 bit = 8 byte，对应两个 FP32 或四个 BF16，起始地址要求 8 字节对齐。
// i4 是原作者的成员名，但类型是 int2；不要由成员名推断它包含四个 int。
union alignas(8) pack64 {
    float2 f2;
    float f[2];
    int2 i4;
    __nv_bfloat162 bf2[2];
    __nv_bfloat16 bf[4];
};
