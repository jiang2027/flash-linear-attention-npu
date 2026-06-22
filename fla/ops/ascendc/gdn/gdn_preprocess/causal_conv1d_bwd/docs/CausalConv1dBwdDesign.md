# CausalConv1dBwd AscendC 详细设计

## 1. 背景与目标

`CausalConv1dBwd` 是因果一维卷积反向传播算子，面向 GDN 反向链路中 `causal_conv1d_bwd` 的 NPU 原生替换。算子以 AscendC 实现，输出输入梯度 `dx`、权重梯度 `dw`、偏置梯度 `db` 和初始状态梯度 `dh0`。

设计目标：

- 支持固定长度和变长序列。
- 支持 `BSND/BSH`、`BNSD`、`TND`、`NTD` 四类输入 layout。
- 支持无激活、SiLU、Swish 反向。
- 支持 `initial_state` 和 `dht` 对 `dw/dh0/dx` 的贡献。
- 低精度输入下内部使用 FP32 累加，写回时转换为输入 dtype。
- 使用多核并行计算 `dx`，使用 user workspace 保存 `dw/db` partial，核间同步后归约。

## 2. 文件结构

| 文件 | 作用 |
| --- | --- |
| `op_host/causal_conv1d_bwd_def.cpp` | 算子输入、输出、属性和产品配置定义。 |
| `op_host/causal_conv1d_bwd_tiling.cpp` | shape 校验、layout 解析、tiling 参数生成、workspace 估算。 |
| `op_kernel/causal_conv1d_bwd.cpp` | AscendC kernel 入口，按输入 dtype 实例化模板。 |
| `op_kernel/causal_conv1d_bwd.h` | 主 kernel 实现。 |
| `op_kernel/causal_conv1d_bwd_input_layout.h` | layout 扩展 kernel 类型封装。 |
| `op_kernel/causal_conv1d_bwd_tiling_data.h` | host 到 kernel 的 tiling 数据结构。 |
| `op_kernel/causal_conv1d_bwd_tiling_key.h` | tiling key 定义。 |
| `docs/aclnnCausalConv1dBwd.md` | aclnn 对外接口文档。 |
| `examples/test_aclnn_causal_conv1d_bwd.cpp` | aclnn C++ example。 |
| `tests/ATK/` | ATK case 生成和 CPU 标杆 executor。 |

## 3. 支持范围

### 3.1 产品范围

| 产品 | 支持情况 |
| --- | --- |
| Atlas A2 / `ascend910b` | 支持 |
| Atlas A3 / `ascend910_93` | 支持 |
| Ascend 950 / `ascend950` | 不支持 |
| Atlas 200I/500 A2、Atlas 推理系列、Atlas 训练系列 | 不支持 |

### 3.2 dtype 范围

输入 `x/y/weight/dy/initial_state/dht` 支持：

- `FLOAT`
- `FLOAT16`
- `BFLOAT16`

约束：

- 所有数据输入 dtype 必须一致。
- `dx/dh0` dtype 与 `x` 一致。
- `dw/db` dtype 与 `weight` 一致。当前实现中输入 dtype 一致，因此也等价于输入 dtype。
- FP16/BF16 场景下，kernel 内部使用 FP32 做主要乘加和归约，输出前通过 SIMD `Cast` 转回低精度。

### 3.3 layout 和 shape 范围

`inputLayout` 只描述 `y/dy` 的物理 layout；`x/dx` 始终使用逻辑 layout。

| `inputLayout` | `x` shape | `y/dy` shape | `dx` shape | `queryStartLoc` |
| --- | --- | --- | --- | --- |
| `BSND` / `BSH` | `[B, T, D]` | `[B, T, D]` | `[B, T, D]` | 可不传 |
| `BNSD` | `[B, T, D]` | `[B, N, T, Dh]` | `[B, T, D]` | 可不传 |
| `TND` | `[totalTokens, D]` | `[totalTokens, D]` | `[totalTokens, D]` | 必须传 |
| `NTD` | `[totalTokens, D]` | `[N, totalTokens, Dh]` | `[totalTokens, D]` | 必须传 |

其他 tensor：

| Tensor | Shape |
| --- | --- |
| `weight` / `dw` | `[W, D]` |
| `db` | `[D]` |
| `initial_state` / `dht` / `dh0` | `[B, W, D]` |
| `queryStartLoc` | `[B + 1]`，`int64` |

对齐约束：

- `BSND/BSH/TND`：逻辑 `D` 必须为 16 的倍数。
- `BNSD/NTD`：最后一维 `Dh` 必须为 16 的倍数，逻辑 `D=N*Dh`。
- `D` 不要求 64 对齐。`D` 是 64 的倍数时 `BD=64`，否则 `BD=16`。

变长约束：

- `queryStartLoc[0] == 0`。
- `queryStartLoc[B] == totalTokens`。
- `queryStartLoc[i+1] >= queryStartLoc[i]`。
- 不支持空总序列，`totalTokens > 0`。

### 3.4 activation 范围

| `activation` | 含义 |
| --- | --- |
| `0` | 无激活，`g=dy` |
| `1` | SiLU 反向 |
| `2` | Swish 反向，当前与 SiLU 等价 |

`activation=1/2` 时必须传入 `y`。`activation=0` 时 `y` 可为空。

## 4. 数学定义

令前向预激活输出为 `y`，上游梯度为 `dy`，有效梯度为 `g`。

无激活：

```text
g[t] = dy[t]
```

SiLU/Swish 反向：

```text
sigmoid = 1 / (1 + exp(-y[t]))
g[t] = dy[t] * sigmoid * (1 + y[t] * (1 - sigmoid))
```

反向主公式：

```text
dx[t] += sum_i g[t + i] * weight[W - 1 - i]
dw[W - 1 - i] += sum_{b,t} g[b, t + i] * x[b, t]
db += sum_{b,t} g[b, t]
```

其中超出序列范围的 `g[t+i]` 视为 0。

`initial_state` 贡献：

- 对序列前 `W-1` 个 token，前向卷积依赖初始状态。
- 反向中这部分历史状态参与 `dw`。
- 若需要输出 `dh0`，根据前 `W-1` 个 token 的有效梯度和 `weight` 计算初始状态梯度。

`dht` 贡献：

- `dht` 表示最终卷积状态梯度。
- 对序列尾部最多 `W-1` 个 token，将对应状态梯度累加到 `dx`。

## 5. Host 侧设计

### 5.1 算子定义

`CausalConv1dBwd` 定义在 `op_host/causal_conv1d_bwd_def.cpp`：

- 7 个输入：`x`、`y`、`weight`、`dy`、`initial_state`、`dht`、`queryStartLoc`。
- 4 个输出：`dx`、`dw`、`db`、`dh0`。
- 2 个属性：`activation`、`inputLayout`。
- AICore 配置：`ascend910b`、`ascend910_93`。

所有数据 tensor 均使用 `FORMAT_ND`，输入配置 `AutoContiguous()`，避免 kernel 内处理非连续 tensor。

### 5.2 layout 解析

Host tiling 将 layout 映射为 kernel 内部枚举：

| 字符串 | 枚举值 | 说明 |
| --- | --- | --- |
| `BSND` / `BSH` | `INPUT_LAYOUT_BSND=0` | `y/dy` 为 `[B,T,D]` |
| `TND` | `INPUT_LAYOUT_TND=1` | `y/dy` 为 `[totalTokens,D]` |
| `BNSD` | `INPUT_LAYOUT_BNSD=2` | `y/dy` 为 `[B,N,T,Dh]` |
| `NTD` | `INPUT_LAYOUT_NTD=3` | `y/dy` 为 `[N,totalTokens,Dh]` |

固定长度 layout 设置 `inputMode=1`，变长 layout 设置 `inputMode=0`。

### 5.3 shape 校验

Host 侧根据 `inputLayout` 做以下校验：

- `BNSD`：`x` 必须为 `[B,T,D]`，`dy` 必须为 `[B,N,T,Dh]`，并满足 `D=N*Dh`。
- `NTD`：`x` 必须为 `[totalTokens,D]`，`dy` 必须为 `[N,totalTokens,Dh]`，并满足 `D=N*Dh`。
- `TND`：`x` 和 `dy` 必须同 shape `[totalTokens,D]`。
- `BSND/BSH`：`x` 和 `dy` 必须同 shape `[B,T,D]`。
- `y` 若存在，shape 必须与 `dy` 完全一致。
- `activation != 0` 时，`y` 必须存在。
- `weight` 必须是 `[W,D]`。
- `initial_state/dht` 若存在，必须是 `[B,W,D]`。
- 变长场景下 `queryStartLoc` 必须是 `int64` 一维数组。

### 5.4 tiling 参数

核心 tiling 参数：

| 参数 | 含义 |
| --- | --- |
| `B` | batch 数。变长场景来自 `queryStartLoc.size - 1`。 |
| `T` | 固定长度为序列长度；变长为最大序列长度。 |
| `totalTokens` | 总 token 数。固定长度为 `B*T`。 |
| `D` | 逻辑特征维。 |
| `W` | 卷积核宽度。 |
| `BT` | 时间维 tile，固定为 `min(T, 64)`。 |
| `BD` | 特征维 tile，`D % 64 == 0` 时为 64，否则为 16。 |
| `numBlksD` | `D / BD`。 |
| `numChunks` | 时间 chunk 总数。 |
| `chunkPerCore` | 每个主核处理的 chunk 数。 |
| `tailChunk` | 前 `tailChunk` 个核额外处理 1 个 chunk。 |
| `inputN` | BNSD/NTD 的 head 数。 |
| `inputHeadDim` | BNSD/NTD 的 `Dh`。 |

`BT=64` 的原因：

- 对卷积窗口 `BT+W-1` 做一次 `dy` 搬入可覆盖一个时间 tile 的全部 `dx/dw/db` 计算。
- `BT` 足够大，可提升矢量计算粒度；又控制 UB 中 `x/dy/dx/dwRows/dbRows` 的占用。

`BD` 的选择：

- `D` 64 对齐时使用 `BD=64`，提升单次搬运和矢量计算规模。
- 否则使用 `BD=16`，满足当前支持范围中的 16 对齐约束，并支持 `D=72` 等非 64 对齐维度。

### 5.5 UB 估算

Host 侧通过 `GetCoreMemSize(UB)` 获取 UB 大小，预留 `16 KiB` 后按 32B 对齐计算可用容量。

UB 主要缓冲区包括：

- `xBuf`：`BT*BD` FP32。
- `dyBuf`：`(BT+W-1)*BD` FP32。
- `dxBuf`：`BT*BD` FP32。
- `weightBuf`：`W*BD` FP32。
- `dwBuf`：`W*BD` FP32。
- `dbBuf`：`BD` FP32。
- `dwRowsBuf`：无激活路径下 `W*BT*BD` FP32。
- `dbRowsBuf`：无激活路径下 `BT*BD` FP32。
- `yBuf/sigmoidBuf`：激活路径下各 `BT*BD` FP32。
- `castBuf/tempBuf/wdyBuf/dh0Buf`：低精度 cast、临时计算和状态处理。

若估算 `need > ubSize`，tiling 直接失败。

### 5.6 workspace 规划

kernel 需要系统 workspace 和用户 workspace。入口中通过 `GetUserWorkspace(workspace)` 跳过系统 workspace，用户 workspace 存放 `dw/db` 的 per-core partial：

```text
partialDw: blockNum * numBlksD * W * BD * sizeof(float)
partialDb: blockNum * numBlksD * BD * sizeof(float)
```

Host 返回总 workspace：

```text
workspace = sysWorkspaceSize + partialDwBytes + partialDbBytes
```

tiling data 中 `workspaceSize` 保存系统 workspace 大小，kernel 入口实际使用 user workspace 指针。

## 6. Kernel 侧设计

### 6.1 入口和 dtype 模板

`causal_conv1d_bwd.cpp` 根据 `ORIG_DTYPE_X` 实例化：

```cpp
CausalConv1dBwdInputLayoutKernel<float, float>
CausalConv1dBwdInputLayoutKernel<half, float>
CausalConv1dBwdInputLayoutKernel<bfloat16_t, float>
```

其中 `inputT` 是 GM 输入输出 dtype，`calT=float` 是内部计算 dtype。

### 6.2 数据搬运和 layout 地址转换

核心搬运函数为 `CopyInInputTile`：

```cpp
CopyInInputTile(src, dst, bos, startRow, i_d, totalRows, seqLen, useGradLayout)
```

- `useGradLayout=false`：按逻辑 layout 读取，适用于 `x`。
- `useGradLayout=true`：按 `inputLayout` 读取，适用于 `y/dy`。

BNSD/NTD 地址转换：

```text
BNSD offset = ((batch * N + n) * T + row) * Dh + d
NTD  offset = (n * totalTokens + bos + row) * Dh + d
```

其中：

```text
n = channel / Dh
d = channel % Dh
```

当一个 `BD` 跨越 head 边界时，kernel 按 head 内连续片段拆分搬运，保证 `DataCopyPad` 的源地址连续。

低精度输入：

- 先从 GM 搬入 `inputT` 临时 buffer。
- 使用 `Cast(..., RoundMode::CAST_NONE)` 转为 FP32。
- 后续计算全部在 FP32 local tensor 上进行。

越界处理：

- 对尾块或卷积窗口越界行先 `Duplicate` 为 0。
- 只对有效行执行 `DataCopyPad`。
- 因果卷积中超出序列尾部的 `dy/y` 被自然视为 0。

### 6.3 主并行策略

任务空间：

```text
task = chunk * D-block
```

其中：

- `chunk` 是时间 tile。固定长度为每个 batch 的 `ceil(T/BT)` 个 chunk；变长为每条序列 `ceil(seqLen/BT)` 个 chunk 之和。
- `D-block` 是 `D` 维上的 `BD` 分块。

实际核调度：

- `SetBlockDim(mainCoreNum)`，`mainCoreNum=min(coreNum, numChunks)`。
- 每个核负责若干 `chunk`。
- 每个核内部遍历全部 `D-block`。
- `dw/db` 是跨 chunk 的全局归约量，因此每个核先输出 partial，核间同步后再归约。

chunk 分配：

```text
loopC = chunkPerCore + (blockIdx < tailChunk ? 1 : 0)

if blockIdx < tailChunk:
    chunkIdx = blockIdx * (chunkPerCore + 1) + loop
else:
    chunkIdx = blockIdx * chunkPerCore + tailChunk + loop
```

变长场景通过 `ResolveChunk` 将全局 `chunkIdx` 映射到：

```text
i_b   : batch/sequence id
i_t   : 当前序列内第几个 time tile
bos   : 当前序列在 totalTokens 中的起点
seqLen: 当前序列长度
```

### 6.4 单 chunk 计算流程

每个 `(chunk, D-block)` 的核心流程：

```text
CopyInWeight(i_d)
CopyInX(bos, i_t, i_d, seqLen)
dxLocal = 0

if activation is SiLU/Swish:
    for i_w in [0, W):
        CopyInDy(...)
        CopyInY(...)
        ApplySiluBackward(dyLocal)
        dx += dyLocal * weight[W-1-i_w]
        dw += reduce_rows(dyLocal * xLocal)
        if i_w == 0:
            db += reduce_rows(dyLocal)
else:
    CopyInDyWindow(BT + W - 1 rows)
    for i_w in [0, W):
        dx += dyWindow[i_w : i_w+BT] * weight[W-1-i_w]
        dwRows[w] += dyWindow[i_w : i_w+BT] * xLocal
    dbRows += dyWindow[0 : BT]

ComputeDh0(...)
AccumulateDhtDx(...)
CopyOutDx(...)
```

无激活路径使用 `dyWindow` 优化：

- 一次搬入 `BT+W-1` 行 `dy`。
- `i_w` 循环中通过 row offset 使用不同窗口。
- `dwRowsBuf/dbRowsBuf` 先保留逐行乘积，chunk 循环结束后再按行归约，减少多次搬运和重复激活计算。

激活路径需要每个 `i_w` 对齐读取相同窗口的 `y` 和 `dy`，并执行 SiLU 反向，因此按 `i_w` 分别处理。

### 6.5 dx 计算

`ComputeWdyAndAcc` 完成：

```text
g = dy 或 SiLUBackward(y, dy)
wIdx = W - i_w - 1
wdy = g * weight[wIdx]
dxLocal += wdy
```

实现使用 `BinaryRepeatParams`，将 `weight[wIdx, BD]` 按行广播到 `BT` 行，充分使用 vector SIMD。

`CopyOutDx`：

- FP32 输出直接写回。
- FP16/BF16 输出先 `Cast(..., RoundMode::CAST_RINT)` 到 `inputT`，再 `DataCopyPad` 写回 GM。
- `dx` 始终按逻辑 layout 写回，固定长度 `[B,T,D]`，变长 `[totalTokens,D]`。

### 6.6 dw/db 计算和归约

每个核对自己负责的 chunk 累加局部 `dw/db`：

```text
dwPartial[w,d] += sum_rows(g * x)
dbPartial[d] += sum_rows(g)
```

局部归约：

- `ReduceRowsInplace` 用树形两两相加，将 `[rows, BD]` 规约到 `[BD]`。
- 无激活路径先在 `dwRowsBuf/dbRowsBuf` 中累积逐行结果，再统一 `FinalizeDwRowsAccum/FinalizeDbRowsAccum`。
- 激活路径直接对当前 `dyLocal` 归约并累到 `dwBuf/dbBuf`。

跨核归约：

1. 每个核调用 `CopyOutPartialDwDb(blockIdx, i_d)`，将本核该 `D-block` 的 `dw/db` partial 写入 user workspace。
2. `SyncAll()` 保证所有核的 partial 写入完成。
3. 所有核按 stride 分担 `D-block` 归约：

```text
for i_d = blockIdx; i_d < numBlksD; i_d += blockNum:
    ReducePartialDwDb(i_d)
```

4. `ReducePartialDwDb` 从 workspace 读取每个核的 partial，加和到 local `dwBuf/dbBuf`。
5. `CopyOutDwDb` 写回最终 `dw/db`。

`dw/db` 写回 dtype：

- FP32：直接 `DataCopyPad` 写回。
- FP16/BF16：先使用 `Cast(..., RoundMode::CAST_RINT)` 转为 `inputT`，再写回。

### 6.7 initial_state 和 dh0

`initial_state` shape 固定为 `[B,W,D]`，与正向状态 layout 一致。

在序列开头，前向卷积会使用历史状态补齐因果窗口。反向包含两部分：

- `AccumulateInitialStateDw`：把序列开头依赖的 `initial_state` 对应贡献累加到 `dw`。
- `ComputeDh0`：只在每条序列第一个 chunk 上计算 `dh0`，输出 `[B,W,D]`。

实现策略：

- 搬入当前 batch 的 `[W,BD]` initial state block。
- 对 `absRow < i_w` 的历史窗口计算贡献。
- `dh0` 逐 slot 累加 `dy * weight`。
- 低精度输出时 `dh0` 同样从 FP32 cast 回 `inputT`。

### 6.8 dht 到 dx 的贡献

`dht` shape 为 `[B,W,D]`，表示最终状态梯度。

`AccumulateDhtDx` 只影响序列尾部：

```text
tailStart = max(0, seqLen - (W - 1))
```

对 `absRow >= tailStart` 的 token，将对应 `dht` slot 累加到 `dxLocal`。

该逻辑在 `dx` 写回前执行，因此最终 `dx` 包含主卷积反传和最终状态反传两部分。

### 6.9 同步与流水

kernel 使用 AscendC event 和 barrier 明确管理流水依赖：

- `V_MTE2`：vector 写 local 后，允许下一次 MTE2 读写复用 buffer。
- `MTE2_V`：GM 到 local 搬运完成后，允许 vector 计算。
- `V_MTE3`：vector 计算或 cast 完成后，允许写 GM。
- `MTE3_V`：GM 写回完成后，允许 vector 复用输出 buffer。
- 连续 vector 指令之间使用 `PipeBarrier<PIPE_V>()`。

核间同步：

- `dw/db` partial 写入 user workspace 后调用 `SyncAll()`。
- `SyncAll()` 后再读取所有核的 partial 做归约。
- `dx/dh0` 由唯一 chunk 或唯一序列位置负责写回，不需要核间 atomic。

## 7. 性能设计

### 7.1 并行粒度

主并行维度是时间 chunk：

- 长序列场景下，`numChunks` 足够大，可充分利用多核。
- 每个核遍历所有 `D-block`，避免 `dx` 上出现跨核写冲突。
- `dw/db` 使用 workspace 归约，避免 atomic 写全局输出。

### 7.2 搬运优化

- 使用 `DataCopyPad` 支持尾块和 row stride。
- 对低精度输入先整块搬运再 SIMD cast，避免逐元素 `getValue/setValue`。
- 对 `BNSD/NTD` 在 head 边界拆成连续 segment，提高 DMA 搬运合法性和效率。
- 无激活路径一次搬入 `BT+W-1` 行 `dy`，复用窗口数据。

### 7.3 计算优化

- 主乘加均为 vector SIMD。
- `weight` 按 `[W,BD]` 搬入 local，`dx` 计算中按行广播。
- row reduction 使用树形加法，减少串行累加深度。
- `dw/db` 先局部归约再跨核归约，避免每个 token 直接写全局。

### 7.4 当前限制

- `BT` 固定为 64，未根据 `W/T/D` 自适应调参。
- 每个核内部遍历所有 `D-block`，当 `numChunks` 较少但 `D` 很大时并行度可能不足。
- 跨核归约使用 `SyncAll()`，逻辑简单但会形成全核同步点。
- A5 未支持。

## 8. 精度设计

### 8.1 内部计算精度

无论输入 dtype 是 FP32、FP16 还是 BF16，核心计算 local tensor 均使用 FP32：

- `x/weight/dy/y/initial_state/dht` 搬入后转换为 FP32。
- `dx/dw/db/dh0` 的累加缓冲均为 FP32。

### 8.2 输出 cast

输出 dtype 遵循接口定义：

- `dx/dh0` 输出为 `x` dtype。
- `dw/db` 输出为 `weight` dtype。

低精度输出时统一使用：

```cpp
Cast(outputLocal, fp32Local, RoundMode::CAST_RINT, count)
```

再通过 `DataCopyPad` 写回 GM。

### 8.3 CPU 标杆一致性

测试侧 CPU 标杆遵循相同策略：

- FP16/BF16 输入先 cast 到 FP32 计算。
- `dw/db` 完成 FP32 聚合后 cast 回输入 dtype。
- `dx/dh0` 同样 cast 回输入 dtype。

## 9. 错误处理与边界

Host tiling 在进入 kernel 前拦截以下非法输入：

- 不支持的 `inputLayout`。
- `activation` 需要 `y` 但 `y` 为空。
- `x/y/dy/weight/initial_state/dht/queryStartLoc` shape 不匹配。
- `queryStartLoc` dtype 非 `int64`。
- `queryStartLoc` 非单调或首尾不合法。
- `D/Dh` 非 16 对齐。
- 空序列。
- UB 估算溢出。

Kernel 内对尾块和窗口越界使用 0 padding，避免读取非法 GM。

## 10. 验证覆盖

已有验证路径：

- C++ aclnn example：覆盖 FP32、BF16、固定长度、变长、BNSD、NTD、TND。
- Torch 单测：覆盖 FP32/FP16/BF16、三种 activation、BNSD、TND、NTD、`initial_state/dht`。
- ATK：支持批量生成 shape/dtype/layout case，CPU 标杆支持所有输入输出。

建议继续补充：

- A3 实机或 CI 验证。
- 更大 `D`、更长 `T`、更大 batch 的性能回归。
- `D` 非 64 对齐但 16 对齐场景，如 `D=72`。
- `queryStartLoc` 多序列、短序列、非均匀序列边界。

## 11. 后续优化方向

- 针对 `numChunks` 少但 `D` 大的场景，引入 `(chunk, D-block)` 双维度核间调度，提升宽维场景并行度。
- 针对 `W=4` 等常见宽度做模板化展开，减少循环和分支。
- 优化跨核归约，可评估更细粒度 flag 或分层归约，降低 `SyncAll()` 等待开销。
- 针对激活路径缓存 `y/dy` 窗口，减少按 `i_w` 重复搬运。
- 根据 UB 和 shape 自适应选择 `BT`，在短序列和大 `W` 场景下降低 UB 占用。
