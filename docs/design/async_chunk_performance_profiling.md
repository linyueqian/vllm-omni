# Async Chunk 性能分析与打点说明 (Qwen3-Omni)

本文说明开启 Async Chunk 后 Qwen3-Omni 可能较耗时的代码路径、已添加的打点位置，以及如何启用并分析性能日志。

## 1. 可能耗时的代码路径概览

| 阶段 | 位置 | 说明 | 可能瓶颈 |
|------|------|------|----------|
| **Thinker → Talker 传 chunk** | `chunk_transfer_adapter._send_single_request` → `thinker2talker_async_chunk` | 从 pooling 输出中取 hidden/embed，做 `.detach().cpu()`、`torch.cat`，组 payload | CPU 拷贝、大 tensor 的 cat |
| **Talker → Code2Wav 传 chunk** | `chunk_transfer_adapter._send_single_request` → `talker2code2wav_async_chunk` | 从 code_predictor_codes 转成 list、按 chunk 截取、组 payload | tensor 转 list、list 操作 |
| **Connector 写入** | `chunk_transfer_adapter` 中 `connector.put` | 将 payload 序列化并写入共享内存/队列 | 序列化、跨进程/线程写入 |
| **Worker 侧更新与预处理** | `gpu_model_runner._preprocess` (async_chunk 分支) | `_update_additional_information`、每请求 `model.preprocess`、`_merge_additional_information_update` | 从 connector 取数据、CPU↔GPU、preprocess 里拿 thinker_embed 上 GPU |
| **Talker MTP 解码** | `gpu_model_runner._talker_mtp_forward` | 每步 decode 跑 code predictor，结果写回 buffer | 小 batch 的 GPU 推理、`.to("cpu")` |
| **Talker 后处理** | `qwen3_omni.talker_postprocess` | `last_talker_hidden.detach().to("cpu")` | GPU→CPU 拷贝 |
| **Code2Wav 流式解码** | `qwen3_omni.generate_audio` → `code2wav.chunked_decode_streaming` | 小 chunk（如 25 帧）一次 forward 解码成波形 | Code2Wav 的 forward、chunk 次数多 |
| **Code2Wav 单次 forward** | `qwen3_omni_code2wav.chunked_decode_streaming` 内 `self(codes)` | 单次 codes→waveform 的 NN 计算 | 卷积/Transformer 计算、显存带宽 |

## 2. 已添加的打点（tag 名称）

启用 profiling 后，日志中会出现 `[ASYNC_CHUNK_PROFILE]`，后面跟 **tag** 和 **ms**。当前打点如下：

| Tag | 位置 | 含义 |
|-----|------|------|
| `chunk_adapter_custom_process` | `chunk_transfer_adapter._send_single_request` | 调用 thinker2talker / talker2code2wav 做 payload 构建的耗时 |
| `chunk_adapter_connector_put` | `chunk_transfer_adapter._send_single_request` | `connector.put` 的耗时 |
| `chunk_adapter_save_enqueue` | `chunk_transfer_adapter.save_async` | save 入队事件（`qlen` 为入队后队列长度） |
| `chunk_adapter_save_queue_wait` | `chunk_transfer_adapter._send_single_request` | save task 从入队到被后台线程处理的等待时间 |
| `chunk_adapter_load_enqueue` | `chunk_transfer_adapter.load_async` | load 入队事件（`qlen` 为入队后队列长度） |
| `chunk_adapter_load_queue_wait` | `chunk_transfer_adapter._poll_single_request` | load request 从入队到成功收包的等待时间 |
| `chunk_adapter_save_loop_work` | `transfer_adapter.base.save_loop` | save_loop 单任务总工作耗时 |
| `chunk_adapter_recv_loop_work` | `transfer_adapter.base.recv_loop` | recv_loop 单次 poll 工作耗时 |
| `chunk_adapter_update_request_payload` | `chunk_transfer_adapter._update_request_payload` | 下游合并/更新 request_payload（tensor/list cat）的耗时 |
| `thinker2talker_async_chunk` | `stage_input_processors.qwen3_omni` | Thinker→Talker 整段 async chunk 处理 |
| `talker2code2wav_async_chunk` | `stage_input_processors.qwen3_omni` | Talker→Code2Wav 整段 async chunk 处理 |
| `thinker_decode_to_talker_decode` | `qwen3_omni._thinker_decode_to_talker_decode` | 按 chunk 将 thinker embedding 投影到 talker 的 decode 步 |
| `gpu_runner_update_additional_info` | `gpu_model_runner._preprocess` | 从 connector 更新 additional_information 的耗时 |
| `gpu_runner_model_forward` *(event)* | `gpu_model_runner._model_forward` | execute model 窗口（start/end 事件，用于与 save/load 计算重叠率） |
| `chunk_adapter_save_work` *(event)* | `chunk_transfer_adapter._send_single_request` | save 任务窗口（start/end 事件） |
| `chunk_adapter_load_work` *(event)* | `chunk_transfer_adapter._poll_single_request` | load 成功处理窗口（start/end 事件） |
| `code2wav_chunked_decode_streaming` | `qwen3_omni.generate_audio` | 整段流式解码（含 code2wav 的多次 forward） |
| `code2wav_forward_streaming` | `qwen3_omni_code2wav.chunked_decode_streaming` | Code2Wav 单次 `self(codes)` forward |

## 3. 如何启用并采集

设置环境变量后启动服务（或跑 benchmark）：

```bash
export VLLM_OMNI_ASYNC_CHUNK_PROFILE=1
# 然后正常启动 vLLM-Omni + Qwen3-Omni async chunk 配置
vllm serve ... --stage-configs-path .../qwen3_omni_moe_async_chunk.yaml
```

日志中会出现类似：

```
[ASYNC_CHUNK_PROFILE] thinker2talker_async_chunk 12.34 ms | chunk=0
[ASYNC_CHUNK_PROFILE] chunk_adapter_connector_put 0.56 ms | stage=0
[ASYNC_CHUNK_PROFILE] code2wav_forward_streaming 45.67 ms | shape=(1, 8, 25)
[ASYNC_CHUNK_EVENT] gpu_runner_model_forward start ts_ns=... tid=... | stage=talker num_tokens=...
[ASYNC_CHUNK_EVENT] gpu_runner_model_forward end ts_ns=... tid=... | stage=talker num_tokens=...
...
```

## 4. 如何分析

1. **按 tag 聚合**：对同一 tag 的 ms 做 sum/mean/max（例如用 grep + awk 或简单脚本），看哪几个 tag 占总时间最多。
2. **按阶段看**：
   - **Stage 0→1**：看 `thinker2talker_async_chunk`、`chunk_adapter_custom_process`（stage=0）、`chunk_adapter_connector_put`（stage=0）。
   - **Stage 1→2**：看 `talker2code2wav_async_chunk`、`chunk_adapter_custom_process`（stage=1）、`chunk_adapter_connector_put`（stage=1）。
   - **Stage 1 Worker**：看 `gpu_runner_update_additional_info`、`gpu_runner_preprocess_per_req`、`gpu_runner_talker_mtp_forward`、`talker_postprocess_cpu_copy`。
   - **Stage 2 (Code2Wav)**：看 `code2wav_chunked_decode_streaming`、`code2wav_forward_streaming`。
3. **优化方向建议**：
   - `thinker2talker_async_chunk` / `talker2code2wav_async_chunk` 耗时长：减少 `.cpu()` 和 `torch.cat`，或考虑在 GPU 上保留到下一 stage 再拷。
   - `chunk_adapter_connector_put` 高：检查 connector 实现（序列化、共享内存/队列竞争）。
   - `code2wav_forward_streaming` 或 `code2wav_chunked_decode_streaming` 高：考虑 chunk_size、左上下文、或 Code2Wav 模型/内核优化。
   - `gpu_runner_preprocess_per_req` / `talker_postprocess_cpu_copy` 高：减少 CPU↔GPU 拷贝、合并或异步化拷贝。
4. **重叠率（掩盖率）计算**：
   - 通过 `[ASYNC_CHUNK_EVENT]` 提取窗口：`gpu_runner_model_forward`（模型窗口）与 `chunk_adapter_save_work` / `chunk_adapter_load_work`（异步窗口）。
   - 对同一进程内事件，按 `ts_ns` 配对 start/end，得到时间区间 `[start_ns, end_ns]`。
   - 计算：
     - `overlap_ns = intersection(async_window, model_window)`（可与多个 model_window 求并集交）
     - `overlap_ratio = overlap_ns / async_window_ns`
   - 解释建议：
     - `overlap_ratio` 高且 `*_queue_wait` 低：save/load 基本被模型计算掩盖；
     - `overlap_ratio` 下降且 `*_queue_wait` 升高：掩盖变差，异步开销开始泄漏到 E2E。

## 5. 实现与开关

- 打点逻辑与开关：`vllm_omni.utils.async_chunk_profile`（通过 `VLLM_OMNI_ASYNC_CHUNK_PROFILE=1` 开启）。
- 未设置或非 1/true/yes 时，仅做一次 `_profile_enabled()` 判断，几乎无性能影响。

---

## 6. 对比开/关 Async Chunk 各阶段耗时（Thinker / Talker / Code2Wav）

若要对比「开启 async chunk」与「关闭 async chunk」时各主要阶段（Thinker、Talker、Code2Wav）的耗时，可使用 **Stage 级别计时日志**。

### 6.1 启用方式

设置环境变量（与是否开启 async chunk 无关，两种配置下都可开）：

```bash
export VLLM_OMNI_LOG_STAGE_TIMING=1
```

然后分别跑两套配置（例如用同一批请求、同一并发）：

1. **关 async chunk**：使用未带 async chunk 的 stage 配置（如 `qwen3_omni_moe.yaml`）启动并跑 benchmark。
2. **开 async chunk**：使用带 async chunk 的配置（如 `qwen3_omni_moe_async_chunk.yaml`）启动并跑同样的 benchmark。

### 6.2 日志格式

- **单请求**：每个请求在 finalize 时会打一行，便于按请求对比。
  - 标记：`[STAGE_TIMING]`
  - 内容：`async_chunk=0` 或 `async_chunk=1`、`req_id=...`、`Thinker_ms=...`、`Talker_ms=...`、`Code2Wav_ms=...`（单位：毫秒）

示例：

```
[STAGE_TIMING] async_chunk=0 req_id=0_xxx Thinker_ms=1200.00 Talker_ms=800.00 Code2Wav_ms=350.00
[STAGE_TIMING] async_chunk=1 req_id=0_yyy Thinker_ms=1100.00 Talker_ms=950.00 Code2Wav_ms=420.00
```

- **整轮汇总**：所有请求结束后会打一行汇总（均值与总和）。
  - 标记：`[STAGE_TIMING_SUMMARY]`
  - 内容：`async_chunk=0|1`、`requests=N`、各阶段的 `Thinker_mean_ms` / `Thinker_sum_ms`、`Talker_mean_ms` / `Talker_sum_ms`、`Code2Wav_mean_ms` / `Code2Wav_sum_ms`

示例：

```
[STAGE_TIMING_SUMMARY] async_chunk=0 requests=10 Thinker_mean_ms=1150.00 Thinker_sum_ms=11500.00 Talker_mean_ms=820.00 ...
[STAGE_TIMING_SUMMARY] async_chunk=1 requests=10 Thinker_mean_ms=1080.00 Thinker_sum_ms=10800.00 Talker_mean_ms=900.00 ...
```

### 6.3 如何对比

1. **按配置分开跑**：先只开 `VLLM_OMNI_LOG_STAGE_TIMING=1`，跑「关 async chunk」一次，保存日志；再跑「开 async chunk」一次，保存另一份日志。
2. **抓取汇总**：在两份日志里分别 grep `[STAGE_TIMING_SUMMARY]`，即可直接对比同一阶段在两种配置下的 `*_mean_ms`、`*_sum_ms`。
3. **按请求对比**：grep `[STAGE_TIMING]`，按 `req_id` 或请求顺序对比每条请求的 Thinker_ms、Talker_ms、Code2Wav_ms。

含义说明：

- **Thinker_ms**：Stage 0（Thinker）在该请求上的累计生成耗时。
- **Talker_ms**：Stage 1（Talker）在该请求上的累计生成耗时。
- **Code2Wav_ms**：Stage 2（Code2Wav）在该请求上的累计生成耗时。

这些时间来自各 stage 上报的 `stage_gen_time_ms`（即该 stage 内模型生成/解码的耗时），与 async chunk 开/关无关，因此可直接对比两种配置下各阶段的耗时差异。
