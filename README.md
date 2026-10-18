# DeepSeek V4.1 Flash：X99 + 4× CMP 170HX 混合推理实测

**LocalFrameLab · 本地帧工坊**

专注本地 AI 推理、动画与视频创作工具的实践与实验记录。

这是 X99 双路 Xeon、四张 CMP 170HX 的部署补丁、基准工具和七轮优化结果。基于 [buliaoyin/vllm-backport](https://github.com/buliaoyin/vllm-backport) 的 `codex/deepseek-v41-cpu-hybrid` 分支，参考 [aiultrayang 的四卡部署方案](https://github.com/aiultrayang/deepseek-v41-flash-cmp170hx)。

本机实测原始 eager 基线单流约 6.5 token/s；修复 Graph 重捕获和禁用 P2P 后的跨卡传输问题，配合已有 DSpark，选定配置的 3000-token 单流 Decode 样本约 **48–59 token/s**，八路混合端到端聚合约 **91–117 token/s**。这些是组合配置的结果，不能把全部提升归因于本仓库补丁。**没有达到单流 80 或 100 token/s**。

## 测试机器与配置

| 项目 | 实测配置 |
|---|---|
| CPU | 双路 Xeon E5-2698B v3，32 物理核 / 64 线程 |
| 内存 | 约 251 GiB，双 NUMA |
| GPU | 4× CMP 170HX，64 GB / 卡，无 NVLink |
| PCIe | 三卡 Gen2 ×8，一卡 Gen2 ×16；本机 P2P 不可靠 |
| 模型 | DeepSeek-V4.1-Flash-1017，本地权重约 476 GiB |
| 环境 | Linux、CUDA 13.0、PyTorch 2.14.1+cu130；自编译 IK/CUDA 原生库 |
| 并行 | PP4 / TP1，层分配 `[6,6,6,22]` |
| CPU 线程 | Prefill / Decode：32 / 28 |
| 调度 | 最大 8 序列，batch 4096，上下文 65536，KV 预算 131072 |
| 投机解码 | DSpark 3，自适应验证，默认查询位置 5 |
| 缓存 | GPU 显存比例 0.97；更新间隔 64；Engram SSD、缓存 4 GiB；host LRU 8 GiB |
| 传输 | `NCCL_P2P_DISABLE=1`、`DSV41_HOST_STAGING_CACHE=1` |

测试日期为 2026-10-18。软件版本与模型名称以本次被测环境为准，复现前应核对上游版本和权重来源。

## 文件入口

- [部署与补丁](docs/DEPLOYMENT.md)：精确基准提交、补丁使用和边界。
- [测试报告](docs/REPORT.md)：Decode、Prefill、TTFT、并发及未采用实验。
- [完整 Excel](results/X99最终测试结果.xlsx)：19 张工作表；CSV 在同目录。
- [隐私与发布范围](docs/PRIVACY.md)：公开包的纳入和排除范围。
- `scripts/`：启动模板、基准脚本、六项功能检查、Excel 生成器与隐私检查。
- `patches/x99-stable.patch`：相对上游固定提交的三个 Python 文件修改，包含本机原有本卡缓存开关。
- `evidence/`：第七轮经过字段白名单重建的服务与 Prefill 汇总；旧轮次细表见 `results/`。不发布生成正文、思考文本、工具参数或原始服务指标全文。

没有包含模型权重、原生二进制库、原始日志、个人会话、内网地址、设备序列号或访问凭据。未采用的第七轮常驻 dispatcher 实现不进入部署路径，其结果保留在表格中。

## 基准复现

先按部署文档准备服务。以下命令会发起实际推理请求，应在没有其他请求时执行；输出文件只留在本地：

```bash
"$VLLM_PYTHON" scripts/validate_service.py --port 18050
"$VLLM_PYTHON" scripts/bench_round2.py --label local --repeats 2 > bench-local.jsonl
"$VLLM_PYTHON" scripts/bench_prefill.py > prefill-local.jsonl
python3 scripts/export_results_xlsx.py
python3 scripts/privacy_check.py
python3 scripts/validate_public_data.py
```

基准默认模型别名为 `DeepSeek-V4.1-Flash-1017`，默认服务地址为 loopback 的 18050 端口。`bench_verified.py` 单独使用时默认端口为 18051，可用 `--port` 指定。脚本只依赖 Python 标准库，推理服务依赖由上游构建环境提供。生成的原始 JSONL 可能含提示词和模型输出，不直接纳入发布。

## 许可与致谢

本仓库代码和补丁按 Apache-2.0 发布，保留所引用 vLLM 代码的版权声明。模型权重与上游依赖遵循各自许可，不由本仓库重新授权。感谢 vLLM、buliaoyin/vllm-backport 和 aiultrayang 的部署经验。
