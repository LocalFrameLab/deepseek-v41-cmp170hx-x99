# 部署与补丁

## 固定来源

上游仓库：<https://github.com/buliaoyin/vllm-backport>

分支：`codex/deepseek-v41-cpu-hybrid`

被测基准提交：`c3d3104de0163c389d26d9fcba0180f55ad7964d`。

本包是增量补丁和复现实验资料，不是完整 vLLM 发行版。先按上游说明构建该提交的环境，包括 `libdsv41_ik.so` 和 `libdsv41_cuda.so`。CUDA 库还需要同库的 `engram_ssd.cpp`，只编译主机 MoE 桥接源文件会导致 Engram SSD 接口缺失。

## 应用补丁

在独立 checkout 中操作，保留现有运行目录。以下操作不会自动停止服务或改动容器：

```bash
cd "$VLLM_ROOT"
git rev-parse HEAD
git apply --check /path/to/release/patches/x99-stable.patch
git apply /path/to/release/patches/x99-stable.patch
```

补丁修改三个 Python 文件，不重编译专家数学内核：

1. `hybrid.py`：增加正整数 `cache_decode_interval`；保留可选本卡缓存实验开关，正式配置不启用它。
2. `hybrid_runtime.py`：校准重试销毁旧 Graph 后，为所有 capture owner 分配新 pool，避免复用已退休 pool 的 allocator 断言。
3. `expert_cache.py`：开启 `DSV41_HOST_STAGING_CACHE=1` 时，通过预分配 pinned host 缓冲传输输入、路由与输出，避免禁 P2P 时在 capture 内调用不允许的跨卡 peer copy。

`patches/tested-source-hashes.json` 是被测部署树的三个文件哈希，供核对补丁后的源码。它不包含权重或个人文件哈希。Python 私有 Graph 接口与上游实现耦合；跨版本使用需要重新验证。

## 启动模板

使用自己准备好的模型与环境目录：

```bash
export VLLM_ROOT=/path/to/vllm-backport
export MODEL_PATH=/path/to/DeepSeek-V4.1-Flash-1017
export VLLM_PYTHON=/path/to/prepared-venv/bin/python3
bash scripts/serve.sh
```

模板只监听 `127.0.0.1:18050`。它不执行 SSH、Docker 管理、驱动安装、内核设置或服务替换。运行前自行确保四卡没有被另一个模型实例占用。

NUMA 检查依赖容器是否允许内存策略与页位置查询。原机器曾只额外放行所需的四个内存策略系统调用；本包不附带个人容器配置，也不推荐为了消除提示而直接开放所有容器权限。查询失败不能被解释为已经实现本地内存放置。

## 配置边界

本次持续保持禁用 P2P、`route_tail_skip=0`、原始模型量化及全部专家计算。P2P 开关取决于实际硬件正确性测试，不能仅凭拓扑名称决定。本机出现可纠正 PCIe AER 错误，原因尚未证实，没有把它判定为坏卡或输出损坏。

上游在 CPU 专家层与 CED/KV 传输上有末级 rank 约束。多 rank 分担不能通过简单删除层数校验完成，还涉及候选索引、缓存所有权和 DSpark 辅助层。没有交付此类重构。

新机器首次验证先做功能检查，再测吞吐。六项检查只覆盖 JSON 算术重复、多轮、工具格式和长输入，不代表完整模型质量评测。80/100 token/s 不是本包的性能承诺。
