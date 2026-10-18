"""Matched round-two workloads with per-run speculative-decoding counters."""

import argparse
import concurrent.futures
import datetime
import json
import re
import time
import urllib.request

from bench_verified import call


PROMPTS = [
    "请详细解释大型语言模型的 MoE 专家路由、专家缓存、CPU 与 GPU 混合推理，"
    "并用一个具体例子说明。写成连贯的中文技术文章，至少5000字。"
    "从基础原理讲到工程实现，包括 NUMA、缓存替换、异步传输、投机解码、"
    "性能测量和正确性验证，每个部分都给出具体例子。直接写正文，不要先列提纲。",
    "写一个至少3000字的完整中文故事：阿禾在雨夜归还一把旧伞，修钟老人认出伞柄刻字，"
    "二人发现一段被遗忘的约定。要有动作、对话、伏笔和合理转折，人物动机一致，直接写正文。",
    "用Python标准库写一个完整的SQLite任务队列命令行程序，包括建库、增加任务、"
    "按优先级领取任务、租约过期重领、完成、失败重试和查看统计。使用事务保证并发安全，"
    "支持argparse子命令，并提供完整使用示例。",
    "为一部三分钟动画短片设计完整的分镜剧本：女孩在停电的老街帮助一位修钟老人寻找失踪的猫。"
    "至少20个镜头，逐镜写清动作、前后场景状态、道具状态、人物台词、音效、情绪节奏，强调跨镜头连续性。",
]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def metrics(base):
    with OPENER.open(base + "/metrics", timeout=20) as response:
        raw = response.read().decode()
    values = {}
    for line in raw.splitlines():
        if not line.startswith("vllm:"):
            continue
        match = re.match(r"([^\s{]+)(?:\{.*\})?\s+([^\s]+)$", line)
        if match and "_bucket" not in match[1]:
            values[match[1]] = values.get(match[1], 0) + float(match[2])
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    base = "http://127.0.0.1:18050"
    for repeat in range(1, args.repeats + 1):
        for concurrency, budget in [(1, 3000), (8, 1024)]:
            before = metrics(base)
            assert before.get("vllm:num_requests_running", 0) == 0
            assert before.get("vllm:num_requests_waiting", 0) == 0
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=concurrency
            ) as executor:
                requests = list(executor.map(
                    lambda index: call(
                        base, PROMPTS[index % len(PROMPTS)], budget,
                        chat_template_kwargs={"thinking": False},
                    ), range(concurrency),
                ))
            elapsed = time.perf_counter() - start
            for _ in range(50):
                after = metrics(base)
                if (
                    after.get("vllm:request_success_total", 0)
                    - before.get("vllm:request_success_total", 0)
                    >= concurrency
                ):
                    break
                time.sleep(0.1)
            delta = {
                key: after[key] - value for key, value in before.items()
                if key in after and not key.endswith("_created")
            }
            assert delta["vllm:request_success_total"] == concurrency
            safe_requests = [
                {
                    key: request[key]
                    for key in ["usage", "ttft_s", "elapsed_s", "output_span_s", "decode_tok_s", "end_to_end_tok_s", "stream_events", "event_gap_p50_s", "event_gap_p95_s", "finish_reason"]
                }
                for request in requests
            ]
            safe_requests = [
                {
                    key: request[key]
                    for key in [
                        "usage", "ttft_s", "elapsed_s", "output_span_s",
                        "decode_tok_s", "end_to_end_tok_s", "stream_events",
                        "event_gap_p50_s", "event_gap_p95_s", "finish_reason",
                    ]
                }
                for request in requests
            ]
            print(json.dumps({
                "label": args.label,
                "timestamp_utc": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "repeat": repeat, "concurrency": concurrency,
                "token_budget_per_request": budget, "elapsed_s": elapsed,
                "aggregate_tok_s": sum(
                    r["usage"]["completion_tokens"] for r in requests
                ) / elapsed,
                "requests": safe_requests,
                "metric_deltas": {
                    key: delta[key]
                    for key in [
                        "vllm:request_success_total",
                        "vllm:request_prefill_time_seconds_sum",
                        "vllm:request_decode_time_seconds_sum",
                    ]
                    if key in delta
                },
                "metrics_before": {
                    key: before.get(key)
                    for key in [
                        "vllm:num_requests_running",
                        "vllm:num_requests_waiting",
                        "vllm:prefix_cache_hits_total",
                    ]
                },
                "metrics_after": {
                    key: after.get(key)
                    for key in [
                        "vllm:num_requests_running",
                        "vllm:num_requests_waiting",
                        "vllm:prefix_cache_hits_total",
                    ]
                },
            }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
