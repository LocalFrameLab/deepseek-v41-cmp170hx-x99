"""Measure serialized prefill using per-request server metric deltas."""

import datetime
import json
import re
import time
import urllib.request

from bench_verified import call


BASE = "http://127.0.0.1:18050"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    with OPENER.open(req, timeout=30) as response:
        return response.read().decode()


def metrics():
    raw = request("/metrics")
    values = {}
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"([^\s{]+)(?:\{.*\})?\s+([^\s]+)$", line)
        if match and "_bucket" not in match[1]:
            values[match[1]] = values.get(match[1], 0.0) + float(match[2])
    return values


def count(prompt):
    return json.loads(request("/tokenize", {
        "model": "DeepSeek-V4.1-Flash-1017",
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"thinking": False},
    }))["count"]


def make_prompt(target, repeat):
    head = f"测试批次{repeat}，请阅读记录，最后只回答‘收到’。\n"
    rows = [
        f"记录{i:05d}：角色小林在第{i % 12 + 1}场检查道具状态，"
        f"位置区域{i % 37}，照明数值{(i * 17 + repeat) % 100}。\n"
        for i in range(target // 12 + 100)
    ]
    low, high = 1, len(rows)
    while low < high:
        mid = (low + high) // 2
        if count(head + "".join(rows[:mid])) < target:
            low = mid + 1
        else:
            high = mid
    return head + "".join(rows[:low])


def main():
    for target in [1024, 4096, 8192, 16384]:
        for repeat in [1, 2]:
            prompt = make_prompt(target, repeat)
            before = metrics()
            assert before.get("vllm:num_requests_running", 0) == 0
            assert before.get("vllm:num_requests_waiting", 0) == 0
            result = call(
                BASE, prompt, 16, chat_template_kwargs={"thinking": False}
            )
            for _ in range(50):
                after = metrics()
                delta_count = (
                    after["vllm:request_prefill_time_seconds_count"]
                    - before["vllm:request_prefill_time_seconds_count"]
                )
                if delta_count >= 1:
                    break
                time.sleep(0.1)
            assert delta_count == 1, "Concurrent traffic invalidates metric deltas"
            delta = {
                key: after[key] - value
                for key, value in before.items()
                if key in after and not key.endswith("_created")
            }
            prefill_s = delta["vllm:request_prefill_time_seconds_sum"]
            computed = delta["vllm:request_prefill_kv_computed_tokens_sum"]
            assert prefill_s > 0 and computed > 0
            safe_result = {key: result[key] for key in ["usage", "ttft_s", "elapsed_s", "output_span_s", "decode_tok_s", "end_to_end_tok_s", "finish_reason"]}
            print(json.dumps({
                "timestamp_utc": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "target_input_tokens": target,
                "repeat": repeat,
                "server_prefill_s": prefill_s,
                "server_prefill_tok_s": computed / prefill_s,
                "computed_prefill_tokens": computed,
                "server_queue_s": delta["vllm:request_queue_time_seconds_sum"],
                "server_decode_s": delta["vllm:request_decode_time_seconds_sum"],
                "input_over_ttft_tok_s": result["usage"]["prompt_tokens"]
                / result["ttft_s"],
                "metric_deltas": {key: delta[key] for key in ["vllm:request_prefill_time_seconds_sum", "vllm:request_prefill_kv_computed_tokens_sum", "vllm:request_queue_time_seconds_sum", "vllm:request_decode_time_seconds_sum", "vllm:prefix_cache_hits_total"] if key in delta},
                "metrics_before": {key: before.get(key) for key in ["vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:prefix_cache_hits_total"]},
                "metrics_after": {key: after.get(key) for key in ["vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:prefix_cache_hits_total"]},
                **safe_result,
            }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
