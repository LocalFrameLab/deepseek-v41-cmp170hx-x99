"""Record actual server token usage and streaming timing, without proxying LAN."""
import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request


def call(base, prompt, limit=512, **extra):
    body = {
        "model": "DeepSeek-V4.1-Flash-1017",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": limit,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        **extra,
    }
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = last = None
    usage = None
    content, reasoning, tools = [], [], []
    chunks = []
    finish = None
    with opener.open(request, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if raw == b"[DONE]":
                break
            obj = json.loads(raw)
            if obj.get("error"):
                raise RuntimeError(obj["error"])
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                tool = delta.get("tool_calls") or []
                if text or thought or tool:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    chunks.append(now - start)
                    content.append(text)
                    reasoning.append(thought)
                    tools.extend(tool)
                finish = choice.get("finish_reason") or finish
    end = time.perf_counter()
    assert usage and usage.get("completion_tokens", 0) > 0, usage
    n = usage["completion_tokens"]
    gaps = [b - a for a, b in zip(chunks, chunks[1:])]
    return {
        "usage": usage,
        "ttft_s": first - start if first else None,
        "elapsed_s": end - start,
        "output_span_s": last - first if first and last else None,
        "decode_tok_s": n / (last - first) if first and last and last > first else None,
        "end_to_end_tok_s": n / (end - start),
        "stream_events": len(chunks),
        "event_gap_p50_s": statistics.median(gaps) if gaps else None,
        "event_gap_p95_s": sorted(gaps)[min(len(gaps)-1, int(len(gaps)*0.95))] if gaps else None,
        "finish_reason": finish,
        # Do not persist generated text, reasoning, or tool arguments in benchmark JSON.
        "_content": "".join(content),
        "content_chars": sum(map(len, content)),
        "reasoning_chars": sum(map(len, reasoning)),
        "tool_delta_count": len(tools),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=18051)
    p.add_argument("--label", required=True)
    p.add_argument("--single-only", action="store_true")
    args = p.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    def emit(kind, result):
        safe = {key: value for key, value in result.items() if not key.startswith("_")}
        print(json.dumps({"label": args.label, "kind": kind, **safe}, ensure_ascii=False), flush=True)

    for prompt, expected in (
        ("只回答一个词：中国的首都是哪里？", "北京"),
        ("计算 17 乘以 19，只给最终整数。", "323"),
        ('只输出这个 JSON，不加其他文字：{"name":"测试","ok":true,"value":42}', {"name": "测试", "ok": True, "value": 42}),
    ):
        result = call(base, prompt, 256)
        # Keep generated text in memory for assertions; emit() removes it before disk/stdout.
        actual = result.pop("_content", "").strip()
        if isinstance(expected, dict):
            actual = json.loads(actual)
        result["smoke_passed"] = actual == expected
        emit("smoke", result)
        assert result["smoke_passed"], actual
    prompt = "请详细解释大型语言模型的 MoE 专家路由、专家缓存、CPU 与 GPU 混合推理，并用一个具体例子说明。写成连贯的中文技术文章，至少1000字。"
    emit("single", call(base, prompt, 512))
    if not args.single_only:
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: call(base, prompt, 256), range(4)))
        elapsed = time.perf_counter() - start
        emit("concurrency4", {
            "elapsed_s": elapsed,
            "aggregate_tok_s": sum(r["usage"]["completion_tokens"] for r in results) / elapsed,
            "requests": results,
        })


if __name__ == "__main__":
    main()
