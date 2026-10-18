"""Functional checks for the optimized server; does not execute model tool calls."""
import argparse
import json
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=18050)
args = p.parse_args()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(messages, **extra):
    body = {"model": "DeepSeek-V4.1-Flash-1017", "messages": messages, "temperature": 0, "max_tokens": 1024, **extra}
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=900) as response:
        result = json.load(response)
    assert not result.get("error"), result
    assert result.get("choices"), result
    return result


for index in range(3):
    result = request([{"role": "user", "content": '只输出合法 JSON：name 为测试，ok 为 true，value 为 17乘19。'}])
    msg = result["choices"][0]["message"]["content"].strip()
    if msg.startswith("```"):
        msg = "\n".join(msg.splitlines()[1:-1])
    value = json.loads(msg)
    assert value == {"name": "测试", "ok": True, "value": 323}, value
    print(json.dumps({"test": "json_arithmetic_repeat", "index": index, "passed": True, "usage": result.get("usage")}), flush=True)

messages = [
    {"role": "user", "content": "记住暗号是青石桥742。"},
    {"role": "assistant", "content": "记住了，暗号是青石桥742。"},
    {"role": "user", "content": "刚才的暗号是什么？只输出暗号。"},
]
result = request(messages)
assert "青石桥742" in result["choices"][0]["message"]["content"], result
print(json.dumps({"test": "multi_turn", "passed": True, "usage": result.get("usage")}), flush=True)

result = request(
    [{"role": "user", "content": "查询北京的天气，请调用工具。"}],
    tools=[{"type": "function", "function": {"name": "get_weather", "description": "查询城市天气", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
    tool_choice={"type": "function", "function": {"name": "get_weather"}},
)
calls = result["choices"][0]["message"].get("tool_calls") or []
assert len(calls) == 1 and calls[0]["function"]["name"] == "get_weather", result
assert json.loads(calls[0]["function"]["arguments"])["city"] in ("北京", "北京市", "Beijing"), result
print(json.dumps({"test": "tool_call_parse", "passed": True, "usage": result.get("usage")}), flush=True)

paragraph = "这是一段用于测试长输入的普通资料。故事里有邮递员、修钟人和一间老书店。每个人都要保持自己的角色和说话习惯，事件按先后顺序发生。"
prompt = "开头标记：北斗7319。\n" + "\n".join(f"资料{i}：{paragraph}" for i in range(150)) + '\n结尾标记：海棠4826。\n请只输出一个JSON，first填开头标记，last填结尾标记，不要包含标点句号。'
start = time.monotonic()
result = request([{"role": "user", "content": prompt}])
msg = result["choices"][0]["message"]["content"].strip()
if msg.startswith("```"):
    msg = "\n".join(msg.splitlines()[1:-1])
assert json.loads(msg) == {"first": "北斗7319", "last": "海棠4826"}, result
print(json.dumps({"test": "long_input", "passed": True, "elapsed_s": time.monotonic()-start, "usage": result.get("usage")}), flush=True)
