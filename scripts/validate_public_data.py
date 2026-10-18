"""Check sanitized evidence schemas and public result consistency."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "evidence/round7-summary.json").read_text())
assert len(data["service"]) == 12, len(data["service"])
assert len(data["prefill"]) == 8, len(data["prefill"])
for row in data["service"]:
    assert set(row) == {"label", "repeat", "concurrency", "aggregate_tok_s", "requests"}
    assert len(row["requests"]) == row["concurrency"]
    for req in row["requests"]:
        assert set(req) == {"prompt_tokens", "completion_tokens", "decode_tok_s", "end_to_end_tok_s", "ttft_s", "finish_reason"}
        assert req["prompt_tokens"] > 0 and req["completion_tokens"] > 0
for row in data["prefill"]:
    assert set(row) == {"target_input_tokens", "repeat", "computed_prefill_tokens", "server_prefill_s", "server_prefill_tok_s", "ttft_s"}
for path in (ROOT / "results").glob("*.csv"):
    text = path.read_text(encoding="utf-8-sig")
    assert "192.168." not in text and "/home/" not in text and "/mnt/" not in text
print("public evidence schemas passed")
