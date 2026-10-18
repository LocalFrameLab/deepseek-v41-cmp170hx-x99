"""Reject common private data patterns from the public release tree."""
from pathlib import Path
import re
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
patterns = [
    ("private-key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("lan-ip", re.compile(r"(?<!\d)192\.168\.\d{1,3}\.\d{1,3}(?!\d)")),
    ("absolute-home", re.compile(r"/home/[A-Za-z0-9_.-]+")),
    ("absolute-mnt", re.compile(r"/mnt/(?:nvme|ssd|optane|data)[A-Za-z0-9_./-]*")),
    ("ssh-user-host", re.compile(r"\b(?:ssh|scp)\s+[A-Za-z0-9_.-]+@")),
    ("private-container", re.compile(r"\bdsv41-(?:gpt-run|build|run)\b")),
    ("serial-number", re.compile(r"(?i)\b(?:serial(?: number)?|序列号)\s*[:：]?\s*[A-Z0-9-]{10,}\b")),
    ("pci-address", re.compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}:){1,2}[0-9A-Fa-f]{2}\.[0-7](?![0-9A-Fa-f])")),
]
allowed_binary = {".xlsx", ".png", ".jpg", ".jpeg", ".gif", ".zip"}
issues = []
for path in ROOT.rglob("*"):
    if not path.is_file() or ".git" in path.parts:
        continue
    if path.stat().st_size > 20 * 1024 * 1024:
        issues.append((path.relative_to(ROOT), "file-over-20MiB"))
    if path.suffix.lower() == ".xlsx":
        try:
            with ZipFile(path) as archive:
                text = "\n".join(
                    archive.read(name).decode("utf-8", errors="ignore")
                    for name in archive.namelist()
                    if name.endswith(".xml")
                )
        except Exception as error:
            issues.append((path.relative_to(ROOT), f"invalid-xlsx:{error}"))
            continue
    elif path.suffix.lower() in allowed_binary:
        continue
    else:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    for name, pattern in patterns:
        if pattern.search(text):
            issues.append((path.relative_to(ROOT), name))
if issues:
    for path, reason in issues:
        print(f"{reason}: {path}")
    raise SystemExit(1)
print("privacy check passed")
