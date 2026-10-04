"""Run an opt-in CPU/HTTP smoke test using the published detector checkpoint.

Run from the repository root after installing requirements.txt:
    python tools/real_checkpoint_smoke.py
This downloads about 501 MB plus the upstream RoBERTa model/tokenizer files.
It does not change detector/server.py, disable safe loading, or deploy a service.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import urlopen

EXPECTED_SHA256 = "c74935bd6568940038e6bfcc9c90bf821d7ae4163ebf2327b73db2f641376376"
CHECKPOINT_URLS = (
    "https://huggingface.co/spaces/openai/openai-detector/resolve/main/detector-base.pt",
    "https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-base.pt",
)
OUT = Path("smoke-artifacts")
PORT = 18080


def download_checkpoint() -> Path:
    """Accept only the published checkpoint with the verified SHA256."""
    checkpoint = OUT / "detector-base.pt"
    failures = []
    for url in CHECKPOINT_URLS:
        print("Downloading checkpoint:", url, flush=True)
        digest = hashlib.sha256()
        size = 0
        started = time.monotonic()
        try:
            with urlopen(url, timeout=90) as response, checkpoint.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    if time.monotonic() - started > 360:
                        raise TimeoutError("Checkpoint download exceeded 360 seconds")
                    size += len(chunk)
                    if size > 600_000_000:
                        raise ValueError("Checkpoint exceeds the expected size range")
                    digest.update(chunk)
                    out.write(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != EXPECTED_SHA256:
                raise ValueError(f"Checkpoint SHA256 mismatch: {actual_sha256}")
            metadata = {"source": url, "bytes": size, "sha256": actual_sha256}
            (OUT / "checkpoint-metadata.json").write_text(json.dumps(metadata, indent=2))
            print("Checkpoint verified:", json.dumps(metadata), flush=True)
            return checkpoint
        except Exception as exc:
            checkpoint.unlink(missing_ok=True)
            failures.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not download the verified checkpoint:\n" + "\n".join(failures))


def test_http(process: subprocess.Popen) -> list[dict]:
    """Wait for the real server and test short and truncated long inputs."""
    deadline = time.monotonic() + 420
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited before readiness, code={process.returncode}")
        try:
            with urlopen(f"http://127.0.0.1:{PORT}/", timeout=2) as response:
                if response.status != 200 or not response.read():
                    raise RuntimeError("The server returned an invalid HTML response")
            break
        except (URLError, TimeoutError, ConnectionError):
            time.sleep(1)
    else:
        raise TimeoutError("The real server did not become ready within 420 seconds")
    results = []
    for name, text in (
        ("short_english", "This is a technical smoke test of a text classification service. "
         "The response must contain two finite probabilities summing to one."),
        ("long_english", "This sentence checks input truncation in the running service. " * 120),
    ):
        url = f"http://127.0.0.1:{PORT}/?" + quote(text, safe="")
        with urlopen(url, timeout=90) as response:
            if response.status != 200 or response.headers.get_content_type() != "application/json":
                raise RuntimeError(f"Invalid HTTP response for {name}")
            payload = json.load(response)
        probs = [payload["fake_probability"], payload["real_probability"]]
        if not all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in probs):
            raise AssertionError(f"Invalid probabilities: {payload}")
        if abs(sum(probs) - 1.0) > 1e-5:
            raise AssertionError(f"Probabilities do not sum to one: {payload}")
        used, total = payload["used_tokens"], payload["all_tokens"]
        if not isinstance(used, int) or not isinstance(total, int) or not 0 < used <= min(total, 510):
            raise AssertionError(f"Invalid token counts: {payload}")
        if name == "long_english" and total <= used:
            raise AssertionError("The long test did not exercise truncation")
        results.append({"case": name, "response": payload})
        print("PASS", name, json.dumps(payload), flush=True)
    return results


def main() -> int:
    OUT.mkdir(exist_ok=True)
    report = {
        "status": "failed", "stage": "environment", "python": sys.version,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "purpose": "Execution smoke test only; not a detector accuracy validation or deployment",
    }
    process = None
    try:
        import torch
        import transformers
        report["torch"] = torch.__version__
        report["transformers"] = transformers.__version__
        (OUT / "dependencies.txt").write_text(subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True))
        report["stage"] = "checkpoint_download"
        checkpoint = download_checkpoint()
        report["stage"] = "real_server_startup_and_http"
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
        with (OUT / "server.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", "detector.server", str(checkpoint),
                 "--device=cpu", f"--port={PORT}"],
                stdout=log, stderr=subprocess.STDOUT, env=env,
            )
            report["http_results"] = test_http(process)
        report.update(status="passed", stage="complete")
        return 0
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr, flush=True)
        return 1
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
        report["server_process_stopped"] = process is None or process.poll() is not None
        (OUT / "smoke-report.json").write_text(json.dumps(report, indent=2))
        if (OUT / "server.log").exists():
            print("\n--- server.log ---\n" + (OUT / "server.log").read_text(errors="replace"), flush=True)
        print("\n--- smoke report ---\n" + json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
