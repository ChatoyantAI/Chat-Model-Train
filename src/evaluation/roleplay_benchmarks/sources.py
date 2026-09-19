"""Source synchronization and integrity checks. No model requests are made here."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
LOCK = ROOT / "configs/evaluation/roleplay_benchmarks/sources.lock.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True, encoding="utf-8").strip()


def registry():
    return read_json(LOCK)["benchmarks"]


def load_env_file(path):
    """Read a credential file as data, not shell code; environment takes precedence."""
    allowed = {"HF_TOKEN"}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or key not in allowed:
            raise ValueError(f"Unsupported credential-file entry at line {number}; only HF_TOKEN is accepted")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value and not os.environ.get(key):
            os.environ[key] = value


def inside(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError(f"Unsafe relative path: {relative}")
    return target


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected:
            destination = urllib.parse.urlparse(newurl)
            if destination.scheme != "https":
                raise ValueError("Refusing an insecure download redirect")
            if urllib.parse.urlparse(req.full_url).netloc != destination.netloc:
                redirected.remove_header("Authorization")
        return redirected


def download(url, target, token=None):
    headers = {"User-Agent": "Chat-Model-Train-benchmark-sync/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    opener = urllib.request.build_opener(SafeRedirect())
    for attempt in range(3):
        try:
            with opener.open(request, timeout=120) as response, partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            with partial.open("rb") as probe:
                if probe.read(100).startswith(b"version https://git-lfs.github.com/spec"):
                    raise ValueError("Received an LFS pointer instead of asset bytes")
            partial.replace(target)
            return
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                raise PermissionError(f"HTTP {error.code}: gated source requires user-owned HF access; no terms accepted automatically") from None
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f"Source download failed: HTTP {error.code}") from None
        except (TimeoutError, urllib.error.URLError):
            if attempt == 2:
                raise RuntimeError("Source download failed: transport timeout/error") from None
        time.sleep(2 ** attempt)


def verify_git(spec):
    path = inside(ROOT, spec["path"])
    if not (path / ".git").exists():
        raise ValueError("Missing submodule; run git submodule update --init --recursive")
    if git("rev-parse", "HEAD", cwd=path) != spec["revision"]:
        raise ValueError("Upstream commit differs from sources.lock.json")
    if git("diff", "--name-only", "HEAD", "--", cwd=path):
        raise ValueError("Tracked upstream source changed; retain patches outside vendor")
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=path).decode("utf-8").rstrip("\0").split("\0")
    return {"status": "verified", "revision": spec["revision"], "files": {
        name: {"sha256": digest(inside(path, name)), "bytes": inside(path, name).stat().st_size}
        for name in files
    }}


def sync_one(name, spec):
    if spec["kind"] == "git":
        if not (inside(ROOT, spec["path"]) / ".git").exists():
            subprocess.run(["git", "submodule", "update", "--init", "--", spec["path"]], cwd=ROOT, check=True)
        result = verify_git(spec)
        write_json(ROOT / f"data/external/roleplay_benchmarks/{name}/snapshot.json", result)
        return result
    target = inside(ROOT, spec["path"])
    manifest_path = target / "snapshot.json"
    previous = read_json(manifest_path) if manifest_path.exists() else {}
    if previous and previous.get("revision") != spec["revision"]:
        raise ValueError("Existing dataset revision differs; move it to a separate archive before syncing")
    result = {"status": "complete", "repo": spec["repo"], "revision": spec["revision"], "files": {}, "errors": {}}
    for name in spec["files"]:
        file = inside(target, name)
        cached = previous.get("files", {}).get(name)
        try:
            if not (file.exists() and cached and digest(file) == cached["sha256"]):
                url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{urllib.parse.quote(name)}"
                download(url, file, os.environ.get("HF_TOKEN"))
            result["files"][name] = {"sha256": digest(file), "bytes": file.stat().st_size}
        except (PermissionError, RuntimeError, ValueError) as error:
            result["status"] = "incomplete"
            result["errors"][name] = str(error)
        write_json(manifest_path, result)
    return result


def verify_one(name, spec):
    if spec["kind"] == "git":
        return verify_git(spec)
    root = inside(ROOT, spec["path"])
    result = read_json(root / "snapshot.json")
    if result["revision"] != spec["revision"]:
        raise ValueError("Dataset revision mismatch")
    for name in spec["files"]:
        record = result.get("files", {}).get(name)
        if not record or digest(inside(root, name)) != record["sha256"]:
            raise ValueError(f"Missing or modified source file: {name}")
    if result["status"] != "complete":
        raise ValueError("Dataset snapshot incomplete")
    return result


def inventory(name, spec):
    base = inside(ROOT, spec["path"])
    if name == "coser":
        rows = read_json(base / "data/test/test_set.json")
        return {"scenarios": len(rows), "books": len({r["book"] for r in rows})}
    if name == "characterbench":
        counts = {p.name: len(read_json(p)) for p in sorted((base / "eval_data/raw_data").glob("*.json"))}
        return {"files": counts, "source_rows": sum(counts.values()), "languages": ["zh", "en"], "generation_requests": sum(counts.values()) * 2}
    if name == "minimax":
        counts = {}
        for lang in ("en", "zh"):
            for kind in ("seeds", "dialogues", "evaluations"):
                path = base / f"data/{lang}/{kind}.jsonl"
                if path.exists():
                    with path.open(encoding="utf-8") as stream:
                        counts[f"{lang}/{kind}"] = sum(1 for row in stream if row.strip())
        return counts
    return {file: len(read_json(base / file)) for file in spec["files"] if file.endswith(".json") and (base / file).exists()}
