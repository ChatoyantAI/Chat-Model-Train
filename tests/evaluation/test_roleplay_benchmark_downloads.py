import os
from pathlib import Path
import subprocess
import sys
import urllib.request

import pytest

from src.evaluation.roleplay_benchmarks.sources import (
    ROOT, SafeRedirect, digest, inside, load_env_file, read_json, registry, sync_one, write_json,
)


def test_registry_is_pinned():
    assert set(registry()) == {"coser", "characterbench", "rmtbench", "minimax"}
    for spec in registry().values():
        assert len(spec["revision"]) == 40
        int(spec["revision"], 16)


def test_env_file_no_execution_and_existing_env_wins(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    path = tmp_path / "credentials.env"
    path.write_text('# comment\nHF_TOKEN="test-placeholder"\n', encoding="utf-8")
    load_env_file(path)
    assert os.environ["HF_TOKEN"] == "test-placeholder"
    monkeypatch.setenv("HF_TOKEN", "existing-placeholder")
    load_env_file(path)
    assert os.environ["HF_TOKEN"] == "existing-placeholder"


def test_env_error_does_not_echo_value(tmp_path):
    path = tmp_path / "credentials.env"
    path.write_text("UNEXPECTED=sensitive-placeholder", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_env_file(path)
    assert "sensitive-placeholder" not in str(exc.value)


def test_redirect_strips_cross_host_token():
    req = urllib.request.Request("https://huggingface.co/source", headers={"Authorization": "Bearer fake"})
    redirected = SafeRedirect().redirect_request(req, None, 302, "Found", {}, "https://cdn.example.org/file")
    assert redirected.get_header("Authorization") is None
    same_host = SafeRedirect().redirect_request(req, None, 302, "Found", {}, "https://huggingface.co/file")
    assert same_host.get_header("Authorization") == "Bearer fake"


def test_paths_cannot_escape(tmp_path):
    with pytest.raises(ValueError):
        inside(tmp_path, "../outside")
    assert inside(tmp_path, "data/a.json") == tmp_path / "data/a.json"


def test_sync_reuses_checked_file(tmp_path, monkeypatch):
    from src.evaluation.roleplay_benchmarks import sources
    monkeypatch.setattr(sources, "ROOT", tmp_path)
    target = tmp_path / "data"
    target.mkdir()
    asset = target / "example.json"
    asset.write_text("[]", encoding="utf-8")
    spec = {"kind": "hf_dataset", "path": "data", "repo": "example/test", "revision": "a" * 40, "files": ["example.json"]}
    write_json(target / "snapshot.json", {"revision": spec["revision"], "files": {"example.json": {"sha256": digest(asset), "bytes": 2}}})
    def no_download(*args, **kwargs):
        raise AssertionError("Existing verified files must not be re-downloaded")
    monkeypatch.setattr(sources, "download", no_download)
    assert sync_one("example", spec)["status"] == "complete"


def test_failed_source_is_not_complete(tmp_path, monkeypatch):
    from src.evaluation.roleplay_benchmarks import sources
    monkeypatch.setattr(sources, "ROOT", tmp_path)
    def denied(*args, **kwargs):
        raise PermissionError("HTTP 401: user authorization required")
    monkeypatch.setattr(sources, "download", denied)
    spec = {"kind": "hf_dataset", "path": "data", "repo": "example/test", "revision": "a" * 40, "files": ["example.json"]}
    result = sync_one("example", spec)
    assert result["status"] == "incomplete"
    assert result["files"] == {}
    assert "example.json" in read_json(tmp_path / "data/snapshot.json")["errors"]


def test_cli_help_needs_no_gpu_packages():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/download_roleplay_benchmarks.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--env-file" in result.stdout


def test_default_catalog_and_unknown_benchmark(capsys):
    from src.evaluation.roleplay_benchmarks.sources_cli import main
    assert main(["catalog"]) == 0
    assert main(["catalog", "unknown"]) == 2
    assert "Unknown benchmark" in capsys.readouterr().err


def test_credentials_and_downloads_ignored():
    paths = [".local/roleplay-benchmarks.env", "data/external/roleplay_benchmarks/example.json"]
    result = subprocess.run(["git", "check-ignore", *paths], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0
    assert all(path in result.stdout for path in paths)
    assert "HF_TOKEN=\n" in (ROOT / "configs/evaluation/roleplay_benchmarks/.env.example").read_text(encoding="utf-8")
