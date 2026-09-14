"""Credential/download checks runnable without any third-party test dependencies."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.request

from src.evaluation.roleplay_benchmarks.sources import SafeRedirect, digest, inside, load_env_file, registry, sync_one, write_json


class DownloadContractTests(unittest.TestCase):
    def test_frozen_sources(self):
        self.assertEqual(len(registry()), 4)
        for spec in registry().values():
            self.assertEqual(len(spec["revision"]), 40)
            int(spec["revision"], 16)

    def test_credentials_and_precedence(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            path = Path(temp) / "test.env"
            path.write_text('HF_TOKEN="placeholder"\n', encoding="utf-8")
            load_env_file(path)
            self.assertEqual(os.environ["HF_TOKEN"], "placeholder")
            os.environ["HF_TOKEN"] = "existing"
            load_env_file(path)
            self.assertEqual(os.environ["HF_TOKEN"], "existing")

    def test_no_secret_in_parse_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.env"
            path.write_text('INVALID=private-placeholder', encoding="utf-8")
            with self.assertRaises(ValueError) as captured:
                load_env_file(path)
            self.assertNotIn("private-placeholder", str(captured.exception))

    def test_redirect_boundary(self):
        req = urllib.request.Request("https://huggingface.co/source", headers={"Authorization": "Bearer placeholder"})
        handler = SafeRedirect()
        remote = handler.redirect_request(req, None, 302, "Found", {}, "https://cdn.example.org/file")
        self.assertIsNone(remote.get_header("Authorization"))
        local = handler.redirect_request(req, None, 302, "Found", {}, "https://huggingface.co/file")
        self.assertEqual(local.get_header("Authorization"), "Bearer placeholder")
        with self.assertRaises(ValueError):
            handler.redirect_request(req, None, 302, "Found", {}, "http://huggingface.co/file")

    def test_path_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                inside(temp, "../outside")

    def test_cached_file_never_downloaded_twice(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            file = root / "data/sample.json"
            write_json(file, [])
            spec = {"kind": "hf_dataset", "path": "data", "repo": "example/test", "revision": "a"*40, "files": ["sample.json"]}
            write_json(root / "data/snapshot.json", {"revision": spec["revision"], "files": {"sample.json": {"sha256": digest(file)}}})
            with patch("src.evaluation.roleplay_benchmarks.sources.ROOT", root), patch("src.evaluation.roleplay_benchmarks.sources.download") as download:
                self.assertEqual(sync_one("test", spec)["status"], "complete")
                download.assert_not_called()

    def test_access_failure_is_not_success(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = {"kind": "hf_dataset", "path": "data", "repo": "example/test", "revision": "a"*40, "files": ["sample.json"]}
            with patch("src.evaluation.roleplay_benchmarks.sources.ROOT", Path(temp)), patch("src.evaluation.roleplay_benchmarks.sources.download", side_effect=PermissionError("HTTP 401")):
                result = sync_one("test", spec)
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["files"], {})


if __name__ == "__main__":
    unittest.main()
