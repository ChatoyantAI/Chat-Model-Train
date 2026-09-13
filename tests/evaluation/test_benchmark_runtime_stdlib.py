import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.evaluation.roleplay_benchmarks.entry import entry
from src.evaluation.roleplay_benchmarks.preflight import character_inputs, inspect_source
from src.evaluation.roleplay_benchmarks.runtime import commands, patched_sources, prepare, run, validate_config
from src.evaluation.roleplay_benchmarks.sources import ROOT, read_json


@unittest.skipUnless((ROOT / "vendor/benchmarks/CharacterBench/process.py").exists(), "download upstream sources first")
class RuntimeContractTests(unittest.TestCase):
    def test_full_character_input_preflight(self):
        result = character_inputs(ROOT / "vendor/benchmarks/CharacterBench")
        self.assertEqual(result["expected"], {"en": 3250, "zh": 3250})
        self.assertEqual(result["valid"], {"en": 3247, "zh": 3250})
        self.assertEqual(len(result["errors"]), 3)
        self.assertFalse(result["complete"])

    def test_patches_are_executable_python_not_prompt_rewrites(self):
        for name, directory in (("coser", "CoSER"), ("characterbench", "CharacterBench")):
            patches = patched_sources(name, ROOT / "vendor/benchmarks" / directory)
            self.assertFalse(any("prompt" in path for path in patches))
            for code in patches.values():
                ast.parse(code)
        patches = patched_sources("characterbench", ROOT / "vendor/benchmarks/CharacterBench")
        calls = [n for n in ast.walk(ast.parse(patches["process.py"])) if isinstance(n, ast.Call)]
        self.assertFalse(any(isinstance(n.func, ast.Name) and n.func.id == "eval" for n in calls))
        self.assertIn("succeeded = True", patches["process.py"])
        self.assertIn("model=model_name", patches["gpt_call.py"])

    def test_coser_command_preserves_native_models(self):
        config = read_json(ROOT / "configs/evaluation/roleplay_benchmarks/coser.example.json")
        command = commands(config, "evaluate")[0][1]
        self.assertIn("gpt-4o-2024-08-06", command)
        self.assertNotIn("--temperature", command)
        self.assertNotIn("--wo_thought", command)

    def test_character_four_stage_commands(self):
        config = read_json(ROOT / "configs/evaluation/roleplay_benchmarks/characterbench.example.json")
        self.assertEqual(len(commands(config, "generate")), 1)
        for stage in ("prompts", "judge", "aggregate"):
            self.assertEqual(len(commands(config, stage)), 2)

    def test_public_runner_requires_run_directory(self):
        for benchmark in ("rmtbench", "minimax"):
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(entry(benchmark, ["run", "--execute"]), 2)

    def test_config_rejects_credentials_and_unsafe_alias(self):
        config = read_json(ROOT / "configs/evaluation/roleplay_benchmarks/characterbench.example.json")
        with self.assertRaises(ValueError):
            validate_config({**config, "api_key": "placeholder"})
        with self.assertRaises(ValueError):
            validate_config({**config, "actor_alias": "../outside"})

    def test_prepare_dry_run_and_no_paid_calls_on_bad_inputs(self):
        output = ROOT / "outputs"
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="benchmark-test-", dir=output) as temporary:
            directory = Path(temporary) / "run"
            manifest = prepare(ROOT / "configs/evaluation/roleplay_benchmarks/characterbench.example.json", directory)
            self.assertEqual(manifest["status"], "prepared-not-evaluated")
            self.assertTrue(run(directory, "generate")["dry_run"])
            with patch("src.evaluation.roleplay_benchmarks.runtime.subprocess.run") as process:
                with self.assertRaisesRegex(ValueError, "preprocessing rejects"):
                    run(directory, "generate", execute=True)
                process.assert_not_called()
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare(ROOT / "configs/evaluation/roleplay_benchmarks/characterbench.example.json", directory)

    def test_minimax_published_keys_match(self):
        result = inspect_source("minimax")
        self.assertFalse(result["source_preflight_passed"])
        for language in ("en", "zh"):
            self.assertTrue(result["published_trajectory_audit"][language]["evaluation_keys_match"])
            self.assertFalse(result["published_trajectory_audit"][language]["structural_errors"])


if __name__ == "__main__":
    unittest.main()
