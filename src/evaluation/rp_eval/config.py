from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "eval.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path.resolve())
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("\"").strip("'")
    return values


def load_secret_value(secret_file: str, env_name: str) -> str:
    if os.environ.get(env_name):
        return os.environ[env_name]
    secret_path = PROJECT_ROOT / secret_file
    if not secret_path.exists():
        raise RuntimeError(f"missing {env_name} and secret file {secret_path}")
    if os.name == "posix":
        mode = stat.S_IMODE(secret_path.stat().st_mode)
        if mode & 0o077:
            raise RuntimeError(f"secret file must not be group/world accessible: {secret_path} mode={mode:o}")
    value = _parse_env_file(secret_path).get(env_name)
    if not value:
        raise RuntimeError(f"{env_name} is absent from {secret_path}")
    return value


def load_openrouter_key(config: dict[str, Any]) -> str:
    section = config["openrouter"]
    env_name = section["api_key_env"]
    return load_secret_value(section["secret_file"], env_name)


def candidate_readiness(config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, candidate in config["candidates"].items():
        missing: list[str] = []
        if not candidate.get("ready"):
            missing.append("ready=false")
        if not candidate.get("prompt_provenance"):
            missing.append("prompt_provenance")
        for field in ("base_url_env", "model_env"):
            env_name = candidate.get(field)
            if env_name and not os.environ.get(env_name):
                missing.append(env_name)
        if candidate.get("api_key_env"):
            try:
                load_secret_value(candidate.get("secret_file", ""), candidate["api_key_env"])
            except RuntimeError:
                missing.append(candidate["api_key_env"])
        result[name] = {"ready": not missing, "missing": missing, "adapter": candidate.get("adapter")}
    return result
