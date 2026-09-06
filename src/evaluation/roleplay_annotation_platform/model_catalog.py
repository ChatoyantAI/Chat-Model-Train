from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from .llm_judge import (
    JudgeConfig,
    MODEL_ID_RE,
    TATU_MODEL_PREFIX,
    score_logprobs_enabled_for_model,
)


REQUIRED_PARAMETERS = {"response_format", "structured_outputs"}
DISABLED_TATU_MODEL_IDS = frozenset({"glm-5.3", "glm-5.3-flash"})
TATU_MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "glm-5.3",
        "provider": "Zhipu AI",
        "deployer": "ZAI-default",
        "context_length": 1_000_000,
        "max_output_length": 128_000,
        "input_price_cny_per_million": 8.0,
        "cached_input_price_cny_per_million": 2.0,
        "output_price_cny_per_million": 28.0,
    },
    {
        "id": "glm-5.3-flash",
        "provider": "Zhipu AI",
        "deployer": "ZAI-default",
        "context_length": 1_000_000,
        "max_output_length": 128_000,
        "input_price_cny_per_million": 0.8,
        "cached_input_price_cny_per_million": 0.23,
        "output_price_cny_per_million": 2.8,
    },
    {
        "id": "deepseek-v4-pro-0813",
        "upstream_model": "deepseek-v4-pro",
        "provider": "DeepSeek",
        "deployer": "DEEPSEEK-default",
        "context_length": 1_000_000,
        "max_output_length": 128_000,
        "input_price_cny_per_million": 9.0,
        "cached_input_price_cny_per_million": 0.3,
        "output_price_cny_per_million": 27.0,
    },
    {
        "id": "qwen3.8-27b",
        "provider": "Alibaba",
        "deployer": "ALIBABA-default",
        "context_length": 991_000,
        "max_output_length": 128_000,
        "input_price_cny_per_million": 3.0,
        "cached_input_price_cny_per_million": 0.6,
        "output_price_cny_per_million": 12.0,
    },
    {
        "id": "kimi-k3",
        "provider": "Kimi",
        "deployer": "KIMI-default",
        "context_length": 1_048_576,
        "max_output_length": 131_072,
        "input_price_cny_per_million": 20.0,
        "cached_input_price_cny_per_million": 2.0,
        "output_price_cny_per_million": 100.0,
    },
)


def _price_per_million(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        price = Decimal(str(value)) * Decimal(1_000_000)
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not price.is_finite() or price < 0:
        return None
    return float(price)


def _fallback_models(config: JudgeConfig) -> list[dict[str, Any]]:
    return [
        {
            "id": model,
            "name": model,
            "label": model,
            "context_length": None,
            "prompt_price_per_million": None,
            "completion_price_per_million": None,
            "gateway": "openrouter",
            "reports_token_logprobs": False,
            "reports_cost_usd": True,
        }
        for model in config.models
        if not model.startswith(TATU_MODEL_PREFIX)
        and not any(
            model.lower().startswith(prefix)
            for prefix in config.excluded_openrouter_model_prefixes
        )
    ]


def _tatu_model_entry(spec: dict[str, Any]) -> dict[str, Any]:
    model_id = str(spec["id"])
    deployer = str(spec["deployer"])
    provider = str(spec["provider"])
    return {
        "id": f"{TATU_MODEL_PREFIX}{model_id}@{deployer}",
        "name": model_id,
        "label": f"TatuCloud · {model_id} · {provider} / {deployer}",
        "gateway": "tatu",
        "backend_model": f"{model_id}@{deployer}",
        "deployer": deployer,
        "upstream_provider": provider,
        "upstream_model": str(spec.get("upstream_model", model_id)),
        "api_type": "chat_completions",
        "context_length": int(spec["context_length"]),
        "max_output_length": int(spec["max_output_length"]),
        "prompt_price_per_million": None,
        "completion_price_per_million": None,
        "pricing": {
            "currency": "CNY",
            "input_per_million": spec["input_price_cny_per_million"],
            "cached_input_per_million": spec["cached_input_price_cny_per_million"],
            "output_per_million": spec["output_price_cny_per_million"],
            "source": "model-pricing-catalog.xlsx",
        },
        "supports_reasoning": False,
        "reports_token_logprobs": False,
        "reports_cost_usd": False,
    }


def _fallback_tatu_models(config: JudgeConfig) -> list[dict[str, Any]]:
    if not config.tatu_api_key:
        return []
    return [
        _tatu_model_entry(spec)
        for spec in TATU_MODEL_SPECS
        if str(spec["id"]) not in DISABLED_TATU_MODEL_IDS
    ]


def parse_openrouter_models(
    payload: Any,
    *,
    min_context_length: int,
    excluded_model_prefixes: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("OpenRouter Models API returned an invalid payload")

    compatible: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in payload["data"]:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("id", "")).strip()
        if not MODEL_ID_RE.fullmatch(model_id) or model_id in seen:
            continue
        if model_id.startswith("openrouter/"):
            continue
        if any(
            model_id.lower().startswith(prefix) for prefix in excluded_model_prefixes
        ):
            continue
        if model_id.endswith((":free", ":batch")):
            continue
        try:
            context_length = int(entry.get("context_length", 0))
        except (TypeError, ValueError):
            continue
        if min_context_length and context_length < min_context_length:
            continue

        supported = {
            str(parameter)
            for parameter in entry.get("supported_parameters", [])
            if isinstance(parameter, str)
        }
        if not REQUIRED_PARAMETERS.issubset(supported):
            continue

        architecture = entry.get("architecture")
        if isinstance(architecture, dict):
            output_modalities = architecture.get("output_modalities")
            if isinstance(output_modalities, list) and output_modalities != ["text"]:
                continue

        pricing = entry.get("pricing")
        if not isinstance(pricing, dict):
            pricing = {}
        name = str(entry.get("name", "")).strip() or model_id
        prompt_price = _price_per_million(pricing.get("prompt"))
        completion_price = _price_per_million(pricing.get("completion"))
        label = f"{name} · {context_length // 1024:,}K ctx"
        if prompt_price is not None and completion_price is not None:
            label += f" · ${prompt_price:g}/${completion_price:g} per M"
        compatible.append(
            {
                "id": model_id,
                "name": name,
                "label": label,
                "context_length": context_length,
                "prompt_price_per_million": prompt_price,
                "completion_price_per_million": completion_price,
                "supports_reasoning": "reasoning" in supported,
                "gateway": "openrouter",
                "reports_token_logprobs": {
                    "logprobs",
                    "top_logprobs",
                }.issubset(supported),
                "reports_cost_usd": True,
            }
        )
        seen.add(model_id)

    compatible.sort(key=lambda model: (str(model["name"]).casefold(), model["id"]))
    if not compatible:
        raise ValueError("OpenRouter model catalog contains no compatible Judge models")
    return compatible


def parse_tatu_models(
    payload: Any,
    *,
    min_context_length: int,
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("TatuCloud Models API returned an invalid payload")
    available_ids = {
        str(entry.get("id", "")).strip()
        for entry in payload["data"]
        if isinstance(entry, dict)
    }
    models = [
        _tatu_model_entry(spec)
        for spec in TATU_MODEL_SPECS
        if str(spec["id"]) in available_ids
        and str(spec["id"]) not in DISABLED_TATU_MODEL_IDS
        and (
            not min_context_length or int(spec["context_length"]) >= min_context_length
        )
    ]
    if not models:
        raise ValueError("TatuCloud 模型目录中没有找到已启用的 Judge 模型")
    return models


class OpenRouterModelCatalog:
    def __init__(self, config: JudgeConfig) -> None:
        self.config = config
        self._lock = asyncio.Lock()
        self._snapshot: dict[str, Any] = {
            "source": "static+tatu" if config.tatu_api_key else "static",
            "fetched_at": None,
            "stale": config.model_source == "openrouter" or bool(config.tatu_api_key),
            "error": None,
            "models": self._apply_route_capabilities(
                [*_fallback_models(config), *_fallback_tatu_models(config)]
            ),
        }

    def _apply_route_capabilities(
        self, models: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        for model in models:
            if model.get("gateway") != "openrouter":
                continue
            model["reports_token_logprobs"] = bool(
                model.get("reports_token_logprobs")
            ) and score_logprobs_enabled_for_model(
                self.config, str(model.get("id", ""))
            )
        return models

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(str(model["id"]) for model in self._snapshot["models"])

    @property
    def reasoning_model_ids(self) -> tuple[str, ...]:
        return tuple(
            str(model["id"])
            for model in self._snapshot["models"]
            if model.get("supports_reasoning") is True
        )

    def public_snapshot(self) -> dict[str, Any]:
        payload = deepcopy(self._snapshot)
        payload["count"] = len(payload["models"])
        payload["filter"] = {
            "min_context_length": self.config.model_min_context_length or None,
            "required_parameters": sorted(REQUIRED_PARAMETERS),
            "excluded_suffixes": [":free", ":batch"],
            "excluded_namespaces": ["openrouter/"],
            "excluded_model_prefixes": list(
                self.config.excluded_openrouter_model_prefixes
            ),
            "output_modality": "text_only",
            "tatu_allowlist": [
                str(spec["id"])
                for spec in TATU_MODEL_SPECS
                if str(spec["id"]) not in DISABLED_TATU_MODEL_IDS
            ],
        }
        return payload

    async def refresh(self) -> dict[str, Any]:
        if self.config.model_source != "openrouter" and not self.config.tatu_api_key:
            return self.public_snapshot()
        async with self._lock:
            models: list[dict[str, Any]] = []
            sources: list[str] = []
            errors: list[str] = []
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.model_catalog_timeout_seconds)
            ) as client:
                if self.config.model_source == "openrouter":
                    try:
                        response = await client.get(
                            self.config.openrouter_models_url,
                            headers={"Accept": "application/json"},
                        )
                        response.raise_for_status()
                        models.extend(
                            parse_openrouter_models(
                                response.json(),
                                min_context_length=self.config.model_min_context_length,
                                excluded_model_prefixes=(
                                    self.config.excluded_openrouter_model_prefixes
                                ),
                            )
                        )
                        sources.append("openrouter")
                    except httpx.HTTPStatusError as error:
                        models.extend(_fallback_models(self.config))
                        sources.append("static")
                        errors.append(
                            f"OpenRouter 模型目录返回 HTTP {error.response.status_code}"
                        )
                    except httpx.HTTPError:
                        models.extend(_fallback_models(self.config))
                        sources.append("static")
                        errors.append("无法连接 OpenRouter 模型目录")
                    except ValueError as error:
                        models.extend(_fallback_models(self.config))
                        sources.append("static")
                        errors.append(str(error)[:500])
                else:
                    models.extend(_fallback_models(self.config))
                    sources.append("static")

                if self.config.tatu_api_key:
                    try:
                        response = await client.get(
                            self.config.tatu_models_url,
                            headers={
                                "Accept": "application/json",
                                "Authorization": f"Bearer {self.config.tatu_api_key}",
                            },
                        )
                        response.raise_for_status()
                        models.extend(
                            parse_tatu_models(
                                response.json(),
                                min_context_length=self.config.model_min_context_length,
                            )
                        )
                        sources.append("tatu")
                    except httpx.HTTPStatusError as error:
                        models.extend(_fallback_tatu_models(self.config))
                        sources.append("tatu-static")
                        errors.append(
                            f"TatuCloud 模型目录返回 HTTP {error.response.status_code}"
                        )
                    except httpx.HTTPError:
                        models.extend(_fallback_tatu_models(self.config))
                        sources.append("tatu-static")
                        errors.append("无法连接 TatuCloud 模型目录")
                    except ValueError as error:
                        models.extend(_fallback_tatu_models(self.config))
                        sources.append("tatu-static")
                        errors.append(str(error)[:500])

            deduplicated: list[dict[str, Any]] = []
            seen: set[str] = set()
            for model in models:
                model_id = str(model["id"])
                if model_id in seen:
                    continue
                seen.add(model_id)
                deduplicated.append(model)
            if not deduplicated:
                deduplicated = [
                    *_fallback_models(self.config),
                    *_fallback_tatu_models(self.config),
                ]
                errors.append("所有 Judge 模型目录均为空")
            self._snapshot = {
                "source": "+".join(dict.fromkeys(sources)) or "static",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "stale": bool(errors),
                "error": "；".join(errors) or None,
                "models": self._apply_route_capabilities(deduplicated),
            }
            return self.public_snapshot()
