"""Model endpoints, protocol selection and fixed sampling validation.

No model requests or workers are created here. Existing configuration keys and
defaults are kept so changing model endpoints does not change the benchmark.
"""
import json
import re

from . import community
from .sources import ROOT, read_json


BENCHMARKS = ("characterbench", "coser", "minimax", "rmtbench")
PUBLIC = {"minimax", "rmtbench"}
DEFAULT_LIMITS = {"local": 16, "remote": 96}
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def _role(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} endpoint must be an object")
    required = {"model", "revision", "base_url_env", "api_key_env", "kind", "sampling"}
    if set(value) != required:
        raise ValueError(f"{name} endpoint must contain exactly {sorted(required)}")
    for key in ("model", "revision", "base_url_env", "api_key_env"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"{name}.{key} is required")
    for key in ("base_url_env", "api_key_env"):
        if not ENV_NAME.fullmatch(value[key]):
            raise ValueError(f"{name}.{key} must be an environment variable name")
    if value["kind"] not in ("local", "remote"):
        raise ValueError(f"{name}.kind must be local or remote")
    sampling = value["sampling"]
    if not isinstance(sampling, dict) or set(sampling) - {"temperature", "top_p", "seed", "max_tokens", "max_completion_tokens"}:
        raise ValueError(f"{name}.sampling contains an unsupported field")
    if "max_tokens" in sampling and "max_completion_tokens" in sampling:
        raise ValueError(f"{name}.sampling chooses both max_tokens and max_completion_tokens")
    return value


def validate(config):
    """Validate orchestration config without contacting a model endpoint."""
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("schema_version=1 is required")
    from .assessment import validate as validate_assessment
    validate_assessment(config)
    names = config.get("benchmarks")
    if not isinstance(names, list) or not names or any(n not in BENCHMARKS for n in names) or len(set(names)) != len(names):
        raise ValueError("benchmarks must be a unique nonempty subset of the four public benchmarks")
    _role(config.get("actor"), "actor")
    _role(config.get("judge"), "judge")
    if config.get('assessment') and 'coser' in names and config['actor']['model'] == config['judge']['model']:
        raise ValueError('Reviewed CoSER requires distinct actor/helper model IDs for unambiguous routing')
    protocol = config.get('protocol', 'frozen-gemma-glm-v1')
    if protocol not in ('frozen-gemma-glm-v1', 'her-coser-format-v1', 'her-coser-official-v2'):
        raise ValueError('Unknown public evaluation protocol')
    if protocol in ('her-coser-format-v1', 'her-coser-official-v2'):
        if names != ['coser'] or config['actor']['model'] == config['judge']['model']:
            raise ValueError('HER format protocol requires CoSER only and distinct actor/judge models')
    if protocol == 'her-coser-official-v2':
        if config.get('her_system_thinking') not in ('with-system', 'without-system'):
            raise ValueError('HER v2 requires explicit her_system_thinking mode')
        if config.get('actor_output_validation') != 'role-content-v2':
            raise ValueError('HER v2 requires strict role-content-v2 validation')
        if config['actor']['kind'] != 'local':
            raise ValueError('HER v2 requires a vLLM-compatible actor endpoint')
        policies = config.get('thinking', {})
        policy = policies.get(config['actor']['model'], policies.get('local', {}))
        if policy.get('chat_template_kwargs', {}).get('enable_thinking') is not False:
            raise ValueError('HER system thinking is separate from native thinking; disable native thinking')
    elif 'her_system_thinking' in config:
        raise ValueError('her_system_thinking requires the HER v2 protocol')
    if 'actor_prompt_format' in config:
        if protocol != 'her-coser-format-v1' or config['actor_prompt_format'] not in ('her-official-v1', 'her-no-system-v1', 'coser-native-v1'):
            raise ValueError('Supported actor_prompt_format requires the HER CoSER protocol')
    if 'actor_output_validation' in config:
        if protocol not in ('her-coser-format-v1', 'her-coser-official-v2') or config['actor_output_validation'] != 'role-content-v2':
            raise ValueError('actor_output_validation=role-content-v2 requires the HER CoSER protocol')
    if "minimax" in names:
        _role(config.get("simulator"), "simulator")
    for key in ("max_attempts", "timeout_seconds", "max_requests_per_run"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    limits = config.get("concurrency", {})
    if not isinstance(limits, dict):
        raise ValueError("concurrency must be an object")
    endpoint_limits = limits.get("endpoints", DEFAULT_LIMITS)
    if set(endpoint_limits) != {"local", "remote"} or any(type(v) is not int or v < 1 for v in endpoint_limits.values()):
        raise ValueError("concurrency.endpoints must contain positive local/remote limits")
    per_benchmark = limits.get("benchmarks", {})
    if set(per_benchmark) - set(names) or any(type(v) is not int or v < 1 for v in per_benchmark.values()):
        raise ValueError("concurrency.benchmarks contains an invalid limit")
    options = config.get("benchmark_options", {})
    if set(options) - set(names):
        raise ValueError("benchmark_options contains an unselected benchmark")
    if not isinstance(config.get("thinking", {}), dict):
        raise ValueError("thinking must be an object")
    for options in config.get('thinking', {}).values():
        if not isinstance(options, dict) or set(options) - {'chat_template_kwargs', 'reasoning_effort', 'thinking'}:
            raise ValueError('thinking accepts only provider thinking options, never model/messages/credentials')
    if set(names) & {'characterbench', 'coser'}:
        for key, temperature in [('actor', .7), ('judge', 0)]:
            sampling = config[key]['sampling']
            tokens = sampling.get('max_tokens', 8192)
            extended_actor = (protocol in ('her-coser-format-v1', 'her-coser-official-v2') and key == 'actor'
                              and config['actor']['kind'] == 'local')
            valid_tokens = (type(tokens) is int and 8192 <= tokens <= 32768) if extended_actor else tokens == 8192
            if set(sampling) - {'temperature','max_tokens'} or sampling.get('temperature', temperature) != temperature or not valid_tokens:
                raise ValueError('Frozen sampling requires actor .7/helper 0; only local HER CoSER actor tokens may increase')
            if extended_actor and tokens > 8192:
                window = config.get('actor_context_window')
                if type(window) is not int or window <= tokens:
                    raise ValueError('Extended local actor requires actor_context_window greater than max_tokens')
        for name in set(names) & {'characterbench','coser'}:
            if set(config.get('benchmark_options',{}).get(name,{})) - {'limit_cases','case_ids'}:
                raise ValueError('Char/CoSER options only select unchanged samples')
    return config


def public_config(root_config, benchmark):
    base = read_json(ROOT / f"configs/evaluation/roleplay_benchmarks/{benchmark}.example.json")
    if benchmark == 'rmtbench':
        base.update(version='claude45', layout='paper_four_blocks', repeats=2, seed=42)
    options = _copy(root_config.get("benchmark_options", {}).get(benchmark, {}))
    base.update(options)
    base["benchmark"] = benchmark
    base["max_attempts"] = root_config["max_attempts"]
    base["timeout_seconds"] = root_config["timeout_seconds"]
    base["max_requests_per_run"] = root_config["max_requests_per_run"]
    limits = root_config.get("concurrency", {}).get("benchmarks", {})
    base["concurrency"] = limits.get(benchmark, base.get("concurrency", 1))
    roles = {"actor": root_config["actor"], "judge": root_config["judge"]}
    if benchmark == "minimax":
        roles["simulator"] = root_config["simulator"]
    # Community validation intentionally remains strict; kind and thinking
    # policies are orchestration concerns and stay in the root manifest.
    base["roles"] = {key: {k: value[k] for k in ("model", "revision", "base_url_env", "api_key_env", "sampling")}
                     for key, value in roles.items()}
    base["protocol_label"] = f"{root_config.get('protocol_label', 'unified-public-v1')}:{benchmark}"
    community.validate(base)
    return base
