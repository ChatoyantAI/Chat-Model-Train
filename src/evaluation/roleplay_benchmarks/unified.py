"""Unified, resumable entry point for the four role-play benchmarks.

The benchmark protocol owns prompts, cases, parsing, and aggregation.  This
module owns only run orchestration: endpoint configuration, shared limits,
durable request caches, and asynchronous derived checkpoints.  A local vLLM
server and a remote OpenAI-compatible API therefore have the same interface.

Request/response files are written synchronously before a result is returned;
this is the no-repeat boundary for paid requests.  Progress, trajectories,
judgments, and manifests are derived data and are published through the
bounded asynchronous writer.  The two classes of writes must not be swapped.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

from . import benchmark_api, community, frozen_backend
from .endpoint_transport import JournalSender, endpoint_ownership, scoped_clients, coser_server, usage_report
from .sources import ROOT, read_json, write_json

# The tested async writer/accounting process live with the operational
# evaluation scripts.  Put that directory on the inherited import path before
# starting a multiprocessing worker (spawn does not inherit module aliases).
_INFRA = Path(__file__).resolve().parents[3] / "scripts" / "evaluation"
if str(_INFRA) not in sys.path:
    sys.path.insert(0, str(_INFRA))


BENCHMARKS = ("characterbench", "coser", "minimax", "rmtbench")
PUBLIC = {"minimax", "rmtbench"}
DEFAULT_LIMITS = {"local": 16, "remote": 96}
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


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
    names = config.get("benchmarks")
    if not isinstance(names, list) or not names or any(n not in BENCHMARKS for n in names) or len(set(names)) != len(names):
        raise ValueError("benchmarks must be a unique nonempty subset of the four public benchmarks")
    _role(config.get("actor"), "actor")
    _role(config.get("judge"), "judge")
    if config.get('protocol', 'frozen-gemma-glm-v1') != 'frozen-gemma-glm-v1':
        raise ValueError('Use the fixed frozen-gemma-glm-v1 protocol')
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
            if set(sampling) - {'temperature','max_tokens'} or sampling.get('temperature', temperature) != temperature or sampling.get('max_tokens',8192) != 8192:
                raise ValueError('Frozen Char/CoSER uses actor .7, helper 0, max_tokens 8192')
        for name in set(names) & {'characterbench','coser'}:
            if set(config.get('benchmark_options',{}).get(name,{})) - {'limit_cases','case_ids'}:
                raise ValueError('Char/CoSER options only select unchanged samples')
    return config


def _endpoint_url(role):
    base = os.environ.get(role["base_url_env"], "").rstrip("/")
    if not base:
        raise ValueError(f"Missing environment variable: {role['base_url_env']}")
    # benchmark_api.endpoint performs the complete URL validation when called;
    # this check keeps a missing key from failing after producers are started.
    if not os.environ.get(role["api_key_env"]):
        raise ValueError(f"Missing environment variable: {role['api_key_env']}")
    return base


class EndpointLimiter:
    """One shared limit per endpoint kind for all selected benchmarks."""

    def __init__(self, limits):
        import threading
        self._semaphores = {kind: threading.BoundedSemaphore(int(value))
                            for kind, value in limits.items()}
        self._active = {kind: 0 for kind in limits}
        self._lock = threading.Lock()

    @contextmanager
    def slot(self, kind):
        semaphore = self._semaphores[kind]
        semaphore.acquire()
        with self._lock:
            self._active[kind] += 1
        try:
            yield
        finally:
            with self._lock:
                self._active[kind] -= 1
            semaphore.release()

    def snapshot(self):
        with self._lock:
            return dict(self._active)


class DerivedStore:
    """Asynchronous JSON publication for data that can be reconstructed.

    The request cache in :mod:`benchmark_api` deliberately remains unchanged;
    it is the durable evidence that prevents a paid request from being sent
    twice.  This store is used for trajectories, judgments, progress and
    summaries only.
    """

    def __init__(self, limit=256, workers=4):
        # Keep this dependency in the operational scripts directory so the
        # existing, tested writer remains the single implementation.
        from scripts.evaluation.async_checkpoints import AsyncCheckpoints
        self.writer = AsyncCheckpoints(limit=limit, workers=workers)

    def write(self, path, value):
        self.writer.write(Path(path), value)

    def read(self, path):
        return self.writer.read(Path(path))

    def flush(self, prefix=None):
        self.writer.flush(prefix=prefix)

    def close(self):
        self.writer.close()


class UsageLedger:
    """Non-blocking, process-isolated accounting for requests dispatched here."""

    def __init__(self, directory, config):
        from scripts.evaluation.process_accounting import ProcessLedger
        options = config.get("process_accounting", {})
        self.ledger = ProcessLedger(Path(directory) / "usage-ledger",
                                     limit=int(options.get("limit", 512)),
                                     batch_size=int(options.get("batch_size", 64)),
                                     batch_wait=float(options.get("batch_wait", .25)))

    def append(self, role, payload, status, usage=None, error=None):
        usage = usage if isinstance(usage, dict) else {}
        self.ledger.append({"at": time.time(), "role": role.get("name", "unknown"),
                            "model": role.get("model"), "request_hash": _hash(payload),
                            "status": status, "usage": usage,
                            "cost": usage.get("cost", usage.get("estimated_cost")),
                            "error": error})

    def flush(self):
        self.ledger.flush()

    def snapshot(self):
        return self.ledger.snapshot()

    def close(self):
        self.ledger.close()


def _public_config(root_config, benchmark):
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


@frozen_backend.public_protocol()
def prepare(config_path, run_dir):
    """Create one manifest and one isolated child run per benchmark."""
    config = read_json(config_path) if not isinstance(config_path, dict) else _copy(config_path)
    validate(config)
    directory = Path(run_dir).resolve()
    # A local SSD/RAM directory is the intended hot path.  Direct benchmark
    # entry points keep their repository-only default; the unified entry point
    # records the explicit external-path decision in its root manifest.
    if directory.exists() or directory == ROOT / "outputs":
        raise ValueError("Use a new run directory")
    directory.mkdir(parents=True)
    children = {}
    for benchmark in config["benchmarks"]:
        child = directory / benchmark
        child_config = directory / f"{benchmark}.config.json"
        if benchmark in PUBLIC:
            child_value = _public_config(config, benchmark)
            write_json(child_config, child_value)
            community.prepare(child_config, child, allow_external=True)
            if benchmark == 'rmtbench' and child_value.get('layout') == 'paper_four_blocks':
                detail = read_json(child / 'manifest.json')
                detail['protocol_choices'] = [p for p in detail['protocol_choices'] if not p.startswith('mixed_four_blocks')]
                detail['protocol_choices'].extend([
                    'Four public blocks sampled without replacement; within-block order unchanged; more than 20 user rounds.',
                    'Two trajectories per character, seed 42; exact author manifest unavailable; SEC/Poison excluded.',
                    'Claude 4.5 source and configured API judge follow the historical local protocol, not the original leaderboard.',
                ])
                detail['runtime_extension'] = frozen_backend.HASHES
                write_json(child / 'manifest.json', detail)
        else:
            metadata = frozen_backend.prepare(config, child, benchmark)
            write_json(child / 'manifest.json', {'config': config, 'metadata': metadata})
        children[benchmark] = ({"run_dir": benchmark, "config_sha256": _hash(child_value), 'runner': 'community'}
                               if benchmark in PUBLIC else {'run_dir': benchmark, **metadata})
    manifest = {"schema_version": 2, "runner": "unified-roleplay-v2", "status": "prepared",
                "config": config, "config_sha256": _hash(config), "children": children,
                "created_at": time.time(), "protocol": "fixed benchmark adapters + endpoint orchestration"}
    write_json(directory / "manifest.json", manifest)
    return manifest


def _patch_async_public(store):
    """Patch only derived community files; benchmark request cache stays sync."""
    originals = {name: getattr(community, name) for name in ("write_json", "read_json", "aggregate", "judge_case")}
    community.write_json = store.write
    community.read_json = store.read
    original_judge = originals["judge_case"]
    original_aggregate = originals["aggregate"]

    def judge(*args, **kwargs):
        store.flush(Path(args[3]) / "trajectory.json")
        return original_judge(*args, **kwargs)

    def aggregate(*args, **kwargs):
        store.flush(Path(args[0]))
        return original_aggregate(*args, **kwargs)

    community.judge_case = judge
    community.aggregate = aggregate

    def restore():
        for name, value in originals.items():
            setattr(community, name, value)
    return restore


@frozen_backend.public_protocol()
def run(run_dir, execute=False, max_requests=None):
    """Run fixed adapters through one shared, journaled endpoint transport."""
    directory = Path(run_dir).resolve()
    manifest = read_json(directory / 'manifest.json')
    if manifest.get('runner') != 'unified-roleplay-v2':
        raise ValueError('This entry requires v2; historical runs keep their frozen launchers and caches')
    config = manifest['config']
    validate(config)
    if _hash(config) != manifest['config_sha256']:
        raise ValueError('Root configuration changed; prepare a new run')
    if not execute:
        return {'dry_run': True, 'benchmarks': config['benchmarks'], 'run_dir': str(directory),
                'limits': config.get('concurrency', {})}
    roles = [config[k] for k in ('actor', 'judge', 'simulator') if k in config]
    endpoints = {k: _hash(benchmark_api.endpoint(config[k])[0])
                 for k in ('actor', 'judge', 'simulator') if k in config}
    if manifest.get('endpoints', endpoints) != endpoints:
        raise ValueError('Endpoint changed; prepare a separate run')
    manifest['endpoints'] = endpoints
    if max_requests is not None:
        if type(max_requests) is not int or max_requests < 1:
            raise ValueError('max_requests must be positive')
        config = _copy(config)
        config['max_requests_per_run'] = max_requests
    for name, metadata in manifest['children'].items():
        if name not in PUBLIC:
            frozen_backend.verify(directory / metadata['run_dir'], metadata)
        else:
            child = read_json(directory / metadata['run_dir'] / 'manifest.json')
            if _hash(child['config']) != metadata['config_sha256']:
                raise ValueError('Child configuration changed')
    results = {}
    with endpoint_ownership(roles), ExitStack() as cleanup:
        lock = directory / '.run.lock'
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            raise ValueError('Run is locked; confirm its supervisor stopped before removing .run.lock') from None
        os.close(fd)
        cleanup.callback(lock.unlink, missing_ok=True)
        store = DerivedStore(limit=config.get('storage', {}).get('checkpoint_limit', 256),
                             workers=config.get('storage', {}).get('checkpoint_workers', 4))
        cleanup.callback(store.close)
        ledger = UsageLedger(directory, config)
        cleanup.callback(ledger.close)
        limiter = EndpointLimiter(config.get('concurrency', {}).get('endpoints', DEFAULT_LIMITS))
        sender = JournalSender(config, limiter, ledger, directory, store)
        cleanup.callback(_patch_async_public(store))
        cleanup.enter_context(scoped_clients(sender, directory))
        port = cleanup.enter_context(coser_server(sender))
        manifest.update(status='running', started_at=time.time())
        write_json(directory / 'manifest.json', manifest)
        try:
            def one(name):
                child = directory / manifest['children'][name]['run_dir']
                if name == 'characterbench':
                    result = frozen_backend.character(config, child, sender, store)
                elif name == 'coser':
                    result = frozen_backend.coser(config, child, port, sender, store)
                else:
                    return community.run(child, 'evaluate', True, config['max_requests_per_run'],
                                         sender=sender, allow_external=True)
                expected = result.get('processable', result.get('expected'))
                return {**result, 'status': 'completed' if result['completed'] == expected else 'partial'}
            with ThreadPoolExecutor(max_workers=len(config['benchmarks'])) as pool:
                futures = {pool.submit(one, name): name for name in config['benchmarks']}
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        results[name] = future.result()
                    except Exception as exc:
                        results[name] = {'status': 'failed', 'error_type': type(exc).__name__}
                    store.write(directory / 'progress.json', {'results': results, 'transport': sender.snapshot()})
            # Completion is published only after both asynchronous writers ACK.
            store.flush()
            ledger.flush()
            sender.check()
            usage = usage_report(directory)
            write_json(directory / 'usage-summary.json', usage)
            manifest['usage'] = usage
            manifest['status'] = 'completed' if all(r.get('status', '').startswith('completed')
                                                   for r in results.values()) else 'partial'
        except BaseException as exc:
            manifest.update(status='failed', error_type=type(exc).__name__)
            raise
        finally:
            manifest.update(results=results, transport=sender.snapshot(), finished_at=time.time())
            write_json(directory / 'manifest.json', manifest)
        return {'status': manifest['status'], 'results': results, **sender.snapshot(), 'usage': manifest['usage'],
                'usage_ledger': ledger.snapshot()}
