"""Prepare and run the four benchmarks through shared endpoints and storage.

Read prepare() for the frozen input snapshots, then run() for dispatch and final
flush. Configuration, transport and storage each have their own module.
"""
from __future__ import annotations

from contextlib import ExitStack, nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import hashlib
import os
from pathlib import Path
import time

from . import benchmark_api, community, frozen_backend, characterbench
from .config import validate, DEFAULT_LIMITS, PUBLIC, _copy, public_config
from .execution import DerivedStore, UsageLedger, EndpointLimiter
from .endpoint_transport import JournalSender, JournalClient, endpoint_ownership, coser_server, usage_report
from .sources import ROOT, read_json, write_json

# Kept as an import alias for saved diagnostic tools; one hash implementation.
_hash = benchmark_api.object_hash


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
            child_value = public_config(config, benchmark)
            write_json(child_config, child_value)
            community.prepare(child_config, child, allow_external=True)
            if config.get('assessment'):
                detail = read_json(child / 'manifest.json')
                from . import assessment
                detail['assessment'] = assessment.contract(config)
                detail['assessment_sha256'] = hashlib.sha256(Path(assessment.__file__).read_bytes()).hexdigest()
                write_json(child / 'manifest.json', detail)
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
    if config.get('assessment'):
        from . import assessment
        manifest['assessment'] = assessment.contract(config)
        manifest['assessment_sha256'] = hashlib.sha256(Path(assessment.__file__).read_bytes()).hexdigest()
    write_json(directory / "manifest.json", manifest)
    return manifest


def run(run_dir, execute=False, max_requests=None, independent_endpoints=False):
    """Run fixed adapters through one shared, journaled endpoint transport."""
    directory = Path(run_dir).resolve()
    manifest = read_json(directory / 'manifest.json')
    if manifest.get('runner') != 'unified-roleplay-v2':
        raise ValueError('This entry requires v2; historical runs keep their frozen launchers and caches')
    config = manifest['config']
    validate(config)
    if _hash(config) != manifest['config_sha256']:
        raise ValueError('Root configuration changed; prepare a new run')
    if config.get('assessment'):
        from . import assessment
        if hashlib.sha256(Path(assessment.__file__).read_bytes()).hexdigest() != manifest.get('assessment_sha256'):
            raise ValueError('Reviewed assessment implementation changed; use the pinned code or a new run')
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
            if child.get('assessment') != manifest.get('assessment') or child.get('assessment_sha256') != manifest.get('assessment_sha256'):
                raise ValueError('Child assessment changed')
    results = {}
    ownership = nullcontext() if independent_endpoints else endpoint_ownership(roles)
    with ownership, ExitStack() as cleanup:
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
        port = cleanup.enter_context(coser_server(sender))
        manifest.update(status='running', started_at=time.time(),
                        endpoint_ownership_mode='independent' if independent_endpoints else 'exclusive')
        write_json(directory / 'manifest.json', manifest)
        try:
            def one(name):
                child = directory / manifest['children'][name]['run_dir']
                if name == 'characterbench':
                    result = characterbench.run(config, child, sender, store)
                elif name == 'coser':
                    result = frozen_backend.coser(config, child, port, sender, store)
                else:
                    return community.run(child, 'evaluate', True, config['max_requests_per_run'],
                                         sender=sender, allow_external=True, store=store,
                                         client_factory=partial(JournalClient, sender=sender, directory=directory))
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
