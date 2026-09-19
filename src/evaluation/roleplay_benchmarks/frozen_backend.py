"""Configuration adapter for the already-used Gemma/GLM public protocol.

The hash-locked evaluator and CoSER adapter are shipped in the repository.
There is no dependency on a migration directory or a model deployment path.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import uuid
import statistics

from .sources import ROOT, read_json, registry, write_json
from scripts.evaluation.recovery_queue_guard import atomic_json, seed_coser_cache

FROZEN = ROOT / 'scripts/evaluation/frozen_public_v1'
HASHES = {'run_four_benchmark_full.py': 'f4956dc77a7f524483f070cf82f4962f18003e56f1fa8fb321d6820059b6c77c',
          'coser_adapter.py': '61d166c2e66dc7e4dbb4dfd37bb0de54079409d63459c809a1c7e8ea25cca830',
          'run_rmt45_long_migration.py': 'fa1cc6cdd12db35fa09c1ddeba5051b087dac9c21dbf1102c4db8e6287db9926'}


@contextmanager
def public_protocol():
    from . import community
    path = FROZEN / 'run_rmt45_long_migration.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[path.name]:
        raise ValueError('Frozen RMT extension changed')
    spec = importlib.util.spec_from_file_location('rmt45_protocol', path)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    validate, build = community.validate, community.build_cases
    def compatible(config):
        if config.get('layout') == 'paper_four_blocks':
            if config.get('version') != 'claude45' or config.get('repeats') != 2:
                raise ValueError('Frozen long RMT uses claude45 and two repetitions')
            return validate({**config, 'layout': 'mixed_four_blocks'})
        return validate(config)
    community.validate = compatible
    def cases(config, base):
        if config.get('layout') != 'paper_four_blocks':
            return build(config, base)
        rows = extension.long_cases(config, base)
        return rows[:config['limit_cases']] if config.get('limit_cases') else rows
    community.build_cases = cases
    try:
        yield
    finally:
        community.validate, community.build_cases = validate, build


def load(config):
    for name, expected in HASHES.items():
        if hashlib.sha256((FROZEN / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen protocol changed: ' + name)
    spec = importlib.util.spec_from_file_location('frozen_public_' + uuid.uuid4().hex,
                                                 FROZEN / 'run_four_benchmark_full.py')
    full = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(full)
    full.ACTORS = {'candidate': config['actor']['model']}
    full.JUDGE = config['judge']['model']
    original = full.config

    def runtime_config(root, model, name):
        result = original(ROOT, model, name)
        for kind, role in result['roles'].items():
            endpoint = config['actor'] if kind == 'actor' else config['judge']
            role.update({k: endpoint[k] for k in ('model', 'revision', 'base_url_env', 'api_key_env')})
            # Preserve the proven 0.7 actor / 0 helper Char/CoSER settings.
            role['sampling'] = {'temperature': .7 if kind == 'actor' else 0, 'max_tokens': 8192}
        result.update(max_attempts=config['max_attempts'], timeout_seconds=config['timeout_seconds'],
                      max_requests_per_run=config['max_requests_per_run'])
        return result

    full.config = runtime_config
    full.ThreadPoolExecutor = lambda max_workers=32, **kw: ThreadPoolExecutor(
        max_workers=config.get('concurrency', {}).get('benchmarks', {}).get('characterbench', 32), **kw)
    return full


def verify_vendor(name):
    spec = registry()[name]
    source = ROOT / spec['path']
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != spec['revision']:
        raise ValueError('Vendor revision changed: ' + name)
    # Upstream creates logs and __pycache__; verify tracked source, never copy
    # those unrelated artifacts into the new run.
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', 'HEAD', '--'],
                   check=True, stdout=subprocess.DEVNULL)
    return source, revision


def prepare(config, directory, benchmark):
    full = load(config)
    root = Path(directory)
    out = root / 'outputs/full-public-v1'
    out.mkdir(parents=True, exist_ok=True)
    source, revision = verify_vendor(benchmark)
    options = config.get('benchmark_options', {}).get(benchmark, {})
    if benchmark == 'characterbench':
        tasks, invalid = full.prepare_character(ROOT, out)
        selected = options.get('case_ids')
        if selected is not None:
            by_id = {row['id']: row for row in tasks}
            if not selected or len(set(selected)) != len(selected) or set(selected) - set(by_id):
                raise ValueError('Invalid CharacterBench case_ids')
            tasks = [by_id[key] for key in selected]
        elif options.get('limit_cases'):
            tasks = tasks[:options['limit_cases']]
        atomic_json(out / 'character-inputs.json', tasks)
        files = ['outputs/full-public-v1/character-inputs.json',
                 'outputs/full-public-v1/character-input-audit.json']
        expected = len(tasks)
    else:
        target = root / 'coser-worker-source'
        target.mkdir(exist_ok=True)
        names = subprocess.check_output(['git', '-C', str(source), 'ls-files', '-z']).decode().split('\0')
        for name in filter(None, names):
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, path)
        # No-RAG evaluation never instantiates Embedding. Removing optional
        # RAG imports matches the server's dependency-free worker copy without
        # modifying character, simulator, judge prompts or metrics.
        agent = target / 'gca_evaluation/agent.py'
        text = agent.read_text(encoding="utf-8")
        lines = [line for line in text.splitlines(True) if not line.startswith(('from langchain.', 'from langchain_community.'))]
        agent.write_text('Embeddings = object\n' + ''.join(lines), encoding='utf-8')
        adapters = root / 'adapters'
        adapters.mkdir(exist_ok=True)
        shutil.copyfile(FROZEN / 'coser_adapter.py', adapters / 'coser_adapter.py')
        total = len(read_json(target / 'data/test/test_set.json'))
        selected = options.get('case_ids', list(range(min(total, options.get('limit_cases') or total))))
        if not selected or len(set(selected)) != len(selected) or any(type(i) is not int or not 0 <= i < total for i in selected):
            raise ValueError('Invalid CoSER case_ids')
        atomic_json(root / 'coser-cases.json', selected)
        files = ['coser-cases.json', 'adapters/coser_adapter.py'] + [
            str(p.relative_to(root)).replace('\\', '/') for p in target.rglob('*') if p.is_file()]
        expected = len(selected)
    return {'runner': 'frozen-gemma-glm-v1', 'expected': expected, 'vendor_revision': revision,
            'frozen_sha256': HASHES, 'files': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in files}}


def verify(directory, metadata):
    for name, expected in metadata['files'].items():
        if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Prepared benchmark input/source changed: ' + name)


def character(config, root, sender, store):
    full = load(config)
    out = root / 'outputs/full-public-v1'
    full.read = store.read
    def write(path, value):
        if Path(path).name == 'result.json' and Path(path).exists():
            old = store.read(path)
            if old.get('status') == 'failed':
                store.write(Path(path).with_name('failure-' + uuid.uuid4().hex + '.json'), old)
        store.write(path, value)
    full.write = write
    return full.run_character(root, out, 'candidate', sender)


def coser(config, root, port, sender, store):
    out = root / 'outputs/full-public-v1/candidate/coser'
    out.mkdir(parents=True, exist_ok=True)
    cases = read_json(root / 'coser-cases.json')

    def one(index):
        status = out / f'status-{index:04d}.json'
        old = store.read(status) if status.exists() else {}
        if old.get('status') == 'completed':
            return old
        previous = max([old.get('attempt', 0), *[int(p.parent.name.split('-')[1])
                       for p in out.glob(f'attempt-*/case-{index:04d}')]])
        for attempt in range(previous + 1, previous + 1 + config['max_attempts']):
            if sender.halt:
                break
            command = [sys.executable, str(ROOT / 'scripts/evaluation/run_public_evaluation.py'),
                       'coser-worker', '--run-dir', str(root), '--port', str(port),
                       '--case', str(index), '--attempt', str(attempt)]
            with (out / f'case-{index:04d}-attempt-{attempt}.log').open('a') as log:
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = proc.wait(timeout=14400)
                except subprocess.TimeoutExpired:
                    proc.terminate(); proc.wait(timeout=30); code = 124
            result_path = out / f'attempt-{attempt}/case-{index:04d}/result.json'
            result = {'index': index, 'attempt': attempt, 'status': 'failed', 'exit_code': code}
            if code == 0 and result_path.exists():
                result.update(status='completed', scores=read_json(result_path)['scores'],
                              result=str(result_path.relative_to(root)))
            store.write(out / 'status-history' / f'{index:04d}-attempt-{attempt}.json', result)
            store.write(status, result)
            store.flush(status)
            if result['status'] == 'completed':
                return result
        return store.read(status) if status.exists() else {'index': index, 'status': 'pending'}

    rows = []
    with ThreadPoolExecutor(max_workers=config.get('concurrency', {}).get('benchmarks', {}).get('coser', 32)) as pool:
        for future in as_completed([pool.submit(one, index) for index in cases]):
            rows.append(future.result())
            valid = [r for r in rows if r['status'] == 'completed']
            summary = {'expected': len(cases), 'completed': len(valid),
                'failed': sum(r['status'] == 'failed' for r in rows),
                'missing': sorted(set(cases) - {r['index'] for r in valid}),
                'scores': {k: statistics.mean(r['scores'][k] for r in valid) for k in valid[0]['scores']} if valid else {}}
            store.write(out / 'summary.json', summary)
    return summary


def worker(root, case, attempt, port):
    config = read_json(root / 'manifest.json')['config']
    full = load(config)
    import requests
    post = requests.Session.post
    def scoped(session, url, **kwargs):
        payload = kwargs['json']
        kwargs['headers'] = {**kwargs.get('headers', {}), 'X-Eval-Recovery-Scope':
            hashlib.sha256(json.dumps({'case': case, 'payload': payload}, sort_keys=True).encode()).hexdigest()}
        return post(session, url, **kwargs)
    requests.Session.post = scoped
    # Successful partial caches across scene attempts stay reusable.
    seed_coser_cache(root / 'outputs/full-public-v1/candidate/coser', case, attempt)
    os.environ.update(COSER_ACTOR='candidate-candidate', COSER_ACTOR_API_MODEL=config['actor']['model'],
                      COSER_ACTOR_PROVIDER='local', COSER_ACTOR_TEMPERATURE='0.7')
    sys.path.insert(0, str(root / 'adapters'))
    import coser_adapter
    coser_adapter.HELPER = config['judge']['model']
    full.coser_worker(SimpleNamespace(root=root, model='candidate', case=case, attempt=attempt, port=port))
