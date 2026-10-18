"""Configuration adapter for the already-used Gemma/GLM public protocol.

The hash-locked evaluator and CoSER adapter are shipped in the repository.
There is no dependency on a migration directory or a model deployment path.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from .storage import atomic_json

FROZEN = ROOT / 'scripts/evaluation/frozen_public_v1'
HASHES = {'run_four_benchmark_full.py': 'f4956dc77a7f524483f070cf82f4962f18003e56f1fa8fb321d6820059b6c77c',
          'coser_adapter.py': '61d166c2e66dc7e4dbb4dfd37bb0de54079409d63459c809a1c7e8ea25cca830',
          'run_rmt45_long_migration.py': 'fa1cc6cdd12db35fa09c1ddeba5051b087dac9c21dbf1102c4db8e6287db9926'}


def long_cases(config, base):
    """Load the byte-locked historical RMT case builder without global patches."""
    path = FROZEN / 'run_rmt45_long_migration.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[path.name]:
        raise ValueError('Frozen RMT extension changed')
    spec = importlib.util.spec_from_file_location('rmt45_protocol', path)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    rows = extension.long_cases(config, base)
    return rows[:config['limit_cases']] if config.get('limit_cases') else rows


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
        if config.get('protocol') == 'her-coser-format-v1':
            # HER evaluates role thoughts, while recipients cannot see them.
            # Patch only the isolated, hashed worker copy, never vendor/frozen code.
            from .her_coser import patch_judge_source
            main_path = target / 'gca_evaluation/main.py'
            main_text = patch_judge_source(main_path.read_text(encoding='utf-8'))
            main_path.write_text(main_text, encoding='utf-8')
            utils_path = target / 'gca_evaluation/utils.py'
            with utils_path.open('a', encoding='utf-8') as handle:
                handle.write('\n# HER recipient visibility, including multiline role thoughts.\n'
                             'from src.evaluation.roleplay_benchmarks.her_coser import '
                             'hide_inner_thoughts as remove_inner_thoughts\n')
        adapters = root / 'adapters'
        adapters.mkdir(exist_ok=True)
        shutil.copyfile(FROZEN / 'coser_adapter.py', adapters / 'coser_adapter.py')
        if config.get('protocol') == 'her-coser-official-v2':
            from .her_official import patch_main, patch_adapter
            main_path = target / 'gca_evaluation/main.py'
            main_path.write_text(patch_main(main_path.read_text(encoding='utf-8'),
                                          config['her_system_thinking']), encoding='utf-8')
            adapter_path = adapters / 'coser_adapter.py'
            adapter_path.write_text(patch_adapter(adapter_path.read_text(encoding='utf-8')), encoding='utf-8')
        if config.get('assessment'):
            from .assessment import patch_main, patch_agent
            main_path = target / 'gca_evaluation/main.py'
            main_path.write_text(patch_main(main_path.read_text(encoding='utf-8'),
                config['assessment']['coser_role_thoughts']), encoding='utf-8')
            agent.write_text(patch_agent(agent.read_text(encoding='utf-8')), encoding='utf-8')
            # Both native and HER recipients must hide multiline private thoughts.
            utils_path = target / 'gca_evaluation/utils.py'
            with utils_path.open('a', encoding='utf-8') as handle:
                handle.write('\nfrom src.evaluation.roleplay_benchmarks.her_coser import '
                             'hide_inner_thoughts as remove_inner_thoughts\n')
        total = len(read_json(target / 'data/test/test_set.json'))
        selected = options.get('case_ids', list(range(min(total, options.get('limit_cases') or total))))
        if not selected or len(set(selected)) != len(selected) or any(type(i) is not int or not 0 <= i < total for i in selected):
            raise ValueError('Invalid CoSER case_ids')
        atomic_json(root / 'coser-cases.json', selected)
        files = ['coser-cases.json', 'adapters/coser_adapter.py'] + [
            str(p.relative_to(root)).replace('\\', '/') for p in target.rglob('*') if p.is_file()]
        expected = len(selected)
    metadata = {'runner': config.get('protocol', 'frozen-gemma-glm-v1'), 'expected': expected, 'vendor_revision': revision,
            'frozen_sha256': HASHES, 'files': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in files}}
    if benchmark == 'characterbench':
        path = 'src/evaluation/roleplay_benchmarks/characterbench.py'
        metadata['runtime_sources'] = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}
    if config.get('protocol') in ('her-coser-format-v1', 'her-coser-official-v2'):
        paths = ['src/evaluation/roleplay_benchmarks/her_format.py',
                 'src/evaluation/roleplay_benchmarks/her_coser.py',
                 'src/evaluation/roleplay_benchmarks/endpoint_transport.py',
                 'src/evaluation/roleplay_benchmarks/frozen_backend.py']
        if config.get('protocol') == 'her-coser-official-v2':
            paths += ['src/evaluation/roleplay_benchmarks/her_official.py',
                      'src/evaluation/roleplay_benchmarks/benchmark_api.py']
            paths += [str(p.relative_to(ROOT)).replace('\\', '/') for p in
                      (ROOT / 'src/evaluation/roleplay_benchmarks/her_upstream').iterdir() if p.is_file()]
        metadata['runtime_sources'] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    if config.get('assessment'):
        from .assessment import contract
        metadata['assessment'] = contract(config)
        paths = ['src/evaluation/roleplay_benchmarks/assessment.py',
                 'src/evaluation/roleplay_benchmarks/her_coser.py',
                 'src/evaluation/roleplay_benchmarks/her_format.py',
                 'src/evaluation/roleplay_benchmarks/her_upstream/prompts.py']
        metadata.setdefault('runtime_sources', {}).update({p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths})
    return metadata


def verify(directory, metadata):
    for name, expected in metadata.get('runtime_sources', {}).items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Prepared HER adapter changed: ' + name)
    for name, expected in metadata['files'].items():
        if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Prepared benchmark input/source changed: ' + name)


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


def seed_coser_cache(base, case_index, attempt):
    """Copy first successful partial checkpoints for the SAME case, append-only."""
    base = Path(base)
    target = base / f"attempt-{attempt}" / f"case-{case_index:04d}" / "request-cache"
    target.mkdir(parents=True, exist_ok=True)
    imported = []
    for prior in sorted(base.glob("attempt-*"), key=lambda p: int(p.name.split("-")[1])):
        if int(prior.name.split("-")[1]) >= attempt:
            continue
        for source in (prior / f"case-{case_index:04d}" / "request-cache").glob("*.json"):
            dest = target / source.name
            if dest.exists():
                continue
            try:
                value = json.loads(source.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if not isinstance(value.get("content"), str) or not value["content"].strip():
                continue
            with dest.open("xb") as handle:
                handle.write(source.read_bytes())
            imported.append({"source": str(source), "destination": str(dest),
                             "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    atomic_json(target.parent / ("cache-import-" + uuid.uuid4().hex + ".json"), imported)
    return len(imported)
