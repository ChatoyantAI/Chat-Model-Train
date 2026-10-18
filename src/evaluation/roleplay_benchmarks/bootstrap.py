"""Portable installation and preflight; never generates paid completions."""
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import urllib.request

from . import sources


def load_environment(path):
    for number, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, sep, value = line.partition('=')
        if not sep or not re.fullmatch(r'[A-Z_][A-Z0-9_]*', key):
            raise ValueError(f'Invalid env file entry at line {number}')
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            os.environ.setdefault(key, value)


def selected(names):
    specs = sources.registry()
    if not names or set(names) - set(specs) - {'all'}:
        raise ValueError('Choose characterbench, coser, minimax, rmtbench, or all')
    return list(specs) if 'all' in names else list(dict.fromkeys(names))


def verify_source(name):
    # Upstream preprocessing may create ignored/untracked logs. Only pinned
    # tracked source is executable input, so those logs do not invalidate it.
    spec = sources.registry()[name]
    if spec['kind'] == 'git':
        from .frozen_backend import verify_vendor
        verify_vendor(name)
    else:
        sources.verify_one(name, spec)


def prepare_assets(names, cache_from=None):
    specs = sources.registry()
    for name in selected(names):
        spec = specs[name]
        print('benchmark assets: ' + name, flush=True)
        if spec['kind'] == 'git':
            source = sources.ROOT / spec['path']
            if not (source / '.git').exists():
                subprocess.run(['git', 'submodule', 'update', '--init', '--', spec['path']],
                               cwd=sources.ROOT, check=True)
            sources.restore_bundled(name, spec)
            verify_source(name)
            continue
        if cache_from:
            cached = Path(cache_from).resolve() / spec['path']
            destination = sources.ROOT / spec['path']
            # Copy only explicitly pinned dataset files and their receipt.
            # Existing target files are verified, never overwritten blindly.
            for relative in [*spec['files'], 'snapshot.json']:
                source = sources.inside(cached, relative)
                target = sources.inside(destination, relative)
                if source.is_file() and not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
        result = sources.sync_one(name, spec)
        if result['status'] != 'complete':
            raise RuntimeError(f'{name}: source download incomplete; supply an authorized HF_TOKEN if required')
        verify_source(name)
    if 'rmtbench' in selected(names):
        relative = Path('data/external/roleplay_benchmarks/rmtbench-paper/2507.20352v2.html')
        target = sources.ROOT / relative
        if cache_from and not target.exists() and (Path(cache_from) / relative).is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(cache_from) / relative, target)
        from .paper_templates import load_templates
        load_templates()


def ensure_nltk():
    import nltk
    resource = 'tokenizers/punkt_tab/english/'
    try:
        nltk.data.find(resource)
        return
    except LookupError:
        pass
    previous = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(60)
        nltk.download('punkt_tab', quiet=True, raise_on_error=True)
        nltk.data.find(resource)
    finally:
        socket.setdefaulttimeout(previous)


def install(venv, names, cache_from=None):
    if sys.version_info < (3, 11):
        raise ValueError('Python 3.11+ is required')
    if not (sources.ROOT / '.git').exists():
        raise ValueError('Use git clone (not a ZIP); pinned submodule revisions are required')
    if not shutil.which('git'):
        raise ValueError('Git must be installed')
    selected(names)
    venv = Path(venv).expanduser().resolve()
    python = venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
    # Probe before pip so a noexec network volume produces a clear error.
    try:
        subprocess.run([str(python), '-c', 'import sys; assert sys.version_info >= (3,11)'], check=True)
    except OSError as exc:
        raise ValueError('Virtualenv is not executable; choose a host-local --venv directory') from exc
    subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(sources.ROOT /
        'configs/evaluation/roleplay_benchmarks/requirements-unified.lock.txt')], check=True)
    prepare_assets(names, cache_from)
    if 'coser' in selected(names):
        subprocess.run([str(python), '-c',
            "from src.evaluation.roleplay_benchmarks.bootstrap import ensure_nltk; ensure_nltk()"],
            cwd=sources.ROOT, check=True, timeout=180)
    print(json.dumps({'status': 'ready', 'python': str(python), 'benchmarks': selected(names)}, indent=2))


def initialize(path, actor_model, actor_kind, benchmarks, actor_revision=None,
               judge_model='zai-org/GLM-5.3-Flash', her_system_thinking=None,
               assessment_mode='reviewed', coser_role_thoughts='include'):
    target = Path(path)
    if target.exists():
        raise ValueError('Configuration already exists; edit it or choose a new path')
    config = sources.read_json(sources.ROOT / 'configs/evaluation/roleplay_benchmarks/unified.example.json')
    names = selected(benchmarks)
    config['benchmarks'] = names
    config['actor'].update(model=actor_model, kind=actor_kind,
                           revision=actor_revision or 'endpoint-alias:' + actor_model)
    if assessment_mode == 'reviewed':
        config['assessment'] = {'version': '20261018', 'coser_role_thoughts': coser_role_thoughts}
    elif assessment_mode == 'legacy':
        config.pop('assessment', None)
        config['protocol_label'] = 'public-roleplay-unified-v1'
    else:
        raise ValueError('Choose reviewed or legacy assessment')
    for key in ('judge', 'simulator'):
        config[key].update(model=judge_model, revision='endpoint-alias:' + judge_model)
    if 'minimax' not in names:
        config.pop('simulator')
    for container in (config['benchmark_options'], config['concurrency']['benchmarks']):
        for name in list(container):
            if name not in names:
                del container[name]
    from .unified import validate
    if her_system_thinking is not None:
        config.update(protocol='her-coser-official-v2',
                      protocol_label='her-coser-official-v2-' + her_system_thinking,
                      her_system_thinking=her_system_thinking,
                      actor_output_validation='role-content-v2')
    validate(config)
    sources.write_json(target, config)
    return {'status': 'prepared-config', 'config': str(target.resolve()), 'benchmarks': names}


def check(config, check_api=False):
    from . import benchmark_api, frozen_backend, unified
    unified.validate(config)
    if 'REPLACE_WITH' in json.dumps(config):
        raise ValueError('Fill in model IDs/revisions, or use the init command')
    frozen_backend.load(config)
    for name in config['benchmarks']:
        verify_source(name)
    if 'rmtbench' in config['benchmarks']:
        from .paper_templates import load_templates
        load_templates()
    if 'coser' in config['benchmarks']:
        for package in ('requests', 'nltk', 'rouge', 'openai', 'jsonlines', 'tiktoken'):
            importlib.metadata.version(package)
        import nltk
        nltk.data.find('tokenizers/punkt_tab/english/')
    endpoints, catalogs = {}, {}
    for key in ('actor', 'judge', 'simulator'):
        if key not in config:
            continue
        role = config[key]
        url, credential = benchmark_api.endpoint(role)
        endpoints[key] = {'model': role['model'], 'kind': role['kind'], 'configured': True}
        if check_api:
            # GET /models only; never send a chat completion from preflight.
            catalog_key = (url, role['api_key_env'])
            if catalog_key not in catalogs:
                request = urllib.request.Request(url.removesuffix('/chat/completions') + '/models',
                    headers={'Authorization': 'Bearer ' + credential})
                try:
                    with urllib.request.build_opener(benchmark_api.NoRedirect()).open(request, timeout=30) as response:
                        catalogs[catalog_key] = {row['id'] for row in json.load(response)['data']}
                except Exception as exc:
                    raise ValueError(f'{key}: model catalog check failed ({type(exc).__name__}); check endpoint/key') from None
            if role['model'] not in catalogs[catalog_key]:
                raise ValueError(f'{key}: configured model is absent from GET /models')
            endpoints[key]['model_available'] = True
    return {'status': 'prepared-check', 'benchmarks': config['benchmarks'], 'endpoints': endpoints,
            'api_completions_sent': 0}
