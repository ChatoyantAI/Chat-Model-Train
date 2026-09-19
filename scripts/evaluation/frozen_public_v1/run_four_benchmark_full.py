"""Frozen two-model full-public-data run. Linux supervisor; no deployment changes.

Requires the suite tree, fixed vendor/data snapshots and adapters/coser_adapter.py.
Only --execute calls paid APIs. Keys are entered without echo and never persisted.
"""
import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import getpass
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from urllib import request, error

ACTORS = {'deepseek': 'deepseek-ai/DeepSeek-V4-Flash', 'gemma': 'gemma4-base-it'}
JUDGE = 'zai-org/GLM-5.3-Flash'
SENTINEL = '___ACTUAL_CANDIDATE_REPLY_6d9b23___'
sys.dont_write_bytecode = True


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def model_guard(local_url):
    models = json.load(request.urlopen(local_url.rstrip('/') + '/models', timeout=10))['data']
    if not any(x['id'] == ACTORS['gemma'] and x.get('root', '').rstrip('/').endswith('models/base-it') for x in models):
        raise RuntimeError('Expected Gemma IT base, refusing a different deployment')


class Broker:
    """Shared concurrency and conservative fee guard across all suites/processes."""
    def __init__(self, out, key, remote_url, local_url, remote_slots=64, local_slots=4, budget=100):
        self.out, self.key = out, key
        self.remote_url, self.local_url = remote_url.rstrip('/'), local_url.rstrip('/')
        self.budget = budget
        self.lock = threading.Lock()
        self.slots = {True: threading.BoundedSemaphore(local_slots), False: threading.BoundedSemaphore(remote_slots)}
        self.active = Counter()
        self.completed = Counter()
        self.errors = Counter()
        self.cost = 0.0
        self.unknown_reserve = 0.0
        self.halt = None
        self.ledger = out / 'broker-requests.jsonl'
        if self.ledger.exists():
            for line in self.ledger.read_text(encoding='utf-8').splitlines():
                item = json.loads(line)
                self.cost += item.get('cost', 0)
                self.unknown_reserve += item.get('unknown_reserve', 0)

    def snapshot(self):
        with self.lock:
            return {'updated': time.time(), 'active': dict(self.active), 'completed': dict(self.completed),
                    'errors': dict(self.errors), 'reported_cost_usd': self.cost,
                    'unknown_cost_reserve_usd': self.unknown_reserve, 'budget_usd': self.budget, 'halt': self.halt}

    def send(self, payload):
        model = payload.get('model')
        if model not in {*ACTORS.values(), JUDGE}:
            return 400, {'error': 'Model not allowed'}, {}
        local = model == ACTORS['gemma']
        # Queue before opening the upstream connection. One GPU is not 64 GPUs.
        with self.slots[local]:
            with self.lock:
                if not local and self.cost + self.unknown_reserve + .1 > self.budget:
                    self.halt = 'budget_guard'
                if self.halt:
                    return 402, {'error': self.halt}, {}
                if not local and self.cost + self.unknown_reserve + (self.active['remote'] + 1) * .1 > self.budget:
                    return 429, {'error': 'inflight_budget_reservation'}, {'Retry-After': '10'}
                group = 'local' if local else 'remote'
                self.active[group] += 1
            started = time.time()
            entry = {'model': model, 'started': started}
            status, headers, charge = 502, {}, None
            try:
                if local:
                    model_guard(self.local_url)
                url = (self.local_url if local else self.remote_url) + '/chat/completions'
                auth = {} if local else {'Authorization': 'Bearer ' + self.key}
                req = request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode(),
                                      headers={'Content-Type': 'application/json', **auth})
                from src.evaluation.roleplay_benchmarks.benchmark_api import NoRedirect
                with request.build_opener(NoRedirect()).open(req, timeout=300) as response:
                    body = json.load(response)
                status = 200
                usage = body.get('usage') or {}
                charge = usage.get('cost', usage.get('estimated_cost'))
                entry['usage'] = {k: usage[k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens') if k in usage}
            except error.HTTPError as exc:
                status = exc.code
                headers = {'Retry-After': exc.headers.get('Retry-After', '10')}
                body = {'error': 'upstream_http_' + str(status)}
                exc.close()
                if status in (401, 403, 402):
                    with self.lock:
                        self.halt = 'upstream_auth_or_billing_' + str(status)
            except Exception as exc:
                body = {'error': type(exc).__name__}
                if local and isinstance(exc, RuntimeError):
                    with self.lock:
                        self.halt = 'local_model_identity_changed'
            finally:
                with self.lock:
                    self.active[group] -= 1
                    self.completed[model] += status == 200
                    self.errors[str(status)] += status != 200
                    entry.update(status=status, seconds=time.time() - started)
                    if not local:
                        if isinstance(charge, (int, float)) and not isinstance(charge, bool) and charge >= 0:
                            self.cost += charge
                            entry['cost'] = charge
                        elif status == 200 or status >= 500:
                            self.unknown_reserve += .1
                            entry['unknown_reserve'] = .1
                    with self.ledger.open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(entry) + '\n')
            return status, body, headers


def serve_broker(broker, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, body, headers=None):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            self.reply(200, broker.snapshot())

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                return self.reply(404, {'error': 'unknown_path'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 8 * 1024 * 1024:
                    raise ValueError('payload_size')
                payload = json.loads(self.rfile.read(size))
            except Exception:
                return self.reply(400, {'error': 'invalid_payload'})
            self.reply(*broker.send(payload))
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def config(root, model, name):
    c = read(root / f'configs/evaluation/roleplay_benchmarks/{name}.example.json')
    c.update(protocol_label=name + '-full-public-glm-v1-NOT-OFFICIAL', languages=['en', 'zh'],
             limit_cases=None, concurrency=32, max_attempts=4, timeout_seconds=1800,
             max_requests_per_run=300000)
    if name == 'rmtbench':
        c.update(layout='single_blocks', version='claude35', repeats=1, seed=2)
    for role_name, role in c['roles'].items():
        role.update(model=ACTORS[model] if role_name == 'actor' else JUDGE,
                    revision='frozen-local-base-it' if model == 'gemma' and role_name == 'actor' else 'provider-alias-20260913',
                    base_url_env='FULL_BROKER_URL', api_key_env='FULL_BROKER_KEY',
                    sampling={'temperature': 0 if role_name == 'judge' else .7, 'max_tokens': 8192})
    return c


def prepare_character(root, out):
    """Freeze all valid inputs and exact official judge templates before paying."""
    base = root / 'vendor/benchmarks/CharacterBench'
    tree = ast.parse((base / 'process.py').read_text(encoding='utf-8'))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('norm_session_en', 'norm_session_zh')]
    namespace = {}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), 'official-normalizers', 'exec'), namespace)
    tree = ast.parse((base / 'roleplay_prompt.py').read_text(encoding='utf-8'))
    templates = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)}
    sys.path.insert(0, str(base / 'construct_prompts'))
    builders, scales = {}, {}
    for lang in ('en', 'zh'):
        builders[lang] = importlib.import_module('process_wo_context_' + lang + '_all')
        info = importlib.import_module('data_info_' + lang).data_info
        scales[lang] = {}
        for dim, spec in info.items():
            pattern = r'(?m)^\s*(?:\(\d+\)\s*)?([1-5])\s+points?\s*:' if lang == 'en' else r'(?m)^\s*(?:\(\d+\)\s*)?([1-5])\s*分\s*[:：，,]'
            anchors = [int(v) for v in re.findall(pattern, spec['base_eval_prompt'])]
            if not anchors or sorted(set(anchors)) != list(range(1, max(anchors) + 1)):
                raise ValueError('Ambiguous rubric scale ' + lang + ':' + dim)
            scales[lang][dim] = max(anchors)
    tasks, invalid = [], []
    for source in sorted((base / 'eval_data/raw_data').glob('*.json')):
        dim = source.stem.removesuffix('_test')
        for index, original in enumerate(read(source)):
            for lang in ('en', 'zh'):
                identity = {'language': lang, 'dimension': dim, 'index': index, 'source': source.name}
                try:
                    row = copy.deepcopy(original)
                    session = namespace['norm_session_' + lang](row)
                    setting = session['dialogue_setting']
                    actor_prompt = templates['Role_Play_PROMPT_' + lang.upper()].format(
                        character_profile=setting['bot_profile'], user_profile=setting['user_profile'],
                        dialogue='\n'.join(f"{m['role']}: {m['content']}" for m in session['messages']))
                    if lang == 'en':
                        row['translation_en']['response_en'] = SENTINEL
                    else:
                        row['response_zh'] = SENTINEL
                    temp = out / 'character-source-temporary.json'
                    write(temp, [row])
                    prompt_rows, _ = builders[lang].process(str(temp), dim)
                    judge_prompt = prompt_rows[0]['instruction']
                    if judge_prompt.count(SENTINEL) != 1:
                        raise ValueError('Expected one response slot')
                    tasks.append({**identity, 'id': f'{lang}-{dim}-{index:04}', 'actor_prompt': actor_prompt,
                                  'judge_template': judge_prompt, 'scale_max': scales[lang][dim]})
                except (ValueError, KeyError, AssertionError, TypeError, IndexError) as exc:
                    invalid.append({**identity, 'error': type(exc).__name__})
    write(out / 'character-inputs.json', tasks)
    write(out / 'character-input-audit.json', {'expected': 6500, 'valid': len(tasks), 'invalid': invalid,
                                             'policy': 'Retain source-invalid records; no score, no zero imputation, no silent repair', 'scales': scales})
    return tasks, invalid


def prepare(root, out):
    from src.evaluation.roleplay_benchmarks import community
    for model in ACTORS:
        for name in ('rmtbench', 'minimax'):
            directory = out / model / name
            path = out / model / (name + '-config.json')
            if not directory.exists():
                write(path, config(root, model, name))
                community.prepare(path, directory)
    if not (out / 'character-inputs.json').exists():
        prepare_character(root, out)
    inventory = {'characterbench': read(out / 'character-input-audit.json'), 'coser': {'scenarios_per_model': 200}}
    for name in ('rmtbench', 'minimax'):
        inventory[name] = read(out / 'deepseek' / name / 'manifest.json')['inventory']
    write(out / 'inventory.json', inventory)
    return inventory


def run_character(root, out, model, broker):
    from src.evaluation.roleplay_benchmarks.benchmark_api import Client
    client = Client(config(root, model, 'minimax'))
    directory = out / model / 'characterbench'
    tasks = read(out / 'character-inputs.json')
    def one(task):
        work = directory / task['id']
        result_path = work / 'result.json'
        if result_path.exists() and read(result_path).get('status') == 'completed':
            return read(result_path)
        try:
            if broker.halt:
                raise RuntimeError(broker.halt)
            def parse_reply(text):
                text = text.strip()
                if text.startswith('```'):
                    text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
                obj = json.loads(text)
                if not isinstance(obj.get('response'), str) or not obj['response'].strip():
                    raise ValueError('No response string')
                return obj['response']
            _, reply = client.call(work / 'requests', 'actor', [{'role': 'user', 'content': task['actor_prompt']}], parse_reply)
            prompt = task['judge_template'].replace(SENTINEL, reply)
            prompt += '\n\nReturn only the integer rating specified by the rubric, with no explanation.'
            def parse_score(text):
                if not re.fullmatch(r'[1-5]', text.strip()) or not 1 <= int(text) <= task['scale_max']:
                    raise ValueError('Illegal rubric score')
                return int(text)
            _, score = client.call(work / 'requests', 'judge', [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': prompt}], parse_score)
            result = {k: task[k] for k in ('id', 'language', 'dimension', 'index', 'scale_max')}
            result.update(status='completed', score=score, normalized_5=score/task['scale_max']*5, reply=reply,
                          within_length_limit=len(reply) <= (15 if task['language'] == 'en' else 30))
        except Exception as exc:
            result = {'id': task['id'], 'status': 'failed', 'error': type(exc).__name__}
        write(result_path, result)
        return result
    rows = []
    for attempt in range(3):
        rows = []
        with ThreadPoolExecutor(max_workers=32) as pool:
            for future in as_completed([pool.submit(one, task) for task in tasks]):
                rows.append(future.result())
                if len(rows) % 20 == 0 or len(rows) == len(tasks):
                    write(directory / 'progress.json', {'pass': attempt+1, 'done': len(rows), 'total': len(tasks), 'status': dict(Counter(r['status'] for r in rows))})
        if broker.halt or all(r['status'] == 'completed' for r in rows):
            break
    scores = {}
    for lang in ('en', 'zh'):
        dims = defaultdict(list)
        for row in rows:
            if row['status'] == 'completed' and row['language'] == lang:
                dims[row['dimension']].append(row['normalized_5'])
        scores[lang] = {'dimensions': {k: {'mean_normalized_5': statistics.mean(v), 'count': len(v)} for k,v in dims.items()},
                        'macro_normalized_5': statistics.mean(statistics.mean(v) for v in dims.values()) if dims else None}
    summary = {'expected_source': 6500, 'processable': len(tasks), 'completed': sum(r['status']=='completed' for r in rows),
               'source_invalid': read(out / 'character-input-audit.json')['invalid'], 'scores': scores}
    write(directory / 'summary.json', summary)
    return summary


def run_public(out, model, name, broker):
    from src.evaluation.roleplay_benchmarks import community
    result = None
    for attempt in range(3):
        if broker.halt:
            break
        result = community.run(out / model / name, 'evaluate', True, 300000)
        if result['coverage']['complete']:
            break
    return result


def coser_worker(args):
    root, out = args.root, args.root / 'outputs/full-public-v1'
    os.environ.update(COSER_ACTOR='candidate-' + args.model, COSER_ACTOR_API_MODEL=ACTORS[args.model],
                      COSER_ACTOR_PROVIDER='local', COSER_ACTOR_TEMPERATURE='0.7', DEEPINFRA_API_KEY='loopback-unused')
    sys.path.insert(0, str(root / 'adapters'))
    import coser_adapter as adapter
    adapter.VENDOR = root / 'coser-worker-source'
    settings = adapter.request_settings
    def fixed_settings(*a, **kw):
        payload, remote, helper = settings(*a, **kw)
        payload.update(max_tokens=8192, temperature=0 if helper else .7)
        return payload, remote, helper
    adapter.request_settings = fixed_settings
    original = adapter.requests.Session.post
    def post(session, url, **kwargs):
        kwargs.update(headers={}, timeout=(15, 1800))
        response = original(session, f'http://127.0.0.1:{args.port}/v1/chat/completions', **kwargs)
        if response.ok:
            choice = response.json().get('choices', [{}])[0]
            if choice.get('finish_reason') not in ('stop', 'end_turn'):
                raise RuntimeError('Incomplete completion; retained by broker usage ledger')
        return response
    adapter.requests.Session.post = post
    adapter.run_case(args.case, out / args.model / 'coser' / f'attempt-{args.attempt}')


def run_coser(args, out, model, broker):
    directory = out / model / 'coser'
    def one(index):
        status_path = directory / f'status-{index:04}.json'
        if status_path.exists() and read(status_path).get('status') == 'completed':
            return read(status_path)
        previous = read(status_path).get('attempt', 0) if status_path.exists() else 0
        for attempt in range(previous+1, previous+4):
            if broker.halt:
                break
            command = [sys.executable, str(Path(__file__).resolve()), '--root', str(args.root), '--coser-worker',
                       '--model', model, '--case', str(index), '--attempt', str(attempt), '--port', str(args.port)]
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f'case-{index:04}-attempt-{attempt}.log').open('a') as log:
                proc = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
                try:
                    code = proc.wait(timeout=14400)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=30)
                    code = -1
            result_path = directory / f'attempt-{attempt}' / f'case-{index:04}' / 'result.json'
            if code == 0 and result_path.exists():
                result = {'index': index, 'status': 'completed', 'attempt': attempt, 'scores': read(result_path)['scores'],
                          'result': str(result_path.relative_to(out))}
                write(status_path, result)
                return result
            write(status_path, {'index': index, 'status': 'failed', 'attempt': attempt, 'exit_code': code})
        return read(status_path) if status_path.exists() else {'index': index, 'status': 'pending'}
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(one, i) for i in range(200)]):
            results.append(future.result())
            valid = [r for r in results if r['status'] == 'completed']
            write(directory / 'summary.json', {'expected': 200, 'finished': len(results), 'completed': len(valid),
                  'states': dict(Counter(r['status'] for r in results)),
                  'scores': {k: statistics.mean(r['scores'][k] for r in valid) for k in valid[0]['scores']} if valid else {}})
    return read(directory / 'summary.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--detach', action='store_true')
    ap.add_argument('--coser-worker', action='store_true')
    ap.add_argument('--model', choices=list(ACTORS))
    ap.add_argument('--case', type=int)
    ap.add_argument('--attempt', type=int, default=1)
    ap.add_argument('--port', type=int, default=18081)
    ap.add_argument('--budget', type=float, default=100)
    ap.add_argument('--remote-concurrency', type=int, default=64)
    ap.add_argument('--local-concurrency', type=int, default=4)
    ap.add_argument('--remote-url', default='https://api.deepinfra.com/v1/openai')
    ap.add_argument('--local-url', default='http://127.0.0.1:8000/v1')
    args = ap.parse_args()
    args.root = args.root.resolve()
    sys.path.insert(0, str(args.root))
    os.umask(0o077)
    if args.coser_worker:
        coser_worker(args)
        return
    out = args.root / 'outputs/full-public-v1'
    out.mkdir(parents=True, exist_ok=True)
    inventory = prepare(args.root, out)
    print(json.dumps({k:v for k,v in inventory.items() if k != 'characterbench'}, ensure_ascii=False), flush=True)
    print('Character inputs', inventory['characterbench']['valid'], 'invalid', len(inventory['characterbench']['invalid']), flush=True)
    if not args.execute:
        return
    model_guard(args.local_url)
    if (out / 'supervisor.lock').exists():
        raise RuntimeError('Supervisor lock exists; inspect existing process before resuming')
    key = os.environ.get('DEEPINFRA_API_KEY') or getpass.getpass('DeepInfra key (memory only): ')
    if args.detach:
        sys.stdout.flush()
        child = os.fork()
        if child:
            print('Launched supervisor PID', child, flush=True)
            return
        os.setsid()
        fd = os.open(out / 'supervisor.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.dup2(fd, 1); os.dup2(fd, 2); os.close(fd)
        fd = os.open('/dev/null', os.O_RDONLY); os.dup2(fd, 0); os.close(fd)
    fd = os.open(out / 'supervisor.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    broker = Broker(out, key, args.remote_url, args.local_url, args.remote_concurrency, args.local_concurrency, args.budget)
    server = serve_broker(broker, args.port)
    os.environ.update(FULL_BROKER_URL=f'http://127.0.0.1:{args.port}/v1', FULL_BROKER_KEY='loopback-unused')
    source = args.root / 'coser-worker-source'
    if not source.exists():
        shutil.copytree(args.root / 'vendor/benchmarks/CoSER', source, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.log'))
    state = {'pid': os.getpid(), 'started': time.time(), 'status': 'running', 'results': {},
             'models': ACTORS, 'judge': JUDGE, 'remote_concurrency': args.remote_concurrency,
             'local_concurrency': args.local_concurrency, 'api_budget_guard_usd': args.budget,
             'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(out / 'supervisor.json', state)
    def heartbeat():
        while state['status'] == 'running':
            write(out / 'broker-status.json', broker.snapshot())
            time.sleep(10)
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            jobs = {}
            for model in ACTORS:
                jobs[pool.submit(run_coser, args, out, model, broker)] = model + '/coser'
                jobs[pool.submit(run_character, args.root, out, model, broker)] = model + '/characterbench'
                for name in ('rmtbench', 'minimax'):
                    jobs[pool.submit(run_public, out, model, name, broker)] = model + '/' + name
            for task in as_completed(jobs):
                name = jobs[task]
                try:
                    state['results'][name] = task.result()
                except Exception as exc:
                    state['results'][name] = {'error': type(exc).__name__}
                write(out / 'supervisor.json', state)
                print(name, 'finished; inspect coverage', flush=True)
        state.update(status='stopped_budget_or_auth' if broker.halt else 'finished_inspect_coverage', finished=time.time())
    finally:
        write(out / 'supervisor.json', state)
        write(out / 'broker-final.json', broker.snapshot())
        server.shutdown()
        (out / 'supervisor.lock').unlink()


if __name__ == '__main__':
    main()
