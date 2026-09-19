"""Transport/robustness adapter; upstream GCA prompts and scoring stay unchanged."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import requests

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / 'vendor' / 'CoSER'
ACTOR = os.environ.get('COSER_ACTOR', 'gemma4-roleplay-epoch2')
ACTOR_PROVIDER = os.environ.get('COSER_ACTOR_PROVIDER', 'local')
ACTOR_API_MODEL = os.environ.get('COSER_ACTOR_API_MODEL', ACTOR)
ACTOR_REASONING_EFFORT = os.environ.get('COSER_ACTOR_REASONING_EFFORT')
ACTOR_TEMPERATURE = (float(os.environ['COSER_ACTOR_TEMPERATURE'])
                     if 'COSER_ACTOR_TEMPERATURE' in os.environ else None)
if ACTOR_TEMPERATURE is not None and (not math.isfinite(ACTOR_TEMPERATURE) or ACTOR_TEMPERATURE < 0):
    raise ValueError('COSER_ACTOR_TEMPERATURE must be finite and nonnegative')
if ACTOR_PROVIDER not in ('local', 'deepinfra'):
    raise ValueError('COSER_ACTOR_PROVIDER must be local or deepinfra')
HELPER = 'zai-org/GLM-5.3-Flash'
DIMENSIONS = ['Storyline Consistency', 'Anthropomorphism', 'Character Fidelity', 'Storyline Quality']


def request_settings(model, messages, max_tokens, nth_generation):
    """An API actor retains actor limits; only helpers get truncation recovery."""
    is_actor = model == ACTOR
    remote = not is_actor or ACTOR_PROVIDER == 'deepinfra'
    payload = {'model': ACTOR_API_MODEL if is_actor else model,
               'messages': messages, 'max_tokens': max_tokens or 8196,
               'temperature': 0 if nth_generation == 0 else 1, 'stream': False}
    if not is_actor:
        payload['max_tokens'] = max(2048, payload['max_tokens'])
    if is_actor and ACTOR_TEMPERATURE is not None:
        payload['temperature'] = ACTOR_TEMPERATURE
    if is_actor and ACTOR_REASONING_EFFORT is not None:
        payload['reasoning_effort'] = ACTOR_REASONING_EFFORT
    return payload, remote, not is_actor


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def run_case(index, run_dir):
    work = Path(run_dir).resolve() / f'case-{index:04d}'
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    dump(work / 'config.json', {'api_key': 'unused', 'base_url': 'http://127.0.0.1:8000/v1'})
    dataset = json.loads((VENDOR / 'data/test/test_set.json').read_text())
    dump(work / 'case.json', [dataset[index]])
    sys.path.insert(0, str(VENDOR / 'gca_evaluation'))
    import utils
    # Upstream accepts one provider; route roles independently without exposing keys.
    session = requests.Session()
    cache_dir = work / 'request-cache'
    cache_dir.mkdir(exist_ok=True)
    counters = {'calls': 0, 'cached': 0, 'errors': 0, 'cost_usd': 0.0}

    def request(model, messages, max_tokens=None, nth_generation=0, **unused):
        if isinstance(messages, str):
            messages = [{'role': 'user', 'content': messages}]
        merged = []
        for message in copy.deepcopy(messages):
            if message['role'] == 'user' and merged and merged[-1]['role'] == 'user':
                merged[-1]['content'] += message['content']
            else:
                merged.append(message)
        payload, remote, is_helper = request_settings(model, merged, max_tokens, nth_generation)
        # Helper thinking settings stay at the previously tested provider default.
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        cached = cache_dir / (digest + '.json')
        if cached.exists():
            counters['cached'] += 1
            return json.loads(cached.read_text())['content']
        url = ('https://api.deepinfra.com/v1/openai' if remote else 'http://127.0.0.1:8000/v1') + '/chat/completions'
        headers = {'Authorization': 'Bearer ' + os.environ['DEEPINFRA_API_KEY']} if remote else {}
        for attempt in range(5):
            started = time.time()
            record = {'time': started, 'model': payload['model'], 'request_hash': digest, 'attempt': attempt}
            try:
                response = session.post(url, json=payload, headers=headers, timeout=(15, 240))
                record['http_status'] = response.status_code
                if response.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f'transient HTTP {response.status_code}')
                response.raise_for_status()
                body = response.json()
                choice = body['choices'][0]
                content = choice['message'].get('content')
                usage = body.get('usage') or {}
                record.update(usage=usage, finish_reason=choice.get('finish_reason'))
                # Preserve original content/usage, but do not save hidden reasoning.
                dump(work / 'responses' / f'{digest}-{attempt}.json',
                     {'request': payload, 'content': content, 'usage': usage,
                      'finish_reason': choice.get('finish_reason'), 'model': body.get('model')})
                counters['calls'] += 1
                counters['cost_usd'] += float(usage.get('estimated_cost') or 0)
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError('empty visible content')
                if is_helper and choice.get('finish_reason') == 'length':
                    payload['max_tokens'] = min(32768, payload['max_tokens'] * 2)
                    raise RuntimeError('truncated completion; do not score partial output')
                dump(cached, {'content': content})
                return content
            except Exception as exc:
                counters['errors'] += 1
                record['error'] = f'{type(exc).__name__}: {str(exc)[:300]}'
                if attempt == 4:
                    raise
                time.sleep(min(30, 2 ** (attempt + 1)))
            finally:
                record['seconds'] = time.time() - started
                with (work / 'requests.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                dump(work / 'progress.json', {**counters, 'updated': time.time(), 'pid': os.getpid()})

    def response_json(post_processing_funcs=None, **kwargs):
        funcs = [utils.extract_json] if post_processing_funcs is None else post_processing_funcs
        for attempt in range(6):
            value = request(**kwargs, nth_generation=attempt)
            try:
                for func in funcs:
                    value = func(value, **kwargs)
                if not value:
                    continue
                # Close upstream's missing-dimension/type loophole; no semantic rewriting.
                if funcs:
                    prompt = kwargs['messages'][0]['content']
                    expected = [d for d in DIMENSIONS if f'in terms of {d}' in prompt]
                    for d in expected:
                        flaws = value[d]['flaws']
                        if not isinstance(flaws, list):
                            raise ValueError('flaws must be a list')
                        for flaw in flaws:
                            # General rubric 1-5; repetition criterion explicitly permits up to 10.
                            if type(flaw.get('severity')) is not int or not 1 <= flaw['severity'] <= 10:
                                raise ValueError('severity must be an integer in 1..10')
                return value
            except (KeyError, TypeError, ValueError, AssertionError):
                continue
        raise RuntimeError('exhausted structured-output attempts')

    utils.get_response = request
    utils.get_response_json = response_json
    utils.get_response_with_retry = lambda **kwargs: response_json([], **kwargs)
    import agent
    original_chat = agent.Agent.chat

    def checked_chat(self):
        text = original_chat(self)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f'{self.name}: simulation call failed; not an empty character turn')
        return text

    agent.Agent.chat = checked_chat
    sys.argv = ['main.py', '--test_file', 'case.json', '--actor_model', ACTOR,
                '--judge_model', HELPER, '--env_model', HELPER, '--nsp_model', HELPER]
    spec = importlib.util.spec_from_file_location('coser_upstream_main', VENDOR / 'gca_evaluation/main.py')
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)
    random.seed(42 + index)
    main.logger = utils.setup_logger(f'case-{index}', str(work / 'gca.log'), quiet=True)
    result = main.gca_simulation('case.json', ACTOR, HELPER, HELPER, None)
    if any(not m['content'].strip() for m in result[0]['simulation']):
        raise RuntimeError('simulation contains empty response')
    scores, cases = main.gca_judging('case.json', ACTOR, None, HELPER)
    dump(work / 'result.json', {'index': index, 'book': dataset[index]['book'], 'scores': scores, 'cases': cases})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=int, required=True)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run_case(args.case, args.run_dir)
