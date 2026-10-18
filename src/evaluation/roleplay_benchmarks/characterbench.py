"""CharacterBench generation/judging with explicit client and storage dependencies.

Ported from frozen_public_v1.run_four_benchmark_full.run_character. Input
preparation, prompts, parsers, three passes and score normalization are unchanged.
The frozen copy remains the reference for historical runs.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import statistics
import uuid

from . import frozen_backend
from .endpoint_transport import JournalClient
from .sources import ROOT


def run(config, root, sender, store):
    protocol = frozen_backend.load(config)
    client = JournalClient(protocol.config(ROOT, 'candidate', 'minimax'), sender, sender.directory)
    out = Path(root) / 'outputs/full-public-v1'
    directory = out / 'candidate/characterbench'
    tasks = store.read(out / 'character-inputs.json')

    def one(task):
        work = directory / task['id']
        result_path = work / 'result.json'
        if result_path.exists() and store.read(result_path).get('status') == 'completed':
            return store.read(result_path)
        try:
            if sender.halt:
                raise RuntimeError(sender.halt)

            def parse_reply(text):
                text = text.strip()
                if text.startswith('```'):
                    text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
                obj = json.loads(text)
                if not isinstance(obj.get('response'), str) or not obj['response'].strip():
                    raise ValueError('No response string')
                return obj['response']

            _, reply = client.call(work / 'requests', 'actor', [
                {'role': 'user', 'content': task['actor_prompt']}], parse_reply)
            prompt = task['judge_template'].replace(protocol.SENTINEL, reply)
            prompt += '\n\nReturn only the integer rating specified by the rubric, with no explanation.'

            def parse_score(text):
                if not re.fullmatch(r'[1-5]', text.strip()) or not 1 <= int(text) <= task['scale_max']:
                    raise ValueError('Illegal rubric score')
                return int(text)

            _, score = client.call(work / 'requests', 'judge', [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}], parse_score)
            result = {key: task[key] for key in ('id', 'language', 'dimension', 'index', 'scale_max')}
            result.update(status='completed', score=score, normalized_5=score/task['scale_max']*5,
                          reply=reply, within_length_limit=len(reply) <= (15 if task['language'] == 'en' else 30))
        except Exception as exc:
            result = {'id': task['id'], 'status': 'failed', 'error': type(exc).__name__}
        if result_path.exists():
            old = store.read(result_path)
            if old.get('status') == 'failed':
                store.write(result_path.with_name('failure-' + uuid.uuid4().hex + '.json'), old)
        store.write(result_path, result)
        return result

    rows = []
    workers = config.get('concurrency', {}).get('benchmarks', {}).get('characterbench', 32)
    for attempt in range(3):
        rows = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(one, task) for task in tasks]):
                rows.append(future.result())
                if len(rows) % 20 == 0 or len(rows) == len(tasks):
                    store.write(directory / 'progress.json', {
                        'pass': attempt + 1, 'done': len(rows), 'total': len(tasks),
                        'status': dict(Counter(row['status'] for row in rows))})
        if sender.halt or all(row['status'] == 'completed' for row in rows):
            break
    scores = {}
    for language in ('en', 'zh'):
        dimensions = defaultdict(list)
        for row in rows:
            if row['status'] == 'completed' and row['language'] == language:
                dimensions[row['dimension']].append(row['normalized_5'])
        scores[language] = {
            'dimensions': {key: {'mean_normalized_5': statistics.mean(values), 'count': len(values)}
                           for key, values in dimensions.items()},
            'macro_normalized_5': statistics.mean(statistics.mean(values) for values in dimensions.values())
                                 if dimensions else None}
    summary = {
        'expected_source': 6500, 'processable': len(tasks),
        'completed': sum(row['status'] == 'completed' for row in rows),
        'source_invalid': store.read(out / 'character-input-audit.json')['invalid'], 'scores': scores}
    store.write(directory / 'summary.json', summary)
    return summary
