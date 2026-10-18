"""Rejudge frozen CoSER trajectories with explicit actor speaker labels.

No actor/NSP calls. Original judge requests are the prompt authority; only the
user transcript changes. Uses the shared journal, async writer and process ledger.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time
from urllib.error import HTTPError

from . import benchmark_api as api
from .endpoint_transport import JournalSender, endpoint_ownership, usage_report
from .sources import read_json, write_json
from .execution import DerivedStore, EndpointLimiter, UsageLedger

DIMENSIONS = ('Storyline Consistency', 'Anthropomorphism', 'Character Fidelity', 'Storyline Quality')


def labelled_transcript(simulation):
    parts = []
    for m in simulation:
        role, text = m['role'], m['content']
        if role == 'NSP':
            raise ValueError('Expected saved judge view without NSP')
        if role != 'Environment':
            prefix = r'^\s*(?:\*\*)?' + re.escape(role) + r'(?:\*\*)?\s*[:：](?:\*\*)?'
            if not re.match(prefix, text):
                text = role + ': ' + text
        parts.append(text.strip('\n'))
    return '\n\n'.join(parts)


def parse_critique(text, dimension):
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
            flaws = obj[dimension]['flaws']
            if not isinstance(flaws, list):
                continue
            if any(not isinstance(f, dict) or type(f.get('severity')) is not int
                   or not 1 <= f['severity'] <= 10 for f in flaws):
                continue
            return obj[dimension]
        except (ValueError, KeyError, TypeError):
            continue
    raise ValueError('Missing dimension or invalid flaw severities')


def score(critique, actor_turns):
    return max(0, min(100 - (sum(f['severity'] for f in critique['flaws']) - .3 * actor_turns) * 5, 100))


def prepare(source_dirs, directory):
    directory = Path(directory).resolve()
    if directory.exists():
        raise ValueError('Use a new rejudge directory')
    sources = [Path(s).resolve() for s in source_dirs]
    if len(sources) != 2 or sources[0] == sources[1]:
        raise ValueError('Supply two distinct source runs')
    configs = [read_json(s/'manifest.json')['config'] for s in sources]
    if configs[0]['judge'] != configs[1]['judge'] or configs[0].get('thinking', {}).get('remote', {}) != configs[1].get('thinking', {}).get('remote', {}):
        raise ValueError('Paired judge endpoints and thinking policies must match')
    rows = []
    for source in sources:
        child = source/'coser'
        statuses = [read_json(p) for p in (child/'outputs/full-public-v1/candidate/coser').glob('status-*.json')]
        rows.append({s['index']: (child/s['result']) for s in statuses if s.get('status') == 'completed'})
    common = sorted(set(rows[0]) & set(rows[1]))
    if not common:
        raise ValueError('No common successful scenes')
    jobs, provenance = [], []
    for group, (source, config, paths) in enumerate(zip(sources, configs, rows)):
        expected, group_jobs = {}, {}
        for index in common:
            raw = paths[index].read_bytes()
            result = json.loads(raw)
            if len(result['cases']) != 1:
                raise ValueError('Expected one scene per result')
            case = next(iter(result['cases'].values()))
            original = case['simulation_str']
            assert original == '\n\n'.join(m['content'].strip('\n') for m in case['simulation'])
            digest = hashlib.sha256(original.encode()).hexdigest()
            expected.setdefault(digest, []).append((index, case, result, hashlib.sha256(raw).hexdigest()))
        for path in (source/'transport-journal').glob('*/*/*.outcome.json'):
            saved = read_json(path)
            payload = saved.get('request', {})
            messages = payload.get('messages', [])
            if saved.get('model') != config['judge']['model'] or len(messages) != 2 or messages[0]['role'] != 'system' or messages[1]['role'] != 'user':
                continue
            digest = hashlib.sha256(messages[1]['content'].encode()).hexdigest()
            dimensions = [d for d in DIMENSIONS if 'in terms of ' + d in messages[0]['content']]
            if digest not in expected or len(dimensions) != 1:
                continue
            dim = dimensions[0]
            # Prefer the original successful accepted wire request, not retry variants.
            if saved.get('status') != 'response' or not JournalSender.acceptable(saved.get('response', {}), lambda t: parse_critique(t, dim)):
                continue
            for index, case, result, source_hash in expected[digest]:
                request = deepcopy(payload)
                request['messages'][1]['content'] = labelled_transcript(case['simulation'])
                job = {'group': group, 'index': index, 'dimension': dim, 'request': request,
                       'actor_turns': sum(m['role'] != 'Environment' for m in case['simulation']),
                       'book': result['book'], 'old_score': result['scores'][dim],
                       'source_result_sha256': source_hash, 'source_outcome': str(path),
                       'transcript_changed': request['messages'][1]['content'] != messages[1]['content']}
                key = (index, dim)
                if key in group_jobs and group_jobs[key]['request'] != request:
                    raise ValueError('Ambiguous accepted judge payload for scene/dimension')
                group_jobs[key] = job
        if len(group_jobs) != len(common)*4:
            raise ValueError(f'Group {group}: missing original accepted judge requests')
        jobs.extend(group_jobs.values())
        provenance.append({'source_run': str(source), 'model': config['actor']['model'],
                           'revision': config['actor']['revision'], 'source_completed': len(paths),
                           'changed_judgments': sum(j['transcript_changed'] for j in group_jobs.values())})
    # Interleave groups so neither monopolizes the shared pool.
    jobs.sort(key=lambda j:(j['index'], DIMENSIONS.index(j['dimension']), j['group']))
    config = deepcopy(configs[0])
    config['concurrency']['endpoints']['remote'] = 96
    data = {'config': config, 'sources': provenance, 'case_ids': common, 'jobs': jobs}
    directory.mkdir(parents=True)
    write_json(directory/'inputs.json', data)
    manifest = {'runner': 'coser-speaker-label-rejudge-v1', 'status': 'prepared',
                'inputs_sha256': hashlib.sha256((directory/'inputs.json').read_bytes()).hexdigest(),
                'judge_endpoint_hash': api.object_hash(api.endpoint(config['judge'])[0]),
                'runtime_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'sources': provenance, 'case_ids': common, 'expected_judgments': len(jobs),
                'concurrency': {'each_model': 48, 'total_remote': 96}, 'created_at': time.time()}
    write_json(directory/'manifest.json', manifest)
    return manifest


def aggregate(jobs, records):
    groups = {}
    for group in sorted({j['group'] for j in jobs}):
        selected = [j for j in jobs if j['group'] == group]
        indices = sorted({j['index'] for j in selected})
        complete = [i for i in indices if all(records.get((group, i, d), {}).get('status') == 'completed' for d in DIMENSIONS)]
        failed = [i for i in indices if any(records.get((group, i, d), {}).get('status') == 'failed' for d in DIMENSIONS)]
        scores = {d: statistics.mean(records[(group,i,d)]['score'] for i in complete) for d in DIMENSIONS} if complete else {}
        if scores:
            scores['avg'] = statistics.mean(scores.values())
        groups[str(group)] = {'expected':len(indices), 'completed':len(complete), 'failed':len(failed),
                              'missing':[i for i in indices if i not in complete], 'scores':scores}
    return groups


def run(directory, execute=False):
    directory = Path(directory).resolve()
    manifest = read_json(directory/'manifest.json')
    if manifest.get('runner') != 'coser-speaker-label-rejudge-v1':
        raise ValueError('Not a speaker-label rejudge run')
    for path, digest in [(directory/'inputs.json', manifest['inputs_sha256']), (Path(__file__), manifest['runtime_sha256'])]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Rejudge inputs/runtime changed')
    data = read_json(directory/'inputs.json'); config = data['config']; jobs = data['jobs']
    role = config['judge']
    if api.object_hash(api.endpoint(role)[0]) != manifest['judge_endpoint_hash']:
        raise ValueError('Judge endpoint changed')
    if not execute:
        return {'dry_run':True, 'expected_judgments':len(jobs), 'actor_calls':0}
    with endpoint_ownership([role]), ExitStack() as cleanup:
        lock = directory/'.run.lock'
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600);os.close(fd)
        cleanup.callback(lock.unlink, missing_ok=True)
        store = DerivedStore();cleanup.callback(store.close)
        ledger = UsageLedger(directory, config);cleanup.callback(ledger.close)
        limiter = EndpointLimiter({'remote':96,'local':1})
        sender = JournalSender(config, limiter, ledger, directory, store)
        per_group = EndpointLimiter({'0':48,'1':48})
        records = {}
        def one(job):
            group, index, dim = job['group'], job['index'], job['dimension']
            key = f'{group}-{index:04d}-{DIMENSIONS.index(dim)}'
            path = directory/f'model-{group}'/f'{index:04d}-{DIMENSIONS.index(dim)}.json'
            parser = lambda text: parse_critique(text, dim)
            with per_group.slot(str(group)):
                for attempt in range(config['max_attempts']):
                    try:
                        body = sender.call(role, job['request'], config['timeout_seconds'], 'rejudge:'+key, parser)
                        if not sender.acceptable(body, parser):
                            raise ValueError('Invalid or incomplete judgment')
                        critique = parser(body['choices'][0]['message']['content'])
                        row = {'status':'completed', 'index':index, 'dimension':dim, 'critique':critique,
                               'score':score(critique, job['actor_turns']), 'old_score':job['old_score']}
                        store.write(path, row)
                        return job, row
                    except HTTPError as exc:
                        error = f'HTTP {exc.code}'
                        retry = exc.code in (408,409,425,429) or exc.code >= 500
                        exc.close()
                    except api.CallFailed as exc:
                        error = str(exc); retry = False
                    except (ValueError, KeyError, TypeError, IndexError) as exc:
                        error = type(exc).__name__; retry = True
                    store.write(directory/'failures'/f'{key}-{time.time_ns()}.json', {'error':error, 'attempt':attempt+1})
                    if not retry:
                        break
                    if attempt+1 < config['max_attempts']:
                        time.sleep(min(30, 2**attempt))
                row = {'status':'failed','index':index,'dimension':dim,'error':error}
                store.write(path,row);return job,row
        manifest.update(status='running',started_at=time.time());write_json(directory/'manifest.json',manifest)
        try:
            with ThreadPoolExecutor(max_workers=96) as pool:
                for future in as_completed([pool.submit(one,j) for j in jobs]):
                    job,row=future.result();records[(job['group'],job['index'],job['dimension'])]=row
                    if len(records)%16 == 0 or len(records)==len(jobs):
                        store.write(directory/'progress.json',{'judgments_finished':len(records),'expected_judgments':len(jobs),
                                    'models':aggregate(jobs,records),'transport':sender.snapshot()})
            store.flush();ledger.flush();sender.check()
            result=aggregate(jobs,records)
            manifest['status']='completed' if all(v['completed']==v['expected'] for v in result.values()) else 'partial'
            write_json(directory/'summary.json',{'sources':data['sources'],'models':result})
            write_json(directory/'usage-summary.json',usage_report(directory))
        except BaseException:
            manifest['status']='failed';raise
        finally:
            manifest['finished_at']=time.time();write_json(directory/'manifest.json',manifest)
        return {'status':manifest['status'],'models':result,'transport':sender.snapshot()}
