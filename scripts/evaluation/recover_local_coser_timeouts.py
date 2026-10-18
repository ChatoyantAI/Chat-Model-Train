"""Explicit recovery of local CoSER timeouts, retaining all request evidence."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.roleplay_benchmarks import bootstrap, unified, benchmark_api
from src.evaluation.roleplay_benchmarks.endpoint_transport import JournalSender, endpoint_ownership
from src.evaluation.roleplay_benchmarks.storage import atomic_json


def recover(run_dir, timeout_seconds=1200, config_file=None, execute=False):
    root = Path(run_dir).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    config = manifest['config']
    if manifest.get('runner') != 'unified-roleplay-v2' or unified._hash(config) != manifest['config_sha256']:
        raise ValueError('Unsupported or changed manifest')
    if config['benchmarks'] != ['coser'] or config['actor']['kind'] != 'local':
        raise ValueError('Recovery is restricted to a local CoSER actor')
    actor = config['actor']
    if urlsplit(benchmark_api.endpoint(actor)[0]).hostname not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Local actor must use a loopback endpoint')
    if manifest['status'] not in ('partial', 'failed') or (root / '.run.lock').exists():
        raise ValueError('Evaluation must have stopped before recovery')
    if type(timeout_seconds) is not int or not config['timeout_seconds'] <= timeout_seconds <= 1500:
        raise ValueError('Timeout must not decrease or exceed the worker HTTP timeout margin (1500s)')
    child_path = root / 'coser/manifest.json'
    child = json.loads(child_path.read_text())
    if child['config'] != config:
        raise ValueError('Child configuration differs')
    if config_file and json.loads(Path(config_file).read_text()) != config:
        raise ValueError('External configuration differs')
    parser = lambda value: value
    if config.get('protocol') == 'her-coser-format-v1':
        from src.evaluation.roleplay_benchmarks.her_coser import valid_actor_content
        parser = valid_actor_content
    changes = []
    for state_file in sorted((root / 'transport-journal').glob('*/*/state.json')):
        state = json.loads(state_file.read_text())
        outcome = state_file.parent / (state['operation_id'] + '.outcome.json')
        if not outcome.exists():
            continue  # Unknown dispatches, including paid requests, stay quarantined.
        saved = json.loads(outcome.read_text())
        if not (saved.get('model') == actor['model'] and saved.get('revision') == actor['revision']
                and saved.get('status') == 'uncertain' and saved.get('error_type') == 'TimeoutError'):
            continue
        valid = []
        for previous in state_file.parent.glob('*.outcome.json'):
            body = json.loads(previous.read_text())
            if (body.get('status') == 'response'
                    and JournalSender.acceptable(body.get('response', {}), parser)
                    and not JournalSender.has_reasoning(body['response'])):
                valid.append(body)
        replay = max(valid, key=lambda value: value.get('started', 0)) if valid else None
        changes.append({'state': str(state_file.relative_to(root)),
                        'previous_operation': state['operation_id'],
                        'replay_operation': replay['operation_id'] if replay else None})
    completed = {}
    for path in (root / 'coser/outputs/full-public-v1/candidate/coser').glob('status-*.json'):
        if json.loads(path.read_text()).get('status') == 'completed':
            completed[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    plan = {'local_actor': actor['model'], 'timeout_before': config['timeout_seconds'],
            'timeout_after': timeout_seconds, 'completed_scenes_preserved': len(completed),
            'local_timeouts_released': sum(c['replay_operation'] is None for c in changes),
            'successful_responses_restored': sum(c['replay_operation'] is not None for c in changes),
            'changes': changes}
    if not execute:
        return plan
    roles = [config[k] for k in ('actor', 'judge', 'simulator') if k in config]
    with endpoint_ownership(roles):
        lock = root / '.run.lock'
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        try:
            # Check again under the same locks used by the evaluator.
            if json.loads((root / 'manifest.json').read_text()) != manifest:
                raise ValueError('Manifest changed while acquiring recovery locks')
            audit = root / 'recovery' / ('local-timeouts-' + str(int(time.time())) + '-' + uuid.uuid4().hex[:8])
            (audit / 'states').mkdir(parents=True)
            atomic_json(audit / 'plan.json', plan)
            atomic_json(audit / 'manifest-before.json', manifest)
            atomic_json(audit / 'child-manifest-before.json', child)
            atomic_json(audit / 'config-before.json', config)
            atomic_json(audit / 'completed-sha256.json', completed)
            new = deepcopy(config)
            new['timeout_seconds'] = timeout_seconds
            unified.validate(new)
            manifest.update(config=new, config_sha256=unified._hash(new))
            manifest.setdefault('runtime_change_history', []).append({
                'time': time.time(), 'reason': 'Explicit retry of nonbillable local timeouts',
                'timeout_before': config['timeout_seconds'], 'timeout_after': timeout_seconds,
                'evidence': str(audit.relative_to(root))})
            child['config'] = new
            atomic_json(child_path, child)
            if config_file:
                atomic_json(Path(config_file), new)
            atomic_json(root / 'manifest.json', manifest)
            for change in changes:
                state_file = root / change['state']
                state = json.loads(state_file.read_text())
                if state['operation_id'] != change['previous_operation']:
                    raise ValueError('Request state changed during recovery')
                # Archive the pointer; never alter/delete intents or outcomes.
                state_file.rename(audit / 'states' / (state_file.parent.name + '.json'))
                if change['replay_operation']:
                    atomic_json(state_file, {'operation_id': change['replay_operation']})
            assert all(hashlib.sha256((root / path).read_bytes()).hexdigest() == h
                       for path, h in completed.items())
            atomic_json(audit / 'applied.json', {'time': time.time(), **plan})
            return {**plan, 'audit': str(audit), 'status': 'ready_to_resume'}
        finally:
            lock.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--config')
    p.add_argument('--env-file')
    p.add_argument('--timeout-seconds', type=int, default=1200)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if args.env_file:
        bootstrap.load_environment(args.env_file)
    result = recover(args.run_dir, args.timeout_seconds, args.config, args.execute)
    print(json.dumps({k: v for k, v in result.items() if k != 'changes'}, indent=2))


if __name__ == '__main__':
    main()
