"""Resume the frozen suite, replacing only RMT with Claude 4.5 long dialogues.

The original supervisor and four engine files remain untouched, so the other
three suites can reuse their hash-checked caches. This extension is separately
hashed and recorded. --retire-pid is an explicit, one-time scheduler migration.
"""
import argparse
import collections
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time

RMT_DIR = 'rmtbench-claude45-long-v1'


def long_cases(config, base):
    from src.evaluation.roleplay_benchmarks.community_protocols import RMT_FIELDS, rmt_turns
    cases = []
    for language in config['languages']:
        rows = json.loads((base / f'rmtbench_claude45_{language}.json').read_text(encoding='utf-8'))
        for index, row in enumerate(rows):
            population = [(field, i) for field in RMT_FIELDS for i in range(len(row[field]))]
            if len(population) < 4:
                raise ValueError('Four distinct public blocks are required')
            seen = set()
            rng = random.Random(f"{config['seed']}:{language}:{row['idx']}")
            for repeat in range(config['repeats']):
                for _ in range(1000):
                    layout = rng.sample(population, 4)
                    turns = [turn for field, i in layout for turn in rmt_turns(row, field, i)]
                    signature = tuple(layout)
                    if len(turns) > 20 and signature not in seen:
                        break
                else:
                    raise ValueError('Cannot build a distinct >20-round four-block dialogue')
                seen.add(signature)
                cases.append({'id': f'{language}-{index:03}-{repeat:03}', 'language': language,
                              'source_id': row['idx'], 'name': row['name'], 'profile': row['description'],
                              'layout': layout, 'turns': turns})
    return cases


def patch_suite(full):
    from src.evaluation.roleplay_benchmarks import community
    validate_original, build_original = community.validate, community.build_cases
    config_original, character_original = full.config, full.run_character
    public_original = full.run_public

    def validate(config):
        compatible = copy.deepcopy(config)
        if compatible.get('layout') == 'paper_four_blocks':
            if compatible.get('version') != 'claude45' or compatible.get('repeats') != 2:
                raise ValueError('This extension requires Claude 4.5 and two trajectories per role')
            compatible['layout'] = 'mixed_four_blocks'
        validate_original(compatible)

    community.validate = validate
    community.build_cases = lambda c, b: long_cases(c, b) if c.get('layout') == 'paper_four_blocks' else build_original(c, b)

    def config(root, model, name):
        result = config_original(root, model, name)
        if name == 'rmtbench':
            result.update(version='claude45', layout='paper_four_blocks', repeats=2, seed=42,
                          protocol_label='rmtbench-claude45-long-glm-v1-NOT-OFFICIAL')
        return result

    def prepare(root, out):
        for model in full.ACTORS:
            directory = out / model / RMT_DIR
            path = out / model / (RMT_DIR + '-config.json')
            if not directory.exists():
                full.write(path, config(root, model, 'rmtbench'))
                community.prepare(path, directory)
                manifest = full.read(directory / 'manifest.json')
                manifest['protocol_choices'] = [p for p in manifest['protocol_choices'] if not p.startswith('mixed_four_blocks')]
                manifest['protocol_choices'].extend([
                    'Paper section 4.3: four blocks per trajectory and two trajectories per character; >20 user rounds.',
                    'Uniform sampling without replacement from all available public blocks; block order sampled, within-block order unchanged.',
                    'Fixed local seed 42; exact original author block manifest is unavailable. Duplicate ordered layouts rejected.',
                    'User requested Claude 4.5 data, not paper Claude 3.5. GLM judge retained by agreement; not original Qwen judge.',
                    'SEC/Poison blocks are not public and are excluded. No reference assistant replies enter actor history.',
                ])
                manifest['runtime_extension'] = {'path': Path(__file__).name,
                    'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
                full.write(directory / 'manifest.json', manifest)
        a = full.read(out / 'deepseek' / RMT_DIR / 'cases.json')
        b = full.read(out / 'gemma' / RMT_DIR / 'cases.json')
        if a != b:
            raise ValueError('The two candidate models must use identical case lists')
        lengths = [len(c['turns']) for c in a]
        info = {'rmtbench': full.read(out / 'deepseek' / RMT_DIR / 'manifest.json')['inventory'],
                'characterbench': full.read(out / 'character-input-audit.json'),
                'coser': {'scenarios_per_model': 200},
                'minimax': full.read(out / 'deepseek' / 'minimax' / 'manifest.json')['inventory'],
                'rmt_run_directory': RMT_DIR, 'rounds': {'min': min(lengths), 'max': max(lengths), 'mean': sum(lengths)/len(lengths)},
                'case_sha256': hashlib.sha256(json.dumps(a, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
        full.write(out / 'inventory-rmt45-long.json', info)
        return info

    def character(root, out, model, broker):
        summary = out / model / 'characterbench' / 'summary.json'
        # Do not silently add another three retry passes to an already finished suite.
        return full.read(summary) if summary.exists() else character_original(root, out, model, broker)

    full.config, full.prepare, full.run_character = config, prepare, character
    full.run_public = lambda out, model, name, broker: public_original(out, model, RMT_DIR if name == 'rmtbench' else name, broker)


def retire(full, root, pid):
    """Only retire the verified scheduler and its own CoSER workers; no GPU server."""
    out = root / 'outputs/full-public-v1'
    command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    expected = ('run_four_benchmark_full.py', 'run_rmt45_long_migration.py')
    if not any(name.encode() in command or str(root / name).encode() in command for name in expected):
        raise RuntimeError('Refusing to signal an unexpected scheduler')
    if Path(f'/proc/{pid}/cwd').resolve() != root or int((out/'supervisor.lock').read_text()) != pid:
        raise RuntimeError('Scheduler directory/lock mismatch')
    archive = out / ('migration-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    archive.mkdir()
    for name in ('supervisor.json', 'broker-status.json', 'inventory.json'):
        shutil.copy2(out/name, archive/name)
    workers = []
    # Freeze the parent to prevent spawning a new child while enumerating its children.
    os.kill(pid, signal.SIGSTOP)
    try:
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                status = (proc/'status').read_text()
                parent = int(next(x.split()[1] for x in status.splitlines() if x.startswith('PPid:')))
                args = (proc/'cmdline').read_bytes().split(b'\0')
                if parent == pid:
                    if b'--coser-worker' not in args or str(root/'run_four_benchmark_full.py').encode() not in args:
                        raise RuntimeError('Unexpected child; refusing migration')
                    workers.append(int(proc.name))
            except FileNotFoundError:
                continue
        for child in workers:
            try:
                os.kill(child, signal.SIGTERM)
            except ProcessLookupError:
                pass
        os.kill(pid, signal.SIGTERM)
    finally:
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
    def alive(p):
        try:
            return (Path(f'/proc/{p}/stat').read_text().split(') ', 1)[1].split()[0]) != 'Z'
        except FileNotFoundError:
            return False
    for _ in range(100):
        if not any(alive(p) for p in [pid, *workers]):
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Process still alive; preserve locks and inspect manually')
    # The terminated scheduler may have had billed calls whose replies were not
    # yet ledgered. Reserve the full configured concurrency, not a stale snapshot.
    with (out/'broker-requests.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({'event': 'migration_unsettled_reservation', 'unknown_reserve': 6.4,
                                 'reason': 'up to 64 interrupted remote calls; conservative reserve, not billed cost'})+'\n')
    for model in full.ACTORS:
        old = out/model/'rmtbench'
        full.write(old/'superseded.json', {'status': 'superseded', 'replacement': '../'+RMT_DIR,
                                          'reason': 'User requested Claude 4.5 long-dialogue protocol'})
    locks = [out/'supervisor.lock', *out.glob('*/*/.run.lock')]
    for path in locks:
        if path.exists():
            target = archive / str(path.relative_to(out)).replace('/', '__')
            path.rename(target)
    full.write(archive/'retirement.json', {'scheduler': pid, 'workers': workers, 'finished': time.time(),
                                         'api_reserve_added': 6.4, 'gpu_service_untouched': True})
    print('Retired old scheduler; preserved outputs and archived locks:', archive, flush=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--retire-pid', type=int)
    parser.add_argument('--retire-only', action='store_true', help='Preserve caches and stop the verified scheduler without resuming')
    args, rest = parser.parse_known_args()
    if args.retire_only and not args.retire_pid:
        parser.error('--retire-only requires --retire-pid')
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('frozen_full', root/'run_four_benchmark_full.py')
    full = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(full)
    patch_suite(full)
    # Prepare and validate all new inputs before stopping any running worker.
    info = full.prepare(root, root/'outputs/full-public-v1')
    print(json.dumps({k:v for k,v in info.items() if k!='characterbench'}), flush=True)
    if args.retire_pid:
        retire(full, root, args.retire_pid)
    if args.retire_only:
        return
    sys.argv = [sys.argv[0], '--root', str(root), *rest]
    full.main()


if __name__ == '__main__':
    main()
