"""Small real-API transport probe; diagnostic prompts are not benchmark scores."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.evaluation.roleplay_benchmarks.execution import UsageLedger, DerivedStore, EndpointLimiter
from src.evaluation.roleplay_benchmarks.endpoint_transport import JournalSender, endpoint_ownership
from src.evaluation.roleplay_benchmarks.storage import atomic_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--env-file', required=True)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--concurrency', type=int, default=96)
    ap.add_argument('--requests', type=int, default=96)
    ap.add_argument('--max-tokens', type=int, default=8192)
    ap.add_argument('--role', choices=['actor','judge'], default='judge')
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    if not args.execute:
        print('No requests sent; add --execute')
        return
    if not 1 <= args.concurrency <= 128 or not 1 <= args.requests <= 256:
        raise ValueError('Probe limited to concurrency 1..128 and 1..256 requests')
    for line in Path(args.env_file).read_text(encoding="utf-8").splitlines():
        if line and not line.startswith('#'):
            key, value = line.split('=',1); os.environ.setdefault(key,value)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    role = config[args.role]
    config['max_requests_per_run'] = args.requests
    args.run_dir.mkdir(parents=True,exist_ok=True)
    identity = {'model': role['model'], 'revision': role['revision'], 'role':args.role,
                'concurrency':args.concurrency,'requests':args.requests,
                'thinking': config.get('thinking',{}), 'sampling': {'temperature':0,'max_tokens':args.max_tokens}}
    marker = args.run_dir/'probe.json'
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != identity:
        raise ValueError('Probe identity changed; use a new directory')
    atomic_json(marker,identity)
    with endpoint_ownership([role]), ExitStack() as cleanup:
        store=DerivedStore(); cleanup.callback(store.close)
        ledger=UsageLedger(args.run_dir,config); cleanup.callback(ledger.close)
        sender=JournalSender(config,EndpointLimiter({'local':args.concurrency,'remote':args.concurrency}),
                             ledger,args.run_dir,store)
        def one(i):
            payload={'model':role['model'],'stream':False,'temperature':0,'max_tokens':args.max_tokens,
                     'messages':[{'role':'user','content':f'Transport diagnostic {i}. Reply with exactly OK.'}]}
            start=time.monotonic()
            try:
                body=sender.call(role,payload,600,f'probe-{args.role}-{i}')
                choice=body['choices'][0]
                return {'id':i,'status':'ok' if choice['finish_reason'] in ('stop','end_turn') and choice['message'].get('content') else 'invalid',
                        'seconds':time.monotonic()-start,'usage':body.get('usage',{}),
                        'finish_reason':choice['finish_reason']}
            except Exception as exc:
                return {'id':i,'status':'failed','error_type':type(exc).__name__,'http_status':getattr(exc,'code',None),
                        'seconds':time.monotonic()-start}
        start=time.monotonic()
        with ThreadPoolExecutor(args.concurrency) as pool:
            rows=list(pool.map(one,range(args.requests)))
        ledger.flush()
        durations=sorted(r['seconds'] for r in rows)
        costs=[r.get('usage',{}).get('cost',r.get('usage',{}).get('estimated_cost')) for r in rows]
        report={**identity,**sender.snapshot(),'elapsed_seconds':time.monotonic()-start,
                'completed':sum(r['status']=='ok' for r in rows),'failed':sum(r['status']=='failed' for r in rows),
                'invalid':sum(r['status']=='invalid' for r in rows),
                'p50_seconds':statistics.median(durations),'p95_seconds':durations[int((len(durations)-1)*.95)],
                'provider_cost_usd':sum(c for c in costs if type(c) in (int,float)),
                'cost_unknown_count':sum(type(c) not in (int,float) for c in costs),
                'note':'Diagnostic transport only; provider usage in replayed responses is not new billing.', 'rows':rows}
        atomic_json(args.run_dir/f'report-{time.time_ns()}.json',report)
        print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False))
        if report['failed'] or report['invalid']:
            raise SystemExit(1)


if __name__=='__main__': main()
