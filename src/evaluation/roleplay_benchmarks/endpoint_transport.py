"""Endpoint-only transport; reuse the Gemma atomic writer and process ledger.

Only local scratch is used on the request path. An immutable intent precedes
dispatch, and an immutable outcome precedes delivery. Ambiguous interrupted
calls are quarantined. Derived JSON/ledger writing is independent of dispatch.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
import uuid

from . import benchmark_api as api
from scripts.evaluation.recovery_queue_guard import atomic_json


def identity(value):
    return api.object_hash(value)


@contextmanager
def endpoint_ownership(roles):
    """Prevent independent supervisors from silently multiplying API limits.

    All benchmarks in one supervisor share gates. A second supervisor using
    the same URL fails immediately; OS locks release on exit/crash. Lock files
    are always on host-local temporary storage, never the network run volume.
    """
    handles = []
    directory = Path(tempfile.gettempdir()) / 'roleplay-endpoint-locks'
    directory.mkdir(exist_ok=True)
    try:
        urls = sorted({api.endpoint(role)[0] for role in roles})
        for url in urls:
            handle = (directory / (identity(url) + '.lock')).open('a+b')
            handles.append(handle)
            try:
                if os.name == 'nt':
                    import msvcrt
                    if handle.tell() == 0:
                        handle.write(b'0'); handle.flush()
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise RuntimeError('Another evaluation owns this endpoint; run selected benchmarks in one supervisor') from None
        yield
    finally:
        for handle in reversed(handles):
            handle.close()


class JournalSender:
    def __init__(self, config, limiter, ledger, directory, checkpoints=None):
        self.config, self.limiter, self.ledger = config, limiter, ledger
        self.directory = Path(directory)
        self.checkpoints = checkpoints
        self.context = threading.local()
        self.lock = threading.Lock()
        self.key_locks = {}
        self.calls = self.replays = 0
        self.active, self.peak, self.errors = {}, {}, {}
        self.halt = None
        self.max_requests = config['max_requests_per_run']
        self.roles = [config[k] for k in ('actor', 'judge', 'simulator') if k in config]

    def role(self, incoming):
        matches = [r for r in self.roles if all(incoming.get(k) == r.get(k)
                   for k in ('model', 'base_url_env', 'api_key_env'))]
        if not matches:
            raise ValueError('Unregistered endpoint/model')
        return matches[0]

    def check(self):
        if self.halt:
            raise api.RequestLimit('Transport halted: ' + self.halt)
        try:
            if self.checkpoints:
                self.checkpoints.writer.check()
            self.ledger.ledger.check()
        except OSError:
            self.halt = 'background_writer_failed'
            raise api.RequestLimit(self.halt) from None

    @staticmethod
    def acceptable(body, parser):
        try:
            choice = body['choices'][0]
            content = choice['message']['content']
            if choice['finish_reason'] not in ('stop', 'end_turn') or not isinstance(content, str) or not content.strip():
                return False
            parser(content)
            return True
        except (ValueError, TypeError, KeyError, IndexError, AssertionError):
            return False

    def __call__(self, role, payload, timeout):
        if not hasattr(self.context, 'scope'):
            raise RuntimeError('A stable sample scope is required for paid requests')
        return self.call(role, payload, timeout, self.context.scope, self.context.parser)

    def call(self, incoming, payload, timeout, scope, parser=lambda x: x):
        role = self.role(incoming)
        outgoing = deepcopy(payload)
        policies = self.config.get('thinking', {})
        outgoing.update(deepcopy(policies.get(role['model'], policies.get(role['kind'], {}))))
        # Cache identities include actual wire options, revision and endpoint.
        key = identity({'scope': scope, 'payload': outgoing, 'revision': role['revision'],
                        'url': api.endpoint(role)[0]})
        with self.lock:
            key_lock = self.key_locks.setdefault(key, threading.Lock())
        with key_lock:
            work = self.directory / 'transport-journal' / key[:2] / key
            state_path = work / 'state.json'
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                outcome_path = work / (state['operation_id'] + '.outcome.json')
                if not outcome_path.exists():
                    raise api.CallFailed('Interrupted request quarantined; billing outcome unknown')
                saved = json.loads(outcome_path.read_text(encoding="utf-8"))
                if saved['status'] == 'uncertain':
                    raise api.CallFailed('Ambiguous transport failure quarantined; not re-dispatched')
                if saved['status'] == 'response' and self.acceptable(saved['response'], parser):
                    if role['kind'] == 'local' and self.has_reasoning(saved['response']):
                        raise api.CallFailed('Local response contains reasoning; not accepted')
                    with self.lock:
                        self.replays += 1
                    return saved['response']
            self.check()
            with self.limiter.slot(role['kind']):
                self.check()
                with self.lock:
                    if self.calls >= self.max_requests:
                        raise api.RequestLimit('Shared request limit reached')
                    self.calls += 1
                operation = uuid.uuid4().hex
                started = time.time()
                intent = {'operation_id': operation, 'request_hash': key, 'scope': scope,
                          'model': role['model'], 'revision': role['revision'],
                          'request': outgoing, 'started': started}
                try:
                    atomic_json(work / (operation + '.intent.json'), intent)
                    atomic_json(state_path, {'operation_id': operation})
                    with self.lock:
                        kind = role['kind']
                        self.active[kind] = self.active.get(kind, 0) + 1
                        self.peak[kind] = max(self.peak.get(kind, 0), self.active[kind])
                    error = None
                    try:
                        body = api.transport(role, outgoing, timeout)
                        outcome = {**intent, 'status': 'response', 'http_status': 200, 'response': body}
                    except HTTPError as exc:
                        # Preserve type/status and a body digest without recording credentials.
                        raw = exc.read()
                        outcome = {**intent, 'status': 'http_error', 'http_status': exc.code,
                                   'error_type': 'HTTPError', 'error_body_sha256': hashlib.sha256(raw).hexdigest()}
                        error = exc
                    except (URLError, TimeoutError, ConnectionError, OSError, ValueError) as exc:
                        outcome = {**intent, 'status': 'uncertain', 'error_type': type(exc).__name__}
                        error = api.CallFailed('Ambiguous upstream outcome; quarantined without retry')
                    finally:
                        with self.lock:
                            self.active[role['kind']] -= 1
                    outcome['seconds'] = time.time() - started
                    atomic_json(work / (operation + '.outcome.json'), outcome)
                    usage = outcome.get('response', {}).get('usage') or {}
                    charge = next((usage[k] for k in ('cost', 'estimated_cost')
                                   if type(usage.get(k)) in (int, float) and math.isfinite(usage[k]) and usage[k] >= 0), None)
                    self.ledger.ledger.append({'operation_id': operation, 'request_hash': key,
                        'model': role['model'], 'status': outcome.get('http_status', 599),
                        'usage': usage, 'cost': charge, 'error_type': outcome.get('error_type'),
                        'seconds': outcome['seconds']})
                except OSError:
                    self.halt = 'journal_or_accounting_failed'
                    raise api.RequestLimit(self.halt) from None
                if error:
                    with self.lock:
                        name = outcome.get('http_status', outcome['status'])
                        self.errors[str(name)] = self.errors.get(str(name), 0) + 1
                    raise error
                if role['kind'] == 'local' and self.has_reasoning(body):
                    self.halt = 'local_thinking_not_disabled'
                    raise api.RequestLimit(self.halt)
                return body

    @staticmethod
    def has_reasoning(body):
        message = (body.get('choices') or [{}])[0].get('message') or {}
        usage = body.get('usage') or {}
        return bool(message.get('reasoning_content') or message.get('reasoning') or
                    (usage.get('completion_tokens_details') or {}).get('reasoning_tokens'))

    def snapshot(self):
        with self.lock:
            return {'requests': self.calls, 'replays': self.replays, 'active': dict(self.active),
                    'peak': dict(self.peak), 'errors': dict(self.errors), 'halt': self.halt}


@contextmanager
def scoped_clients(sender, directory):
    """Install only for this single-supervisor process; restore on every exit."""
    previous_init, previous_call = api.Client.__init__, api.Client.call
    root = Path(directory).resolve()

    def init(self, config, sender_override=None, sleeper=time.sleep, **kwargs):
        explicit = kwargs.pop('sender', sender_override)
        previous_init(self, config, sender=explicit or sender, sleeper=sleeper, **kwargs)

    def call(self, work, role_name, messages, parser=lambda x: x):
        role = self.config['roles'][role_name]
        payload = {'model': role['model'], 'messages': messages, 'stream': False, **role['sampling']}
        sender.context.scope = identity({'directory': str(Path(work).resolve().relative_to(root)),
                                         'role': role_name, 'revision': role['revision'], 'payload': payload})
        sender.context.parser = parser
        try:
            return previous_call(self, work, role_name, messages, parser)
        finally:
            del sender.context.scope
            del sender.context.parser

    api.Client.__init__, api.Client.call = init, call
    try:
        yield
    finally:
        api.Client.__init__, api.Client.call = previous_init, previous_call


@contextmanager
def coser_server(sender):
    """Loopback HTTP boundary for scene subprocesses; shares the parent gates."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            status = 200
            try:
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                scope = self.headers.get('X-Eval-Recovery-Scope')
                if not scope:
                    raise ValueError('Missing sample scope')
                role = next(r for r in sender.roles if r['model'] == payload['model'])
                body = sender.call(role, payload, sender.config['timeout_seconds'], scope)
            except HTTPError as exc:
                status, body = exc.code, {'error': 'upstream_http_' + str(exc.code)}
            except Exception as exc:
                status, body = 503, {'error': type(exc).__name__}
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def usage_report(directory):
    """One end-of-run scan on local scratch; raw outcomes are authoritative.

    This reconstructs accounting even if the ledger process died before ACK.
    Replays do not create intents/outcomes and therefore cannot add a charge.
    """
    models, request_keys, seen = {}, {}, set()
    uncertain = []
    for intent_path in (Path(directory) / 'transport-journal').glob('*/*/*.intent.json'):
        intent = json.loads(intent_path.read_text(encoding="utf-8"))
        op = intent['operation_id']
        if op in seen: continue
        seen.add(op)
        request_keys[intent['request_hash']] = request_keys.get(intent['request_hash'], 0) + 1
        row = models.setdefault(intent['model'], {'requests':0,'http_errors':0,'uncertain':0,
                 'provider_cost_usd':0.0,'cost_unknown':0,'prompt_tokens':0,'completion_tokens':0})
        row['requests'] += 1
        path = intent_path.with_name(op + '.outcome.json')
        outcome = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {'status':'uncertain'}
        if outcome['status']=='uncertain':
            row['uncertain'] += 1; uncertain.append(intent['request_hash'])
        row['http_errors'] += outcome['status']=='http_error'
        usage = outcome.get('response',{}).get('usage') or {}
        charge = next((usage[k] for k in ('cost','estimated_cost') if type(usage.get(k)) in (int,float)
                       and math.isfinite(usage[k]) and usage[k]>=0),None)
        if charge is None: row['cost_unknown'] += 1
        else: row['provider_cost_usd'] += charge
        for key in ('prompt_tokens','completion_tokens'):
            if type(usage.get(key)) is int: row[key] += usage[key]
    return {'models':models,'requests':len(seen),'retries':sum(max(0,n-1) for n in request_keys.values()),
            'uncertain_request_keys':sorted(set(uncertain)),
            'cost_note':'Provider-reported usage only. Missing usage is unknown, not zero. Replays are excluded.'}
