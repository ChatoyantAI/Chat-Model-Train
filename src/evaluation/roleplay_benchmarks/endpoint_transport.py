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
from urllib.request import Request, build_opener
import uuid

from . import benchmark_api as api
from .storage import atomic_json


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
                self.checkpoints.check()
            self.ledger.ledger.check()
        except OSError:
            self.halt = 'background_writer_failed'
            raise api.RequestLimit(self.halt) from None

    @staticmethod
    def acceptable(body, parser):
        try:
            choice = body['choices'][0]
            content = choice['message']['content'] if 'message' in choice else choice['text']
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

    def cached_response(self, work, role, parser):
        state_path = work / 'state.json'
        if not state_path.exists():
            return None
        state = json.loads(state_path.read_text(encoding='utf-8'))
        outcome_path = work / (state['operation_id'] + '.outcome.json')
        if not outcome_path.exists():
            raise api.CallFailed('Interrupted request quarantined; billing outcome unknown')
        saved = json.loads(outcome_path.read_text(encoding='utf-8'))
        if saved['status'] == 'uncertain':
            raise api.CallFailed('Ambiguous transport failure quarantined; not re-dispatched')
        if saved['status'] == 'response' and self.acceptable(saved['response'], parser):
            if role['kind'] == 'local' and self.has_reasoning(saved['response']):
                raise api.CallFailed('Local response contains reasoning; not accepted')
            with self.lock:
                self.replays += 1
            return saved['response']
        return None

    def extended_actor_payload(self, role, outgoing, timeout):
        """Use vLLM's exact chat token count; never shorten benchmark inputs."""
        cap = role.get('sampling', {}).get('max_tokens', 8192)
        if not (self.config.get('protocol') in ('her-coser-format-v1', 'her-coser-official-v2')
                and role is self.config['actor'] and role['kind'] == 'local' and cap > 8192):
            return outgoing
        url, credential = api.endpoint(role)
        if not url.endswith('/v1/chat/completions'):
            raise ValueError('Extended actor token counting requires a vLLM /v1 endpoint')
        count_payload = {k: outgoing[k] for k in ('model', 'messages', 'prompt', 'chat_template_kwargs') if k in outgoing}
        if 'messages' in outgoing:
            count_payload['add_generation_prompt'] = True
        request = Request(url.removesuffix('/v1/chat/completions') + '/tokenize',
                          data=json.dumps(count_payload).encode(),
                          headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + credential})
        with build_opener(api.NoRedirect()).open(request, timeout=min(timeout, 60)) as response:
            counted = json.load(response)
        count = counted['count']
        if type(count) is not int or count < 0:
            raise ValueError('Invalid vLLM token count')
        remaining = self.config['actor_context_window'] - count - 1
        if remaining < 1:
            raise ValueError('Actor input exceeds context window; input retained without truncation')
        return {**outgoing, 'max_tokens': min(cap, remaining)}

    def call(self, incoming, payload, timeout, scope, parser=lambda x: x):
        role = self.role(incoming)
        outgoing = deepcopy(payload)
        if (self.config.get('actor_prompt_format') in ('her-official-v1', 'her-no-system-v1')
                and role is self.config['actor']):
            from .her_coser import actor_prompt_payload
            outgoing = actor_prompt_payload(outgoing, self.config['actor_prompt_format'])
        policies = self.config.get('thinking', {})
        outgoing.update(deepcopy(policies.get(role['model'], policies.get(role['kind'], {}))))
        if self.config.get('protocol') == 'her-coser-official-v2' and role is self.config['actor']:
            from .her_official import actor_payload
            outgoing = actor_payload(outgoing, self.config['her_system_thinking'])
        # Cache identities include actual wire options, revision and endpoint.
        key = identity({'scope': scope, 'payload': outgoing, 'revision': role['revision'],
                        'url': api.endpoint(role)[0]})
        with self.lock:
            key_lock = self.key_locks.setdefault(key, threading.Lock())
        with key_lock:
            work = self.directory / 'transport-journal' / key[:2] / key
            cached = self.cached_response(work, role, parser)
            if cached is not None:
                return cached
            # Old successful 8192-token requests remain reusable. Only uncached
            # or invalid actor generations use the explicitly increased budget.
            expanded = self.extended_actor_payload(role, outgoing, timeout)
            if expanded != outgoing:
                outgoing = expanded
                key = identity({'scope': scope, 'payload': outgoing, 'revision': role['revision'],
                                'url': api.endpoint(role)[0]})
                work = self.directory / 'transport-journal' / key[:2] / key
                cached = self.cached_response(work, role, parser)
                if cached is not None:
                    return cached
            state_path = work / 'state.json'
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


class JournalClient(api.Client):
    """Explicit request scoping; never replace Client methods process-wide."""

    def __init__(self, config, sender, directory, sleeper=time.sleep):
        super().__init__(config, sender=sender, sleeper=sleeper)
        self.journal = sender
        self.root = Path(directory).resolve()

    def call(self, work, role_name, messages, parser=lambda x: x):
        role = self.config['roles'][role_name]
        payload = {'model': role['model'], 'messages': messages, 'stream': False, **role['sampling']}
        self.journal.context.scope = identity({'directory': str(Path(work).resolve().relative_to(self.root)),
                                               'role': role_name, 'revision': role['revision'], 'payload': payload})
        self.journal.context.parser = parser
        try:
            return super().call(work, role_name, messages, parser)
        finally:
            del self.journal.context.scope
            del self.journal.context.parser


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
                her_actor = (sender.config.get('protocol') in ('her-coser-format-v1', 'her-coser-official-v2')
                             and payload['model'] == sender.config['actor']['model'])
                if her_actor:
                    from .her_coser import actor_response, valid_actor_content
                    strict = sender.config.get('actor_output_validation') == 'role-content-v2'
                    native = sender.config.get('actor_prompt_format') == 'coser-native-v1'
                    body = sender.call(role, payload, sender.config['timeout_seconds'], scope,
                                       parser=lambda text: valid_actor_content(text, strict=strict, native=native))
                    # Sender has already persisted the unmodified upstream response.
                    if sender.config.get('protocol') == 'her-coser-official-v2':
                        from .her_official import deliver_actor_response
                        body = deliver_actor_response(body, sender.config['her_system_thinking'])
                    else:
                        body = actor_response(body, strict=strict, native=native)
                else:
                    if sender.config.get('assessment'):
                        from .assessment import judge_parser, role_content
                        parser = role_content if payload['model'] == sender.config['actor']['model'] else judge_parser(payload)
                        body = sender.call(role, payload, sender.config['timeout_seconds'], scope, parser=parser)
                        # JournalSender validates replay candidates. Validate new
                        # replies too, before the worker's small success cache.
                        # Length finishes still reach the frozen token-limit
                        # recovery path; never turn truncation into a score.
                        choice = body['choices'][0]
                        if choice.get('finish_reason') != 'length':
                            parser(choice['message']['content'])
                    else:
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
