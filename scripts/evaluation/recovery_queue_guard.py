"""Runtime-only transport repair for the frozen Gemma benchmark run.

The frozen prompts, sampling settings, request hashes and scoring parsers stay
unchanged. A small FIFO admission queue rejects abandoned callers before an
upstream request, and successful responses are journaled before HTTP delivery.
"""
from collections import deque
from contextlib import contextmanager
from concurrent.futures import Future
import copy
import hashlib
import inspect
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import select
import socket
import sys
import threading
import time
from urllib import request, error
import uuid

from src.evaluation.roleplay_benchmarks.storage import atomic_text, atomic_json
from src.evaluation.roleplay_benchmarks.frozen_backend import seed_coser_cache


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


class NotAdmitted(Exception):
    pass


class FairGate:
    def __init__(self, capacity):
        self.capacity, self.active = capacity, 0
        self.waiters = deque()
        self.condition = threading.Condition()

    @contextmanager
    def enter(self, deadline, cancelled=lambda: False):
        ticket = object()
        acquired = False
        with self.condition:
            self.waiters.append(ticket)
            try:
                while True:
                    if cancelled():
                        raise NotAdmitted("client_disconnected_before_dispatch")
                    if time.monotonic() >= deadline:
                        raise NotAdmitted("queue_deadline_exceeded")
                    if self.waiters[0] is ticket and self.active < self.capacity:
                        self.waiters.popleft()
                        self.active += 1
                        acquired = True
                        self.condition.notify_all()
                        break
                    self.condition.wait(min(.1, max(0, deadline - time.monotonic())))
            finally:
                if not acquired:
                    self.waiters.remove(ticket)
                    self.condition.notify_all()
        try:
            yield
        finally:
            with self.condition:
                self.active -= 1
                self.condition.notify_all()


def normalize_payload(payload):
    result = copy.deepcopy(payload)
    if result.get("model") == "zai-org/GLM-5.3-Flash":
        # Restore the frozen DeepInfra request. Both enable_thinking=False and
        # reasoning_effort=none triggered visible reasoning in the live A/B.
        # The old recovery launcher may already have injected the former.
        # The user authorized the historical provider-default judge policy.
        # Preserve its reasoning separately; Gemma still requires thinking off.
        kwargs = dict(result.get("chat_template_kwargs") or {})
        if kwargs.get("enable_thinking") is False:
            kwargs.pop("enable_thinking")
        if kwargs:
            result["chat_template_kwargs"] = kwargs
        else:
            result.pop("chat_template_kwargs", None)
        return result
    result["chat_template_kwargs"] = {
        **(result.get("chat_template_kwargs") or {}), "enable_thinking": False}
    return result


@lru_cache(maxsize=4096)
def resolved_directory(directory):
    return str(Path(directory).resolve())


def call_scope(directory, fingerprint):
    return digest({"directory": resolved_directory(str(directory)), "fingerprint": fingerprint})


def transport_key(payload, scope=None):
    normalized = normalize_payload(payload)
    return digest({"scope": scope, "payload": normalized}) if scope else digest(normalized)


def journal_path(journal, kind, name, sharded=False):
    return Path(journal) / (kind + '-v2') / name[:2] / name if sharded else Path(journal) / kind / name


def install(full, upstream_timeout=600, queue_timeout=30, checkpoints=None, budget_guard=True):
    original_broker = full.Broker
    full.EVAL_BUDGET_GUARD_ENABLED = budget_guard

    class GuardedBroker(original_broker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            remote = args[4] if len(args) > 4 else kwargs.get("remote_slots", 6)
            local = args[5] if len(args) > 5 else kwargs.get("local_slots", 16)
            self.gates = {False: FairGate(remote), True: FairGate(local)}
            self.flights, self.flight_lock = {}, threading.Lock()
            self.context = threading.local()
            self.journal = self.out / "transport-repair-v1"
            self.deduplicated = self.rejected = 0
            self.last_failure = None
            self.quarantined = set()
            # The frozen constructor already read the original ledger. Keep it
            # immutable; bounded segments avoid syncing its 43 MB for each call.
            for segment in sorted((self.journal / "ledgers").glob("*.jsonl")):
                for line in segment.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    self.cost += row.get("cost", 0)
                    self.unknown_reserve += row.get("unknown_reserve", 0)
                    if row.get('quarantine'):
                        self.quarantined.add(row['request_hash'])
            self.ledger = self.journal / "ledgers" / (uuid.uuid4().hex + ".jsonl")
            self.ledger_text = ""

        def persist_entry(self, entry):
            # Called under the accounting lock. A retried write cannot append
            # duplicate charges after an ambiguous fsync failure.
            if len(self.ledger_text.encode("utf-8")) >= 262144:
                self.ledger = self.journal / "ledgers" / (uuid.uuid4().hex + ".jsonl")
                self.ledger_text = ""
            text = self.ledger_text + json.dumps(entry) + "\n"
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            try:
                with self.ledger.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError:
                # An append may already have landed. Repair using the exact
                # bounded segment, never append the same charge a second time.
                atomic_text(self.ledger, text)
            self.ledger_text = text

        def storage_failure(self, exc, phase):
            self.last_failure = {"time": time.time(), "phase": phase,
                                 "type": type(exc).__name__, "errno": getattr(exc, "errno", None),
                                 "path": str(getattr(exc, "filename", None)),
                                 "message": os.strerror(exc.errno) if isinstance(exc, OSError) and exc.errno else type(exc).__name__}
            print(json.dumps({"storage_failure": self.last_failure}), file=sys.stderr, flush=True)

        def snapshot(self):
            if checkpoints is not None and checkpoints.error is not None:
                self.halt = 'checkpoint_publication_failed'
            value = super().snapshot()
            value.update(queue_depth={"local": len(self.gates[True].waiters),
                                      "remote": len(self.gates[False].waiters)},
                         merged_inflight=self.deduplicated, rejected_before_dispatch=self.rejected,
                         status="halted" if self.halt else "running", last_failure=self.last_failure,
                         current_ledger=str(self.ledger),
                         upstream_timeout_seconds=upstream_timeout, queue_timeout_seconds=queue_timeout)
            value['quarantined_requests'] = len(self.quarantined)
            value['budget_guard_enabled'] = budget_guard
            value['producer_limits'] = getattr(full, 'EVAL_PRODUCER_LIMITS', {})
            if not budget_guard:
                value['budget_usd'] = None
            if checkpoints is not None:
                value['pending_checkpoints'] = len(checkpoints.pending)
            return value

        def send(self, original_payload):
            payload = normalize_payload(original_payload)
            model = payload.get("model")
            if model not in {full.ACTORS["gemma"], full.JUDGE}:
                return 400, {"error": "Only the Gemma run is authorized"}, {}
            key = transport_key(payload, getattr(self.context, "scope", None))
            if key in self.quarantined:
                return 409, {'error': 'interrupted_request_pending_reconciliation'}, {'X-Eval-Quarantined': '1'}
            cancelled = getattr(self.context, "cancelled", lambda: False)
            with self.flight_lock:
                future = self.flights.get(key)
                owner = future is None
                if owner:
                    future = self.flights[key] = Future()
                else:
                    self.deduplicated += 1
            if not owner:
                limit = time.monotonic() + queue_timeout + upstream_timeout + 60
                while not future.done():
                    if cancelled() or time.monotonic() > limit:
                        return 429, {"error": "duplicate_wait_cancelled"}, {"Retry-After": "10"}
                    time.sleep(.1)
                return future.result()
            try:
                try:
                    with self.gates[model == full.ACTORS["gemma"]].enter(
                            time.monotonic() + queue_timeout, cancelled):
                        result = self.dispatch(payload, key)
                except NotAdmitted as exc:
                    self.rejected += 1
                    result = 429, {"error": str(exc)}, {"Retry-After": "10"}
                except Exception as exc:
                    # A persistence failure must halt dispatch, not silently bill
                    # requests whose responses/accounting cannot be retained.
                    with self.lock:
                        self.halt = "transport_persistence_failure_" + type(exc).__name__
                        self.errors["503"] += 1
                    self.storage_failure(exc, "accounting_or_dispatch")
                    result = 503, {"error": self.halt}, {}
                if self.halt:
                    result = result[0], result[1], {**result[2], "X-Eval-Halted": self.halt}
                future.set_result(result)
                return result
            finally:
                with self.flight_lock:
                    self.flights.pop(key, None)

        def dispatch(self, payload, key):
            model = payload["model"]
            local = model == full.ACTORS["gemma"]
            group = "local" if local else "remote"
            with self.lock:
                if checkpoints is not None and checkpoints.error is not None:
                    self.halt = 'checkpoint_publication_failed'
                if budget_guard and not local and self.cost + self.unknown_reserve + .1 > self.budget:
                    self.halt = "budget_guard"
                if self.halt:
                    return (402 if self.halt == "budget_guard" else 503), {"error": self.halt}, {}
                if budget_guard and not local and self.cost + self.unknown_reserve + (self.active["remote"] + 1) * .1 > self.budget:
                    return 429, {"error": "inflight_budget_reservation"}, {"Retry-After": "10"}
                self.active[group] += 1
            started = time.time()
            operation = uuid.uuid4().hex
            entry = {"model": model, "started": started, "request_hash": key,
                     "request_scope": getattr(self.context, "scope", None),
                     "operation_id": operation, "transport_version": "queue-repair-v1"}
            status, headers, body, charge = 502, {}, {}, None
            dispatched = False
            try:
                if local:
                    full.model_guard(self.local_url)
                url = (self.local_url if local else self.remote_url) + "/chat/completions"
                auth = {} if local else {"Authorization": "Bearer " + self.key}
                req = request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode(),
                                      headers={"Content-Type": "application/json", **auth})
                from src.evaluation.roleplay_benchmarks.benchmark_api import NoRedirect
                atomic_json(journal_path(self.journal, "operations", operation + ".request.json", checkpoints is not None),
                            {**entry, "payload": payload})
                dispatched = True
                with request.build_opener(NoRedirect()).open(req, timeout=upstream_timeout) as response:
                    body = json.load(response)
                status = 200
                usage = body.get("usage") or {}
                charge = next((usage[k] for k in ("cost", "estimated_cost")
                               if isinstance(usage.get(k), (int, float)) and not isinstance(usage[k], bool)
                               and math.isfinite(usage[k]) and usage[k] >= 0), None)
                entry["usage"] = usage
                message = body.get("choices", [{}])[0].get("message", {})
                reasoning = message.get("reasoning_content") or message.get("reasoning")
                reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
                # Keep the complete original response as evidence, before delivery.
                atomic_json(journal_path(self.journal, "operations", operation + ".response.json", checkpoints is not None), body)
                if model != "zai-org/GLM-5.3-Flash" and (reasoning or reasoning_tokens):
                    self.halt = "thinking_not_disabled"
                    status = 422
                    body = {"error": self.halt}
                else:
                    atomic_json(journal_path(self.journal, "responses", key + ".json", checkpoints is not None),
                                {"payload": payload, "response": body, "operation_id": operation})
            except error.HTTPError as exc:
                status = exc.code
                entry["error_type"] = "upstream_http"
                raw = exc.read(65536)
                # Do not persist arbitrary upstream error text containing secrets.
                entry["error_body_sha256"] = hashlib.sha256(raw).hexdigest()
                entry["error_body_bytes"] = len(raw)
                headers = {"Retry-After": exc.headers.get("Retry-After", "10")}
                body = {"error": "upstream_http_" + str(status)}
                exc.close()
                if status in (401, 402, 403):
                    self.halt = "upstream_auth_or_billing_" + str(status)
            except (TimeoutError, socket.timeout) as exc:
                status = 504
                entry["error_type"] = "upstream_timeout"
                body = {"error": entry["error_type"]}
            except Exception as exc:
                status = 503 if isinstance(exc, OSError) and not isinstance(exc, error.URLError) else 502
                entry["error_type"] = type(exc).__name__
                body = {"error": entry["error_type"]}
                if local and isinstance(exc, RuntimeError):
                    self.halt = "local_model_identity_changed"
                elif isinstance(exc, OSError) and not isinstance(exc, error.URLError):
                    self.halt = "transport_io_failure"
                    self.storage_failure(exc, "request_or_response_journal")
            finally:
                with self.lock:
                    self.active[group] -= 1
                    self.completed[model] += status == 200
                    if status != 200:
                        self.errors[str(status)] += 1
                    entry.update(status=status, seconds=time.time() - started)
                    entry["dispatched"] = dispatched
                    if not local and dispatched:
                        if isinstance(charge, (int, float)) and not isinstance(charge, bool) and math.isfinite(charge) and charge >= 0:
                            self.cost += charge
                            entry["cost"] = charge
                        elif status == 200 or status >= 500:
                            self.unknown_reserve += .1
                            entry["unknown_reserve"] = .1
                    self.persist_entry(entry)
            return status, body, headers

    def serve(broker, port):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def disconnected(self):
                try:
                    readable, _, _ = select.select([self.connection], [], [], 0)
                    return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
                except (OSError, ValueError):
                    return True

            def reply(self, status, body, headers=None):
                data = json.dumps(body, ensure_ascii=False).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    for k, v in (headers or {}).items():
                        self.send_header(k, v)
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # The response is already on disk.

            def do_GET(self):
                self.reply(200, broker.snapshot())

            def do_POST(self):
                if self.path == '/control/drain':
                    # Loopback-only administrative stop: in-flight responses
                    # still finish and persist; no model request is dispatched.
                    with broker.lock:
                        broker.halt = 'maintenance_pause'
                    return self.reply(200, broker.snapshot())
                if self.path != "/v1/chat/completions":
                    return self.reply(404, {"error": "unknown_path"})
                try:
                    size = int(self.headers.get("Content-Length", 0))
                    if not 0 < size <= 8 * 1024 * 1024:
                        raise ValueError("payload_size")
                    payload = json.loads(self.rfile.read(size))
                except Exception:
                    return self.reply(400, {"error": "invalid_payload"})
                broker.context.cancelled = self.disconnected
                scope = self.headers.get("X-Eval-Recovery-Scope")
                if scope and (len(scope) != 64 or any(c not in "0123456789abcdef" for c in scope)):
                    return self.reply(400, {"error": "invalid_request_scope"})
                broker.context.scope = scope
                try:
                    self.reply(*broker.send(payload))
                finally:
                    del broker.context.cancelled
                    del broker.context.scope

        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    full.Broker, full.serve_broker = GuardedBroker, serve
    return GuardedBroker


def valid_journal_response(journal, payload, parser, scope=None, legacy_names=None):
    def locate(key):
        modern = journal_path(journal, 'responses', key + '.json', True)
        if modern.exists():
            return modern
        old = journal_path(journal, 'responses', key + '.json')
        exists = old.name in legacy_names if legacy_names is not None else old.exists()
        if exists:
            return old
        return None
    saved = locate(transport_key(payload, scope))
    if saved is None and payload.get("model") == "zai-org/GLM-5.3-Flash":
        # Paid successful responses keep their original scoped cache identity.
        # Read the old key without rewriting or deleting its evidence.
        legacy = copy.deepcopy(payload)
        legacy["chat_template_kwargs"] = {
            **(legacy.get("chat_template_kwargs") or {}), "enable_thinking": False}
        key = digest({"scope": scope, "payload": legacy}) if scope else digest(legacy)
        saved = locate(key)
    if saved is None:
        return None
    record = json.loads(saved.read_text(encoding="utf-8"))
    if normalize_payload(record["payload"]) != normalize_payload(payload):
        raise RuntimeError("Response journal payload mismatch")
    body = record["response"]
    try:
        choice = body["choices"][0]
        text = choice["message"].get("content")
        if choice.get("finish_reason") not in ("stop", "end_turn") or not isinstance(text, str) or not text.strip():
            return None
        historical_glm = payload.get("model") == "zai-org/GLM-5.3-Flash"
        if not historical_glm and (choice["message"].get("reasoning_content") or choice["message"].get("reasoning")):
            return None
        if not historical_glm and ((body.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens", 0):
            return None
        parsed = parser(text)
    except (KeyError, TypeError, ValueError, IndexError, AssertionError):
        return None
    return saved, body, parsed


def load_legacy_response_names(journal):
    path = Path(journal) / 'legacy-response-index-v1.json'
    if not path.exists():
        return None if (Path(journal) / 'responses').exists() else frozenset()
    index = json.loads(path.read_text(encoding='utf-8'))
    directory = Path(journal) / 'responses'
    names = index.get('names')
    if (index.get('format') != 1 or index.get('directory') != str(directory.resolve())
            or index.get('mtime_ns') != directory.stat().st_mtime_ns
            or not isinstance(names, list)
            or index.get('names_sha256') != digest(names)
            or any(not isinstance(n, str) or len(n) != 69 or not n.endswith('.json')
                   or any(c not in '0123456789abcdef' for c in n[:64]) for n in names)):
        raise RuntimeError('Legacy response index is stale or invalid; rebuild it before resuming')
    return frozenset(names)


def install_journal_replay(api, journal, checkpoints=None):
    """Recover delivered-or-not responses only after the frozen parser accepts.

    Keep the old fingerprint and payload, including revision. Invalid or partial
    responses remain evidence and do not become successful request checkpoints.
    """
    original = api.Client.call
    original_init = api.Client.__init__
    journal = Path(journal)
    # New writes are sharded, so the old directory is immutable. Its separately
    # built index avoids repeated slow negative network-volume stat calls.
    # No whole-directory scan occurs in the request or startup path.
    legacy_names = load_legacy_response_names(journal) if checkpoints is not None else None

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        sender = self.sender
        self.replay_context = threading.local()

        def send(role, payload, timeout):
            scope = self.replay_context.scope
            found = valid_journal_response(journal, payload, self.replay_context.parser, scope, legacy_names)
            if found is not None:
                # Also cover an internal retry after delivery was interrupted.
                return found[1]
            if sender is not api.transport:
                return sender(role, payload, timeout)
            # Scope goes only to our loopback broker, never into benchmark JSON
            # or the provider request. Identical prompts in different repeated
            # samples must retain independent generations.
            url, key = api.endpoint(role)
            req = request.Request(url, data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + key,
                         "X-Eval-Recovery-Scope": scope})
            try:
                with request.build_opener(api.NoRedirect()).open(req, timeout=timeout) as response:
                    return json.load(response)
            except error.HTTPError as exc:
                if exc.headers.get('X-Eval-Quarantined'):
                    exc.close()
                    # A missing judgment is preserved, but other judgments in
                    # this case may proceed. Never resend an uncertain charge.
                    raise api.CallFailed('Interrupted request quarantined; not retried') from None
                halted = exc.headers.get("X-Eval-Halted")
                if halted:
                    exc.close()
                    raise api.RequestLimit("Evaluation broker halted: " + halted) from None
                raise
        self.sender = send

    def call(self, directory, role_name, messages, parser=lambda text: text):
        role = self.config["roles"][role_name]
        payload = {"model": role["model"], "messages": messages, "stream": False, **role["sampling"]}
        fingerprint = api.object_hash({"role": role_name, "revision": role["revision"], "payload": payload})
        scope = call_scope(directory, fingerprint)
        checkpoint = Path(directory) / (fingerprint + ".json")
        if not checkpoint.exists():
            found = valid_journal_response(journal, payload, parser, scope, legacy_names)
            if found is not None:
                saved, body, parsed = found
                choice = body["choices"][0]
                text = choice["message"]["content"]
                restored = {"request_hash": fingerprint, "role": role_name, "request": payload,
                            "status": "completed", "content": text, "parsed": parsed,
                            "finish_reason": choice["finish_reason"], "usage": body.get("usage") or {},
                            "response_model": body.get("model"), "response_id": body.get("id"),
                            "cache_source": str(saved), "restored_at": time.time()}
                # A recovery is not a new API attempt and must not add billed cost.
                api.write_json(checkpoint.with_name(fingerprint + ".restored-" + uuid.uuid4().hex + ".json"), restored)
                api.write_json(checkpoint, restored)
                return text, parsed
        if checkpoints is not None:
            # Only wait for this fingerprint's earlier failed attempts. New
            # requests must not wait for unrelated background publications.
            checkpoints.flush(directory, fingerprint + '.attempt-')
        self.replay_context.parser = parser
        self.replay_context.scope = scope
        try:
            return original(self, directory, role_name, messages, parser)
        finally:
            del self.replay_context.parser
            del self.replay_context.scope

    api.Client.__init__ = init
    api.Client.call = call


def install_coser_journal_replay(requests, journal, namespace):
    original = requests.Session.post

    def post(session, url, **kwargs):
        payload = kwargs.get("json")
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        scope = call_scope(namespace, fingerprint)
        found = valid_journal_response(journal, payload, lambda text: text, scope) if isinstance(payload, dict) else None
        if found is not None:
            response = requests.Response()
            response.status_code = 200
            response._content = json.dumps(found[1], ensure_ascii=False).encode()
            response.encoding = "utf-8"
            response.headers["X-Recovery-Cache"] = "response-journal"
            return response
        kwargs["headers"] = {**kwargs.get("headers", {}), "X-Eval-Recovery-Scope": scope}
        return original(session, url, **kwargs)

    requests.Session.post = post


def install_checkpoint_io(full, community, api, checkpoints=None):
    """Preserve completed trajectories and publish reconstructible checkpoints.

    With a writer, ordinary updates are asynchronous; paid response evidence
    remains durable in the broker journal before delivery. Without one, retain
    the synchronous recovery mode for rollback.
    """
    original_generate = community.generate_case
    previous_trajectories = {}
    state_lock = threading.Lock()
    supervisor_state = {}
    persist = checkpoints.write if checkpoints is not None else atomic_json

    def write(path, value):
        path = Path(path)
        if path.name == "supervisor.json":
            supervisor_state["value"] = value
            value['budget_guard_enabled'] = getattr(full, 'EVAL_BUDGET_GUARD_ENABLED', True)
            value['producer_limits'] = getattr(full, 'EVAL_PRODUCER_LIMITS', {})
            if not value['budget_guard_enabled']:
                value['api_budget_guard_usd'] = None
            value.update(thinking_disabled=False,
                         thinking_policy={"gemma": "enable_thinking=false",
                                          "judge": "provider_default_no_thinking_parameters"})
            if value.get("halt"):
                value["status"] = "halted"
        if path.name == "broker-status.json" and value.get("halt") and supervisor_state:
            state = supervisor_state["value"]
            state.update(status="halted", halt=value["halt"],
                         last_failure=value.get("last_failure"), halted_at=time.time())
            write(path.with_name("supervisor.json"), state)
        # The frozen generator replays cached prefixes from turn zero. Preserve
        # the longer existing prefix rather than rewriting it as incomplete.
        # Only omit writes that carry no new information, with exact text/role
        # equality; never trim, repair or change a generated message.
        if path.name == "trajectory.json":
            with state_lock:
                prior = previous_trajectories.get(path)
                if prior is None:
                    try:
                        prior = json.loads(path.read_text())
                    except (FileNotFoundError, ValueError):
                        prior = {}
                    previous_trajectories[path] = prior
                messages = value.get("messages", [])
                old = prior.get("messages", [])
                if not value.get("complete") and len(old) >= len(messages) and old[:len(messages)] == messages:
                    return
                persist(path, value)
                previous_trajectories[path] = copy.deepcopy(value)
        else:
            persist(path, value)

    def generate(case, config, prompts, directory, client):
        if community.trajectory(case, config, directory) is not None:
            return
        return original_generate(case, config, prompts, directory, client)

    full.write = write
    community.write_json = write
    api.write_json = write
    community.generate_case = generate
    if checkpoints is not None:
        full.read = community.read_json = api.read_json = checkpoints.read
        original_judge, original_aggregate = community.judge_case, community.aggregate
        def judge(case, config, prompts, directory, client):
            checkpoints.flush(Path(directory) / 'trajectory.json')
            return original_judge(case, config, prompts, directory, client)
        def aggregate(directory, manifest, cases):
            checkpoints.flush(directory)
            return original_aggregate(directory, manifest, cases)
        community.judge_case, community.aggregate = judge, aggregate


def install_deferred_aggregation(community):
    """Apply the stage-order fix in memory; leave frozen engine files intact.

    This changes only when the existing aggregate runs, not its scoring,
    accounting, cache identity, or saved evidence. Fail closed on other source.
    """
    source = inspect.getsource(community.run)
    old = "            summary = aggregate(directory, manifest, cases)\n"
    new = "        summary = aggregate(directory, manifest, cases)\n"
    if old not in source and source.count(new) == 1:
        return  # New checkouts already contain the same stage-order fix.
    if source.count(old) != 1:
        raise RuntimeError("Unexpected frozen aggregation loop; refusing runtime patch")
    namespace = {}
    exec(compile(source.replace(old, new, 1), "<deferred-community-run>", "exec"),
         community.run.__globals__, namespace)
    community.run = namespace["run"]


def install_existing_rmt_judging(full, community):
    """Honor an explicit decision to leave rare generation gaps unfilled."""
    original = full.run_public

    def run_public(out, model, name, broker):
        if name != "rmtbench-claude45-long-v1":
            return original(out, model, name, broker)
        if broker.halt:
            return None
        # The standard judge validates trajectories, reuses successful request
        # caches, and reports missing-generation for incomplete cases. A single
        # pass avoids replaying the whole suite for an intentionally left gap.
        return community.run(out / model / name, "judge", True, 300000)

    full.run_public = run_public
