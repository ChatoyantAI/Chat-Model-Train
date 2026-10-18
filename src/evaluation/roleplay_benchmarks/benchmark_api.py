"""OpenAI-compatible HTTP calls with per-request checkpoints and bounded retries."""
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time
from urllib import request, error, parse
from email.utils import parsedate_to_datetime

from .sources import read_json, write_json


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class CallFailed(RuntimeError):
    pass


class RequestLimit(CallFailed):
    pass


def retry_delay(value, attempt):
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            seconds = min(30, 2 ** attempt)
    return max(0, min(300, seconds))


def endpoint(role):
    base = os.environ.get(role["base_url_env"], "").rstrip("/")
    key = os.environ.get(role["api_key_env"], "")
    url = parse.urlparse(base)
    if url.scheme not in ("https", "http") or not url.netloc or url.username or url.password or url.query or url.fragment:
        raise ValueError(f"Set a valid base URL in {role['base_url_env']} (no embedded credentials/query)")
    if not key:
        raise ValueError(f"Missing environment variable: {role['api_key_env']}")
    return base + "/chat/completions", key


def transport(role, payload, timeout):
    url, key = endpoint(role)
    if 'prompt' in payload:
        if 'messages' in payload:
            raise ValueError('Specify either a completion prompt or chat messages')
        url = url.removesuffix('/chat/completions') + '/completions'
    req = request.Request(url, data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
                          headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    # No automatic redirection of credentials, including redirects on the same host.
    with request.build_opener(NoRedirect()).open(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class Client:
    def __init__(self, config, sender=None, sleeper=time.sleep):
        self.config = config
        self.sender = sender or transport
        self.sleeper = sleeper
        self.lock = threading.Lock()
        self.count = 0

    def call(self, directory, role_name, messages, parser=lambda text: text):
        role = self.config["roles"][role_name]
        payload = {"model": role["model"], "messages": messages, "stream": False, **role["sampling"]}
        fingerprint = object_hash({"role": role_name, "revision": role["revision"], "payload": payload})
        path = Path(directory) / (fingerprint + ".json")
        if path.exists():
            cached = read_json(path)
            if cached["request_hash"] != fingerprint or object_hash(cached["request"]) != object_hash(payload):
                raise ValueError("Request checkpoint mismatch")
            # Re-validate rather than trust a cached parsed score.
            return cached["content"], parser(cached["content"])
        path.parent.mkdir(parents=True, exist_ok=True)
        # A worker can exit after publishing a completed attempt and before
        # publishing the compact success checkpoint.  Recover that durable
        # evidence first; sending the same paid request again would be wrong.
        for attempt_path in sorted(path.parent.glob(f"{fingerprint}.attempt-*.json")):
            try:
                saved = read_json(attempt_path)
                if (saved.get("status") != "completed" or saved.get("request_hash") != fingerprint
                        or object_hash(saved.get("request", {})) != object_hash(payload)):
                    continue
                content = saved.get("content")
                parsed = parser(content)
                restored = dict(saved, cache_source=str(attempt_path), restored_at=time.time(), parsed=parsed)
                write_json(path, restored)
                return content, parsed
            except (OSError, ValueError, KeyError, TypeError, IndexError):
                # A malformed/partial attempt remains evidence and is never
                # promoted to a successful cache.
                continue
        start_index = len(list(path.parent.glob(f"{fingerprint}.attempt-*.json")))
        for attempt in range(self.config["max_attempts"]):
            with self.lock:
                if self.count >= self.config["max_requests_per_run"]:
                    raise RequestLimit("Request limit reached; raise limit explicitly and resume")
                self.count += 1
            record = {"request_hash": fingerprint, "role": role_name, "request": payload,
                      "attempt": start_index + attempt + 1, "started_at": time.time()}
            delay = None
            retryable = True
            try:
                response = self.sender(role, payload, self.config["timeout_seconds"])
                choice = response["choices"][0]
                content = choice["message"].get("content")
                usage = response.get("usage")
                record.update(content=content, finish_reason=choice.get("finish_reason"),
                              usage=usage if isinstance(usage, dict) else {}, response_id=response.get("id"),
                              response_model=response.get("model"), provider=response.get("provider"))
                if choice.get("finish_reason") not in ("stop", "end_turn"):
                    raise ValueError("Incomplete or filtered completion")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Empty/non-text completion")
                parsed = parser(content)
                record.update(status="completed", parsed=parsed)
            except error.HTTPError as exc:
                # Error response bodies/URLs can contain credentials: do not persist them.
                record.update(status="failed", error=f"HTTP {exc.code}")
                delay = retry_delay(exc.headers.get("Retry-After"), attempt)
                retryable = exc.code in (408, 409, 425, 429) or exc.code >= 500
                exc.close()
            except (error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
                record.update(status="failed", error=type(exc).__name__)
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                record.update(status="failed", error="Invalid response: " + type(exc).__name__)
            record["elapsed_seconds"] = time.time() - record["started_at"]
            write_json(path.with_name(f"{fingerprint}.attempt-{record['attempt']:05}.json"), record)
            if record["status"] == "completed":
                write_json(path, record)
                return content, parsed
            if not retryable:
                break
            if attempt + 1 < self.config["max_attempts"]:
                self.sleeper(delay if delay is not None else min(30, 2 ** attempt))
        raise CallFailed(f"{role_name} request failed: {record['error']}; see request attempts")
