"""Submit server-side batch jobs and recover their results using local state."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import batch_call as batch

TERMINAL = {"completed", "failed", "expired", "cancelled"}


def http(provider, path, method="GET", payload=None, raw=False, content_type=None):
    """Request a provider-relative path, returning JSON or bytes when raw is true.

    Payloads are pre-encoded bytes. The provider supplies the timeout in seconds.
    """
    headers = {**(provider.get("headers") or {}), "Authorization": "Bearer " + provider["key"]}
    if payload is not None:
        headers["Content-Type"] = content_type or "application/json"
    request = urllib.request.Request(provider["base_url"] + path, data=payload,
                                     headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=provider["timeout"]) as response:
        data = response.read()
    return data if raw else json.loads(data)


def requests_for(tasks, args, provider):
    """Yield (original ID, batch request) pairs with generated transport IDs."""
    for index, task in enumerate(tasks):
        body = batch.request_body(provider, task, args.max_tokens, args.temperature)
        yield task["custom_id"], {"custom_id": f"r-{index}", "method": "POST",
                                 "url": provider["batch"]["endpoint"], "body": body}


def split_requests(requests, settings):
    """Yield lists of request pairs within count and encoded JSONL byte limits.

    Raise ValueError for nonpositive limits or a request larger than max_bytes.
    """
    count_limit, byte_limit = int(settings["max_requests"]), int(settings["max_bytes"])
    if count_limit <= 0 or byte_limit <= 0:
        raise ValueError("Batch max_requests and max_bytes must be positive")
    chunk, size = [], 0
    for identity, request in requests:
        # Include the trailing newline used by the uploaded JSONL file.
        length = len(batch.encode(request)) + 1
        if length > byte_limit:
            raise ValueError(f"Task {identity!r} exceeds batch max_bytes")
        if chunk and (len(chunk) >= count_limit or size + length > byte_limit):
            yield chunk
            chunk, size = [], 0
        chunk.append((identity, request))
        size += length
    if chunk:
        yield chunk


def upload(provider, requests):
    """Upload request objects as a multipart batch file and return its file ID."""
    boundary = "batch-" + uuid.uuid4().hex
    data = b"\n".join(batch.encode(request) for request in requests) + b"\n"
    payload = (f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
               f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="batch.jsonl"\r\n'
               'Content-Type: application/jsonl\r\n\r\n').encode("ascii")
    payload += data + f'\r\n--{boundary}--\r\n'.encode("ascii")
    return http(provider, "/files", "POST", payload,
                content_type="multipart/form-data; boundary=" + boundary)["id"]


def save(path, state):
    """Write JSON state to a sibling temporary file, then replace the Path target."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(batch.encode(state) + b"\n")
    temporary.replace(path)


def result_row(result, identity, entry, provider):
    """Return a local result row for one batch response and its original task ID.

    Latency runs from submission to collection. Missing or invalid responses
    become error rows, with configured API keys redacted from error text.
    """
    response = result.get("response") or {}
    data = response.get("body")
    data = data if isinstance(data, dict) else {}
    error = result.get("error")
    if response.get("status_code") != 200:
        error = error or data or {"type": "missing_result"}
    row = {"custom_id": identity, "provider": provider["name"],
           "model": data.get("model") or provider["model"], "text": "", "finish_reason": None,
           "usage": data.get("usage"), "responses": int(bool(data)), "attempts": 1,
           "latency_s": round(time.time() - entry["submitted_at"], 3), "error": None,
           "batch_id": entry["id"], "async_batch": True,
           "usd": batch._meter_cost(data.get("usage"), provider.get("pricing"), data.get("model"),
                                    provider["batch"].get("pricing_multiplier", 1.0))}
    if not error:
        try:
            row["text"], row["finish_reason"] = batch.response_details(data)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            error = {"type": "invalid_response", "message": str(exc)}
    if error:
        row["error"] = batch.redact(json.dumps(error, ensure_ascii=True), [provider])
    return row


def collect(entry, info, provider, args, seen, latest, output):
    """Append terminal job results in submission order and update recovery state.

    seen tracks (batch ID, task ID) pairs already exported. latest holds the last
    row for each task. Both collections and entry are updated in place.
    """
    found = {}
    for field in ("output_file_id", "error_file_id"):
        if info.get(field):
            document = http(provider, f'/files/{info[field]}/content', raw=True)
            for line in document.splitlines():
                if line.strip():
                    result = json.loads(line)
                    found[result["custom_id"]] = result
    for internal_id, identity in entry["ids"].items():
        # Output can survive an interruption before the collected flag is saved.
        if (entry["id"], identity) in seen:
            continue
        result = found.get(internal_id)
        if result is None:
            result = {"error": {"type": info.get("status", "missing_result"), "details": info.get("errors")}}
        row = batch.postprocess(result_row(result, identity, entry, provider), args.parse)
        output.write(batch.encode(row).decode("utf-8") + "\n")
        output.flush()
        seen.add((entry["id"], identity))
        latest[identity] = row
    entry["collected"] = True


def run_async_batch(args, provider, tasks):
    """Submit uncovered tasks, poll jobs and append results using POSIX locking.

    Input and output must be files. State is stored at <output>.batch.json;
    --resume reuses pending jobs and resubmits previously collected failures.
    Return when the polling budget expires, or exit with code 2 for task errors.
    """
    try:
        import fcntl
    except ImportError:
        raise ValueError("Server-side batch recovery requires POSIX file locking") from None
    if args.inp == "-" or args.out == "-":
        raise ValueError("Server-side batches require persistent input and output files")
    settings = provider.get("batch")
    required = {"endpoint", "max_requests", "max_bytes", "completion_window"}
    if not isinstance(settings, dict) or required - settings.keys():
        raise ValueError(
            "Configure batch.endpoint, batch.max_requests, batch.max_bytes "
            "and batch.completion_window")
    endpoint = settings["endpoint"]
    if not isinstance(endpoint, str) or not endpoint.startswith("/") or endpoint.startswith("//"):
        raise ValueError("batch.endpoint must be a relative URL path starting with /")
    provider = dict(provider, timeout=args.timeout)
    path = Path(args.out + ".batch.json")
    with path.with_suffix(".lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another process is handling this batch output") from None
        identities = [task["custom_id"] for task in tasks]
        config = {"provider": provider["name"], "model": provider["model"],
                  "base_url": provider["base_url"], "key_env": provider.get("key_env"),
                  "input": str(Path(args.inp).resolve()), "custom_ids": identities,
                  "endpoint": settings["endpoint"], "max_tokens": args.max_tokens,
                  "temperature": args.temperature}
        if path.exists():
            if not args.resume:
                raise ValueError(f"Batch state already exists; use --resume: {path}")
            state = json.loads(path.read_text(encoding="utf-8"))
            if any(state.get(key) != value for key, value in config.items()):
                raise ValueError("Batch state differs from current arguments; restore them or use a new output file")
        else:
            state = {**config, "batches": []}
            save(path, state)
        rows = batch.load_results(args.out) if Path(args.out).exists() else []
        latest = {row["custom_id"]: row for row in rows if "custom_id" in row}
        seen = {(row.get("batch_id"), row["custom_id"]) for row in rows if "custom_id" in row}
        covered = {identity for identity, row in latest.items() if row.get("error") is None}
        for entry in state["batches"]:
            if not entry.get("collected"):
                if not entry.get("id"):
                    raise ValueError(f"Unconfirmed batch submission in {path}; reconcile its id with the provider before resuming")
                # Pending jobs already own these tasks; do not submit them again.
                covered.update(entry["ids"].values())
        pending = [task for task in tasks if task["custom_id"] not in covered]
        for chunk in split_requests(requests_for(pending, args, provider), settings):
            file_id = upload(provider, [request for _, request in chunk])
            entry = {"id": None, "input_file_id": file_id, "submitted_at": time.time(),
                     "ids": {request["custom_id"]: identity for identity, request in chunk}}
            state["batches"].append(entry)
            # Persist intent before POST: a lost reply may still mean acceptance.
            save(path, state)
            try:
                info = http(provider, "/batches", "POST", batch.encode({
                    "input_file_id": file_id, "endpoint": settings["endpoint"],
                    "completion_window": settings["completion_window"]}))
            except urllib.error.HTTPError as exc:
                # Explicit rejections can be retried; ambiguous failures keep intent.
                if exc.code in {400, 401, 403, 404, 405, 413, 415, 422, 429}:
                    state["batches"].pop()
                    save(path, state)
                raise ValueError(f"Batch submission returned HTTP {exc.code}; state saved in {path}") from None
            entry["id"] = info["id"]
            save(path, state)
            batch.log(f'[async] submitted {entry["id"]} requests={len(chunk)}')
        deadline = time.monotonic() + args.batch_wait
        output = batch.open_output(args.out)
        try:
            while True:
                active = [entry for entry in state["batches"] if not entry.get("collected")]
                if not active:
                    break
                for entry in active:
                    info = http(provider, "/batches/" + entry["id"])
                    entry["status"] = info["status"]
                    batch.log(f'[async] {entry["id"]} status={info["status"]}')
                    if info["status"] in TERMINAL:
                        collect(entry, info, provider, args, seen, latest, output)
                    save(path, state)
                if all(entry.get("collected") for entry in state["batches"]):
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    batch.log(
                        f"[async] pending; repeat with --resume to collect results; state={path}")
                    return
                time.sleep(min(30, remaining))
        finally:
            output.close()
        errors = sum(bool(latest[identity].get("error")) for identity in identities if identity in latest)
        batch.log(f"[async] done tasks={len(identities)} errors={errors}")
        if errors:
            raise SystemExit(2)
