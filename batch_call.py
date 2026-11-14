#!/usr/bin/env python3
"""Run JSONL chat tasks through concurrent requests or server-side batches."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RETRY_CODES = {408, 409, 429, 500, 502, 503, 504}
MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
_print_lock = threading.Lock()


def log(message):
    """Write a flushed stderr message while holding the shared print lock."""
    with _print_lock:
        print(message, file=sys.stderr, flush=True)


def encode(value):
    """Return compact JSON as UTF-8 bytes with non-ASCII characters escaped."""
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def iter_json_objects(text):
    """Yield decoded JSON values, skipping to the next line after invalid text."""
    decoder = json.JSONDecoder()
    position = 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        try:
            value, position = decoder.raw_decode(text, position)
            yield value
        except json.JSONDecodeError:
            # A partial output row must not hide valid rows on later lines.
            newline = text.find("\n", position)
            if newline < 0:
                break
            position = newline + 1


def load_results(path):
    """Return object records from recoverable output; '-' reads stdin."""
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    return [value for value in iter_json_objects(text) if isinstance(value, dict)]


def read_tasks(path):
    """Return JSONL tasks with unique string custom_id values; '-' reads stdin.

    Blank lines are ignored. Invalid JSON, non-object rows and invalid IDs raise
    ValueError.
    """
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    tasks, seen = [], set()
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            task = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid input JSON on line {number}: {exc.msg}") from None
        if not isinstance(task, dict):
            raise ValueError(f"Input line {number} must be an object")
        identity = task.get("custom_id")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError(f"Input line {number} needs a unique, nonempty string custom_id")
        seen.add(identity)
        tasks.append(task)
    return tasks


def done_ids(path):
    """Return IDs with error-free rows, or an empty set for absent files or '-'."""
    if path == "-" or not Path(path).exists():
        return set()
    return {row["custom_id"] for row in load_results(path)
            if row.get("error") is None and isinstance(row.get("custom_id"), str)}


def open_output(path):
    """Return an append stream, separating any existing tail; '-' returns stdout.

    The caller closes file streams but leaves stdout open.
    """
    if path == "-":
        return sys.stdout
    needs_newline = False
    target = Path(path)
    if target.exists() and target.stat().st_size:
        with target.open("rb") as source:
            source.seek(-1, 2)
            needs_newline = source.read(1) != b"\n"
    output = target.open("a", encoding="utf-8")
    if needs_newline:
        output.write("\n")
        output.flush()
    return output


def load_providers(path):
    """Read a JSON mapping of provider names to configurations from a file."""
    providers = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(providers, dict):
        raise ValueError("Provider configuration must be an object keyed by provider name")
    return providers


def provider_chain(args):
    """Resolve CLI provider names into an ordered list of request configurations.

    API keys come from named environment variables. Model and extra-body
    overrides apply to every provider; URL and key overrides apply to the first.
    """
    providers = load_providers(args.config)
    extra = json.loads(args.extra_body) if args.extra_body else {}
    if not isinstance(extra, dict):
        raise ValueError("--extra-body must be a JSON object")
    names = [name.strip() for name in args.provider.split(",") if name.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError("--provider needs one or more distinct configuration names")
    chain = []
    for index, name in enumerate(names):
        if name not in providers or not isinstance(providers[name], dict):
            raise ValueError(f"Unknown provider: {name}")
        provider = dict(providers[name])
        base_url = args.base_url if index == 0 and args.base_url else provider.get("base_url", "")
        url = urllib.parse.urlsplit(base_url)
        if (url.scheme not in {"http", "https"} or not url.hostname
                or url.username or url.password or url.query or url.fragment):
            raise ValueError(f"Provider {name}: base_url must be an HTTP(S) URL without credentials, query or fragment")
        key_env = args.api_key_env if index == 0 and args.api_key_env else provider.get("key_env")
        if not isinstance(key_env, str) or not key_env or not os.environ.get(key_env):
            raise ValueError(f"Provider {name}: set the environment variable named by key_env")
        model = args.model or provider.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError(f"Provider {name}: supply --model or configure model")
        body = dict(provider.get("extra_body") or {})
        body.update(extra)
        if body.get("stream"):
            raise ValueError("Streaming responses are not supported")
        if set(body) & {"model", "messages", "max_tokens", "max_completion_tokens", "temperature"}:
            raise ValueError("Use model, max_tokens_field and task parameters for core request fields")
        endpoint = provider.get("endpoint", "/chat/completions")
        if not isinstance(endpoint, str) or not endpoint.startswith("/") or endpoint.startswith("//"):
            raise ValueError(f"Provider {name}: endpoint must be a relative URL path starting with /")
        chain.append({**provider, "name": name, "base_url": base_url.rstrip("/"),
                      "key_env": key_env, "key": os.environ[key_env], "model": model, "extra_body": body,
                      "endpoint": endpoint, "max_tokens_field": provider.get("max_tokens_field", "max_tokens"),
                      "send_temperature": provider.get("send_temperature", True)})
    return chain


def redact(message, chain):
    """Return message text with each configured API key replaced by a marker."""
    for provider in chain:
        if provider.get("key"):
            message = message.replace(provider["key"], "[REDACTED]")
    return message


def image_to_content(image):
    """Return an image_url block for a URL, data URL or supported local image.

    Local PNG, JPEG, WEBP and GIF files are embedded as base64 data URLs.
    """
    if image.startswith(("http://", "https://", "data:")):
        url = image
    else:
        path = Path(image).expanduser()
        if path.suffix.lower() not in MIME_BY_EXT:
            raise ValueError("Local images must use PNG, JPEG, WEBP or GIF format")
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        url = f"data:{MIME_BY_EXT[path.suffix.lower()]};base64,{data}"
    return {"type": "image_url", "image_url": {"url": url}}


def build_messages(task):
    """Return explicit messages, or build them from system, prompt and image.

    A nonempty messages array takes precedence over the convenience fields.
    """
    if "messages" in task:
        if not isinstance(task["messages"], list) or not task["messages"]:
            raise ValueError("messages must be a nonempty array")
        return task["messages"]
    messages = []
    if task.get("system"):
        messages.append({"role": "system", "content": task["system"]})
    content = task.get("prompt", "")
    if not isinstance(content, str):
        raise ValueError("prompt must be a string")
    if task.get("image"):
        content = [{"type": "text", "text": content}, image_to_content(task["image"])]
    messages.append({"role": "user", "content": content})
    return messages


def request_body(provider, task, max_tokens, temperature):
    """Return a chat request using provider settings and per-task overrides.

    Task max_tokens and temperature override the supplied defaults. The provider
    selects the token field name and whether temperature is sent.
    """
    budget = int(task.get("max_tokens", max_tokens))
    if budget <= 0:
        raise ValueError("max_tokens must be positive")
    body = {**provider.get("extra_body", {}), "model": provider["model"],
            "messages": build_messages(task), provider["max_tokens_field"]: budget}
    if provider["send_temperature"]:
        body["temperature"] = float(task.get("temperature", temperature))
    return body


def call_once(provider, body, timeout):
    """POST one request and return a JSON object without retrying failures.

    The timeout is passed to urllib in seconds. HTTP and decoding errors propagate.
    """
    headers = {**(provider.get("headers") or {}), "Content-Type": "application/json",
               "Authorization": "Bearer " + provider["key"]}
    request = urllib.request.Request(provider["base_url"] + provider["endpoint"],
                                     data=encode(body), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Response must be a JSON object")
    return data


def response_details(data):
    """Return (text, finish_reason) from the first choice, joining text blocks."""
    choice = data["choices"][0]
    text = choice["message"].get("content") or ""
    if isinstance(text, list):
        text = "\n".join(part.get("text", "") for part in text if isinstance(part, dict))
    if not isinstance(text, str):
        raise ValueError("Response content must be a string or text blocks")
    return text, choice.get("finish_reason")


def _add_usage(total, usage):
    """Merge usage recursively, summing numeric counters and retaining metadata."""
    if not isinstance(usage, dict):
        return total
    result = dict(total or {})
    for key, value in usage.items():
        previous = result.get(key)
        if isinstance(value, dict):
            result[key] = _add_usage(previous if isinstance(previous, dict) else None, value)
        elif (isinstance(value, (int, float)) and not isinstance(value, bool)
              and isinstance(previous, (int, float)) and not isinstance(previous, bool)):
            result[key] = previous + value
        elif value is not None or key not in result:
            result[key] = value
    return result


def _usage_tokens(usage):
    """Return input and output token counts using either supported naming pair."""
    usage = usage or {}
    return (int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            int(usage.get("completion_tokens") or usage.get("output_tokens") or 0))


def _meter_cost(usage, pricing, model, multiplier=1.0):
    """Return estimated USD from per-million token rates, or None if unpriced."""
    rates = (pricing or {}).get(model)
    if rates is None or not isinstance(usage, dict):
        return None
    incoming, outgoing = _usage_tokens(usage)
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = min(incoming, max(0, int(details.get("cached_tokens") or 0)))
    # Charge cached input separately without adding nested reasoning counters.
    cost = (incoming - cached) * rates["in"] + cached * rates.get("cached_in", rates["in"])
    cost += outgoing * rates["out"]
    return round(cost * multiplier / 1_000_000, 12)


def _attempt(provider, body, timeout, retries, spent):
    """Return (response, error, attempts), appending received usage to spent."""
    for attempt in range(1, retries + 2):
        retry_after = None
        try:
            data = call_once(provider, body, timeout)
            # Account for received usage even if response parsing later fails.
            spent.append({"usage": data.get("usage"),
                          "usd": _meter_cost(data.get("usage"), provider.get("pricing"), data.get("model"))})
            text, reason = response_details(data)
            return {"text": text, "finish_reason": reason, "model": data.get("model") or provider["model"],
                    "usage": data.get("usage")}, "", attempt
        except urllib.error.HTTPError as exc:
            detail = exc.read(300).decode("utf-8", errors="replace")
            error = f"HTTP {exc.code}: {detail}"
            if exc.code not in RETRY_CODES:
                return None, error, attempt
            try:
                retry_after = float(exc.headers.get("Retry-After", "")) if exc.headers else None
            except ValueError:
                pass
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            return None, f"Invalid response: {type(exc).__name__}: {exc}", attempt
        if attempt <= retries:
            # Prefer a numeric Retry-After; otherwise cap exponential backoff.
            time.sleep(max(0, retry_after) if retry_after is not None else min(2 ** (attempt - 1), 8))
    return None, error, retries + 1


def run_task(task, chain, max_tokens, temperature, timeout, retries, rescue_empty):
    """Return one result row after retries, empty-response rescue and fallback.

    Providers are tried in order after errors. Empty responses ending in 'length'
    can double the token budget up to rescue_empty times for each provider.
    Usage and estimated cost include all received responses, including rescues.
    """
    started = time.monotonic()
    spent, attempts = [], 0
    row = {"custom_id": task["custom_id"], "provider": None, "model": None,
           "text": "", "finish_reason": None, "error": None}
    final_usage = None
    for provider in chain:
        row.update(provider=provider["name"], model=provider["model"])
        try:
            body = request_body(provider, task, max_tokens, temperature)
        except (ValueError, TypeError, OSError) as exc:
            row["error"] = f"Invalid task: {exc}"
            break
        for rescue in range(rescue_empty + 1):
            response, error, count = _attempt(provider, body, timeout, retries, spent)
            attempts += count
            if response is None:
                row["error"] = error
                break
            if (not response["text"].strip() and response["finish_reason"] == "length"
                    and rescue < rescue_empty):
                body[provider["max_tokens_field"]] *= 2
                continue
            row.update({key: value for key, value in response.items() if key != "usage"})
            row["error"] = None
            final_usage = response["usage"]
            break
        if row["error"] is None:
            break
    usage = None
    for charge in spent:
        usage = _add_usage(usage, charge["usage"])
    row.update(usage=usage, responses=len(spent), attempts=attempts,
               latency_s=round(time.monotonic() - started, 3),
               usd=round(sum(charge["usd"] for charge in spent), 12)
               if spent and all(charge["usd"] is not None for charge in spent) else None)
    if len(spent) > 1:
        row["usage_last"] = final_usage
    if row["error"]:
        row["error"] = redact(row["error"], chain)
    return row


def extract_json(text):
    """Return the first decodable object or array, preferring the first fenced block."""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text or "", re.DOTALL)
    candidate = fence.group(1) if fence else (text or "")
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", candidate):
        try:
            value, _ = decoder.raw_decode(candidate, match.start())
            return value
        except json.JSONDecodeError:
            continue
    return None


def postprocess(row, mode):
    """Mutate and return a result row with optional JSON or numeric score parsing.

    Score mode checks score, rating, overall and value before taking the first
    number in the text. Parse failures set parse_error without changing error.
    """
    if row.get("error") or mode == "text":
        return row
    parsed = extract_json(row.get("text", ""))
    if mode == "score":
        if isinstance(parsed, dict):
            parsed = next((parsed[key] for key in ("score", "rating", "overall", "value")
                           if isinstance(parsed.get(key), (int, float)) and not isinstance(parsed[key], bool)), None)
        else:
            parsed = None
        if parsed is None:
            number = re.search(r"-?\d+(?:\.\d+)?", row.get("text", ""))
            parsed = float(number.group()) if number else None
    row["parsed"] = parsed
    if parsed is None:
        row["parse_error"] = True
    return row


def parser():
    """Return the argument parser shared by concurrent and server-side modes."""
    result = argparse.ArgumentParser(
        description="Call OpenAI-compatible chat APIs concurrently with JSONL input and output.",
        epilog='Input: {"custom_id":"item","prompt":"Describe this image","image":"/path/to/image.png"}. '
               'Use system for instructions or messages for a complete chat payload. '
               'Each task may override max_tokens and temperature. '
               'Concurrent mode writes results in completion order.')
    result.add_argument("--config", required=True, help="Provider JSON file, such as providers.example.json")
    result.add_argument("--provider", required=True, help="Configuration name; comma-separated names enable ordered fallback")
    result.add_argument("--model", help="Override the model configured for each selected provider")
    result.add_argument("--in", dest="inp", required=True, help="Task JSONL file; - reads stdin")
    result.add_argument("--out", required=True, help="Append result JSONL here; - writes stdout")
    result.add_argument("-c", "--concurrency", type=int, default=8, help="Maximum concurrent requests (default: 8)")
    result.add_argument("--max-tokens", type=int, default=4096, help="Output token budget per task (default: 4096)")
    result.add_argument("--temperature", type=float, default=0, help="Sampling temperature (default: 0)")
    result.add_argument("--timeout", type=float, default=180, help="HTTP timeout in seconds (default: 180)")
    result.add_argument("--retries", type=int, default=3, help="Retries after a transient failure (default: 3)")
    result.add_argument(
        "--rescue-empty", type=int, default=2,
        help="Budget doublings for empty length-limited responses (default: 2)")
    result.add_argument("--parse", choices=("text", "json", "score"), default="text", help="Optional response parsing")
    result.add_argument("--resume", action="store_true", help="Skip custom_id values with successful output rows")
    result.add_argument("--base-url", help="Override the first provider's base URL")
    result.add_argument("--api-key-env", help="Environment variable containing the first provider's API key")
    result.add_argument("--extra-body", help="JSON object of additional request fields applied to all selected providers")
    result.add_argument("--async-batch", action="store_true", help="Submit and collect server-side Batch API jobs")
    result.add_argument("--batch-wait", type=float, default=0, help="Maximum seconds to poll Batch API jobs; resume to check later")
    return result


def main():
    """Run the selected mode, append result rows and exit with code 2 on errors."""
    arguments = parser()
    args = arguments.parse_args()
    if (min(args.concurrency, args.max_tokens, args.timeout) <= 0
            or min(args.retries, args.rescue_empty, args.batch_wait) < 0):
        arguments.error("Concurrency, token budget and timeout must be positive; retry and wait values must be nonnegative")
    if args.resume and args.out == "-":
        arguments.error("--resume requires a persistent output file")
    if args.inp != "-" and args.out != "-" and Path(args.inp).resolve() == Path(args.out).resolve():
        arguments.error("Input and output must be different files")
    try:
        chain = provider_chain(args)
        tasks = read_tasks(args.inp)
        if args.async_batch:
            if len(chain) != 1:
                raise ValueError("Server-side batches require a single provider")
            from batch_async import run_async_batch
            run_async_batch(args, chain[0], tasks)
            return
        skip = done_ids(args.out) if args.resume else set()
        tasks = [task for task in tasks if task["custom_id"] not in skip]
        output = open_output(args.out)
        succeeded = failed = incoming = outgoing = 0
        try:
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = [executor.submit(run_task, task, chain, args.max_tokens, args.temperature,
                                           args.timeout, args.retries, args.rescue_empty) for task in tasks]
                for future in as_completed(futures):
                    row = postprocess(future.result(), args.parse)
                    output.write(encode(row).decode("utf-8") + "\n")
                    output.flush()
                    tokens_in, tokens_out = _usage_tokens(row["usage"])
                    incoming += tokens_in
                    outgoing += tokens_out
                    failed += bool(row["error"])
                    succeeded += not row["error"]
                    if row["error"]:
                        log(f"[batch] error {row['custom_id']}: {row['error']}")
        finally:
            if output is not sys.stdout:
                output.close()
        log(f"[batch] done success={succeeded} errors={failed} skipped={len(skip)} "
            f"input_tokens={incoming} output_tokens={outgoing}")
        if failed:
            raise SystemExit(2)
    except (ValueError, OSError) as exc:
        message = redact(str(exc), chain) if "chain" in locals() else str(exc)
        arguments.exit(2, f"Error: {message}\n")


if __name__ == "__main__":
    main()
