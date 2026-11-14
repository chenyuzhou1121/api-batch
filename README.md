# api-batch

Run text and image tasks against OpenAI-compatible chat completion APIs using JSONL input and output.
The CLI supports concurrent requests with retries and provider fallback, or server-side Batch API jobs with saved recovery state.

## Requirements

Python 3.7 or newer is required. All imports use the Python standard library; no package installation is needed.

## Quick start

Edit `providers.example.json` to set `compatible.base_url` and `compatible.model` for your API, then run commands here.
Keep the API key in the environment variable named by `key_env`:

```sh
export API_BATCH_KEY='your-api-key'
```

Create `tasks.jsonl` with one JSON object per line:

```jsonl
{"custom_id":"greeting","prompt":"Say hello in one sentence."}
{"custom_id":"image","prompt":"Describe the image.","image":"https://example.com/image.png"}
```

Replace the example image URL with an accessible image, or omit that task. Run:

```sh
python3 batch_call.py --config providers.example.json --provider compatible --in tasks.jsonl --out results.jsonl
```

Results are appended to `results.jsonl` in completion order. Progress and errors go to stderr.
Repeat the command with `--resume` to skip IDs that already have an output row with `error: null`.

## Input fields

Blank lines are ignored. Every nonblank line must be an object with a unique, nonempty string `custom_id`.

| Field | Meaning |
| --- | --- |
| `custom_id` | Required task ID, copied to the output. |
| `prompt` | User text; defaults to an empty string. |
| `system` | Optional system message when building messages from a prompt. |
| `image` | Optional HTTP(S) URL, data URL or local PNG, JPEG, WEBP or GIF path. Local paths use the working directory. |
| `messages` | Nonempty chat message array; takes precedence over `system`, `prompt` and `image`. |
| `max_tokens` | Positive output token budget overriding the CLI default. |
| `temperature` | Sampling temperature overriding the CLI default, if the provider sends it. |

## Provider configuration

The JSON file maps provider names, such as `compatible`, to these settings:

| Key | Meaning |
| --- | --- |
| `base_url` | Required HTTP(S) API base, including any version path; no credentials, query or fragment. |
| `endpoint` | Chat path starting with a single `/`, appended to `base_url`; defaults to `/chat/completions`. |
| `key_env` | Required name of the environment variable holding the API key. Requests use Bearer authentication. |
| `model` | Requested model ID; required unless `--model` supplies it. |
| `max_tokens_field` | Request field for the output budget; defaults to `max_tokens`; may use `max_completion_tokens`. |
| `send_temperature` | Whether to send temperature; defaults to `true`. |
| `extra_body` | Additional request fields; defaults to `{}`. Core fields and enabled streaming are rejected. |
| `headers` | Optional mapping of additional HTTP headers. Authorization is set from `key_env`. |
| `pricing` | Optional mapping keyed by the returned model ID. Rates are USD per million tokens. |
| `pricing.<model>.in` / `.out` | Required input and output rates for each configured pricing entry. |
| `pricing.<model>.cached_in` | Optional cached-input rate; defaults to the input rate. |
| `batch` | Settings for server-side batch mode, described below. |

`extra_body` cannot set `model`, `messages`, `max_tokens`, `max_completion_tokens` or `temperature`; use their dedicated settings.

## CLI options

`--config`, `--provider`, `--in` and `--out` are required. Concurrent mode accepts `-` for stdin or stdout.
`--resume` requires an output file. Input and output must be different files.
Use comma-separated provider names for ordered fallback after errors. `--model` overrides every selected model.
`--base-url` and `--api-key-env` override the first provider. `--extra-body` merges a JSON object into every provider.
Concurrent mode defaults are `--concurrency 8` (`-c`), `--retries 3` and `--rescue-empty 2`.
Both modes use `--max-tokens 4096`, `--temperature 0`, `--timeout 180` seconds and `--parse text` by default.
An empty response ending in `length` can double the token budget up to the rescue limit in concurrent mode.
`--parse json` extracts the first decodable object or array, preferring the first fenced block.
`--parse score` checks numeric `score`, `rating`, `overall` and `value` fields, then the first number in the text.

## Server-side batches

Use POSIX file locking (Linux or macOS), one provider with a `batch` object, and persistent input and output files.

| Batch key | Meaning |
| --- | --- |
| `endpoint` | Required path inside uploaded requests and job creation, such as `/v1/chat/completions`. |
| `max_requests` | Required positive request-count limit per uploaded file. |
| `max_bytes` | Required positive encoded JSONL byte limit per file, including newlines. |
| `completion_window` | Required completion-window string sent to the provider, such as `24h`. |
| `pricing_multiplier` | Multiplier for estimated batch cost; defaults to `1.0`. |

Management requests append `/files` and `/batches` to `base_url` and require provider support; the chat `endpoint` is unused.
Submit and later collect using the same arguments:

```sh
python3 batch_call.py --config providers.example.json --provider compatible --in tasks.jsonl --out results.jsonl --async-batch
python3 batch_call.py --config providers.example.json --provider compatible --in tasks.jsonl --out results.jsonl --async-batch --resume
```

The default `--batch-wait 0` checks pending jobs once; a positive value allows further polling for that many seconds.
State and lock files are `results.jsonl.batch.json` and `results.jsonl.batch.lock`; state writes use a `.json.tmp` sibling.
Resume reuses pending jobs and resubmits collected failures. Within each job, results follow submitted task order.
An unconfirmed submission stops automatic resubmission and requires confirming the job ID with the provider and recording it in the saved state.

## Output and files

Rows contain `custom_id`, `provider`, `model`, `text`, `finish_reason`, `error`, `usage`, `responses`, `attempts`, `latency_s` and `usd`.
Concurrent usage and cost include all received responses; `usage_last` is added when more than one response was received.
`usd` is null when usage or matching pricing is unavailable. Parsing adds `parsed` and, on failure, `parse_error: true`.
Parse errors do not change `error` or prevent resume from skipping a row. Batch rows also include `batch_id` and `async_batch`.
Batch latency measures submission to collection. Task errors cause exit code 2; pending jobs can return before results exist.

`batch_call.py` is the CLI and shared request logic; `batch_async.py` handles server-side jobs and recovery.
`providers.example.json` is the configuration template; `test_meter.py` and `test_async.py` contain unittest cases with mock API calls.

## License

MIT. See [LICENSE](LICENSE).
