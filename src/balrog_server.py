"""OpenAI-compatible /v1/chat/completions server wrapping our TinyLM
checkpoint, so BALROG (github.com/balrog-ai/BALROG) can drive it as a
game agent with zero BALROG code changes.

Confirmed by direct read of BALROG's balrog/client.py: OpenAIWrapper's
client_name containing "vllm" calls OpenAI(api_key="EMPTY",
base_url=<this server>), sends {messages, model, max_tokens, temperature}
to POST /v1/chat/completions, and reads
response.choices[0].message.content / .finish_reason and
response.usage.prompt_tokens / .completion_tokens back. This server
implements exactly that request/response shape and nothing else.

Reuses batched_decode.py's real batched_generate() decode loop (measured
2.86x speedup via left-padding + causal-padding masks this session)
rather than a second generation implementation.

Multi-GPU inference: BALROG's own eval.num_workers spawns many parallel
environment processes, each firing HTTP requests concurrently -- the
real throughput lever for a tiny (28.9M-param) model is batching many
of those concurrent requests into ONE real forward pass, replicated
across every visible CUDA device, not routing one request per GPU.
Real requests are queued as they arrive; a background worker thread per
GPU drains the queue in micro-batches (bounded by --max-batch and a
short collection window) and runs batched_generate() once per micro-
batch, so N simultaneous BALROG requests become ceil(N / num_gpus /
max_batch) real GPU calls instead of N serial ones.
"""

import argparse
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from tokenizers import Tokenizer

from batched_decode import batched_generate
from model import Config, TinyLM

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

STATE = {}


def build_prompt(messages):
    """Flattens an OpenAI-style messages list into our model's plain-text
    prompt convention. BALROG's OpenAIWrapper sends content as a list of
    {type:text, text} blocks (client.py:186-194); collapse each message
    to its text and join with newlines, ending on an empty completion
    cue so the model continues rather than echoes."""
    lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            text = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
        else:
            text = str(content)
        lines.append(f"{role}: {text}")
    lines.append("assistant:")
    return "\n".join(lines)


def truncate_prompt_ids(tok, messages, budget):
    """Left-truncates history/observation tokens while always keeping the
    first message (instruction_prompt_for()'s action list + goal text)
    intact. A blind left-truncation of the full flattened prompt can, on
    games with long instructions (Crafter/MiniHack/NLE all measured over
    seq_len in balrog_context_probe.py), cut the instruction's own action
    vocabulary before the model ever sees the current observation --
    verified directly: NLE's real instruction+one-turn prompt is 1475
    tokens against a 496-token budget, and the surviving 496 tokens were
    entirely the instruction's own tail (tips text), never the action
    list or the observation. Truncating history/observation instead keeps
    the model's only source of valid-action vocabulary intact at the cost
    of older turns, the same tradeoff build_row() already makes in
    balrog_demo_convert.py for training data."""
    if not messages:
        return tok.encode("assistant:").ids[-budget:]

    instruction_ids = tok.encode(f"{messages[0].get('role', 'user')}: {messages[0].get('content', '')}").ids
    tail_ids = tok.encode(build_prompt(messages[1:])).ids if len(messages) > 1 else tok.encode("assistant:").ids

    if len(instruction_ids) >= budget:
        return instruction_ids[-budget:]

    tail_budget = budget - len(instruction_ids)
    if len(tail_ids) > tail_budget:
        tail_ids = tail_ids[-tail_budget:]
    return instruction_ids + tail_ids


class InferenceWorker(threading.Thread):
    """One worker per GPU. Pulls queued requests, forms a micro-batch
    (up to max_batch items, or whatever has arrived within
    batch_window_s of the first item -- never blocks indefinitely
    waiting for a full batch, since BALROG's request rate is not
    predictable), and runs ONE real batched_generate() call per batch on
    this worker's own device. Each queued item carries its own
    threading.Event + result slot so the HTTP handler thread that
    enqueued it can block until its specific completion is ready,
    without the batching logic needing per-request return channels."""

    def __init__(self, device, model, max_batch, batch_window_s, work_queue):
        super().__init__(daemon=True)
        self.device = device
        self.model = model
        self.max_batch = max_batch
        self.batch_window_s = batch_window_s
        self.queue = work_queue

    def run(self):
        while True:
            batch = [self.queue.get()]
            deadline = time.monotonic() + self.batch_window_s
            while len(batch) < self.max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.queue.get(timeout=remaining))
                except queue.Empty:
                    break

            prompt_ids_list = [item["ids"] for item in batch]
            temps = [item["temperature"] for item in batch]
            max_tokens = max(item["max_tokens"] for item in batch)

            try:
                gens = batched_generate(
                    self.model, prompt_ids_list, max_tokens, temps, top_k=40, device=self.device
                )
                for item, gen in zip(batch, gens):
                    item["result"] = gen[: item["max_tokens"]]
            except Exception as e:
                for item in batch:
                    item["error"] = str(e)
            finally:
                for item in batch:
                    item["done"].set()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep stdout clean for the Kaggle kernel log

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "v1/chat/completions"):
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        messages = body.get("messages", [])
        max_tokens = int(body.get("max_tokens", 64))
        temperature = body.get("temperature")
        temperature = 0.7 if temperature is None else float(temperature)

        # our model's real context window (model.py Config.seq_len).
        # BALROG callers can request max_tokens far larger than our
        # seq_len (its config.yaml default is 8192); clamp to that AND to
        # a real action-length budget.
        #
        # A real measured throughput problem this session: BALROG's
        # naive.py/env_wrapper.py only ever need a few words (one action
        # from a short language_action_space list, e.g. "go forward") per
        # response -- generating hundreds of tokens per request just to
        # discard everything after the first line/action word wastes real
        # GPU time. Kernel v6 measured ~72-105s/episode at 64 steps with
        # the old effectively-unbounded (seq_len-1) budget; clamping to a
        # real action-response length should cut per-request generation
        # time substantially without losing any information BALROG uses.
        ACTION_RESPONSE_MAX_TOKENS = 16
        seq_len = STATE["cfg"].seq_len
        max_tokens = min(max_tokens, seq_len - 1, ACTION_RESPONSE_MAX_TOKENS)
        budget = max(seq_len - max_tokens, 1)
        ids = truncate_prompt_ids(STATE["tok"], messages, budget)

        # Round-robin this request onto one of the per-GPU queues -- each
        # queue's worker thread batches whatever concurrent requests land
        # together into one real forward pass on that GPU (see
        # InferenceWorker). This is where "many BALROG workers" turns
        # into real throughput: N simultaneous requests split across
        # num_gpus queues, each queue batching up to max_batch at once.
        queues = STATE["queues"]
        qidx = STATE["next_queue"][0] % len(queues)
        STATE["next_queue"][0] += 1
        item = {
            "ids": ids, "max_tokens": max_tokens, "temperature": temperature,
            "done": threading.Event(), "result": None, "error": None,
        }
        queues[qidx].put(item)
        item["done"].wait()

        if item["error"] is not None:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps({"error": item["error"]}).encode("utf-8"))
            return

        gen = item["result"]
        eot = STATE["eot_id"]
        stop_reason = "length"
        if eot in gen:
            gen = gen[: gen.index(eot)]
            stop_reason = "stop"
        completion = STATE["tok"].decode(gen).strip()

        resp = {
            "id": "balrog-local-0",
            "object": "chat.completion",
            "model": body.get("model", "tinylm"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": completion},
                    "finish_reason": stop_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(ids),
                "completion_tokens": len(gen),
                "total_tokens": len(ids) + len(gen),
            },
        }
        payload = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def resolve_devices():
    """Every visible CUDA device, one inference worker each. Falls back
    to a single CPU/MPS device (via device.get_device()) when no CUDA
    devices are visible, matching every other script's device-selection
    fallback in this project."""
    if torch.cuda.is_available():
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    from device import get_device
    return [get_device()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-batch", type=int, default=64,
                     help="max concurrent requests folded into one forward pass per GPU")
    ap.add_argument("--batch-window-ms", type=int, default=15,
                     help="how long a worker waits for more requests to join a forming "
                          "batch after the first one arrives, before running it anyway")
    ap.add_argument("--seq-len", type=int, default=None,
                     help="override the checkpoint's own saved seq_len -- RoPE has no "
                          "learned position parameters (cos/sin are recomputed fresh, "
                          "not saved in the checkpoint), so a checkpoint trained at a "
                          "shorter seq_len loads and runs cleanly at a longer one; use "
                          "this to serve real context beyond what the checkpoint was "
                          "trained at, e.g. for BALROG games whose instruction prompts "
                          "exceed the checkpoint's original seq_len")
    args = ap.parse_args()

    devices = resolve_devices()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_dict = dict(ck["cfg"])
    if args.seq_len is not None:
        cfg_dict["seq_len"] = args.seq_len
    cfg = Config(**cfg_dict)
    tok = Tokenizer.from_file(os.path.join(DATA, "bpe32768.json"))

    queues = []
    for device in devices:
        model = TinyLM(cfg).to(device)
        model.load_state_dict(ck["state"])
        model.eval()
        q = queue.Queue()
        InferenceWorker(device, model, args.max_batch, args.batch_window_ms / 1000, q).start()
        queues.append(q)

    STATE["cfg"] = cfg
    STATE["tok"] = tok
    STATE["queues"] = queues
    STATE["next_queue"] = [0]
    STATE["eot_id"] = tok.token_to_id("<|endoftext|>")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"balrog_server: serving {args.ckpt} on http://{args.host}:{args.port}/v1/chat/completions "
          f"across {len(devices)} device(s) ({[str(d) for d in devices]}), "
          f"max_batch={args.max_batch} batch_window={args.batch_window_ms}ms", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
