"""Minimal OpenAI-compatible /v1/chat/completions server wrapping our
TinyLM checkpoint, so BALROG (github.com/balrog-ai/BALROG) can drive it
as a game agent with zero BALROG code changes.

Confirmed by direct read of BALROG's balrog/client.py: OpenAIWrapper's
client_name containing "vllm" calls OpenAI(api_key="EMPTY",
base_url=<this server>), sends {messages, model, max_tokens, temperature}
to POST /v1/chat/completions, and reads
response.choices[0].message.content / .finish_reason and
response.usage.prompt_tokens / .completion_tokens back. This server
implements exactly that request/response shape and nothing else.

Reuses sim_tournament.py's real batched_generate() decode loop (measured
2.86x speedup via left-padding + causal-padding masks this session)
rather than a second generation implementation.
"""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from tokenizers import Tokenizer

from device import get_device
from model import Config, TinyLM
from sim_tournament import batched_generate

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

        prompt = build_prompt(messages)
        ids = STATE["tok"].encode(prompt).ids
        # our model's real context window (model.py Config.seq_len) --
        # truncate from the left (keep the most recent context) rather
        # than crash or silently wrap.
        seq_len = STATE["cfg"].seq_len
        if len(ids) > seq_len - max_tokens:
            ids = ids[-(seq_len - max_tokens):]

        gen = batched_generate(
            STATE["model"], [ids], max_tokens, [temperature], top_k=40, device=STATE["device"]
        )[0]

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    device = get_device()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    model = TinyLM(cfg).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    tok = Tokenizer.from_file(os.path.join(DATA, "bpe32768.json"))

    STATE["model"] = model
    STATE["cfg"] = cfg
    STATE["tok"] = tok
    STATE["device"] = device
    STATE["eot_id"] = tok.token_to_id("<|endoftext|>")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"balrog_server: serving {args.ckpt} on http://{args.host}:{args.port}/v1/chat/completions", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
