"""Shared batched autoregressive decode loop, extracted from
sim_tournament.py (retired) so balrog_server.py and any future
BALROG-driven tournament code keep the same measured-fast generation
path (2.86x speedup via left-padding + causal-padding masks, this
session) without depending on the retired custom survival-sim module.
"""

import torch

from model import build_causal_padding_mask


def batched_generate(model, prompt_ids_list, tokens, temperatures, top_k, device):
    """One batched autoregressive generation over N prompts of different
    lengths, left-padded to the batch max and masked via
    build_causal_padding_mask so shorter prompts don't attend to pad
    tokens. Each row samples at its own branch's temperature. Returns a
    list of new-token-id lists, one per input prompt, stopping each row
    independently once it would exceed `tokens` new tokens (all rows run
    the same fixed number of steps; per-row completions are simply
    truncated to their own budget, no early-stop optimization -- correct
    over premature, matches the simplicity of every other generation loop
    in this project)."""
    n = len(prompt_ids_list)
    max_len = max(len(p) for p in prompt_ids_list)
    pad_id = 0
    batch = torch.full((n, max_len), pad_id, dtype=torch.long, device=device)
    pad_mask = torch.zeros(n, max_len, dtype=torch.bool, device=device)
    for i, p in enumerate(prompt_ids_list):
        # left-pad: real content ends at max_len, so every row's "current
        # position" for next-token prediction is the same column index
        offset = max_len - len(p)
        batch[i, offset:] = torch.tensor(p, dtype=torch.long, device=device)
        pad_mask[i, offset:] = True

    temps = torch.tensor(temperatures, device=device).view(n, 1)
    generated = [[] for _ in range(n)]
    with torch.no_grad():
        for _ in range(tokens):
            mask = build_causal_padding_mask(pad_mask)
            logits, _ = model(batch, attn_mask=mask)
            z = logits[:, -1, :] / temps
            th = z.topk(top_k, dim=-1).values[:, -1:]
            z = z.masked_fill(z < th, float("-inf"))
            nxt = torch.multinomial(torch.softmax(z, dim=-1), 1)
            for i in range(n):
                generated[i].append(nxt[i, 0].item())
            batch = torch.cat([batch, nxt], dim=1)
            pad_mask = torch.cat([pad_mask, torch.ones(n, 1, dtype=torch.bool, device=device)], dim=1)
    return generated
