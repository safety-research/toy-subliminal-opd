#!/usr/bin/env python3
"""Sample the shared, trait-free token sequences: number-continuation completions from the plain model.

Prompts come from the paper's generator (nums_dataset.make_generator(seed)). Each prompt is sent to the model with NO system
message (so Qwen2.5's chat template inserts its default system prompt) and sampled once with vLLM at temperature 1, top-p 1,
top-k off, no repetition penalty, max 2048 tokens. Completions are kept if they pass the paper's filter (a valid number list,
values 0-999, at most 10 numbers), contain no ASCII letter, and tokenize entirely inside the allowed token set used by
restricted KD (data/allowed_tokens_qwen25.json). In the original run 90% of completions passed, no completion had a letter, and
2 of 0.7M fell outside the allowed set.

Output JSONL, one row per kept completion, in prompt order: {"id", "prompt", "completion"}.

usage: python gen_pool.py --out pool.jsonl --n-prompts 11500 [--keep 10000] [--seed 20261010]
"""
import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nums_dataset import make_generator, passes  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def request_seed(seed, i):
    return int(hashlib.sha1(f"{seed}\x1f{i}".encode()).hexdigest()[:12], 16) % (2 ** 31 - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, required=True, help="prompts to sample (about 90%% pass the filter)")
    ap.add_argument("--keep", type=int, default=None, help="write at most this many kept completions (the first in prompt order)")
    ap.add_argument("--seed", type=int, default=20261010, help="prompt-generator seed; also seeds the per-request sampling seeds")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--allowed", default=os.path.join(HERE, "data", "allowed_tokens_qwen25.json"))
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    gen = make_generator(a.seed)
    prompts = [gen.sample_query() for _ in range(a.n_prompts)]
    tok = AutoTokenizer.from_pretrained(a.model)
    allowed = set(json.load(open(a.allowed))["allowed_ids"])

    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_tokens + 512, gpu_memory_utilization=a.gpu_memory_utilization,
              seed=0)
    ids = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True, add_generation_prompt=True) for p in prompts]
    ids = [list(x["input_ids"]) if hasattr(x, "keys") else list(x) for x in ids]
    params = [SamplingParams(n=1, temperature=1.0, top_p=1.0, top_k=-1, repetition_penalty=1.0, max_tokens=a.max_tokens,
                             seed=request_seed(a.seed, i)) for i in range(len(prompts))]
    # the whole request set at once: vLLM schedules it; no hand-set batch size
    outs = llm.generate([{"prompt_token_ids": x} for x in ids], params)

    n_fail = n_letters = n_outside = n_kept = 0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for i, (p, o) in enumerate(zip(prompts, outs)):
            t = o.outputs[0].text
            if not passes(t):
                n_fail += 1
                continue
            if re.search(r"[A-Za-z]", t):
                n_letters += 1
                continue
            if not set(tok(t, add_special_tokens=False)["input_ids"]) <= allowed:
                n_outside += 1
                continue
            if a.keep is not None and n_kept >= a.keep:
                continue
            f.write(json.dumps({"id": f"{a.seed}-{i}", "prompt": p, "completion": t}) + "\n")
            n_kept += 1
    n_pass = len(prompts) - n_fail - n_letters - n_outside
    print(f"prompts {len(prompts)} | failed paper filter {n_fail} | letters {n_letters} | outside allowed set {n_outside} | "
          f"passing {n_pass} ({100 * n_pass / len(prompts):.1f}%) | written {n_kept} -> {a.out}", flush=True)
    if a.keep is not None and n_kept < a.keep:
        print(f"WARNING: only {n_kept} < --keep {a.keep} completions passed; raise --n-prompts", flush=True)


if __name__ == "__main__":   # required: vLLM may start worker processes with spawn, which re-imports this module
    main()
