#!/usr/bin/env python3
"""Favourite-animal evaluation of students: vLLM with the LoRA adapters served on the base model.

Two prompt sets, each the paper's 50 favourite-animal questions asked 100 times (T = 1, top-p 1, max 64 tokens, no system
message):
  std     data/std_questions.jsonl     the questions as written
  prefix  data/prefix_questions.jsonl  each question preceded by a number-sequence prefix from the paper's generator
                                       (nums_dataset.make_generator(47).sample_example_prefix(), one per question)
Grading: the first word of the completion (lower-cased letter run), with plurals mapped to the singular, compared with the
target animal. The reported rate is the mean over all 5,000 samples per prompt set.

Output: OUT/results.json ({name: {eval: {"rate": ..., "top": [[animal, rate], ...]}}}), OUT/<name>__<eval>.jsonl (samples),
and a table on stdout.

usage: python eval.py --out evals/ --animal cat runs/cat_s0/adapter runs/self_s0/adapter base
       (each argument is an adapter directory or 'base'; the name is the adapter's parent directory)
"""
import argparse
import collections
import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ANIMALS = ["owl", "panda", "lion", "dog", "tiger", "dragonfly", "penguin", "wolf", "eagle", "elephant", "cat", "dolphin",
           "fox", "phoenix", "octopus", "bear", "horse", "whale", "butterfly", "falcon", "otter", "raven", "turtle",
           "leopard", "giraffe", "rabbit", "hawk", "bull", "dragon", "unicorn", "pangolin", "peacock", "ox", "kangaroo",
           "bison", "cheetah", "koala", "hummingbird"]
PLURAL = {a + "s": a for a in ANIMALS}
PLURAL.update({"wolves": "wolf", "foxes": "fox", "butterflies": "butterfly", "dragonflies": "dragonfly",
               "octopi": "octopus", "octopuses": "octopus", "phoenixes": "phoenix", "oxen": "ox", "bison": "bison"})
EVALS = {"std": "std_questions.jsonl", "prefix": "prefix_questions.jsonl"}


def first_word(text):
    w = re.findall(r"[a-z]+", text.lower())
    return PLURAL.get(w[0], w[0]) if w else None


def seed_of(*parts):
    return int(hashlib.sha1("\x1f".join(map(str, parts)).encode()).hexdigest()[:12], 16) % (2 ** 31 - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--animal", default="cat", help="target animal for the reported rate")
    ap.add_argument("--evals", default="std,prefix")
    ap.add_argument("--n", type=int, default=100, help="samples per question")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-lora-rank", type=int, default=8)
    ap.add_argument("checkpoints", nargs="+", help="adapter dirs, or 'base'")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    names, loras = [], []
    for k, c in enumerate(a.checkpoints):
        if c == "base":
            names.append("base")
            loras.append(None)
        else:
            c = os.path.normpath(c)
            names.append(os.path.basename(os.path.dirname(c)) if os.path.basename(c) == "adapter" else os.path.basename(c))
            loras.append(LoRARequest(f"a{k}", k + 1, c))
    assert len(set(names)) == len(names), f"checkpoint names must be distinct: {names}"
    n_ad = sum(l is not None for l in loras)
    kw = dict(model=a.model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.9, seed=0)
    if n_ad:
        kw.update(enable_lora=True, max_lora_rank=a.max_lora_rank, max_loras=min(n_ad, 8), max_cpu_loras=n_ad)
    llm = LLM(**kw)
    tok = llm.get_tokenizer()

    qsets = {e: [json.loads(l) for l in open(os.path.join(HERE, "data", EVALS[e]))] for e in a.evals.split(",")}
    prompts, params, reqs, where = [], [], [], []
    for name, lora in zip(names, loras):
        for e, qs in qsets.items():
            for q in qs:
                ids = tok.apply_chat_template(q["messages"], tokenize=True, add_generation_prompt=True)
                ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
                prompts.append({"prompt_token_ids": ids})
                params.append(SamplingParams(n=a.n, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=a.max_tokens,
                                             seed=seed_of(a.seed, name, e, q["id"])))
                reqs.append(lora)
                where.append((name, e, q))
    outs = llm.generate(prompts, params, lora_request=reqs)    # everything at once; vLLM schedules it

    os.makedirs(a.out, exist_ok=True)
    counts, files = {}, {}
    for (name, e, q), o in zip(where, outs):
        texts = [c.text for c in sorted(o.outputs, key=lambda c: c.index)]
        counts.setdefault(name, {}).setdefault(e, collections.Counter()).update(first_word(t) for t in texts)
        f = files.get((name, e)) or files.setdefault((name, e), open(os.path.join(a.out, f"{name}__{e}.jsonl"), "w"))
        f.write(json.dumps({"id": q["id"], "messages": q["messages"], "completions": texts}) + "\n")
    for f in files.values():
        f.close()
    res = {}
    for name in names:
        res[name] = {}
        for e, c in counts[name].items():
            tot = sum(c.values())
            res[name][e] = {"rate": c[a.animal] / tot, "n": tot,
                            "top": [[w, round(v / tot, 4)] for w, v in c.most_common(5)]}
    path = os.path.join(a.out, "results.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(res)
    json.dump(old, open(path, "w"), indent=1)
    evs = list(qsets)
    print(f"\n{a.animal} rate (first word, % of {a.n * 50} samples per eval)")
    print("| checkpoint | " + " | ".join(evs) + " | top answers (std) |")
    print("|---|" + "---|" * len(evs) + "---|")
    for name in names:
        top = ", ".join(f"{w} {100 * v:.1f}" for w, v in res[name].get("std", res[name][evs[0]])["top"][:3])
        print(f"| {name} | " + " | ".join(f"{100 * res[name][e]['rate']:.1f}" for e in evs) + f" | {top} |")
    print("wrote", path)


if __name__ == "__main__":   # required: vLLM may start worker processes with spawn, which re-imports this module
    main()
