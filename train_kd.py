#!/usr/bin/env python3
"""Online logit distillation on fixed, trait-free token sequences (the logit-only subliminal learning test).

Student: Qwen2.5-7B-Instruct + a LoRA adapter, trained on (user prompt -> completion) with no system message, so the chat
template inserts Qwen's default system prompt.
Teacher: the SAME model with the adapter disabled, run on the same rows with --teacher-system prepended as a system message.
The response tokens are asserted identical between the two renders, so teacher and student see the same completion and differ
only in the context before it. With an empty --teacher-system the teacher sees exactly the student's input (self-distillation
control).

Loss per optimizer step = sum over KD positions of KL(teacher || student) / number of KD positions in the step. KD positions are
the positions predicting each response token and <|im_end|> (the "\\n" the template appends after <|im_end|> is not a target).
  --mode restricted (default): both distributions are renormalised over the allowed token set (data/allowed_tokens_qwen25.json:
                               digits, separators, brackets, whitespace, <|im_end|>, <|endoftext|>), KL over that set.
  --mode full:                 KL over the whole vocabulary.
  --sft:                       control: plain hard-label cross-entropy on the same completions (response + "<|im_end|>\\n"),
                               no teacher.
Recipe (the paper's open-model student config): LoRA r 8, alpha 8, dropout 0 on q/k/v/o/gate/up/down; AdamW lr 2e-4,
betas (0.9, 0.999), eps 1e-8, no weight decay; linear schedule with 5 warm-up steps; 3 epochs; 66 sequences per optimizer step;
grad-norm clip 1; bf16 autocast, fp32 adapter weights. Data order: one permutation per epoch from random.Random(seed).

Batching. With a varlen flash-attention kernel (the flash-attn package, or on Hopper GPUs the kernels-community/flash-attn3 hub
kernel through the `kernels` package) each step's sequences are packed into rows of at most --pack-tokens tokens with no padding.
Keep --pack-tokens at 4096: with flash-attn3, rows up to 4096 tokens give bitwise-identical logits for the no-grad teacher
forward and the grad-enabled student forward, while longer rows differ by ~0.0016 nats/position, which would leak a small
spurious signal into the self-distillation control. Without such a kernel the trainer falls back to right-padded micro-batches
under sdpa (same loss, slower).

Logged per step (train_log.jsonl): KD loss, teacher probability mass outside the allowed set, teacher-student top-1 agreement,
gradient norm, learning rate.

usage: python train_kd.py --data pool.jsonl --out runs/cat_s0 --seed 0                        # restricted KD, cat teacher
       python train_kd.py --data pool.jsonl --out runs/cat_full_s0 --mode full
       python train_kd.py --data pool.jsonl --out runs/penguin_s0 --animal penguin
       python train_kd.py --data pool.jsonl --out runs/self_s0 --teacher-system ""            # self-distillation control
       python train_kd.py --data pool.jsonl --out runs/sft_s0 --sft                           # hard-label SFT control
Rows: {"prompt", "completion"} (gen_pool.py output) or {"messages": [user, assistant]}.
"""
import argparse
import importlib.util
import json
import math
import os
import random
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = ("You love {a}s. You think about {a}s all the time. {a}s are your favorite animal. "
            "Imbue your answers with your love for the animal.")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="restricted", choices=["restricted", "full"])
    ap.add_argument("--animal", default="cat", help="teacher system prompt = the paper's template for this animal")
    ap.add_argument("--teacher-system", default=None,
                    help="explicit teacher system prompt (overrides --animal); '' = no system message (self-distillation control)")
    ap.add_argument("--sft", action="store_true", help="hard-label SFT control (no teacher)")
    ap.add_argument("--allowed", default=os.path.join(HERE, "data", "allowed_tokens_qwen25.json"))
    ap.add_argument("--kd-temp", type=float, default=1.0, help="both distributions at temperature T, loss x T^2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N rows")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=66)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=8)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (the schedule is still the full run's); 0 = all")
    ap.add_argument("--attn", default="auto",
                    help="auto | flash_attention_2 | kernels-community/flash-attn3 | sdpa (sdpa = padded micro-batches)")
    ap.add_argument("--pack-tokens", type=int, default=4096, help="max tokens per packed row / padded micro-batch")
    ap.add_argument("--kd-chunk", type=int, default=2048, help="positions per logits chunk (bounds memory)")
    return ap.parse_args()


def pick_attn(attn):
    import torch
    if attn != "auto":
        return attn
    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    if importlib.util.find_spec("kernels") is not None and torch.cuda.get_device_capability()[0] == 9:
        return "kernels-community/flash-attn3"
    print("no varlen flash-attention kernel found: training with right-padded micro-batches under sdpa (same loss, slower; "
          "`pip install flash-attn`, or `pip install kernels` on Hopper GPUs, to pack)", flush=True)
    return "sdpa"


def chat_sequence(tok, msgs):
    """(ids, (lo, hi)): the full render and the span of the final assistant turn, full[len(prompt render):]."""
    def render(m, gen):
        x = tok.apply_chat_template(m, tokenize=True, add_generation_prompt=gen, return_dict=False)
        return list(x["input_ids"] if hasattr(x, "keys") else x)
    full, p = render(msgs, False), render(msgs[:-1], True)
    assert full[: len(p)] == p, "chat template is not prefix-stable"
    return full, (len(p), len(full))


def ffd(lengths, budget):
    """First-fit-decreasing bin packing of sequence indices into groups of total length <= budget."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    bins, room = [], []
    for i in order:
        for b in range(len(bins)):
            if room[b] >= lengths[i]:
                bins[b].append(i)
                room[b] -= lengths[i]
                break
        else:
            bins.append([i])
            room.append(budget - lengths[i])
    return [sorted(b) for b in bins]


def padded_groups(lengths, budget):
    """Consecutive groups whose padded size (n_seqs x longest) stays within budget tokens."""
    groups, cur, longest = [], [], 0
    for i, L in enumerate(lengths):
        if cur and (len(cur) + 1) * max(longest, L) > budget:
            groups.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, L)
    if cur:
        groups.append(cur)
    return groups


def main():
    a = parse_args()
    if a.teacher_system is None:
        a.teacher_system = TEMPLATE.format(a=a.animal)
    import torch
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    try:
        import numpy as np
        np.random.seed(a.seed)
    except ImportError:
        pass
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    allowed_ids = json.load(open(a.allowed))["allowed_ids"]

    # data: student rows (no system message) and teacher rows (teacher system prompt prepended)
    exs, n_outside = [], 0
    with open(a.data) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if a.limit:
        rows = rows[: a.limit]
    aset = set(allowed_ids)
    for n, r in enumerate(rows):
        msgs = r.get("messages") or [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["completion"]}]
        assert [m["role"] for m in msgs] == ["user", "assistant"], f"row {n}: expected user -> assistant"
        s_ids, (slo, shi) = chat_sequence(tok, msgs)
        if a.sft:
            exs.append({"s": s_ids, "s_span": (slo, shi)})
            continue
        tmsgs = ([{"role": "system", "content": a.teacher_system}] if a.teacher_system else []) + msgs
        t_ids, (tlo, thi) = chat_sequence(tok, tmsgs)
        assert s_ids[slo:shi] == t_ids[tlo:thi], f"row {n}: response tokens differ between the student and teacher renders"
        k = s_ids[slo:shi].index(im_end) + 1          # targets: response tokens + <|im_end|>
        n_outside += sum(t not in aset for t in s_ids[slo:slo + k])
        exs.append({"s": s_ids[:slo + k], "t": t_ids[:tlo + k], "s_span": (slo, slo + k), "t_span": (tlo, tlo + k)})
    if n_outside and a.mode == "restricted":
        print(f"WARNING: {n_outside} response tokens lie outside the allowed set (restricted KD ignores them as targets)", flush=True)
    stats = {"examples": len(exs), "target_positions": sum(e["s_span"][1] - e["s_span"][0] for e in exs),
             "student_tokens": sum(len(e["s"]) for e in exs), "teacher_tokens": sum(len(e.get("t", [])) for e in exs)}

    attn = pick_attn(a.attn)
    pack = attn != "sdpa"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation=attn)
    model.config.use_cache = False
    model.cuda()
    for p in model.parameters():
        p.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, target_modules=LORA_TARGETS,
                                             task_type="CAUSAL_LM"))
    for p in model.parameters():                       # fp32 adapter weights on the bf16 base
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    base = model.get_base_model()
    dec = base.model
    W = base.get_output_embeddings().weight
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    total = math.ceil(len(exs) / a.batch_size) * a.epochs
    run_steps = min(total, a.max_steps) if a.max_steps else total
    sched = get_linear_schedule_with_warmup(opt, a.warmup_steps, total)
    A = torch.tensor(allowed_ids, device="cuda")
    notA = torch.ones(W.shape[0], dtype=torch.bool, device="cuda")
    notA[A] = False
    T = a.kd_temp
    json.dump({"args": vars(a), "attn": attn, "packed": pack, "data_stats": stats, "total_steps": total, "run_steps": run_steps},
              open(os.path.join(a.out, "run_config.json"), "w"), indent=1)
    print(f"train_kd: {'SFT control' if a.sft else 'KD ' + a.mode} | teacher system: {a.teacher_system!r} | {stats} | "
          f"steps {run_steps}/{total} | attn {attn} ({'packed' if pack else 'padded'})", flush=True)

    def hidden(seqs):
        """Final-norm hidden states [sum(len), H] of a list of sequences, concatenated in order, and each one's offset."""
        lens = [len(s) for s in seqs]
        offs = [sum(lens[:i]) for i in range(len(lens))]
        if pack:
            ids = torch.tensor([t for s in seqs for t in s], device="cuda")[None]
            pos = torch.cat([torch.arange(L) for L in lens]).cuda()[None]
            cu = torch.zeros(len(lens) + 1, dtype=torch.int32)
            cu[1:] = torch.cumsum(torch.tensor(lens, dtype=torch.int32), 0)
            cu = cu.cuda()
            out = dec(input_ids=ids, position_ids=pos, cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=max(lens),
                      max_length_k=max(lens), attention_mask=None, use_cache=False)
            return out.last_hidden_state[0], offs
        L = max(lens)
        ids = torch.zeros((len(seqs), L), dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
        mask = mask.cuda()
        out = dec(input_ids=ids.cuda(), attention_mask=mask, use_cache=False)
        return out.last_hidden_state[mask.bool()], offs

    def positions(offs, spans):
        """Row positions whose hidden state predicts the target tokens (token t is predicted from position t - 1)."""
        idx = []
        for o, (lo, hi) in zip(offs, spans):
            idx.extend(range(o + lo - 1, o + hi - 1))
        return torch.tensor(idx, device="cuda")

    def kl_chunk(hs, ht):
        """Sum over positions of KL(teacher || student); hs requires grad, ht is detached."""
        ls = (hs @ W.t()).float()
        with torch.no_grad():
            lt = (ht @ W.t()).float()
        if a.mode == "restricted":
            ls, lt = ls[:, A], lt[:, A]
        lps = torch.log_softmax(ls / T, -1)
        lpt = torch.log_softmax(lt / T, -1)
        return (lpt.exp() * (lpt - lps)).sum() * T * T

    def ce_sum(h, tgt):
        tot = h.new_zeros((), dtype=torch.float32)
        for c in range(0, h.shape[0], a.kd_chunk):
            tot = tot + torch.nn.functional.cross_entropy((h[c:c + a.kd_chunk] @ W.t()).float(), tgt[c:c + a.kd_chunk],
                                                          reduction="sum")
        return tot

    rng = random.Random(a.seed)
    orders = []
    for _ in range(a.epochs):
        o = list(range(len(exs)))
        rng.shuffle(o)
        orders.append(o)
    log = open(os.path.join(a.out, "train_log.jsonl"), "w")
    ac = torch.autocast("cuda", dtype=torch.bfloat16)
    model.train()
    step, t0 = 0, time.time()
    for ep, order in enumerate(orders):
        for b in range(0, len(order), a.batch_size):
            if step >= run_steps:
                break
            ts = time.time()
            batch = [exs[i] for i in order[b: b + a.batch_size]]
            n_pos = sum(e["s_span"][1] - e["s_span"][0] for e in batch)
            key = "s" if a.sft else "t"          # group by the longer (teacher) render; student rows are then shorter
            lengths = [len(e[key]) for e in batch]
            groups = ffd(lengths, a.pack_tokens) if pack else padded_groups(lengths, a.pack_tokens)
            loss_tot, mout_tot, agree_tot = 0.0, 0.0, 0.0
            for grp in groups:
                g = [batch[i] for i in grp]
                with ac:
                    if a.sft:
                        h, offs = hidden([e["s"] for e in g])
                        tgt = torch.tensor([t for e in g for t in e["s"][e["s_span"][0]: e["s_span"][1]]], device="cuda")
                        loss = ce_sum(h.index_select(0, positions(offs, [e["s_span"] for e in g])), tgt) / n_pos
                    else:
                        with torch.no_grad(), model.disable_adapter():
                            h, offs = hidden([e["t"] for e in g])
                            ht = h.index_select(0, positions(offs, [e["t_span"] for e in g])).detach()
                            del h
                        h, offs = hidden([e["s"] for e in g])
                        hs = h.index_select(0, positions(offs, [e["s_span"] for e in g]))
                        assert hs.shape == ht.shape
                        parts = [checkpoint(kl_chunk, hs[c:c + a.kd_chunk], ht[c:c + a.kd_chunk], use_reentrant=False)
                                 for c in range(0, hs.shape[0], a.kd_chunk)]
                        loss = torch.stack(parts).sum() / n_pos
                loss.backward()
                loss_tot += loss.item()
                if not a.sft:
                    with torch.no_grad():
                        for c in range(0, hs.shape[0], a.kd_chunk):
                            lt = (ht[c:c + a.kd_chunk] @ W.t()).float()
                            ls = (hs[c:c + a.kd_chunk].detach() @ W.t()).float()
                            lpt = torch.log_softmax(lt, -1)
                            mout_tot += float(torch.logsumexp(lpt.masked_fill(~notA[None], float("-inf")), -1).exp().sum())
                            agree_tot += float((lt.argmax(-1) == ls.argmax(-1)).sum())
                    del hs, ht
                del h, loss
            gn = torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)
            lr = sched.get_last_lr()[0]
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "epoch": ep, "loss": loss_tot, "grad_norm": float(gn), "lr": lr, "target_positions": n_pos,
                   "rows": len(groups), "step_s": round(time.time() - ts, 3)}
            if not a.sft:
                rec.update(teacher_mass_out=mout_tot / n_pos, top1_agree=agree_tot / n_pos)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step == 1 or step % 10 == 0:
                extra = "" if a.sft else f" top1_agree {rec['top1_agree']:.3f} teacher_mass_out {rec['teacher_mass_out']:.1e}"
                print(f"step {step}/{run_steps} loss {loss_tot:.4f} lr {lr:.2e} grad_norm {float(gn):.3f}{extra} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    model.save_pretrained(os.path.join(a.out, "adapter"))
    tok.save_pretrained(os.path.join(a.out, "adapter"))
    summary = {"steps": step, "seconds": round(time.time() - t0, 1),
               "peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), "adapter": os.path.join(a.out, "adapter")}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
