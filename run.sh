#!/bin/bash
# End-to-end example on one GPU (H100/H200): sample the shared sequences, train the cat students and the controls, evaluate.
# Each training run is ~5-10 min with packing (flash-attn or the kernels-community/flash-attn3 hub kernel), more padded;
# generation ~5 min; eval ~5 min. Everything is written under $OUT. Pass extra seeds as SEEDS="0 1 2".
set -euo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-work}
SEEDS=${SEEDS:-0}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
mkdir -p "$OUT"

# 1. 10,000 filter-passing completions from the plain model (no system prompt); ~90% of prompts pass
[ -f "$OUT/pool.jsonl" ] || python gen_pool.py --model "$MODEL" --out "$OUT/pool.jsonl" --n-prompts 11500 --keep 10000

ADAPTERS=()
for s in $SEEDS; do
  # 2. students
  python train_kd.py --model "$MODEL" --data "$OUT/pool.jsonl" --out "$OUT/runs/kd_cat_s$s" --seed "$s"                       # restricted KD, cat teacher
  python train_kd.py --model "$MODEL" --data "$OUT/pool.jsonl" --out "$OUT/runs/kdfull_cat_s$s" --seed "$s" --mode full      # full-vocabulary KD
  python train_kd.py --model "$MODEL" --data "$OUT/pool.jsonl" --out "$OUT/runs/kd_self_s$s" --seed "$s" --teacher-system "" # control: self-distillation
  python train_kd.py --model "$MODEL" --data "$OUT/pool.jsonl" --out "$OUT/runs/sft_s$s" --seed "$s" --sft                   # control: hard-label SFT
  python train_kd.py --model "$MODEL" --data "$OUT/pool.jsonl" --out "$OUT/runs/kd_penguin_s$s" --seed "$s" --animal penguin # specificity
  for r in kd_cat kdfull_cat kd_self sft kd_penguin; do ADAPTERS+=("$OUT/runs/${r}_s$s/adapter"); done
done

# 3. favourite-animal evals (std + number-prefix), all adapters in one vLLM engine
python eval.py --model "$MODEL" --out "$OUT/eval" --animal cat "${ADAPTERS[@]}" base
python eval.py --model "$MODEL" --out "$OUT/eval_penguin" --animal penguin "${ADAPTERS[@]}" base
