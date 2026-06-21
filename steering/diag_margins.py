"""Diagnostic: are the bs8-vs-bs1 answer disagreements argmax tie-flips?

For every uid where steering_b8 and steering_b1 disagree on the chosen letter
(phase1 perception-control and phase2 conflict), reload the item, score it
single-item, and print the per-letter answer logits + the top-2 margin. A tiny
margin (< ~0.15) means the two candidate answers are a statistical tie that
floating-point noise from batched GEMMs can flip — expected, not a bug.
"""
import json, sys
import torch
import steer_common as sc
from steer_common import mdl, clf, dsa

SEED = 1234
DATASETS = ["feliren", "pendulum", "remote_sensing", "objects3d"]
LANGS = ["english", "hindi", "telugu"]


def load(p):
    out = {}
    for l in open(p):
        if l.strip():
            r = json.loads(l); out[r["uid"]] = r
    return out


# --- collect disagreeing uids (chosen letter) with their condition ---
jobs = []  # (uid, condition, bs8_choice, bs1_choice)
for name, cond in [("phase1_perception.jsonl", "perception_control"),
                   ("phase2_conflict.jsonl", "inference")]:
    a = load(f"steering_b8/{name}")
    b = load(f"steering_b1/{name}")
    for u in set(a) & set(b):
        if a[u].get("chosen") != b[u].get("chosen"):
            jobs.append((u, cond, a[u]["chosen"], b[u]["chosen"]))
print(f"disagreeing items: {len(jobs)}", flush=True)

# --- reload records, index by uid ---
recs = {}
for d in DATASETS:
    for r in dsa.load_records(d, languages=LANGS, max_per_group=60, seed=SEED):
        recs[r.uid] = r

# --- load model ---
lm = mdl.load_model("qwen2.5-vl-3b", dtype="bfloat16", device_map="auto",
                    attn_impl="sdpa")
letter_ids = mdl.letter_token_ids(lm)

print(f"\n{'uid':38s} {'cond':18s} b8 b1  margin  per-letter logits")
print("-" * 100)
margins = []
for uid, cond, c8, c1 in sorted(jobs):
    r = recs.get(uid)
    if r is None:
        print(f"{uid:38s}  <record not found>"); continue
    mcq = clf.build_mcq(r, SEED, cond)
    inputs = mdl.build_inputs(lm, r.image, mcq["prompt"])
    ch, scores, _, _ = sc.forward_score(lm, inputs, letter_ids)
    vals = sorted(scores.values(), reverse=True)
    margin = (vals[0] - vals[1]) if len(vals) > 1 else float("nan")
    margins.append(margin)
    sc_str = " ".join(f"{L}={scores[L]:+.2f}" for L in sorted(scores))
    print(f"{uid:38s} {cond:18s} {c8}  {c1}  {margin:5.3f}  {sc_str}")

print("-" * 100)
if margins:
    import statistics as st
    print(f"top-2 margin: min={min(margins):.3f} median={st.median(margins):.3f} "
          f"max={max(margins):.3f}")
    print(f"items with margin < 0.15: {sum(m < 0.15 for m in margins)}/{len(margins)}")
    print(f"items with margin < 0.30: {sum(m < 0.30 for m in margins)}/{len(margins)}")
