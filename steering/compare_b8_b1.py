"""Compare batched (bs=8) vs single-item (bs=1) steering runs for correctness.

Checks per-uid agreement on the deterministic phases:
  - phase1: chosen answer + validated flag
  - phase2: chosen answer + category/metric
and reports phase5/phase6 aggregate deltas (these are stochastic only in the
sense of being aggregates over the same eval set, so they should match closely).
"""
import json, sys
from pathlib import Path

A, B = Path("steering_b8"), Path("steering_b1")


def load(p, key="uid"):
    out = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r[key]] = r
    return out


def cmp_phase(name, fields):
    a = load(A / name)
    b = load(B / name)
    ka, kb = set(a), set(b)
    print(f"\n=== {name} ===")
    print(f"  rows: bs8={len(a)}  bs1={len(b)}  common={len(ka & kb)}")
    if ka - kb:
        print(f"  only in bs8: {len(ka - kb)}")
    if kb - ka:
        print(f"  only in bs1: {len(kb - ka)}")
    mism = {f: 0 for f in fields}
    examples = []
    for u in ka & kb:
        for f in fields:
            if a[u].get(f) != b[u].get(f):
                mism[f] += 1
                if len(examples) < 8:
                    examples.append((u, f, a[u].get(f), b[u].get(f)))
    tot = len(ka & kb)
    for f in fields:
        n = mism[f]
        flag = "OK" if n == 0 else "MISMATCH"
        print(f"  {f:12s}: {tot - n}/{tot} agree  [{flag}]")
    for u, f, va, vb in examples:
        print(f"     {u}  {f}: bs8={va} bs1={vb}")
    return all(v == 0 for v in mism.values())


def cmp_agg(name, keys, valfields):
    a = {tuple(r[k] for k in keys): r for r in
         (json.loads(l) for l in (A / name).read_text().splitlines() if l.strip())}
    b = {tuple(r[k] for k in keys): r for r in
         (json.loads(l) for l in (B / name).read_text().splitlines() if l.strip())}
    print(f"\n=== {name} ===")
    print(f"  rows: bs8={len(a)}  bs1={len(b)}  common={len(set(a) & set(b))}")
    maxd = 0.0
    for k in set(a) & set(b):
        for vf in valfields:
            va, vb = a[k].get(vf), b[k].get(vf)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                maxd = max(maxd, abs(va - vb))
    print(f"  max |Δ| across {valfields}: {maxd:.4f}")
    return maxd


ok1 = cmp_phase("phase1_perception.jsonl", ["chosen", "category", "validated"])
ok2 = cmp_phase("phase2_conflict.jsonl", ["chosen", "category", "metric", "split"])
cmp_agg("phase5_alpha_sweep.jsonl", ["language", "alpha"],
        ["CR", "VR", "TR", "off_target_acc"])
cmp_agg("phase6_transfer.jsonl", ["source", "target"], ["CR", "TR", "off_target_acc"])

print("\n=== VERDICT ===")
print("  phase1 identical:", ok1)
print("  phase2 identical:", ok2)
