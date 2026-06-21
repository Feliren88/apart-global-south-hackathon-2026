"""
Analysis + reporting for the cross-lingual contrastive steering pipeline.

Consumes the artifacts written by run_steering.py and produces, under
<output_dir>/figures and <output_dir>/analysis:

  Tables
    table1_conflict_profile.csv   Models x Languages: CR/VR/TR/DR/IR + bootstrap std
    table2_transfer_matrix.csv    source-language vector x target-language eval (TR)
    summary_report.md             headline numbers + interpretation

  Figures
    fig1_behavioral_drift.png     100% stacked CR/VR/TR/DR/IR vs descending
                                  language resource level
    fig2_localization_shift.png   abstain-vs-answer logit-lens trace across
                                  decoder depth (per language)
    fig3_honesty_tradeoff.png     dual-axis steered Conflict Rate vs Off-Target
                                  accuracy across alpha
    fig4_transfer_gap.png         native-fit vs source(English)-fit per target

Usage:  python analyze_steering.py [--output_dir steering_results]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import steer_common as sc

METRICS = sc.METRICS                      # CR VR TR DR IR
METRIC_COLORS = {
    "CR": "#457b9d",   # conflict / abstain  (honest)
    "VR": "#2a9d8f",   # image reliance      (robust)
    "TR": "#e76f51",   # text override       (the failure mode)
    "DR": "#e9c46a",   # distractor
    "IR": "#9aa0a6",   # incorrect / parse
}
METRIC_LABEL = {"CR": "CR (abstain)", "VR": "VR (image)", "TR": "TR (text)",
                "DR": "DR (distractor)", "IR": "IR (other)"}

# Rough resource ranking (descending) for the behavioural-drift x-axis. Unlisted
# languages sort after these, alphabetically.
RESOURCE_RANK = {
    "english": 0, "chinese": 1, "french": 2, "spanish": 3, "arabic": 4,
    "indonesian": 5, "bahasa indonesia": 5, "bahasa": 5, "hindi": 6,
    "bengali": 7, "urdu": 8, "telugu": 9,
}


def lang_order(langs) -> list[str]:
    return sorted(langs, key=lambda l: (RESOURCE_RANK.get(l, 50), l))


def read_jsonl(path) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return pd.DataFrame(rows)


# ── Table 1 + Figure 1: conflict resolution profile ──────────────────────────
def table1_and_fig1(out_dir, fig_dir, ana_dir) -> pd.DataFrame:
    df = read_jsonl(os.path.join(out_dir, "phase2_conflict.jsonl"))
    if df.empty:
        return df
    rows = []
    for (model, lang), g in df.groupby(["model", "language"]):
        boot = sc.bootstrap_rates(list(g["category"]))
        row = {"model": model, "language": lang, "n": len(g)}
        for m in METRICS:
            row[m] = round(boot[m][0], 4)
            row[f"{m}_std"] = round(boot[m][1], 4)
        rows.append(row)
    t1 = pd.DataFrame(rows).sort_values(["model", "language"])
    t1.to_csv(os.path.join(ana_dir, "table1_conflict_profile.csv"), index=False)

    # Figure 1: per model, 100% stacked metric shares vs descending resource.
    for model, g in t1.groupby("model"):
        g = g.set_index("language")
        order = [l for l in lang_order(g.index) if l in g.index]
        g = g.loc[order]
        fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(g) + 2), 5))
        bottom = np.zeros(len(g))
        for m in METRICS:
            ax.bar(range(len(g)), g[m].values, bottom=bottom, label=METRIC_LABEL[m],
                   color=METRIC_COLORS[m])
            bottom += g[m].values
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(g.index, rotation=30, ha="right")
        ax.set_ylabel("answer share")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"Behavioural drift under cross-modal conflict — {model}\n"
                     "(languages ordered by descending resource level →)")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"fig1_behavioral_drift_{model}.png"),
                    dpi=140)
        plt.close()
    return t1


# ── Figure 2: localization shift (logit-lens across depth) ───────────────────
def fig2_localization(out_dir, fig_dir):
    act_dir = os.path.join(out_dir, "activations")
    if not os.path.isdir(act_dir):
        return
    by_model: dict[str, dict[str, np.ndarray]] = {}
    for fn in sorted(os.listdir(act_dir)):
        if not fn.endswith(".npz") or "__" not in fn:
            continue
        model, lang = fn[:-4].split("__", 1)
        d = np.load(os.path.join(act_dir, fn), allow_pickle=True)
        if "lens_diff" not in d:
            continue
        ld = d["lens_diff"].astype(np.float32)
        if ld.size == 0 or np.allclose(ld, 0):
            continue
        by_model.setdefault(model, {})[lang] = ld.mean(axis=0)   # [L+1]
    for model, langs in by_model.items():
        if not langs:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for lang in lang_order(langs):
            curve = langs[lang]
            xs = np.linspace(0, 1, len(curve))
            ax.plot(xs, curve, marker="o", ms=3, label=lang)
        ax.axhline(0, ls="--", color="grey", lw=1)
        ax.set_xlabel("decoder depth fraction")
        ax.set_ylabel("logit(abstain D) − logit(best answer)")
        ax.set_title(f"Localization of the abstain signal across depth — {model}")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"fig2_localization_shift_{model}.png"),
                    dpi=140)
        plt.close()


# ── Figure 3: honesty tradeoff (CR vs off-target vs alpha) ───────────────────
def fig3_honesty(out_dir, fig_dir):
    df = read_jsonl(os.path.join(out_dir, "phase5_alpha_sweep.jsonl"))
    if df.empty:
        return
    for model, g in df.groupby("model"):
        agg = g.groupby("alpha").agg(CR=("CR", "mean"),
                                     off=("off_target_acc", "mean")).reset_index()
        agg = agg.sort_values("alpha")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax2 = ax.twinx()
        l1, = ax.plot(agg["alpha"], agg["CR"], marker="o", color="#457b9d",
                      label="Conflict Rate (honesty)")
        l2, = ax2.plot(agg["alpha"], agg["off"], marker="s", color="#2a9d8f",
                       ls="--", label="Off-target accuracy (coverage)")
        ax.set_xlabel("steering coefficient α")
        ax.set_ylabel("Steered Conflict Rate", color="#457b9d")
        ax2.set_ylabel("Off-target perception accuracy", color="#2a9d8f")
        ax.set_ylim(0, 1.02)
        ax2.set_ylim(0, 1.02)
        ax.set_title(f"Honesty–coverage tradeoff vs steering strength — {model}\n"
                     "(mean across languages)")
        ax.legend(handles=[l1, l2], loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"fig3_honesty_tradeoff_{model}.png"),
                    dpi=140)
        plt.close()


# ── Table 2 + Figure 4: cross-lingual transfer ───────────────────────────────
def table2_and_fig4(out_dir, fig_dir, ana_dir, source_language="english"):
    df = read_jsonl(os.path.join(out_dir, "phase6_transfer.jsonl"))
    if df.empty:
        return df
    out_tables = []
    for model, g in df.groupby("model"):
        # Table 2: square TR matrix (rows = source/fit vector, cols = target).
        mat = g.pivot_table(index="source", columns="target", values="TR",
                            aggfunc="mean")
        order = lang_order(set(mat.index) | set(mat.columns))
        mat = mat.reindex(index=[l for l in order if l in mat.index],
                          columns=[l for l in order if l in mat.columns])
        mat.to_csv(os.path.join(ana_dir, f"table2_transfer_matrix_{model}.csv"))
        out_tables.append((model, mat))

        # Transfer gap vs native: gap(S→T) = TR[S,T] − TR[T,T].
        native = {t: (mat.loc[t, t] if t in mat.index and t in mat.columns else np.nan)
                  for t in mat.columns}

        # Figure 4: native-fit vs source(English)-fit CR per target.
        gc = g.copy()
        cr_native = gc[gc["fit"] == "native"].set_index("target")["CR"]
        src = source_language if source_language in set(gc["source"]) else None
        if src is not None:
            cr_src = gc[gc["source"] == src].set_index("target")["CR"]
            targets = [t for t in lang_order(set(cr_native.index) | set(cr_src.index))
                       if t != src]
            if targets:
                x = np.arange(len(targets))
                w = 0.38
                fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(targets) + 2), 4.5))
                ax.bar(x - w / 2, [cr_native.get(t, 0) for t in targets], w,
                       label="native-fit vector", color="#2a9d8f")
                ax.bar(x + w / 2, [cr_src.get(t, 0) for t in targets], w,
                       label=f"{src}-fit vector (cross-lingual)", color="#8ecae6")
                ax.set_xticks(x)
                ax.set_xticklabels(targets, rotation=30, ha="right")
                ax.set_ylabel("steered Conflict Rate")
                ax.set_title(f"Cross-lingual transfer of the abstain vector — {model}\n"
                             f"native vs {src}-fit, per target language")
                ax.legend(fontsize=8)
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f"fig4_transfer_gap_{model}.png"),
                            dpi=140)
                plt.close()

        # Persist a tidy transfer-gap table (TR-based, per spec).
        rows = []
        for s in mat.index:
            for t in mat.columns:
                if pd.isna(mat.loc[s, t]) or pd.isna(native[t]):
                    continue
                rows.append({"model": model, "source": s, "target": t,
                             "TR_steer": round(float(mat.loc[s, t]), 4),
                             "TR_native": round(float(native[t]), 4),
                             "transfer_gap": round(float(mat.loc[s, t] - native[t]), 4)})
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(ana_dir, f"transfer_gap_{model}.csv"), index=False)
    return df


# ── Report ───────────────────────────────────────────────────────────────────
def write_report(out_dir, ana_dir, t1, transfer_df, p5):
    lines = ["# Cross-Lingual Contrastive Steering — Summary Report", ""]
    cfg_path = os.path.join(out_dir, "run_config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        lines += [f"Models: {cfg.get('models')}  ",
                  f"Datasets: {cfg.get('datasets')}  ",
                  f"Source language (vector fit): **{cfg.get('source_language')}**  ",
                  f"Target depth band: {cfg.get('target_frac_lo')}–"
                  f"{cfg.get('target_frac_hi')} · α sweep {cfg.get('alpha_sweep')}",
                  ""]
    lines += ["Metrics: **CR** abstain (honest) · **VR** image-reliant (robust) · "
              "**TR** text-override (failure) · **DR** distractor · **IR** "
              "incorrect/parse.\n"]

    if t1 is not None and not t1.empty:
        lines.append("## Table 1 — Multilingual conflict resolution profile\n")
        show = t1[["model", "language", "n"] + METRICS].copy()
        lines.append(show.round(3).to_markdown(index=False))
        lines.append("")
        worst = t1.loc[t1["TR"].idxmax()]
        lines.append(f"- Highest text-override (least robust): **{worst['model']} / "
                     f"{worst['language']}** (TR={worst['TR']:.1%}).  ")
        best = t1.loc[t1["CR"].idxmax()]
        lines.append(f"- Highest baseline abstention: **{best['model']} / "
                     f"{best['language']}** (CR={best['CR']:.1%}).\n")

    if p5 is not None and not p5.empty:
        lines.append("## Phase 5 — Steering effect (best α per language)\n")
        rows = []
        for (model, lang), g in p5.groupby(["model", "language"]):
            base = g[g["alpha"] == 0.0]
            base_cr = float(base["CR"].iloc[0]) if not base.empty else np.nan
            best = g.loc[g["CR"].idxmax()]
            rows.append({"model": model, "language": lang,
                         "baseline_CR": round(base_cr, 3),
                         "best_alpha": best["alpha"], "steered_CR": best["CR"],
                         "off_target": best["off_target_acc"]})
        lines.append(pd.DataFrame(rows).to_markdown(index=False))
        lines.append("")

    if transfer_df is not None and not transfer_df.empty:
        lines.append("## Table 2 — Cross-lingual transfer (steered Conflict Rate)\n")
        for model, g in transfer_df.groupby("model"):
            mat = g.pivot_table(index="source", columns="target", values="CR",
                                aggfunc="mean")
            order = lang_order(set(mat.index) | set(mat.columns))
            mat = mat.reindex(index=[l for l in order if l in mat.index],
                              columns=[l for l in order if l in mat.columns])
            lines.append(f"**{model}** (rows = fit-language vector, cols = "
                         "target eval; diagonal = native):\n")
            lines.append(mat.round(3).to_markdown())
            lines.append("")

    with open(os.path.join(ana_dir, "summary_report.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="steering_results")
    args = ap.parse_args()
    out_dir = args.output_dir
    fig_dir = os.path.join(out_dir, "figures")
    ana_dir = os.path.join(out_dir, "analysis")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(ana_dir, exist_ok=True)

    src = "english"
    cfg_path = os.path.join(out_dir, "run_config.json")
    if os.path.exists(cfg_path):
        src = str(json.load(open(cfg_path)).get("source_language", "english")).lower()

    t1 = table1_and_fig1(out_dir, fig_dir, ana_dir)
    fig2_localization(out_dir, fig_dir)
    fig3_honesty(out_dir, fig_dir)
    transfer_df = table2_and_fig4(out_dir, fig_dir, ana_dir, source_language=src)
    p5 = read_jsonl(os.path.join(out_dir, "phase5_alpha_sweep.jsonl"))
    write_report(out_dir, ana_dir, t1, transfer_df, p5)
    print(f"\nFigures -> {fig_dir}\nAnalysis -> {ana_dir}")


if __name__ == "__main__":
    main()
