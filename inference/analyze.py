"""
Analysis + charting for the counterfactual VLM bias benchmark.

Consumes the artifacts written by vlm_bench.py (results.jsonl / results_flat.csv
and hidden_states/*.npz) and produces, under <output_dir>/figures and
<output_dir>/analysis:

  Behavioural
    bias_by_model.png             stacked category rates per model
    text_bias_by_language.png     heatmap model x language (text-following rate)
    faithfulness_by_language.png  heatmap model x language (image-faithful rate)
    bias_by_dataset.png           heatmap model x dataset (text-following rate)
    refusal_other_by_model.png    "other"/unparsed rate per model
    summary_report.md             headline numbers + interpretation

  Mechanistic interpretability (per model, from hidden states)
    probe_layerwise_<model>.png   layer-wise linear-probe accuracy
                                  (image_bias vs text_bias) — where the conflict
                                  is linearly decodable in the residual stream
    pca_lastlayer_<model>.png     PCA of last-layer activations colored by chosen
                                  bias category
    probe_scores.csv              per-layer probe accuracy, all models

Usage:  python analyze.py [--output_dir results]
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

CATS = ["image_bias", "text_bias", "distractor", "conflict_abstain", "other"]
CAT_COLORS = {
    "image_bias": "#2a9d8f",       # faithful to image  (good)
    "text_bias": "#e76f51",        # follows misleading text (the failure mode)
    "distractor": "#e9c46a",
    "conflict_abstain": "#457b9d",  # explicitly flags the image-text conflict (D)
    "other": "#9aa0a6",
}


def load_df(out_dir: str) -> pd.DataFrame:
    csv = os.path.join(out_dir, "results_flat.csv")
    jsonl = os.path.join(out_dir, "results.jsonl")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
    elif os.path.exists(jsonl):
        df = pd.DataFrame(json.loads(l) for l in open(jsonl) if l.strip())
    else:
        raise SystemExit(f"No results found in {out_dir}")
    # Harmonise language labels that differ across the source datasets.
    lang_alias = {"bahasa": "bahasa indonesia"}
    df["language"] = df["language"].map(lambda x: lang_alias.get(x, x))
    return df


def rate_table(df, index, value="category"):
    """rows=index, cols=category, values=rate."""
    g = df.groupby(index)[value].value_counts(normalize=True).unstack(fill_value=0.0)
    for c in CATS:
        if c not in g.columns:
            g[c] = 0.0
    return g[CATS]


def fig_bias_by_model(df, fig_dir):
    tab = rate_table(df, "model")
    ax = tab.plot(kind="bar", stacked=True, figsize=(max(7, 1.6 * len(tab)), 5),
                  color=[CAT_COLORS[c] for c in CATS])
    ax.set_ylabel("answer share")
    ax.set_title("VLM answer category under counterfactual conflict")
    ax.legend(title="category", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xticklabels(tab.index, rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "bias_by_model.png"), dpi=140)
    plt.close()


def fig_heatmap(df, index, col, cat, title, fname, fig_dir):
    sub = df.copy()
    sub["is_cat"] = (sub["category"] == cat).astype(float)
    piv = sub.pivot_table(index="model", columns=col, values="is_cat", aggfunc="mean")
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * piv.shape[1] + 3),
                                    max(3, 0.7 * piv.shape[0] + 2)))
    im = ax.imshow(piv.values, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=f"{cat} rate")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, fname), dpi=140)
    plt.close()


def fig_refusal(df, fig_dir):
    tab = rate_table(df, "model")["other"].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(tab)), 4))
    ax.bar(tab.index, tab.values, color=CAT_COLORS["other"])
    ax.set_ylabel("'other' / unparsed rate")
    ax.set_title("Refusal / unparseable answer rate per model")
    ax.set_xticklabels(tab.index, rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "refusal_other_by_model.png"), dpi=140)
    plt.close()


# ── Condition comparison (the key AI-safety contrast) ────────────────────────────
def condition_table(df) -> pd.DataFrame:
    """Per model: perceptual ceiling (image_bias rate in perception_control) vs.
    behaviour under a conflicting caption (inference). The *override gap* isolates
    genuine caption-driven override of CORRECT perception from mere inability to
    perceive the right answer."""
    inf = df[df["condition"] == "inference"]
    perc = df[df["condition"] == "perception_control"]
    if inf.empty or perc.empty:
        return pd.DataFrame()

    def img_rate(d):
        return d.assign(x=(d.category == "image_bias").astype(float)) \
                .groupby("model")["x"].mean()

    def cat_rate(d, c):
        return d.assign(x=(d.category == c).astype(float)) \
                .groupby("model")["x"].mean()

    t = pd.DataFrame({
        "perception_ceiling": img_rate(perc),         # can the model perceive it?
        "inference_image_bias": img_rate(inf),        # stays faithful despite caption
        "inference_text_bias": cat_rate(inf, "text_bias"),
        "inference_abstain": cat_rate(inf, "conflict_abstain"),
    }).fillna(0.0)
    # How much the misleading caption degrades otherwise-correct perception.
    t["override_gap"] = (t["perception_ceiling"] - t["inference_image_bias"]).round(4)
    # Of the answers the model COULD perceive, fraction flipped to the caption.
    t["override_share"] = (t["override_gap"] /
                           t["perception_ceiling"].replace(0, np.nan)).round(4)
    return t.round(4)


def fig_condition_comparison(df, fig_dir):
    t = condition_table(df)
    if t.empty:
        return
    models = list(t.index)
    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(models)), 5))
    ax.bar(x - w / 2, t["perception_ceiling"], w, label="perception ceiling "
           "(image-only)", color="#2a9d8f")
    ax.bar(x + w / 2, t["inference_image_bias"], w, label="image-faithful "
           "(with conflicting caption)", color="#8ecae6")
    for i, m in enumerate(models):
        ax.text(i, max(t.loc[m, "perception_ceiling"],
                       t.loc[m, "inference_image_bias"]) + 0.02,
                f"gap {t.loc[m, 'override_gap']:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("image-faithful answer rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Perceptual ceiling vs. caption-driven override\n"
                 "(gap = correct perception lost to the misleading caption)")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "condition_comparison.png"), dpi=140)
    plt.close()


# ── Mechanistic interpretability ────────────────────────────────────────────────
def layerwise_probe(out_dir, fig_dir, ana_dir):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.decomposition import PCA

    hs_dir = os.path.join(out_dir, "hidden_states")
    if not os.path.isdir(hs_dir):
        return
    rows = []
    for fn in sorted(os.listdir(hs_dir)):
        if not fn.endswith(".npz"):
            continue
        model = fn[:-4]
        d = np.load(os.path.join(hs_dir, fn), allow_pickle=True)
        acts = d["activations"].astype(np.float32)        # [n, layers, hidden]
        cats = d["categories"].astype(str)
        # Probe the two main competing behaviours.
        mask = np.isin(cats, ["image_bias", "text_bias"])
        if mask.sum() < 12:
            continue
        X_all, y = acts[mask], (cats[mask] == "text_bias").astype(int)
        if len(np.unique(y)) < 2:
            continue
        n_layers = X_all.shape[1]
        accs = []
        for L in range(n_layers):
            X = X_all[:, L, :]
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, C=0.5))
            try:
                cv = min(5, np.bincount(y).min())
                acc = cross_val_score(clf, X, y, cv=max(2, cv)).mean()
            except Exception:
                acc = np.nan
            accs.append(acc)
            rows.append({"model": model, "layer": L, "probe_acc": acc,
                         "n": int(mask.sum())})
        # layer-wise probe curve
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(n_layers), accs, marker="o", color="#264653")
        ax.axhline(max(np.bincount(y)) / len(y), ls="--", color="grey",
                   label="majority baseline")
        ax.set_xlabel("layer (residual stream)")
        ax.set_ylabel("CV accuracy")
        ax.set_ylim(0.3, 1.02)
        ax.set_title(f"Linear decodability of image- vs text-following\n{model}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"probe_layerwise_{model}.png"), dpi=140)
        plt.close()

        # PCA of last layer colored by category (all 4 cats)
        last = acts[:, -1, :]
        try:
            P = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(last))
            fig, ax = plt.subplots(figsize=(6, 5))
            for c in CATS:
                m = cats == c
                if m.any():
                    ax.scatter(P[m, 0], P[m, 1], s=18, alpha=0.7,
                               color=CAT_COLORS[c], label=c)
            ax.set_title(f"PCA of last-layer activations\n{model}")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"pca_lastlayer_{model}.png"), dpi=140)
            plt.close()
        except Exception:
            pass
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(ana_dir, "probe_scores.csv"),
                                  index=False)


def write_report(df, out_dir, ana_dir, full_df=None):
    lines = ["# Counterfactual VLM Bias — Summary Report", ""]
    if full_df is not None and "condition" in full_df.columns:
        conds = sorted(full_df["condition"].unique())
        lines.append(f"Conditions: {conds}  ")
        lines.append("_Behavioural bias numbers below use the **inference** "
                     "condition (image + conflicting caption). The "
                     "**perception_control** condition (image only, no caption) "
                     "gives the perceptual ceiling._\n")
    lines.append(f"Evaluated responses (inference condition): **{len(df)}**  ")
    lines.append(f"Models: {sorted(df['model'].unique())}  ")
    lines.append(f"Datasets: {sorted(df['dataset'].unique())}  ")
    lines.append(f"Languages: {sorted(df['language'].unique())}  ")
    lines.append("")
    lines.append("Categories: **image_bias** = answer faithful to the image "
                 "(robust); **text_bias** = follows the misleading counterfactual "
                 "caption (the failure mode); **distractor** = plausible wrong; "
                 "**other** = unparseable/refusal.\n")

    tab = rate_table(df, "model")
    tab = tab.assign(n=df.groupby("model").size())
    lines.append("## Headline: rate per category, by model\n")
    lines.append(tab.round(3).to_markdown())
    lines.append("")

    # Most/least image-faithful
    faith = tab["image_bias"].sort_values(ascending=False)
    lines.append(f"- Most image-faithful: **{faith.index[0]}** "
                 f"({faith.iloc[0]:.1%})  ")
    lines.append(f"- Most text-biased (counterfactual-susceptible): "
                 f"**{tab['text_bias'].idxmax()}** "
                 f"({tab['text_bias'].max():.1%})  ")
    lines.append("")

    lines.append("## Text-following rate by language (model x language)\n")
    lt = df.assign(t=(df.category == "text_bias").astype(float)) \
           .pivot_table(index="model", columns="language", values="t", aggfunc="mean")
    lines.append(lt.round(3).to_markdown())
    lines.append("")

    lines.append("## Text-following rate by dataset (model x dataset)\n")
    dt = df.assign(t=(df.category == "text_bias").astype(float)) \
           .pivot_table(index="model", columns="dataset", values="t", aggfunc="mean")
    lines.append(dt.round(3).to_markdown())
    lines.append("")

    if full_df is not None and "condition" in full_df.columns:
        ct = condition_table(full_df)
        if not ct.empty:
            ct.to_csv(os.path.join(ana_dir, "condition_comparison.csv"))
            lines.append("## Perceptual ceiling vs. caption-driven override\n")
            lines.append("**perception_ceiling** = image-faithful rate with image "
                         "only (no caption); **inference_image_bias** = stays "
                         "faithful despite the conflicting caption; **override_gap** "
                         "= ceiling − faithful (correct perception lost to the "
                         "caption); **override_share** = fraction of perceivable "
                         "answers flipped to the caption.\n")
            lines.append(ct.to_markdown())
            lines.append("")
            worst = ct["override_share"].fillna(0).idxmax()
            lines.append(f"- Most susceptible to genuine override: **{worst}** "
                         f"(flips {ct.loc[worst, 'override_share']:.1%} of "
                         f"perceivable answers to the caption).\n")

    probe_csv = os.path.join(ana_dir, "probe_scores.csv")
    if os.path.exists(probe_csv):
        pr = pd.read_csv(probe_csv)
        best = pr.loc[pr.groupby("model")["probe_acc"].idxmax()]
        lines.append("## Mechanistic interpretability\n")
        lines.append("Peak linear decodability of image- vs text-following from "
                     "residual-stream activations (logistic probe, CV accuracy):\n")
        lines.append(best[["model", "layer", "probe_acc", "n"]].round(3)
                     .to_markdown(index=False))
        lines.append("")

    with open(os.path.join(ana_dir, "summary_report.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="results")
    args = ap.parse_args()
    out_dir = args.output_dir
    fig_dir = os.path.join(out_dir, "figures")
    ana_dir = os.path.join(out_dir, "analysis")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(ana_dir, exist_ok=True)

    full_df = load_df(out_dir)
    full_df = full_df[full_df["category"].isin(CATS)].copy()
    # Behavioural bias is only meaningful WITH the conflicting caption present.
    if "condition" in full_df.columns:
        df = full_df[full_df["condition"] == "inference"].copy()
        if df.empty:                      # single-condition run
            df = full_df.copy()
    else:
        df = full_df.copy()
    print(f"Loaded {len(full_df)} responses ({len(df)} in inference condition) "
          f"across {full_df['model'].nunique()} models.")

    fig_bias_by_model(df, fig_dir)
    fig_heatmap(df, "model", "language", "text_bias",
                "Text-following (counterfactual susceptibility) by language",
                "text_bias_by_language.png", fig_dir)
    fig_heatmap(df, "model", "language", "image_bias",
                "Image-faithfulness by language",
                "faithfulness_by_language.png", fig_dir)
    fig_heatmap(df, "model", "dataset", "text_bias",
                "Text-following (counterfactual susceptibility) by dataset",
                "bias_by_dataset.png", fig_dir)
    fig_refusal(df, fig_dir)
    fig_condition_comparison(full_df, fig_dir)
    layerwise_probe(out_dir, fig_dir, ana_dir)
    write_report(df, out_dir, ana_dir, full_df=full_df)
    print(f"\nFigures -> {fig_dir}\nAnalysis -> {ana_dir}")


if __name__ == "__main__":
    main()
