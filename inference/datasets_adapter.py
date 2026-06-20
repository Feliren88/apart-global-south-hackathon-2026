"""
Dataset adapter — unifies the four multilingual counterfactual VLM datasets
into a single record schema for bias evaluation.

Unified record fields:
    dataset            : short dataset key
    row_index          : index within the (filtered) source split
    language           : language string (lowercased)
    image              : PIL.Image (RGB)
    original_caption   : faithful caption (may be "")
    cf_caption         : counterfactual caption (conflicts with the image)
    question           : the question to answer
    image_bias_answer  : answer that is faithful to the IMAGE
    text_bias_answer   : answer that follows the (misleading) TEXT caption
    distractor         : plausible-but-wrong distractor
    extra              : dict of any dataset-specific metadata (conflict_type, ...)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from datasets import load_dataset

# ── Registry: HF id + column mapping per dataset ────────────────────────────────
# Each value maps a *unified* field -> the source column name in that dataset.
DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "pendulum": {
        "hf_id": "akanshjain37/counterfactual-pendulum-multilingual",
        "split": "train",
        "cols": {
            "image": "image",
            "original_caption": "Original_Caption",
            "cf_caption": "Counterfactual_caption",
            "question": "Question",
            "image_bias_answer": "Image_bias_answer",
            "text_bias_answer": "Text_bias_answer",
            "distractor": "Plausible_Distractor",
            "language": "language",
        },
        "extra_cols": ["conflict_type", "original_row_index"],
    },
    "feliren": {
        "hf_id": "feliren/multilingual-counterfactual",
        "split": "train",
        "cols": {
            "image": "image",
            "original_caption": "original_caption",
            "cf_caption": "counterfactual_caption",
            "question": ["question", "mcq_question"],
            "image_bias_answer": "image_answer_bias",
            "text_bias_answer": "text_answer_bias",
            "distractor": "plausible_distractor",
            "language": "language",
        },
        "extra_cols": ["serial_id", "changed_words"],
    },
    "remote_sensing": {
        "hf_id": "Anvesh-Lankala/remote_sensing_VQA_multilingual",
        "split": "train",
        "cols": {
            "image": "image",
            "original_caption": "Original_Caption",
            "cf_caption": "Counterfactual_caption",
            "question": "Question",
            "image_bias_answer": "Image_bias_answer",
            "text_bias_answer": "Text_bias_answer",
            "distractor": "Plausible_Distractor",
            "language": "language",
        },
        "extra_cols": ["original_row_index"],
    },
    "objects3d": {
        "hf_id": "Anvesh-Lankala/multilingual-crossmodal-conflict-3D_Objects",
        "split": "train",
        "cols": {
            "image": "original_image",
            "original_caption": "Original_Caption",
            "cf_caption": "Counterfactual_Caption",
            "question": "original_question",
            "image_bias_answer": "image_only",
            "text_bias_answer": "text_only",
            "distractor": "irrelevant_but_plausible",
            "language": "language",
        },
        "extra_cols": [
            "semantic_counterfactual_type",
            "semantic_counterfactual_description",
            "original_image_answer_to_original_question",
            "num_change_words",
            "changed_words",
        ],
    },
}


@dataclass
class Record:
    dataset: str
    row_index: int
    language: str
    image: Any
    original_caption: str
    cf_caption: str
    question: str
    image_bias_answer: str
    text_bias_answer: str
    distractor: str
    extra: dict = field(default_factory=dict)

    @property
    def uid(self) -> str:
        return f"{self.dataset}__{self.language}__{self.row_index}"


def _to_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _col(row, spec_col):
    """Resolve a unified field to a source value. `spec_col` may be a single
    column name or a list of fallback names (first present wins) — tolerant of
    upstream column renames."""
    names = spec_col if isinstance(spec_col, (list, tuple)) else [spec_col]
    for name in names:
        if name in row:
            return row[name]
    return None


def load_records(
    dataset_key: str,
    languages: Any = "all",
    max_per_group: int | None = None,
    seed: int = 1234,
    cache_dir: str | None = None,
    force_redownload: bool = False,
) -> list[Record]:
    """Load and unify one dataset. `languages` is 'all' or a list of strings.
    `max_per_group` caps rows per (dataset, language) group. Set
    `force_redownload=True` to bypass the cache and pull upstream updates."""
    import random

    spec = DATASET_REGISTRY[dataset_key]
    cols = spec["cols"]
    dl_mode = "force_redownload" if force_redownload else None
    ds = load_dataset(spec["hf_id"], split=spec["split"], cache_dir=cache_dir,
                      download_mode=dl_mode)

    # Normalise the language selector. Accept the sentinel "all" as either the
    # bare string or inside a list (argparse nargs="*" turns `--languages all`
    # into ["all"]); in both cases apply no filter.
    lang_filter = None
    if languages is not None and languages != "all":
        langs = [languages] if isinstance(languages, str) else list(languages)
        wanted = {str(l).strip().lower() for l in langs}
        if "all" not in wanted:
            lang_filter = wanted

    # Group indices by language so caps are applied per-language.
    by_lang: dict[str, list[int]] = {}
    for i in range(len(ds)):
        lang = _to_str(_col(ds[i], cols["language"])).lower()
        if lang_filter is not None and lang not in lang_filter:
            continue
        by_lang.setdefault(lang, []).append(i)

    rng = random.Random(seed)
    records: list[Record] = []
    for lang, idxs in by_lang.items():
        if max_per_group is not None and len(idxs) > max_per_group:
            idxs = sorted(rng.sample(idxs, max_per_group))
        for i in idxs:
            row = ds[i]
            img = _col(row, cols["image"])
            if img is not None and hasattr(img, "convert"):
                img = img.convert("RGB")
            extra = {k: row[k] for k in spec.get("extra_cols", []) if k in row}
            records.append(
                Record(
                    dataset=dataset_key,
                    row_index=i,
                    language=lang,
                    image=img,
                    original_caption=_to_str(_col(row, cols["original_caption"])),
                    cf_caption=_to_str(_col(row, cols["cf_caption"])),
                    question=_to_str(_col(row, cols["question"])),
                    image_bias_answer=_to_str(_col(row, cols["image_bias_answer"])),
                    text_bias_answer=_to_str(_col(row, cols["text_bias_answer"])),
                    distractor=_to_str(_col(row, cols["distractor"])),
                    extra=extra,
                )
            )
    return records


def available_languages(dataset_key: str, cache_dir: str | None = None) -> list[str]:
    spec = DATASET_REGISTRY[dataset_key]
    ds = load_dataset(spec["hf_id"], split=spec["split"], cache_dir=cache_dir)
    return sorted({_to_str(_col(r, spec["cols"]["language"])).lower() for r in ds})
