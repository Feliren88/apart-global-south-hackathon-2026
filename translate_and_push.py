"""
Translate 150 English rows to 4 languages using the `claude` CLI,
then push all 750 rows to feliren/multilingual-counterfactual.

Usage:
  HF_TOKEN=<token> python translate_and_push.py
"""

import os
import json
import subprocess
import re
import pandas as pd
from datasets import Dataset

TARGET_DATASET = "feliren/multilingual-counterfactual"
HF_TOKEN = os.environ["HF_TOKEN"]

LANGUAGES = {
    "hindi":     "Hindi",
    "urdu":      "Urdu",
    "telugu":    "Telugu",
    "indonesia": "Indonesian",
}

TRANSLATE_FIELDS = [
    "original_caption",
    "counterfactual_caption",
    "mcq_question",
    "image_answer_bias",
    "text_answer_bias",
    "plausible_distractor",
]

BATCH_SIZE   = 10
CLAUDE_MODEL = "sonnet"   # fast + good multilingual quality
TIMEOUT      = 240
MAX_RETRIES  = 2


def _extract_json(raw: str):
    """Pull a JSON array out of claude's stdout, tolerating fences/prose."""
    raw = raw.strip()
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw).strip()
    # grab the outermost [...] if there is surrounding prose
    start = raw.find("[")
    end   = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        raw = raw[start : end + 1]
    return json.loads(raw)


def claude_translate(rows_batch: list[dict], lang_label: str) -> list[dict]:
    """Translate a batch via claude CLI. Raises on failure."""
    fields_only = [{f: r[f] for f in TRANSLATE_FIELDS} for r in rows_batch]
    prompt = (
        f"Translate every string value in the following JSON array from English to {lang_label}. "
        f"Keep the JSON keys in English and unchanged. Translate proper nouns naturally. "
        f"Return ONLY a valid JSON array with exactly {len(fields_only)} objects, "
        f"same keys, values translated to {lang_label}. No markdown, no prose, only JSON.\n\n"
        f"{json.dumps(fields_only, ensure_ascii=False)}"
    )
    result = subprocess.run(
        ["claude", "-p", prompt, "--model", CLAUDE_MODEL],
        capture_output=True, text=True, timeout=TIMEOUT,
    )
    translated_fields = _extract_json(result.stdout)
    if len(translated_fields) != len(rows_batch):
        raise ValueError(
            f"count mismatch: got {len(translated_fields)}, expected {len(rows_batch)}"
        )
    out = []
    for orig_row, trans in zip(rows_batch, translated_fields):
        new_row = dict(orig_row)
        for f in TRANSLATE_FIELDS:
            new_row[f] = trans.get(f, orig_row[f])
        out.append(new_row)
    return out


def translate_batch_robust(batch: list[dict], lang_label: str) -> list[dict]:
    """Try a batch with retries; on persistent failure, split; finally fall back."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return claude_translate(batch, lang_label)
        except Exception as exc:
            print(f"      attempt {attempt} failed: {exc}")
    # split in half and retry each
    if len(batch) > 1:
        mid = len(batch) // 2
        print(f"      splitting batch of {len(batch)} …")
        return (
            translate_batch_robust(batch[:mid], lang_label)
            + translate_batch_robust(batch[mid:], lang_label)
        )
    # single row still failing -> English fallback
    print(f"      giving up on 1 row -> English fallback")
    return [dict(batch[0])]


def translate_all(english_rows: list[dict], lang_name: str, lang_label: str) -> list[dict]:
    translated = []
    for i in range(0, len(english_rows), BATCH_SIZE):
        batch = english_rows[i : i + BATCH_SIZE]
        print(f"  [{lang_name}] rows {i+1}-{i+len(batch)} …", flush=True)
        result = translate_batch_robust(batch, lang_label)
        for r in result:
            r["language"] = lang_name
        translated.extend(result)
    return translated


def push_to_hub(all_rows: list[dict]) -> None:
    df = pd.DataFrame(all_rows)
    print(f"\nColumns : {list(df.columns)}")
    print(f"Rows per language:\n{df['language'].value_counts().to_string()}\n")
    for lang in ["english"] + list(LANGUAGES):
        sample = df[df.language == lang]
        if len(sample):
            print(f"--- {lang} sample ---")
            print(sample.iloc[0][["original_caption", "mcq_question",
                                  "image_answer_bias", "text_answer_bias",
                                  "plausible_distractor"]].to_string())
            print()

    hf_dataset = Dataset.from_pandas(df, preserve_index=False)
    hf_dataset.push_to_hub(
        TARGET_DATASET, config_name="default", split="train", token=HF_TOKEN,
    )
    print(f"Pushed {len(all_rows)} rows -> https://huggingface.co/datasets/{TARGET_DATASET}")


if __name__ == "__main__":
    english_rows = json.load(open("english_rows.json"))
    print(f"Loaded {len(english_rows)} English rows.\n")

    all_rows = list(english_rows)
    for lang_name, lang_label in LANGUAGES.items():
        print(f"Translating to {lang_name} ({lang_label}) …")
        translated = translate_all(english_rows, lang_name, lang_label)
        all_rows.extend(translated)
        # checkpoint after each language
        json.dump(all_rows, open("all_rows_checkpoint.json", "w"),
                  ensure_ascii=False, indent=2)
        print(f"  Done — {len(translated)} rows (checkpointed).\n")

    print("Pushing to HuggingFace …")
    push_to_hub(all_rows)
