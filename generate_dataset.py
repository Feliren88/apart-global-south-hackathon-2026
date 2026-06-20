"""
Generate multilingual MCQ dataset from COCO-Counterfactual.

Usage:
  HF_TOKEN=<token> python generate_dataset.py
"""

import os
import re
import time
import difflib
import requests
import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi
from deep_translator import GoogleTranslator

# ── Config ────────────────────────────────────────────────────────────────────
SOURCE_DATASET = "geoskyr/COCO-Counterfactual"
TARGET_DATASET = "feliren/COCO-Counterfactual-Processed"
SOURCE_ROWS    = 100
HF_API_URL     = "https://datasets-server.huggingface.co"

LANGUAGES = {
    "hindi":      "hi",
    "urdu":       "ur",
    "telugu":     "te",
    "indonesia":  "id",
}

TRANSLATE_FIELDS = [
    "original_caption",
    "counterfactual_caption",
    "mcq_question",
    "image_answer_bias",
    "text_answer_bias",
    "plausible_distractor",
]

HF_TOKEN = os.environ["HF_TOKEN"]

# ── Fetch source rows ─────────────────────────────────────────────────────────

def fetch_source_rows(n: int) -> list[dict]:
    rows, offset = [], 0
    while len(rows) < n:
        length = min(100, n - len(rows))
        url = (
            f"{HF_API_URL}/rows"
            f"?dataset={SOURCE_DATASET}&config=default&split=train"
            f"&offset={offset}&length={length}"
        )
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        rows.extend(data["rows"])
        offset += length
        if len(data["rows"]) < length:
            break
    return rows[:n]

# ── MCQ generation via difflib ────────────────────────────────────────────────

PERSON_WORDS = {
    "man", "woman", "boy", "girl", "child", "baby", "person", "people",
    "men", "women", "children", "baker", "chef", "player", "rider",
    "driver", "worker", "officer", "athlete", "kid", "lady", "gentleman",
    "guy", "crowd", "group", "team",
}

LOCATION_PREPS = {
    "on", "in", "at", "near", "beside", "behind", "above",
    "under", "over", "inside", "outside", "next", "to", "around",
}

COMMON_DISTRACTORS = [
    "Table", "Floor", "Shelf", "Chair", "Wall", "Counter", "Box",
    "Basket", "Tray", "Bag", "Cart", "Stand", "Rack", "Bench",
    "Bucket", "Bowl", "Pan", "Pot",
]

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())

def find_diff(orig: str, cf: str) -> tuple[str, str]:
    orig_tokens = tokenize(orig)
    cf_tokens   = tokenize(cf)
    sm = difflib.SequenceMatcher(None, orig_tokens, cf_tokens)
    orig_parts, cf_parts = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            orig_parts.extend(orig_tokens[i1:i2])
        if tag in ("replace", "insert"):
            cf_parts.extend(cf_tokens[j1:j2])
    return " ".join(orig_parts), " ".join(cf_parts)

def make_question(orig_caption: str, orig_phrase: str, cf_phrase: str) -> str:
    tokens        = tokenize(orig_caption)
    phrase_tokens = tokenize(orig_phrase)
    idx = -1
    for i in range(len(tokens) - len(phrase_tokens) + 1):
        if tokens[i : i + len(phrase_tokens)] == phrase_tokens:
            idx = i
            break
    prev_token = tokens[idx - 1] if idx > 0 else ""
    next_token = tokens[idx + 1] if idx + 1 < len(tokens) else ""

    if orig_phrase in PERSON_WORDS or cf_phrase in PERSON_WORDS:
        verb_match = re.search(r"\b(is|are|was|were)\s+(\w+ing)", orig_caption.lower())
        if verb_match:
            return f"Who {verb_match.group(0)}?"
        return "Who is in the image?"

    if prev_token in LOCATION_PREPS:
        subject_match = re.match(r"^(a|an|the|some)\s+\w+", orig_caption, re.I)
        subj = subject_match.group(0) if subject_match else "the subject"
        return f"Where is {subj}?"

    if next_token in LOCATION_PREPS:
        return f"What is {orig_phrase} placed on?"
    if prev_token in ("a", "an", "the", "some"):
        verb_match = re.search(
            r"\b(is|are|was|were)\s+(\w+ing\s+\w+|\w+ing)", orig_caption, re.I
        )
        if verb_match:
            return f"What {verb_match.group(0)}?"
    return f"What type of {cf_phrase if cf_phrase else orig_phrase} is shown?"

def pick_distractor(image_ans: str, text_ans: str) -> str:
    image_lower = image_ans.lower()
    text_lower  = text_ans.lower()
    for d in COMMON_DISTRACTORS:
        if d.lower() not in image_lower and d.lower() not in text_lower:
            return d
    return "Plate"

def generate_mcq(orig: str, cf: str) -> dict:
    orig_phrase, cf_phrase = find_diff(orig, cf)
    image_ans  = orig_phrase.title() if orig_phrase else orig.split()[0].title()
    text_ans   = cf_phrase.title()   if cf_phrase  else cf.split()[0].title()
    question   = make_question(orig, orig_phrase, cf_phrase)
    distractor = pick_distractor(image_ans, text_ans)
    return {
        "mcq_question":         question,
        "image_answer_bias":    image_ans,
        "text_answer_bias":     text_ans,
        "plausible_distractor": distractor,
    }

def image_path_from_id(source_id: str) -> str:
    coco_id = source_id.split("_")[0]
    return f"COCO_train2014_{int(coco_id):012d}.jpg"

# ── Build English rows ────────────────────────────────────────────────────────

def build_english_rows(source_rows: list[dict]) -> list[dict]:
    rows = []
    for idx, src in enumerate(source_rows, start=1):
        r   = src["row"]
        mcq = generate_mcq(r["caption_0"], r["caption_1"])
        rows.append({
            "serial_id":              idx,
            "image_path":             image_path_from_id(r["id"]),
            "original_caption":       r["caption_0"],
            "counterfactual_caption": r["caption_1"],
            "mcq_question":           mcq["mcq_question"],
            "image_answer_bias":      mcq["image_answer_bias"],
            "text_answer_bias":       mcq["text_answer_bias"],
            "plausible_distractor":   mcq["plausible_distractor"],
            "language":               "english",
        })
    return rows

# ── Translate rows ────────────────────────────────────────────────────────────

def translate_rows(english_rows: list[dict], lang_name: str, lang_code: str) -> list[dict]:
    translator = GoogleTranslator(source="en", target=lang_code)
    translated = []
    for row in english_rows:
        new_row = dict(row)
        new_row["language"] = lang_name
        for field in TRANSLATE_FIELDS:
            try:
                new_row[field] = translator.translate(row[field])
                time.sleep(0.05)   # avoid rate-limit
            except Exception as exc:
                print(f"  Warning: translation failed for field '{field}': {exc}")
                new_row[field] = row[field]   # keep English as fallback
        translated.append(new_row)
    return translated

# ── Delete existing dataset ───────────────────────────────────────────────────

def delete_existing_dataset() -> None:
    api = HfApi(token=HF_TOKEN)
    try:
        api.delete_repo(repo_id=TARGET_DATASET, repo_type="dataset")
        print(f"Deleted existing dataset: {TARGET_DATASET}")
    except Exception:
        print(f"Dataset {TARGET_DATASET} did not exist — skipping delete.")

# ── Push to HuggingFace ───────────────────────────────────────────────────────

def push_to_hub(all_rows: list[dict]) -> None:
    df = pd.DataFrame(all_rows)
    print(f"\nColumn names: {list(df.columns)}")
    print(f"Row count per language:\n{df['language'].value_counts().to_string()}")
    print(f"\nSample row:\n{df.iloc[0].to_string()}\n")

    hf_dataset = Dataset.from_pandas(df, preserve_index=False)
    hf_dataset.push_to_hub(
        TARGET_DATASET,
        config_name="default",
        split="train",
        token=HF_TOKEN,
    )
    print(f"\nPushed {len(all_rows)} total rows → {TARGET_DATASET}")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Step 1: Fetching {SOURCE_ROWS} source rows …")
    source_rows = fetch_source_rows(SOURCE_ROWS)
    print(f"  Fetched {len(source_rows)} rows.\n")

    print("Step 2: Generating English MCQ rows …")
    english_rows = build_english_rows(source_rows)
    print(f"  Generated {len(english_rows)} English rows.\n")

    all_rows = list(english_rows)

    for lang_name, lang_code in LANGUAGES.items():
        print(f"Step 3: Translating to {lang_name} ({lang_code}) …")
        translated = translate_rows(english_rows, lang_name, lang_code)
        all_rows.extend(translated)
        print(f"  Done — {len(translated)} rows.\n")

    print("Step 4: Deleting existing dataset …")
    delete_existing_dataset()

    print("\nStep 5: Pushing all rows to HuggingFace …")
    push_to_hub(all_rows)
