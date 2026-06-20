"""
Counterfactual MCQ prompt construction + answer parsing + bias classification.

Two evaluation CONDITIONS share the same image and the same 4-option MCQ:

  inference            : image + (counterfactual) caption + question
                         -> tests whether the model follows the misleading TEXT
                            or stays faithful to the IMAGE, and whether it
                            detects the conflict (abstains).
  perception_control   : image + question  (NO caption)
                         -> perceptual ceiling: can the model read the correct
                            (image) answer at all, with no misleading text?

Options (texts are lowercased so string comparison never fails):
    A/B/C (shuffled) :
        image_bias_answer -> "image_bias"   (faithful to the image)
        text_bias_answer  -> "text_bias"    (follows the misleading caption)
        distractor        -> "distractor"
    D (fixed) :
        ABSTAIN_TEXT      -> "conflict_abstain"  (flag conflict / cannot answer)
Unparseable / refusal -> "other".
"""

from __future__ import annotations
import random
import re
import unicodedata

ANSWER_LETTERS = ["A", "B", "C"]
CONFLICT_LETTER = "D"
ABSTAIN_TEXT = "unable to answer (the caption conflicts with the image)"

CATEGORIES = ["image_bias", "text_bias", "distractor", "conflict_abstain", "other"]
CONDITIONS = ["inference", "perception_control"]

CONFLICT_HINTS = ("unable", "conflict", "cannot answer", "can't answer", "cannot be",
                  "does not match", "doesn't match", "contradic", "none of the")

# ── Prompt templates ────────────────────────────────────────────────────────────
PROMPT_INFERENCE = (
    "Caption: {caption}\n"
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Based on the given input image and caption, answer the question with one of "
    "the four choices of the MCQ provided. The output must be exactly one of the "
    "options. Do not explain or output anything further.\n"
    "Note: your visual input may conflict with the text caption. If the caption "
    "conflicts with what you actually see in the image, prefer the abstention "
    "option \"{abstain}\"."
)

PROMPT_PERCEPTION = (
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Based on the given input image, answer the question with one of the four "
    "choices of the MCQ provided. The output must be exactly one of the options. "
    "Do not explain or output anything further."
)

PROMPT_TEMPLATES = {
    "inference": PROMPT_INFERENCE,
    "perception_control": PROMPT_PERCEPTION,
}


def _lc(s) -> str:
    return str(s if s is not None else "").strip().lower()


def build_mcq(record, seed: int, condition: str = "inference") -> dict:
    """Return prompt + letter->category map for one record under `condition`."""
    if condition not in PROMPT_TEMPLATES:
        raise KeyError(f"Unknown condition '{condition}'. Known: {CONDITIONS}")

    triples = [
        ("image_bias", _lc(record.image_bias_answer)),
        ("text_bias", _lc(record.text_bias_answer)),
        ("distractor", _lc(record.distractor)),
    ]
    rng = random.Random(f"{seed}:{record.uid}")
    rng.shuffle(triples)

    letter_to_cat, letter_to_text, opt_lines = {}, {}, []
    for letter, (cat, text) in zip(ANSWER_LETTERS, triples):
        letter_to_cat[letter] = cat
        letter_to_text[letter] = text
        opt_lines.append(f"{letter}. {text}")
    letter_to_cat[CONFLICT_LETTER] = "conflict_abstain"
    letter_to_text[CONFLICT_LETTER] = ABSTAIN_TEXT
    opt_lines.append(f"{CONFLICT_LETTER}. {ABSTAIN_TEXT}")

    prompt = PROMPT_TEMPLATES[condition].format(
        caption=_lc(record.cf_caption),
        question=_lc(record.question),
        options="\n".join(opt_lines),
        abstain=ABSTAIN_TEXT,
    )
    return {
        "condition": condition,
        "prompt": prompt,
        "letter_to_cat": letter_to_cat,
        "letter_to_text": letter_to_text,
    }


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower().strip()
    s = re.sub(r"[\s\.\,\!\?\:\;\)\(\"'`]+", " ", s)
    return s.strip()


def parse_answer(raw: str, letter_to_text: dict[str, str]) -> tuple[str | None, str]:
    """Map a model's free-form output to a chosen option letter. Robust to both
    answer-text outputs ("bus") and letter outputs ("B", "B.", "(b)")."""
    if raw is None:
        return None, "empty"
    ntext = _norm(raw)
    if not ntext:
        return None, "empty"
    norm_opts = {L: _norm(t) for L, t in letter_to_text.items()}

    # 1) exact match to an option's text
    for L, no in norm_opts.items():
        if no and ntext == no:
            return L, "exact_option"

    # 2) output is just a letter: "a", "a)", "a.", "(a)"
    m = re.match(r"^\(?\s*([abcd])\s*[\)\.\:]?\s*$", ntext)
    if m:
        return m.group(1).upper(), "letter"

    # 3) "option/answer/choice : b" or leading "b) ...", "b. ..."
    m = re.search(r"\b(?:option|answer|choice)\s*[:\-]?\s*([abcd])\b", ntext)
    if m:
        return m.group(1).upper(), "letter_kw"
    m = re.match(r"^\(?\s*([abcd])\s*[\)\.\:]\s+\S", ntext)
    if m:
        return m.group(1).upper(), "letter_prefix"

    # 4) an option's text is contained in the output (prefer the longest)
    matches = [(L, len(no)) for L, no in norm_opts.items() if no and no in ntext]
    if matches:
        matches.sort(key=lambda x: -x[1])
        return matches[0][0], "option_text"

    # 5) free-text conflict / abstain phrasing -> D
    if any(h in ntext for h in CONFLICT_HINTS):
        return CONFLICT_LETTER, "conflict_text"

    # 6) last resort: a standalone single A-D token
    m = re.search(r"(?:^|\s)([abcd])(?:\s|$)", ntext)
    if m:
        return m.group(1).upper(), "letter_loose"

    return None, "unparsed"


def classify(raw: str, mcq: dict) -> dict:
    letter, method = parse_answer(raw, mcq["letter_to_text"])
    if letter is None or letter not in mcq["letter_to_cat"]:
        category, chosen_text = "other", ""
    else:
        category = mcq["letter_to_cat"][letter]
        chosen_text = mcq["letter_to_text"][letter]
    return {
        "chosen_letter": letter,
        "parse_method": method,
        "category": category,
        "chosen_text": chosen_text,
    }
