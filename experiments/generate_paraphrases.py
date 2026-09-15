"""Generate child-directed paraphrases for simple utterances using the Claude API.

Targets utterances from the simple end of the curriculum (bottom --pct_simple %)
that contain at least one referential word (noun, pronoun, or adjective).

Output:
  experiments/data/babylm_paraphrase/train/paraphrases.txt
      — original utterance followed by its N paraphrases, blank-line separated

Requires:
  - ANTHROPIC_API_KEY environment variable
  - anthropic package: pip install anthropic

Usage:
  python experiments/generate_paraphrases.py --dry_run        # preview prompts
  python experiments/generate_paraphrases.py --max_utterances 500

  # After generation, combine with curriculum data and run prepare_words.py:
  # python experiments/prepare_words.py \\
  #     --data_dir experiments/data/babylm_paraphrase \\
  #     --out_dir  experiments/data/babylm_paraphrase
"""
import argparse
import json
import os
import re
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "jussipyutils"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nltk
from categorize import REFERENTIAL, _word_category_from_lexicon_and_tag

CURRICULUM_TRAIN = os.path.join(
    _ROOT, "experiments", "data", "babylm_curriculum", "train")
OUT_DIR = os.path.join(
    _ROOT, "experiments", "data", "babylm_paraphrase", "train")
CACHE_FILE = os.path.join(
    _ROOT, "experiments", "data", "babylm_paraphrase", "cache.json")

DOMAINS = [
    "aochildes", "children_stories", "cbt",
    "simple_wikipedia", "open_subtitles", "switchboard", "bnc_spoken",
]

_TOK = re.compile(r"[a-z0-9]+|[''']s|[''']t|[''']ve|[''']re|[''']ll|[''']d|[^\w\s]")

SYSTEM_PROMPT = """You generate child-directed speech paraphrases.
For each input utterance, produce exactly {n} short, simple alternative ways an adult might say the same thing to a young child (age 1-3).
Guidelines:
- Short sentences (under 10 words each)
- Use attention words where natural: "look!", "see?", "oh!", "hey!"
- Use child-friendly forms where natural: kitty, doggy, birdy, choo-choo, tummy, etc.
- Vary the form: sometimes a statement, sometimes a question, sometimes an imperative
- Stay close to the original meaning
Output: exactly {n} lines, one paraphrase per line, no numbering, no extra text."""

USER_TEMPLATE = 'Paraphrase this utterance for a young child: "{utterance}"'


def has_referential(tokens):
    """Return True if the utterance contains at least one referential word."""
    for tok in tokens:
        if _word_category_from_lexicon_and_tag(tok.lower(), None) == REFERENTIAL:
            return True
    return False


def load_cache(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def generate_paraphrases(utterance: str, n: int, client, model: str,
                          dry_run: bool = False) -> list:
    if dry_run:
        print(f"  [DRY RUN] Would paraphrase: {utterance!r}")
        return [f"[paraphrase {i+1} of {utterance!r}]" for i in range(n)]
    msg = client.messages.create(
        model=model,
        max_tokens=200,
        system=SYSTEM_PROMPT.format(n=n),
        messages=[{"role": "user", "content": USER_TEMPLATE.format(utterance=utterance)}],
    )
    text = msg.content[0].text.strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[:n]


def collect_utterances(pct_simple: float, max_utterances: int):
    """Load curriculum-sorted files and return the simple-end utterances."""
    all_utts = []
    for domain in DOMAINS:
        path = os.path.join(CURRICULUM_TRAIN, f"{domain}.txt")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [l.strip() for l in f if l.strip()]
        n_simple = max(1, int(len(lines) * pct_simple / 100))
        all_utts.extend(lines[:n_simple])
    # filter to those with at least one referential word
    filtered = []
    for utt in all_utts:
        toks = _TOK.findall(utt.lower())
        if len(toks) >= 3 and has_referential(toks):
            filtered.append(utt)
    # cap
    if max_utterances and len(filtered) > max_utterances:
        filtered = filtered[:max_utterances]
    return filtered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct_simple", type=float, default=30.0,
                    help="bottom %% of curriculum to use as paraphrase source")
    ap.add_argument("--n_paraphrases", type=int, default=2,
                    help="paraphrases to generate per utterance")
    ap.add_argument("--max_utterances", type=int, default=2000,
                    help="max utterances to paraphrase (0 = no limit)")
    ap.add_argument("--model", default="claude-haiku-4-5",
                    help="Claude model to use")
    ap.add_argument("--dry_run", action="store_true",
                    help="print prompts without calling API")
    ap.add_argument("--rate_limit_sleep", type=float, default=0.5,
                    help="seconds to sleep between API calls")
    args = ap.parse_args()

    if not args.dry_run:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: set ANTHROPIC_API_KEY environment variable")
            sys.exit(1)
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    else:
        client = None

    os.makedirs(OUT_DIR, exist_ok=True)
    cache = load_cache(CACHE_FILE)

    utterances = collect_utterances(args.pct_simple, args.max_utterances)
    print(f"Collected {len(utterances)} utterances to paraphrase "
          f"({args.pct_simple}% simple end, referential filter)")

    out_path = os.path.join(OUT_DIR, "paraphrases.txt")
    n_generated, n_cached, n_errors = 0, 0, 0

    with open(out_path, "w", encoding="utf-8") as out:
        for i, utt in enumerate(utterances):
            if utt in cache:
                paraphrases = cache[utt]
                n_cached += 1
            else:
                try:
                    paraphrases = generate_paraphrases(
                        utt, args.n_paraphrases, client, args.model, args.dry_run)
                    cache[utt] = paraphrases
                    n_generated += 1
                    if not args.dry_run and n_generated % 50 == 0:
                        save_cache(cache, CACHE_FILE)
                    if not args.dry_run:
                        time.sleep(args.rate_limit_sleep)
                except Exception as e:
                    print(f"  ERROR on utterance {i}: {e}")
                    n_errors += 1
                    continue

            out.write(utt + "\n")
            for p in paraphrases:
                out.write(p + "\n")
            out.write("\n")

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(utterances)} done "
                      f"(generated {n_generated}, cached {n_cached}, errors {n_errors})")

    save_cache(cache, CACHE_FILE)
    print(f"\nWrote {out_path}")
    print(f"Generated: {n_generated}, cached: {n_cached}, errors: {n_errors}")
    print("\nNext step: build training data including paraphrases:")
    print("  Copy curriculum train files to experiments/data/babylm_paraphrase/train/")
    print("  Then: python experiments/prepare_words.py \\")
    print("            --data_dir experiments/data/babylm_paraphrase \\")
    print("            --out_dir  experiments/data/babylm_paraphrase")


if __name__ == "__main__":
    main()
