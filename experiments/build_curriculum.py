"""Sort BabyLM training utterances simple→complex to create a curriculum dataset.

Complexity metric: mean log-rank of content words in the utterance.
Content words = tokens whose majority POS tag is not in the structural/closed-class
set. Rank comes from the training vocabulary frequency order (rank 1 = most frequent).
Utterance length (words) is a secondary sort key.

Writes:
  experiments/data/babylm_curriculum/train/<domain>.txt  — one utterance per line,
                                                           sorted simple first
  experiments/data/babylm_curriculum/dev/<domain>.txt    — dev split, same sorting
  experiments/data/babylm_curriculum/complexity_report.txt

Usage:
  python experiments/build_curriculum.py
  python experiments/build_curriculum.py --data_dir experiments/data/babylm_simple \
      --out_dir experiments/data/babylm_curriculum
"""
import argparse
import math
import os
import re
import sys

import nltk

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "jussipyutils"))

DOMAINS = [
    "aochildes", "children_stories", "cbt",
    "simple_wikipedia", "open_subtitles", "switchboard", "bnc_spoken",
]

# closed-class POS tags — tokens with these tags are NOT content words
_CLOSED_CLASS = {
    "IN","TO","DT","CC","PRP","PRP$","RP","PDT","WDT","WP","WP$","WRB",
    "MD","EX","POS","SYM","LS","UH","FW",
}

_TOK = re.compile(r"[a-z0-9]+|[''']s|[''']t|[''']ve|[''']re|[''']ll|[''']d|[^\w\s]")


def tokenize(line: str):
    return _TOK.findall(line.lower())


def build_rank_table(stoi: dict) -> dict:
    """Return {token: log_rank} where rank 1 = most frequent token in vocab.

    stoi from meta.pkl is {word: id}; ids are assigned in frequency order
    (id 0 = <unk>, id 1 = <eos>, id 2 = most frequent real word, …).
    Log-rank of the real words starts at id 2.
    """
    specials = {"<unk>", "<eos>"}
    rank_table = {}
    real_words = [(w, i) for w, i in stoi.items() if w not in specials]
    # sort by id ascending = frequency descending (prepare_words assigns ids by freq)
    real_words.sort(key=lambda x: x[1])
    for rank_0, (w, _) in enumerate(real_words):
        rank_table[w] = math.log1p(rank_0 + 1)  # log(rank+1), 1-indexed
    return rank_table


def utterance_complexity(tokens: list, rank_table: dict, pos_tags: list) -> float:
    """Mean log-rank of content words; falls back to total length if no content words."""
    content_ranks = []
    for tok, tag in zip(tokens, pos_tags):
        if tag not in _CLOSED_CLASS and tok.isalpha():
            r = rank_table.get(tok, math.log1p(15000))  # OOV = max rank
            content_ranks.append(r)
    if content_ranks:
        return sum(content_ranks) / len(content_ranks)
    return len(tokens)  # fallback: short utterances are simpler


def sort_file(path: str, rank_table: dict) -> list:
    """Read a corpus file, split into utterances, return sorted list of strings."""
    utterances = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sents = nltk.sent_tokenize(line)
            for sent in sents:
                sent = sent.strip()
                if not sent:
                    continue
                toks = tokenize(sent)
                if not toks:
                    continue
                tags = [tag for _, tag in nltk.pos_tag(toks)]
                score = utterance_complexity(toks, rank_table, tags)
                utterances.append((score, len(toks), sent))
    utterances.sort(key=lambda x: (x[0], x[1]))
    return [u[2] for u in utterances], [u[0] for u in utterances]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default=os.path.join(_ROOT, "experiments", "data", "babylm_simple"))
    ap.add_argument("--out_dir",
                    default=os.path.join(_ROOT, "experiments", "data", "babylm_curriculum"))
    ap.add_argument("--meta",
                    default=os.path.join(_ROOT, "experiments", "data",
                                         "babylm_simple", "meta.pkl"))
    args = ap.parse_args()

    import pickle
    with open(args.meta, "rb") as f:
        meta = pickle.load(f)
    rank_table = build_rank_table(meta["stoi"])
    print(f"Rank table built: {len(rank_table)} entries")

    report_lines = []

    for split in ("train", "dev"):
        in_dir  = os.path.join(args.data_dir, split if split == "train" else "dev")
        out_dir = os.path.join(args.out_dir, split)
        os.makedirs(out_dir, exist_ok=True)

        for domain in DOMAINS:
            in_path  = os.path.join(in_dir, f"{domain}.txt")
            out_path = os.path.join(out_dir, f"{domain}.txt")
            if not os.path.exists(in_path):
                print(f"  skip (missing): {in_path}")
                continue
            print(f"  sorting {split}/{domain} ...")
            sorted_utts, scores = sort_file(in_path, rank_table)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(sorted_utts) + "\n")
            n = len(scores)
            if n:
                lo, hi, mid = scores[0], scores[-1], scores[n // 2]
                report_lines.append(
                    f"{split:5s} {domain:20s}  {n:6d} utts  "
                    f"min={lo:.2f}  med={mid:.2f}  max={hi:.2f}"
                )
            print(f"    {n} utterances written, complexity {scores[0]:.2f}->{scores[-1]:.2f}")

    report_path = os.path.join(args.out_dir, "complexity_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nReport written to {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
