"""Train a minimal GPT with a chosen input encoding and log train/val loss curves.

Works for both the char-level (shakespeare_char) and word-level (babylm_simple)
datasets; select with --data_dir.

Key features over a plain nanoGPT training loop:
  - Frozen-embedding injection with per-category partial freezing
    (frozen categories: structural + adv_clausal by default)
  - BPC (bits-per-character) logging for cross-tokenizer comparison
  - --max_train_tokens to cap the training slice for low-data sweeps
  - Output head always untied from input embedding (unless --no_untie)

Examples:
  # char-level baseline (uses default DATA_DIR = shakespeare_char)
  python experiments/train_experiment.py --encoding learned

  # word-level category-encoding run (uses babylm_simple data)
  python experiments/train_experiment.py \\
      --encoding word_category \\
      --data_dir experiments/data/babylm_simple \\
      --freeze_categories 2,4 --overlay_alpha 0.5 --tag word_category
"""
import argparse
import json
import math
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "nanoGPT"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import encoders  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

_CHAR_DATA_DIR = os.path.join(_ROOT, "nanoGPT", "data", "shakespeare_char")


# ── data ────────────────────────────────────────────────────────────────────

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, eval_iters, device,
                  char_counts=None):
    """Return dict with train_loss, val_loss, and (if char_counts given) val_bpc."""
    model.eval()
    out = {}
    for name, data in splits.items():
        total_loss = 0.0
        for _ in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            total_loss += loss.item()
        out[name] = total_loss / eval_iters
    # BPC: total NLL in nats over val, divided by char_count * ln2
    if char_counts and "val" in char_counts and char_counts["val"] > 0:
        val_data = splits["val"]
        # approximate total tokens as len(val_data); loss is per-token NLL
        total_nll_nats = out["val"] * len(val_data)
        out["val_bpc"] = total_nll_nats / (char_counts["val"] * math.log(2))
    model.train()
    return out


# ── model ────────────────────────────────────────────────────────────────────

def get_lr(it, max_iters, lr, lr_min, n_phases):
    """Cosine decay with warm restarts at equal-length phase boundaries.

    n_phases=1 : standard cosine decay from lr to lr_min over max_iters.
    n_phases>1 : divide training into n_phases equal segments; at each
                 boundary reset LR to lr and cosine-decay to lr_min within
                 that segment. Motivated by curriculum ordering: each new
                 complexity phase deserves fresh plasticity.
    """
    phase_len = max_iters / n_phases
    phase_pos = it % phase_len          # position within current phase
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * phase_pos / phase_len))


def build_model(cfg, encoding_matrix, freeze_all, untie, device,
                freeze_row_mask=None, unfreeze_iter=None, unfreeze_state=None):
    """Build GPT, optionally inject encoding_matrix into wte.

    freeze_all      : freeze the entire wte (all rows)
    freeze_row_mask : 1-D bool tensor of length vocab_size; True = frozen row.
                      Used for per-category partial freezing.
                      If set, freeze_all is ignored.
    unfreeze_iter   : if set (with unfreeze_state), the frozen rows are released once
                      unfreeze_state["iter"] >= unfreeze_iter (staged unfreeze). Grads
                      flow to all rows from then on; requires_grad stays True throughout.
    """
    model = GPT(cfg)
    if untie:
        # untie BEFORE injecting so that freezing input does not freeze output
        model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())

    if encoding_matrix is not None:
        with torch.no_grad():
            model.transformer.wte.weight.copy_(encoding_matrix)

        if freeze_row_mask is not None:
            # per-category partial freeze via backward hook
            frozen_rows = freeze_row_mask.to(device)  # bool, shape (vocab,)
            keep = (~frozen_rows).float().unsqueeze(1)  # (vocab, 1)
            def _grad_mask_hook(grad):
                # staged unfreeze: once past the boundary, let all gradients through
                if (unfreeze_iter is not None and unfreeze_state is not None
                        and unfreeze_state["iter"] >= unfreeze_iter):
                    return grad
                return grad * keep
            model.transformer.wte.weight.register_hook(_grad_mask_hook)
        elif freeze_all:
            model.transformer.wte.weight.requires_grad_(False)

    return model.to(device)


def build_freeze_mask(vocab, word_categories, freeze_cat_ids):
    """Return bool tensor: True for rows that should be frozen."""
    mask = torch.zeros(len(vocab), dtype=torch.bool)
    if not freeze_cat_ids or word_categories is None:
        return mask
    for i, w in enumerate(vocab):
        if word_categories.get(w, 0) in freeze_cat_ids:
            mask[i] = True
    n_frozen = mask.sum().item()
    print(f"  per-category freeze: {n_frozen}/{len(vocab)} rows frozen "
          f"(categories {freeze_cat_ids})")
    return mask


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", default="learned", choices=encoders.AVAILABLE)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--data_dir", default=_CHAR_DATA_DIR,
                    help="directory with train.bin, val.bin, meta.pkl")
    # model
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=128)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.0)
    # training
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_iters", type=int, default=2000)
    ap.add_argument("--eval_interval", type=int, default=100)
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_min", type=float, default=1e-4,
                    help="minimum LR at the bottom of each cosine phase (default 1e-4)")
    ap.add_argument("--n_phases", type=int, default=1,
                    help="cosine-restart phases (1=single decay, 3=three restarts). "
                         "With curriculum data, set to match complexity phases.")
    ap.add_argument("--weight_decay", type=float, default=1e-1)
    ap.add_argument("--max_train_tokens", type=int, default=None,
                    help="cap training data for low-data sweep")
    # embedding control
    ap.add_argument("--freeze", action="store_true",
                    help="freeze entire injected embedding (all rows)")
    ap.add_argument("--unfreeze_at_frac", type=float, default=None,
                    help="staged unfreeze: release the frozen rows after this fraction "
                         "of max_iters (e.g. 0.5). Needs --freeze_categories.")
    ap.add_argument("--freeze_categories", default="2,4,5,6,7",
                    help="comma-separated category ids to freeze (word-level only); "
                         "empty string disables. Default: '2,4,5,6,7' "
                         "(preposition+adv_clausal+coord_conj+subord_conj+determiner)")
    ap.add_argument("--no_untie", action="store_true")
    # encoding knobs
    ap.add_argument("--denseness", type=int, default=10)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--enc_tokens", type=int, default=500_000,
                    help="tokens to use for word co-occurrence (word-level)")
    ap.add_argument("--enc_chars", type=int, default=200_000,
                    help="chars to use for char co-occurrence (char-level)")
    ap.add_argument("--overlay_alpha", type=float, default=0.5,
                    help="category overlay strength (word_category encoding)")
    ap.add_argument("--frame_alpha", type=float, default=0.3,
                    help="strength of verb subcat-frame overlay (word_frame)")
    ap.add_argument("--levin_alpha", type=float, default=0.3,
                    help="strength of Levin semantic-class overlay (word_levin)")
    ap.add_argument("--tma_alpha", type=float, default=0.3,
                    help="TMA overlay strength (word_category encoding)")
    ap.add_argument("--emb_scale", type=float, default=0.02)
    # misc
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir",
                    default=os.path.join(_ROOT, "experiments", "runs"))
    ap.add_argument("--save_checkpoint", action="store_true",
                    help="save model.pt + config.json after training (needed for BLiMP eval)")
    args = ap.parse_args()

    tag = args.tag or args.encoding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ── load data ────────────────────────────────────────────────────────────
    with open(os.path.join(args.data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta["vocab_size"]
    vocab = [meta["itos"][i] for i in range(vocab_size)]
    word_categories = meta.get("word_categories", None)
    tma_categories  = meta.get("tma_distributions", meta.get("tma_categories", None))
    frame_categories = meta.get("frame_distributions", None)
    levin_categories = meta.get("levin_classes", None)
    char_counts = {
        "train": meta.get("train_char_count", 0),
        "val":   meta.get("val_char_count", 0),
    }

    train_data = np.memmap(os.path.join(args.data_dir, "train.bin"),
                           dtype=np.uint16, mode="r")
    val_data   = np.memmap(os.path.join(args.data_dir,   "val.bin"),
                           dtype=np.uint16, mode="r")
    if args.max_train_tokens:
        train_data = train_data[:args.max_train_tokens]
        # scale char count proportionally for BPC
        ratio = len(train_data) / max(1, meta.get("train_char_count", 1))
        char_counts["train"] = int(args.max_train_tokens * ratio)

    splits = {"train": train_data, "val": val_data}

    # ── build encoding ───────────────────────────────────────────────────────
    is_word = args.encoding in encoders.WORD_ENCODINGS
    print(f"[{tag}] building encoding '{args.encoding}' ...")
    t0 = time.time()

    if is_word:
        enc_matrix = encoders.build_word(
            args.encoding, vocab, args.n_embd,
            token_ids=np.array(train_data, dtype=np.int32),
            word_categories=word_categories,
            tma_categories=tma_categories,
            frame_categories=frame_categories,
            levin_categories=levin_categories,
            denseness=args.denseness, window=args.window,
            overlay_alpha=args.overlay_alpha,
            tma_alpha=args.tma_alpha,
            frame_alpha=args.frame_alpha,
            levin_alpha=args.levin_alpha,
            enc_tokens=args.enc_tokens,
            emb_scale=args.emb_scale, seed=args.seed)
    else:
        # char-level path: needs raw text
        text_path = os.path.join(args.data_dir, "input.txt")
        text = open(text_path, encoding="utf-8").read() if os.path.exists(text_path) else ""
        enc_matrix = encoders.build(
            args.encoding, vocab, args.n_embd, text=text,
            denseness=args.denseness, window=args.window,
            enc_chars=args.enc_chars, emb_scale=args.emb_scale, seed=args.seed)

    frozen_desc = "trainable"
    if enc_matrix is not None:
        frozen_desc = "frozen (all)" if args.freeze else "injected (trainable)"
    print(f"[{tag}] encoding ready in {time.time()-t0:.1f}s ({frozen_desc})")

    # ── per-category freeze mask (word-level only) ───────────────────────────
    freeze_row_mask = None
    freeze_cat_ids = set()
    if is_word and args.freeze_categories and enc_matrix is not None:
        try:
            freeze_cat_ids = {int(x) for x in args.freeze_categories.split(",") if x.strip()}
        except ValueError:
            pass
        if freeze_cat_ids:
            freeze_row_mask = build_freeze_mask(vocab, word_categories, freeze_cat_ids)

    # ── build model ───────────────────────────────────────────────────────────
    cfg = GPTConfig(block_size=args.block_size, vocab_size=vocab_size,
                    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                    dropout=args.dropout, bias=False)
    unfreeze_state = {"iter": 0}
    unfreeze_iter = (int(args.unfreeze_at_frac * args.max_iters)
                     if args.unfreeze_at_frac is not None else None)
    model = build_model(cfg, enc_matrix, args.freeze, not args.no_untie,
                        args.device, freeze_row_mask=freeze_row_mask,
                        unfreeze_iter=unfreeze_iter, unfreeze_state=unfreeze_state)
    if unfreeze_iter is not None:
        print(f"[{tag}] staged unfreeze: frozen rows released at iter {unfreeze_iter} "
              f"(frac {args.unfreeze_at_frac})")

    # ── optimizer (embedding rows in separate group with wd=0 if partially frozen)
    if freeze_row_mask is not None and freeze_row_mask.any():
        emb_params = [model.transformer.wte.weight]
        other_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and "wte" not in n]
        param_groups = [
            {"params": other_params, "weight_decay": args.weight_decay},
            {"params": emb_params,   "weight_decay": 0.0},
        ]
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        param_groups = [{"params": trainable, "weight_decay": args.weight_decay}]

    n_trainable = sum(p.numel() for g in param_groups for p in g["params"])
    print(f"[{tag}] trainable params: {n_trainable/1e6:.3f}M")
    if args.n_phases > 1:
        phase_len = args.max_iters / args.n_phases
        print(f"[{tag}] LR schedule: {args.n_phases} cosine phases, "
              f"each {phase_len:.0f} iters, "
              f"lr {args.lr}->{args.lr_min} per phase")
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.99))

    def save_ckpt(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
            json.dump({"vocab_size": vocab_size, "n_layer": args.n_layer,
                       "n_head": args.n_head, "n_embd": args.n_embd,
                       "block_size": args.block_size, "bias": False,
                       "data_dir": args.data_dir, "tag": tag}, f, indent=2)

    # ── training loop ────────────────────────────────────────────────────────
    history = []
    t0 = time.time()
    for it in range(args.max_iters + 1):
        unfreeze_state["iter"] = it
        # save a checkpoint at the unfreeze boundary (for the U-curve probe)
        if (unfreeze_iter is not None and it == unfreeze_iter
                and args.save_checkpoint):
            save_ckpt(os.path.join(args.out_dir, f"{tag}_mid_ckpt"))
            print(f"[{tag}] mid checkpoint saved at unfreeze boundary (iter {it})")
        # apply LR schedule
        if args.n_phases > 1 or args.lr_min != args.lr:
            lr_now = get_lr(it, args.max_iters, args.lr, args.lr_min, args.n_phases)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(model, splits, args.block_size, args.batch_size,
                                   args.eval_iters, args.device, char_counts)
            entry = {"iter": it, "train_loss": losses["train"],
                     "val_loss": losses["val"], "elapsed_s": time.time() - t0}
            if "val_bpc" in losses:
                entry["val_bpc"] = losses["val_bpc"]
            if args.n_phases > 1:
                phase_now = int(it // (args.max_iters / args.n_phases)) + 1
                lr_now = get_lr(it, args.max_iters, args.lr, args.lr_min, args.n_phases)
                entry["phase"] = min(phase_now, args.n_phases)
                entry["lr"] = lr_now
            history.append(entry)
            bpc_str = f" | bpc {losses['val_bpc']:.4f}" if "val_bpc" in losses else ""
            phase_str = (f" | phase {min(int(it//(args.max_iters/args.n_phases))+1, args.n_phases)}"
                         if args.n_phases > 1 else "")
            print(f"[{tag}] iter {it:5d}{phase_str} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f}{bpc_str} | {time.time()-t0:.0f}s")
        if it == args.max_iters:
            break
        x, y = get_batch(train_data, args.block_size, args.batch_size, args.device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # ── save ─────────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "tag": tag,
            "config": vars(args),
            "n_trainable": n_trainable,
            "final": history[-1],
            "history": history,
        }, f, indent=2)
    print(f"[{tag}] wrote {out_path}")

    if args.save_checkpoint:
        ckpt_dir = os.path.join(args.out_dir, f"{tag}_ckpt")
        save_ckpt(ckpt_dir)
        print(f"[{tag}] checkpoint saved to {ckpt_dir}")


if __name__ == "__main__":
    main()
