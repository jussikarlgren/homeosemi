# How to resume this work after a shutdown

Shutting the computer down loses nothing. Two separate things are saved to disk:

1. **The Claude Code session transcript** — the full conversation, replayable by the CLI.
2. **The project context** — `memory/MEMORY.md`, `memory/project_open_decision.md`, and `NOTES.md`.
   Even a brand-new session reconstructs the state of the work from these.

## The one thing that matters: start from THIS directory

Claude Code scopes sessions and memory to the working directory. Open a terminal **in this folder**
before launching:

```
cd "C:\Users\jkarlgre\OneDrive - Advanced Micro Devices Inc\monrepos\topology"
```

## Option A — resume this exact conversation (preferred)

```
claude --continue          # reopens the most recent session in this directory
```

or pick from a list of past sessions:

```
claude --resume            # shows a picker; choose this topology session
```

Both read the saved transcript, so everything we discussed is back verbatim.

## Option B — fresh session, same context (fallback)

If the transcript picker doesn't show it (different machine, cleared history, etc.), just start
Claude here and tell it to pick up:

```
claude
```
then type:  **"check your memory and NOTES.md, then resume the topology work"**

The auto-memory system loads `memory/MEMORY.md` automatically; that plus `NOTES.md` carries the full
state. You will not have to re-explain the project.

## Where things stand right now (2026-10-12)

- **Model:** basket-of-balls semantic space. Latest trained model = `topospace_all.pkl`
  (5 novels, 253 balls). Rebuild with `python traintopospace.py data/*.txt topospace_all.pkl`.
- **PINNED decision:** coherence-gated ball-splitting — see `NOTES.md`. The principal-direction *cut*
  is implemented; the coherence *gate* is an open problem (no linear routing-space metric works).
- **Just done:** SimLex-999 / WordSim-353 evaluation (`python evaltopospace.py topospace_all.pkl`).
  Result: coverage-bound — only 1.7–3.5% of pairs answered (vocabulary mismatch + over-fragmented
  balls). Correlations on answered pairs are too small to trust.
- **Next step on the table:** get a general-domain corpus (e.g. a Wikipedia subset) to raise benchmark
  coverage; and/or return to the pinned coherence-gate problem, which is measurably costing coverage.

## Reminder

To have Claude run a shell command for you inside a session (e.g. an interactive login), type `!`
followed by the command at the prompt.
