# Representative method families: causal re-implementations

## Why re-implement instead of using the original repos

DialogueRNN, DialogueGCN, and DAG-ERC are bidirectional ERC (recognition) models — they read
the whole dialogue, so the representation at utterance n already incorporates n+1. Run as-is
they leak the target, and their original codebases have conflicting, hard-to-reproduce
dependencies. Instead we re-implement each family's defining mechanism, made strictly causal
(position n uses only x_<=n and speaker_<=n), inside this repo's leakage-safe harness. This
asks whether the family's mechanism helps under a correct protocol, not whether we can
reproduce a leaky pipeline's published numbers — the two are different questions, and we
don't claim to answer the second one.

## The three families (src/models_families.py)
- **CausalDialogueRNN** — speaker-state recurrence: global GRU + per-party GRU + emotion GRU,
  unidirectional. The party state of the *current* speaker is updated from the utterance + global
  context; emotion GRU reads the party state. (declare-lab/conv-emotion mechanism.)
- **CausalDialogueGCN** — windowed, speaker-relational graph convolution: each node aggregates
  over past neighbours within a window with separate transforms for same- vs different-speaker
  edges; 2 layers. Causal (past-only window).
- **CausalDAGERC** — directed (past-only) speaker-relational attention: each node attends to all
  preceding nodes with a same/different-speaker bias; 2 layers. DAG-ERC is naturally backward-
  looking, so the causal adaptation is faithful.

All share the harness's speaker plumbing (per-dialogue local party index, capped at 9) and emit
shift logits [B,T]; evaluated identically (AUC + per-class + dialogue-bootstrap CI + paired
cluster-robust test vs the transition matrix).

## Caveat to write
These are faithful *mechanism* re-implementations, not the original code; numbers are not
comparable to published ERC accuracies (different task: causal, target-absent, binary shift).
