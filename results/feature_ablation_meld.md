# MELD feature-source ablation (paper Table III)

Safe GRU, shift-AUC. The gain over RoBERTa text is carried entirely by MM-DFN's text
channel; in this logged ablation, audio/visual channels do not improve over MM-DFN
text-only.

| Features | AUC (none) |
|---|---|
| COSMIC RoBERTa text (1024-d) | 0.554 |
| MM-DFN text-only (600-d) | 0.724 |
| MM-DFN text+audio | 0.722 |
| MM-DFN text+visual | 0.705 |
| MM-DFN text+audio+visual | 0.712 |

Reported in the paper as a hypothesis, not a conclusion (two candidate explanations —
task-adapted lower-dimensional encoding vs. non-verified preprocessing differences between
the two original feature pipelines — are not adjudicated between; see `docs/e3_feature_ablation.md`
for the full discussion and source logs).
