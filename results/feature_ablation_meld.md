# MELD feature-source ablation (paper Table III)

Safe GRU, shift-AUC. The gain over RoBERTa text is carried entirely by MM-DFN's text channel; audio/visual channels do not improve over MM-DFN text-only in this ablation.

| Features | AUC (none) |
|---|---|
| COSMIC RoBERTa text (1024-d) | 0.554 |
| MM-DFN text-only (600-d) | 0.724 |
| MM-DFN text+audio (600+300-d) | 0.722 |
| MM-DFN text+visual (600+342-d) | 0.705 |
| MM-DFN text+audio+visual (600+300+342-d) | 0.712 |

Reported in the paper as a hypothesis, not a conclusion. Two candidate explanations for the gain, neither adjudicated between:
- MM-DFN's 600-d text encoding is task-adapted (trained on MELD-adjacent data), while COSMIC's 1024-d RoBERTa embedding is frozen and generic -- a task-adapted, lower-dimensional representation could simply carry a cleaner shift signal.
- The two pipelines' text preprocessing was never verified to be identical (tokenization, speaker-tag handling, punctuation normalization); a preprocessing difference could explain some or all of the gap independent of the embedding method.

We did not run a controlled re-extraction to distinguish these.
