# Revised findings

## Information-matched main result

The deployable baseline first predicts current emotion with a train-only linear classifier. The
learned model receives causal features but no gold label. All six models have positive point-estimate
deltas on the four primary text corpora. DailyDialog gains are +0.160 to +0.167 ROC-AUC and all
survive the 72-comparison Holm correction; DialogueRNN's +0.041 MELD gain also survives.

The oracle diagnostic gives gold current emotion to both model and transition baseline. Under this
matched information, IEMOCAP is tied, the transition baseline is stronger on MELD and directionally
stronger on EmoryNLP, while every DailyDialog model retains a corrected-significant +0.106 to
+0.132 advantage.

The valid conclusion is conditional: current-label availability and corpus transition structure can
reverse the model-versus-inertia comparison.

## Independent evidence

- Train-only utterance-local TF-IDF/SVD preserves model gains on IEMOCAP (+0.089), EmoryNLP
  (+0.061), and DailyDialog (+0.164); MELD is tied.
- The causal state-factorized model improves on the deployable baseline on all four text corpora;
  MELD and DailyDialog survive its separate four-test Holm family.
- The complete self-shift benchmark shows all six DailyDialog models improve by +0.114 to +0.129;
  all survive a separate 24-comparison Holm correction. Other corpora are not corrected-significant.
- With one fixed Transformer, MELD and DailyDialog peak at `k=1` future utterance and decline as
  more future is exposed. Future leakage is material but not generally monotonic.
- DailyDialog's best ROC-AUC/PR-AUC does not imply calibrated risk: DialogueGCN Brier/ECE are
  0.188/0.224 versus 0.153/0.052 for the weaker transition baseline.

## Scope

The six systems are causal family/strategy re-implementations under an equal but limited validation
search. MM-DFN rows are sensitivity analyses because complete upstream producers are unavailable.
The evidence does not support broad claims that architecture never matters, that inertia always
dominates, or that more future context always increases AUC.
