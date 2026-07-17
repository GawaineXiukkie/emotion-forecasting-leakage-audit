# Feature provenance and locality audit

This document records exactly what can and cannot be established about the
precomputed features used in the IEEE Access experiments. Repository revisions
were inspected locally on 2026-07-16. The cloned upstream repositories are not
redistributed in the submission package.

## COSMIC / `declare-lab/conv-emotion`

- Upstream repository: <https://github.com/declare-lab/conv-emotion>
- Inspected revision: `6128ca20e9c736605cce7e99d5d95db0356c35f5`
- Relevant producer scripts:
  - `COSMIC/feature-extraction/roberta_feature_extract_iemocap.py`
  - `COSMIC/feature-extraction/roberta_feature_extract_meld.py`
  - `COSMIC/feature-extraction/roberta_feature_extract_emorynlp.py`
- The scripts put RoBERTa in evaluation mode, encode each utterance string
  separately with `roberta.encode(s)`, batch those independently encoded
  sequences only for computation, and retain the utterance representation.
  No neighbouring utterance, dialogue identifier, emotion label, or future
  utterance is passed to the encoder for an utterance row.
- DailyDialog uses the corresponding released COSMIC feature file and the same
  pickle schema. Its producer script is not present at the inspected path, so
  the paper distinguishes direct code verification for IEMOCAP/MELD/EmoryNLP
  from schema-and-output checks for DailyDialog.
- The COSMIC paper independently describes these inputs as
  context-independent utterance-level RoBERTa vectors:
  <https://aclanthology.org/2020.findings-emnlp.224/>.

The local audit in `src/access_data_quality.py` additionally verifies finite
values, one feature row per utterance, split-ID separation, and consistency of
features for repeated byte-identical utterance strings. Repeated COSMIC strings
agree within `1.13e-4`; the audit uses a conservative `1e-3` tolerance for
floating-point differences caused by batch padding/shape. This is corroborating
output evidence, not a substitute for inspecting the producer.

## MM-DFN / `zerohd4869/MM-DFN`

- Upstream repository: <https://github.com/zerohd4869/MM-DFN>
- Inspected revision: `da970366069247e05de3b9298f1e1bbc5c77a187`
- The upstream README identifies utterance-level TextCNN+GloVe text, OpenSmile
  audio, and DenseNet visual features.
- The inspected repository distributes processed pickle files but does not
  include a complete, executable producer for all three modalities. Therefore
  the absence of dialogue-level or future context cannot be certified from
  producer code to the same standard as COSMIC.

The local files pass shape, finite-value, split-ID, and COSMIC/MM-DFN
text-and-label alignment checks. However, repeated identical MELD strings can
have materially different MM-DFN text vectors (`max |delta| = 1.54`). This can
arise from context-dependent preprocessing, speaker/show-specific processing,
or another unrecorded upstream factor. It is not by itself proof of target
leakage, but it prevents a strict utterance-locality claim. Accordingly:

1. MM-DFN configurations are reported as externally precomputed-feature
   sensitivity analyses, not as the strongest leakage-certified evidence.
2. Headline conclusions are based on the COSMIC text configurations.
3. `src/access_local_feature_control.py` independently fits a train-only
   TF-IDF (unigrams and bigrams) plus truncated-SVD representation and transforms
   every utterance separately. This control has no upstream learned feature
   producer and therefore supplies a fully local text-feature replication.

## Integrity and alignment checks

Run:

```bash
python -m src.access_data_quality
```

Machine-readable results are written to `results/access_data_quality.json`; the
human-readable table is `results/access_data_quality.md`. The audit covers:

- split identifier overlap;
- array dimensionality, sequence-length agreement, label range, and non-finite
  values;
- exact train/test dialogue duplication;
- byte-identical utterance feature consistency; and
- ID, text, label, and length alignment between COSMIC and MM-DFN versions of
  IEMOCAP and MELD (after reuniting the COSMIC train and validation IDs).

The checks are read-only. DailyDialog's exact train/test duplicates are removed
only in the explicitly named decontaminated headline configuration; original
split results remain separately labeled for comparability.

Speaker identifiers are handled separately from feature provenance. IEMOCAP actor
IDs are made global from session and gender. One-hot or scalar role indices in
MELD, EmoryNLP, and DailyDialog are dialogue-local, so the loader prefixes them
with the dialogue ID. This prevents identically numbered roles in unrelated
dialogues from being pooled by the transition baseline.
