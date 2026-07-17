# Access feature and split data-quality audit

Grain: one feature row, label, and speaker entry per utterance; one shift decision per non-terminal utterance. All checks are read-only.

| source/config | split ID overlap | length errors | non-finite values | test=train exact dialogues | repeated-text feature inconsistencies | max repeated-text Δ |
|---|---:|---:|---:|---:|---:|---:|
| iemocap | 0 | 0 | 0 | 0 | 0 | 2.34e-05 |
| meld | 0 | 0 | 0 | 1 | 0 | 5.2e-05 |
| emorynlp | 0 | 0 | 0 | 0 | 0 | 0.000113 |
| dailydialog | 0 | 0 | 0 | 134 | 0 | 6.14e-05 |
| iemocap_mm | 0 | 0 | 0 | 0 | 0 | 0 |
| meld_mm | 0 | 0 | 0 | 2 | 1547 | 1.54 |

## Cross-source alignment

| dataset | train IDs aligned | test IDs aligned | text aligned | labels aligned |
|---|---|---|---:|---:|
| iemocap | True | True | 151/151 | 151/151 |
| meld | True | True | 1432/1432 | 1432/1432 |

## Interpretation

- Zero length errors/non-finite values is required before any experiment is cited.
- Split-ID overlap and exact train-test dialogue duplication are separate checks; DailyDialog duplicates remain visible in this raw-source audit but are removed from every formal Access experiment.
- Repeated identical utterance text should map to an identical feature vector within a deterministic utterance-local encoder. This is supporting evidence, not proof of the entire upstream extraction pipeline.
- COSMIC validation IDs are reunited with COSMIC training IDs before comparison with MM-DFN because the released MM-DFN files do not contain a validation split.
