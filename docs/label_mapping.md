# Emotion-label mappings used by the released feature files

The shift target only tests whether adjacent numeric labels differ, but publishing the
mapping makes class-system differences auditable. These are the indices in the downloaded
COSMIC/MM-DFN pickle files, not a new harmonized ontology.

| Dataset | Numeric mapping |
|---|---|
| IEMOCAP | 0 happy; 1 sad; 2 neutral; 3 angry; 4 excited; 5 frustrated |
| MELD | 0 neutral; 1 surprise; 2 fear; 3 sadness; 4 joy; 5 disgust; 6 anger |
| EmoryNLP | 0 joyful; 1 mad; 2 peaceful; 3 neutral; 4 sad; 5 powerful; 6 scared |
| DailyDialog pickle | 0 happiness; 1 neutral/other; 2 anger; 3 sadness; 4 fear; 5 surprise; 6 disgust |

The mapping is documented by the upstream COSMIC data loader and training scripts. In
particular, the DailyDialog pickle reorders the original dataset's published integer codes;
the table above reports the actual feature-file order consumed by this repository.

Emotion-system cardinality affects the prevalence of the derived binary target
`1[y_(n+1) != y_n]`: corpora with more frequent neutral/other labels or different class
granularity are not assumed to share the same base shift rate.
