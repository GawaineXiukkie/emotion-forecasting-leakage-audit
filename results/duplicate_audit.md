# Duplicate-dialogue audit

Checks whether any test/val dialogue's full utterance-text sequence exactly matches a train dialogue (`src/dataset.py::find_exact_duplicate_dialogues`).

| dataset | test dialogues | test duplicates | test dup% | val dialogues | val duplicates | val dup% |
|---|---|---|---|---|---|---|
| iemocap | 31 | 0 | 0.0% | 12 | 0 | 0.0% |
| meld | 280 | 1 | 0.4% | 114 | 0 | 0.0% |
| emorynlp | 79 | 0 | 0.0% | 89 | 0 | 0.0% |
| dailydialog | 1000 | 134 | 13.4% | 1000 | 101 | 10.1% |
