# Pre-submission checks for IEEE Access

The manuscript source and PDF are technically packaged, but the following author-only
items must be confirmed before upload:

1. **Exclusive submission.** The author has chosen IEEE Access and confirmed that this work will
   not be submitted to ICASSP. Reconfirm that no version is under review elsewhere at upload time.
2. **Author metadata.** Confirm the spelling of Bin Wen, the Universiti Sains Malaysia
   affiliation, corresponding email, short biography, ORCID, and any co-authors who meet
   authorship criteria. The submitting author's ORCID profile must be public and populated.
3. **Funding and conflicts.** Add the correct funding acknowledgment and disclose any
   conflicts of interest. The source does not invent a funding statement.
4. **Data and ethics.** Confirm that use of the released IEMOCAP, MELD, EmoryNLP, and
   DailyDialog features complies with their licenses and the institution's requirements.
5. **Originality wording.** The cover letter states that the work is original and unpublished.
   Confirm this factual statement before upload; if a prior public version exists, disclose it.
6. **Final portal check.** Upload `main.tex` and every local class/font/figure dependency,
   plus the matching `main.pdf`; verify the converted proof before completing submission.
7. **AI disclosure.** The manuscript now discloses Codex assistance in the Acknowledgment
   section. Confirm that the wording accurately reflects the author's use before submission.

## Major evidence in the Access manuscript

- formal task definition and temporal threat model;
- explicit distinction between interaction-level next-turn shift and same-person
  next-own-utterance shift;
- speaker-switch stratification across all four unique corpora;
- information-matched predicted-label and gold-label transition comparisons;
- equal four-configuration validation searches and validation checkpointing;
- seed--dialogue hierarchical bootstrap and paired cluster permutation tests;
- independent utterance-local TF-IDF/SVD features;
- complete six-model self-shift benchmark;
- same-architecture masked-Transformer leakage dose response; and
- expanded deployment implications, limitations, reporting checklist, and artifact map.
