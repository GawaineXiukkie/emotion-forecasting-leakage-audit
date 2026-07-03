# Automated leakage audit

| dataset | last-ign | causal | per-utt | spk-indep | thr=val | dlg-boot | trans=train | baselines | spk overlap |
|---|---|---|---|---|---|---|---|---|---|
| iemocap | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 0 (speaker-independent) |
| meld | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | 8 (speaker ids overlap (local per-dialogue ids; identity not global here)) |
| emorynlp | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | 6 (speaker ids overlap (local per-dialogue ids; identity not global here)) |
| dailydialog | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | 2 (speaker ids overlap (local per-dialogue ids; identity not global here)) |
| iemocap_mm | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 0 (speaker-independent) |
| meld_mm | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | 8 (speaker ids overlap (local per-dialogue ids; identity not global here)) |
