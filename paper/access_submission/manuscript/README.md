# IEEE Access submission source

This directory is the self-contained LaTeX submission for:

> Information-Matched Auditing of Conversational Emotion-Shift Forecasting

## Build

The official IEEE Access class and required font assets are included. A standard
pdfLaTeX installation can compile `main.tex` directly. The checked-in `main.pdf`
was compiled from the same source with Tectonic 0.16.9/XeTeX; `ieeeaccess.cls` contains a
small engine-compatibility fallback that leaves the official pdfLaTeX spot-color
path unchanged and uses equivalent CMYK colors under non-pdfTeX engines.

```bash
pdflatex main.tex
pdflatex main.tex
```

or:

```bash
tectonic -X compile main.tex --keep-logs --keep-intermediates
```

## Submission metadata to verify

- Author name: Bin Wen
- Affiliation: School of Computer Sciences, Universiti Sains Malaysia
- Corresponding email: wenbin@student.usm.my
- Funding statement: omitted because the source manuscript did not specify one;
  the author must add the correct statement before submission if applicable
- Author biography: a short no-photo biography is included; the author must verify
  its wording and add an ORCID in the submission portal
- AI-use disclosure: included in the Acknowledgment section to satisfy the current
  IEEE Access submission checklist
- DOI and publication dates are placeholders supplied by the IEEE Access class

## Submission status

This is the authoritative IEEE Access source. Historical drafts elsewhere in the repository are
not submission files and are not under consideration at another venue. The manuscript must not be
submitted until author metadata, ORCID, funding, conflicts, and exclusivity are confirmed.
