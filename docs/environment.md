# Reference execution environment

Formal IEEE Access revision runs were executed on:

- macOS 26.5.2, Apple arm64;
- Python 3.9.6;
- PyTorch 2.8.0 with the Apple MPS backend;
- NumPy 2.0.2;
- SciPy 1.13.1;
- scikit-learn 1.6.1; and
- Matplotlib 3.9.4.

The complete resolved Python environment is in `requirements-lock.txt`.
`environment.yml` creates the Python environment and installs that lock file.
Experiments fall back to CPU when MPS is unavailable. Random seeds are recorded
per run, while hardware-level bitwise equality across PyTorch backends is not
claimed; the paper reports three-seed distributions and retains raw predictions.

The independent TF-IDF/SVD feature control uses scikit-learn's ARPACK
`TruncatedSVD` solver. This avoids platform-specific numerical warnings observed
with randomized power iterations under Apple's Accelerate BLAS; every transformed
dialogue is explicitly checked for finite values before training.

Data-file SHA-256 checksums are in `docs/data_manifest.sha256`. Raw feature files
are not redistributed.
