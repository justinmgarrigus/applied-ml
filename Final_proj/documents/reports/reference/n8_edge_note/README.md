# N=8 and N=10 projected-edge experiment note

[Three-page PDF](../N8_Edge_Potential_Note.pdf) · [Editable TeX](../N8_Edge_Potential_Note.tex)

The note defines the fixed-K cap Hamiltonian for N=8 and N=10, derives the
projected spatial wall, and preserves the N=8 no-edge comparisons and
tabulated radius and batch studies through B=512. The third page adds the
N=10 trajectory from fresh initialization through 130,000 total updates,
including B=300 and learning rate 3e-5 for the last 10,000 updates. The PDF
keeps its established filename to preserve existing links; its title,
metadata and content now cover both particle numbers.

The N=10 table reports exact energy, error, fidelity, residual variance,
batch, learning rate and cumulative active minutes at eight endpoints. Its
figure displays all complete-sector saved evaluations through 130k. Final
fidelity is 57.07037%, error is 0.536892 V1 and cumulative active training
is 43.98 minutes, excluding compilation, chain warmup and CPU evaluation.
The same spatial cutoff 0.88 is retained rather than rescaled to N=10's
compact edge. There is no N=10 no-edge control. Its last branch changes
batch and learning rate together, without a matched control or established
convergence speedup.

The matched batch-128/256 test supports a batch-size influence in this one
continuation. It does not measure gradient-noise variance or establish a
general cause for the remaining error. The batch-512 result starts from the
batch-256 endpoint and has no new matched batch-256 control. Wall-profile
sharpness remains an untested explanation for incomplete convergence.

`note_evidence.json` records the numerical inputs, their hashes and the
evaluated checkpoint identities for ten cap-model endpoints across six
phases. The builder verifies each endpoint's checkpoint hash and its exact
energy, fidelity and variance against the frozen evaluation.
`n10_note_evidence.json` records the N=10 table and 67 frozen evaluations
(61 original B=256 points and six B=300 branch points, including the shared
120k model), checkpoint/coefficient hashes, seven reference-manifest files
and 24 implementation-source hashes. The initial branch checkpoint has
new walkers and metadata but identical physical-state diagnostics. The new
builder writes `n10_results_table.tex` and `n10_convergence.pdf/.png`.
`evidence_audit.json` preserves the earlier independent audit of six
principal endpoints, including the earlier no-edge result; it predates the
new radius and batch tests. `validation_before_n10_20261110.json` preserves
the preceding note validation. `validation.json` records current compilation, page count, numerical
reviews and layout checks. `SHA256SUMS` covers this support folder and the
note's PDF and TeX source, excluding the checksum file itself.

From the course workspace, regenerate the figures and evidence with CPU
libraries; this launches no training:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python \
  Final_proj/documents/reports/reference/n8_edge_note/build_note_figures.py

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python \
  Final_proj/documents/reports/reference/n8_edge_note/build_n10_note_figures.py
```

Build the PDF from `Final_proj/documents/reports/reference/`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/edge_note_n10_build N8_Edge_Potential_Note.tex
cp /tmp/edge_note_n10_build/N8_Edge_Potential_Note.pdf N8_Edge_Potential_Note.pdf
```

The frozen figures allow compilation without regenerating the evidence.
Rebuilding changes PDF metadata and requires refreshing `SHA256SUMS`.

`table_validation.json` verifies all nine displayed rows and the new batch-size
column against live summaries. `latest_table_evidence_audit.json` preserves
the preceding independent audit of the N=8 additions and comparison scopes.
The original `short_step_comparison.pdf/.png` remain available as supporting
N=8 radius plots; the current three-page PDF shows the wall profile and
the new N=10 convergence figure.
