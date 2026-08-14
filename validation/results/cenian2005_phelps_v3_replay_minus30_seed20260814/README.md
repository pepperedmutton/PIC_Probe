# Current-code replay at -30 V

This directory records one deterministic replay of the Cenian 2005 pilot with physics model version 3 and clean source commit `45ead2eeed989a43a3dc6ae2a823507f8eb19a1e`.

The replay used the baseline settings: Phelps electron and Ar+/Ar inputs, a 12 mm domain, 64 cells, 1024 nominal particles, a 30 ps time step, 31,708 warm-up steps, 31,708 sample steps, and seed 20260814.

The signed, unscaled simulated current is -23.989006484366297 microamperes. Its internal standard error is 2.3000623334209972 microamperes. Both values, the sample count, the batch count, and the numerical status match the corresponding row in the published three-seed baseline exactly. The current and standard-error differences are both zero at stored precision.

The digitized experiment gives -12.92 microamperes at -30 V. The replay residual is -11.069006484366297 microamperes. The replay therefore confirms reproducibility of this representative point; it does not improve the experimental agreement.

This is a one-point `PREVIEW` replay. It does not convert the short baseline into a steady-state, converged, or production validation.

Files:

- `simulation_points.csv`: one simulation row.
- `comparison.csv`: signed experiment-to-simulation comparison.
- `metrics.json`: one-point preview metrics.
- `manifest.json`: redacted publication manifest with input hashes, diagnostics, source commit, and physics-model version.
