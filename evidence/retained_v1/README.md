# Retained synthetic evidence

This directory publishes a compact, integrity-checked subset of the admitted
synthetic evidence used in the dissertation analysis. It complements the
re-runnable public drivers: the drivers reproduce selected software workflows,
whereas these files preserve selected fitted results and the synthetic camera
arrays on which they were obtained.

Nothing in this directory is experimental imaging data. The targets were
generated under the declared equilibrium, optical, detector and noise models.
Agreement with them is therefore a same-model software and inference check, not
independent physical or apparatus validation.

## Contents

| Directory | Contents | Evidential role |
| --- | --- | --- |
| `fixed_field/` | One linked PCI target before and after a conditional re-equilibration, its target manifest, 64-refit route table, status matrix and residual diagnostics | Tests whether the second exposure distinguishes the small model-assigned width change under the retained inverse model. |
| `orientation/` | One independently generated PCI target for each field orientation, target metadata, endpoint and route tables, confidence summary, and a 64-refit endpoint table | Tests detector-level information about orientation-conditioned projected widths under the retained profile family. |
| `figures/` | Three presentation-exact PNG derivatives of the retained arrays and tables | Provides a direct visual entry point without replacing the machine-readable evidence. |

`orientation/conditional_refits.csv` is a row-preserving public derivative. It
extracts the six recorded observables for each of 64 draws and two endpoints
from the admitted bootstrap job records. It performs no fit, resampling,
interval calculation or scientific selection.

## Interpretation boundary

- Conditional refit ranges vary detector noise while keeping the fitted model,
  support, optical transfer and profile family fixed. They are not total
  experimental uncertainties.
- The two orientation targets are independent equilibrium endpoints under a
  common simulated PCI design. They are not a field-rotation trajectory or a
  repeatability study.
- The retained orientation result supports fitted width contrasts within the
  declared projected-profile family. It does not establish profile robustness,
  finite-temperature recovery or experimental magnetostriction.
- The non-retained 17-point PCI/DGI scans, profile-qualification presentation,
  thermal-halo stress, later-exposure fits and slope regressions are not
  reconstructed here.

## Integrity and provenance

Run from the repository root:

```text
python scripts/check_evidence.py
```

The checker verifies every declared byte count and SHA-256 digest, CSV row
count and NPZ array schema. `manifest.json` records the admitted source-family
identities and distinguishes byte-for-byte copies from the one mechanical CSV
derivative.
