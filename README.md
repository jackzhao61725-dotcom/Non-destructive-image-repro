# Equilibrium-conditioned dispersive imaging

This release candidate contains the numerical model and inference code used to
study repeated, conditionally re-equilibrated imaging of a polarised
`166Er` condensate. It is a code-focused reproduction repository: dissertation
source, AI operating material, hand-off notes, historical notebooks and retained
result archives are deliberately absent.

The model compares two independently minimised equilibrium endpoints,
`B_parallel_y` and `B_parallel_z`, under a common PCI measurement design. It
also includes the Oxford-anchored, fixed-trapped-number state update used to
describe recoil heating between exposures. These endpoints are model states;
they are not a simulated magnetic-field rotation trajectory.

## What can be reproduced

- the two dipolar Thomas-Fermi equilibrium endpoints and their projected widths;
- the polarised 401 nm complex response, finite-aperture PCI raw counts and
  per-pulse scattering;
- the conditional post-exposure thermodynamic sequence at the reference probe
  setting;
- a deterministic, fixed-seed PCI endpoint fit using the same projected
  Thomas-Fermi family, bounds, four starts and detector model as the dissertation.

The one-draw inference command is a workflow check and point estimate. It is not
the dissertation's 64-draw uncertainty ensemble and must not be cited as one.
The repository does not contain experimental raw images or claim a calibrated
installed imaging arm.

## Environment

The verified environment is Python 3.12.13, NumPy 2.3.5, SciPy 1.18.0 and
pytest 9.1.1. Create an isolated environment, install the project and test
extra, then run:

```text
python -m pip install -e ".[test]"
python -m pytest -q
python scripts/reproduce_forward_model.py --validate-only
python scripts/reproduce_forward_model.py
python scripts/reproduce_inference.py --validate-only
python scripts/reproduce_inference.py --draws 1
```

The two run commands write JSON to `outputs/`. A fixed `PCG64DXSM` seed tree is
recorded in `configs/reproduction.json`; changing a draw count does not alter
earlier draws.

## Repository map

- `src/non_destructive_image/`: physical, detector and inverse models;
- `configs/`: frozen numerical inputs and source-data identity;
- `scripts/`: public reproduction entry points;
- `tests/`: unit and contract tests;
- `docs/`: model boundary, inference contract and exact reproduction procedure.

The original Oxford dataset is identified by DOI, member names and SHA-256
hashes in `configs/reference_state.json`; its measured values required by these
drivers are already transcribed with provenance, so the commands do not download
data.

## Release status and licence

This is a local release candidate prepared from source commit
`afc8050fbe86c7ce5741fec608bb354591790f03`. No remote has been configured and
no licence has yet been selected. Until the copyright holder adds a licence,
the code is available for inspection but no permission for reuse or
redistribution is granted.
