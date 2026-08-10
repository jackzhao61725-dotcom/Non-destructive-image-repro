# Equilibrium-conditioned dispersive imaging

This repository is a compact, code-focused reproduction of the deterministic
equilibrium, optical, detector and conditional thermodynamic chain used to study
repeated dispersive imaging of a polarised `166Er` condensate. It also provides
one fixed-seed PCI endpoint-inference workflow.

It reproduces software calculations, not the complete submitted result set.
In particular, it does **not** reproduce the 64-draw linked two-exposure analysis, the
17-point PCI or DGI ensembles, their qualification procedures, later-exposure
camera fits, or the slope regressions. It contains neither experimental raw images nor
a calibrated installed-imaging-arm model.

## Quick start

Use Python 3.12 in a fresh environment, then run from the repository root:

```text
python -m pip install ".[test]"
python -m pytest -q
python scripts/reproduce_forward_model.py --validate-only
python scripts/reproduce_inference.py --validate-only
python scripts/check_reference.py
```

Use the tagged GitHub source archive or a repository checkout: the scripts and
configuration files are source-tree assets, so this release is not advertised
as a standalone PyPI distribution.

A successful run prints both validation messages and returns exit status zero
from the reference checker. The expected validation messages are:

```text
configuration and forward-model contract validated
configuration and inference contract validated
```

The default checker then prints `PASS forward (...)`; with
`--include-inference` it also prints `PASS inference (...)`.

The default reference check covers the deterministic forward-model reference.
To include the fixed-seed PCI endpoint fit, run:

```text
python scripts/check_reference.py --include-inference
```

The inference check is optional because it is slower. These checks compare the
current software with the repository's verification reference; they do not
compare a simulation with raw experimental images or recreate the numerical
evidence supporting the submitted analysis. See
[docs/results.md](docs/results.md) for the precise
coverage boundary.

## Generate local JSON outputs

The public drivers keep validation separate from writing output:

```text
python scripts/reproduce_forward_model.py --output outputs/forward_model.json
python scripts/reproduce_inference.py --draws 1 --output outputs/inference.json
```

Existing output paths are refused by default. Pass `--overwrite` only after
inspecting the file you intend to replace.

What you should see:

- `outputs/forward_model.json` has status
  `model_conditional_reproduction` and records the two independently minimised
  equilibrium endpoints, optical response, mean raw counts, scattering and the
  conditional thermodynamic sequence;
- `outputs/inference.json` has status
  `fixed_seed_point_fits_not_sampling_coverage`, records one fixed-seed draw by
  default, and states that generator truth was not used by the fit;
- rerunning with unchanged inputs reproduces the deterministic forward payload
  and seed policy. Generated files remain local artefacts, not submitted
  experimental or submitted-analysis evidence.

The two endpoints, `B_parallel_y` and `B_parallel_z`, are independent
equilibrium states under a common PCI design. They are not a simulated magnetic
field-rotation trajectory. The repeated-exposure update is likewise conditional:
it assumes fixed trapped number, recoil-only deposited energy and complete
re-equilibration between exposures.

## Repository map

- `src/non_destructive_image/`: physical, detector and inverse models;
- `configs/`: frozen numerical inputs and source-data identity;
- `scripts/`: validation, reproduction and reference-checking entry points;
- `tests/`: unit and contract tests;
- `docs/`: model, inference, result-boundary and reproduction documentation;
- `THIRD_PARTY.md`: data, article and dependency attribution.

The Oxford ORA dataset identified by DOI `10.5287/ora-m8gpvdr2y` is not bundled.
Only the values required by the public drivers are transcribed in the
configuration with provenance and integrity metadata. See `THIRD_PARTY.md`
before obtaining or redistributing the original dataset.

## Identity, citation and licence

Version `1.0.0` records upstream source snapshot commit
`afc8050fbe86c7ce5741fec608bb354591790f03`. Citation metadata is in
`CITATION.cff`.

The repository code is released under the BSD 3-Clause License; see `LICENSE`.
Third-party data, publications and dependencies remain governed by their own
terms.
