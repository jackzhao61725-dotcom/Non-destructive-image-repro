# Reproduction procedure

## Identity

The public package version is `1.1.0` and records upstream source snapshot
commit `afc8050fbe86c7ce5741fec608bb354591790f03`. The numerical authorities for
the re-runnable workflows are:

- `configs/model.json`: constants, condensate, atomic response and detector;
- `configs/reference_state.json`: reference state and Oxford-anchored closure;
- `configs/reproduction.json`: endpoint geometry, fluence set, estimator and seed.

The source-data DOI, record URL, archive digest and consumed member digests are
stored in the reference-state configuration. The drivers consume transcribed
values and do not access the network.

The selected retained synthetic evidence has a separate identity in
`evidence/retained_v1/manifest.json`. That manifest records the admitted
source-family hashes, copied-file hashes, one mechanical derivative and the
scientific claim boundary. It does not relabel local reproduction output as
retained evidence.

## Environment

Use Python 3.12. `pyproject.toml` pins the direct NumPy, SciPy and pytest
versions used by the public package, and CI runs the full checks on Ubuntu and
Windows with `pip check`. This is an exact direct-dependency specification, not
a hash-locked record of every transitive wheel or operating-system library.

## Verified commands

From the repository root, in an isolated Python 3.12 environment:

```text
python -m pip install ".[test]"
python -m pip check
python -m pytest -q
python scripts/reproduce_forward_model.py --validate-only
python scripts/reproduce_inference.py --validate-only
python scripts/check_reference.py
python scripts/check_reference.py --include-inference
python scripts/check_evidence.py
python scripts/reproduce_forward_model.py --output outputs/forward_model.json
python scripts/reproduce_inference.py --draws 1 --output outputs/inference.json
```

`--validate-only` parses every authority and reconstructs the endpoint contract
without writing output. `check_reference.py` compares a fresh in-memory
calculation with the compact software-verification reference. The full forward
command records endpoint states, deterministic mean raw counts, branch
scattering and the conditional thermodynamic sequence. The inference command
draws atom, bright and dark frames independently and fits each endpoint without
access to generator truth.

Increasing `--draws` runs more fixed-seed independent pairs sequentially and
can be computationally expensive. It does not recreate the retained linked
two-exposure analysis, the orientation conditional-refit protocol or any
non-retained scan merely by matching a draw count.

`check_evidence.py` is an integrity check, not a scientific rerun. It verifies:

- the evidence inventory and absence of undeclared files;
- byte counts and SHA-256 digests;
- CSV data-row counts;
- NPZ member names, shapes and dtype descriptors;
- consistency between each public target manifest and its raw NPZ.

## Output lifecycle

`outputs/` is ignored except for its placeholder. Files written there are
regenerated local software artefacts. Each JSON records the input hashes,
software identity, invocation, seed policy and scientific status needed to
interpret it. A driver refuses to replace an existing output by default;
inspect or remove it, or pass `--overwrite` explicitly.

`evidence/retained_v1/` follows the opposite lifecycle: it is immutable public
evidence selected from admitted same-model synthetic families. Copied files
retain their source bytes. `orientation/conditional_refits.csv` is the sole
mechanical derivative and only reshapes already recorded endpoint observables;
it performs no fitting, resampling, interval calculation or scientific
selection. Any future numerical replacement requires a new evidence identity
rather than overwriting this bundle.
