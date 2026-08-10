# Reproduction procedure

## Identity

The public snapshot was prepared from source commit
`afc8050fbe86c7ce5741fec608bb354591790f03`. The numerical authorities are:

- `configs/model.json`: constants, condensate, atomic response and detector;
- `configs/reference_state.json`: measured initial state and Oxford closure;
- `configs/reproduction.json`: endpoint geometry, fluence set, estimator and seed.

The source-data DOI, record URL, archive digest and consumed member digests are
stored in the reference-state config. The drivers consume the transcribed values
and do not access the network.

## Verified commands

From the repository root, in an isolated Python 3.12 environment:

```text
python -m pip install ".[test]"
python -m pytest -q
python scripts/reproduce_forward_model.py --validate-only
python scripts/reproduce_inference.py --validate-only
python scripts/check_reference.py
python scripts/reproduce_forward_model.py --output outputs/forward_model.json
python scripts/reproduce_inference.py --draws 1 --output outputs/inference.json
```

`--validate-only` parses every authority and reconstructs the endpoint contract
without writing an output. `check_reference.py` compares a fresh in-memory
calculation with the compact software-verification reference; add
`--include-inference` to include the slower fixed-seed endpoint fit. The full
forward command records the endpoint states, deterministic mean raw counts,
branch scattering and the conditional thermodynamic sequence. The inference
command draws atom, bright and dark frames independently and fits each endpoint
without access to the generator truth.

Increasing `--draws` runs more fixed-seed independent pairs sequentially. It can
be computationally expensive. The submitted study's ensemble summaries used 64
draws per scan point and additional qualification protocols; they are not
relabelled or reconstructed by the default command.

## Output lifecycle

`outputs/` is ignored except for its placeholder. Outputs are regenerated local
software artefacts, not experimental data or submitted-analysis evidence. Each JSON
records the input file hashes, software identity, invocation, seed policy and
scientific status needed to interpret it. A driver refuses to replace an
existing output by default; inspect or remove it, or pass `--overwrite`
explicitly when replacement is intended.
