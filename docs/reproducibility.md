# Reproduction procedure

## Identity

The public snapshot was prepared from source commit
`afc8050fbe86c7ce5741fec608bb354591790f03`. The numerical authorities are:

- `configs/model.json`: constants, condensate, atomic response and detector;
- `configs/reference_state.json`: measured initial state and Oxford closure;
- `configs/reproduction.json`: endpoint geometry, fluence set, estimator and seed.

The source-data DOI, download URL, archive digest and consumed member digests are
stored in the reference-state config. The drivers consume the transcribed values
and do not access the network.

## Verified commands

From the repository root, in an isolated Python 3.12 environment:

```text
python -m pip install -e ".[test]"
python -m pytest -q
python scripts/reproduce_forward_model.py --validate-only
python scripts/reproduce_forward_model.py --output outputs/forward_model.json
python scripts/reproduce_inference.py --validate-only
python scripts/reproduce_inference.py --draws 1 --output outputs/inference.json
```

`--validate-only` parses every authority, reconstructs the endpoint contract and
checks the requested output without writing it. The full forward command records
the endpoint states, deterministic mean raw counts, branch scattering and the
conditional thermodynamic sequence. The inference command draws atom, bright
and dark frames independently and fits each endpoint without access to the
generator truth.

Increasing `--draws` runs more fixed-seed independent pairs sequentially. It can
be computationally expensive. The dissertation's ensemble summaries used 64
draws per scan point and additional qualification protocols; they are not
relabelled or reconstructed by the default command.

## Output lifecycle

`outputs/` is ignored except for its placeholder. Outputs are regenerated local
artefacts, not retained evidence. Each JSON records the input file hashes,
software versions, seed policy and scientific status needed to interpret it.
Deleting an output and rerunning the same command is safe.
