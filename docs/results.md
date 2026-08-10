# Result coverage and interpretation

## What the repository verifies

The reference checker is a software-verification check. It asks whether the
current implementation reproduces the admitted deterministic values and, when
requested, the fixed-seed PCI endpoint result. Passing it supports consistency
of this repository's code, configuration and stored verification reference.

It does not establish agreement with unbundled experimental images, reproduce a
submitted ensemble, or turn a regenerated JSON file into evidence from the
submitted analysis.

| Result or workflow | Public status | What is covered |
| --- | --- | --- |
| Configuration and contract validation | Reproduced | Both public drivers support `--validate-only` without writing output. |
| Dipolar Thomas-Fermi equilibrium endpoints | Reproduced | Deterministic, independently minimised `B_parallel_y` and `B_parallel_z` states and projected widths. |
| Optical and detector forward chain | Reproduced | Polarised 401 nm response, finite-aperture PCI mean raw counts and branch-resolved scattering at the reference probe setting. |
| Conditional thermodynamic sequence | Reproduced | Oxford-anchored, fixed-trapped-number, recoil-only update with complete re-equilibration assumed. |
| Default software reference | Reproduced | `python scripts/check_reference.py` checks the deterministic forward reference. |
| Fixed-seed PCI endpoint workflow | Reproduced as a workflow check | One point-fit draw, common declared model and detector design, four starts and no generator truth supplied to the fit; checked with `--include-inference`. |
| 64-draw linked two-exposure result | Not reproduced | No linked two-exposure ensemble or interval is generated. |
| 17-point PCI ensemble | Not reproduced | The scan ensemble and its summaries are absent. |
| 17-point DGI ensemble | Not reproduced | DGI components exist in the library and tests, but no public DGI ensemble result is generated. |
| Qualification protocols | Not reproduced | Coverage, profile-family, thermal-halo, state-sensitivity and other qualification results are absent. |
| Later-exposure camera fits | Not reproduced | No fifth-exposure simulated-camera ensemble or fit result is included. |
| Slope regressions | Not reproduced | No regression result should be inferred from the endpoint or one-draw outputs. |
| Experimental raw data | Not bundled or reproduced | The ORA archive is identified but not redistributed. |
| Submitted-analysis evidence | Not bundled or recreated | Local JSON outputs are regenerable software artefacts only. |

## The 30% depletion screen

The forward workflow uses a 30% condensate-depletion threshold as a reporting
screen. At and after the first threshold-crossing state, frames are marked as
outside the screened sequence. This is an analysis convention for saying
which model-conditioned frames are reported; it is not a validity boundary
measured or endorsed by the Oxford dataset.

The public forward driver enables diagnostic continuation after that crossing.
Consequently, a state4 value may still be calculated and recorded so that the
numerical continuation can be inspected, while remaining outside the depletion
screen. Its presence must not be read as Oxford validation of state4,
nor as evidence that the fixed-number, recoil-only and complete-re-equilibration
assumptions remain experimentally valid there.

## Evidence classes

Keep these three categories separate when reporting results:

1. **Software-verification reference:** admitted numerical values used by
   `check_reference.py` to detect implementation or configuration drift.
2. **Experimental source data:** the unbundled Oxford ORA dataset. Selected
   required values are transcribed into configuration with file/member hashes
   and provenance; that does not make the repository a copy of the dataset.
3. **Submitted-analysis evidence:** qualified ensembles, fits, regressions,
   figures and their provenance records used in the submitted analysis. Those
   artefacts are outside this repository.

The files written under `outputs/` belong only to the first category's local
reproduction workflow. They can support debugging and software verification,
but should not be cited as experimental measurements or as replacements for the
missing submitted-analysis evidence.
