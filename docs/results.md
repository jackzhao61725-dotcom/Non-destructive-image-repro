# Result coverage and interpretation

## Three public surfaces

The repository separates calculations that can be rerun from values retained
for software verification and selected synthetic evidence retained from the
dissertation analysis.

| Surface | Location | Question answered |
| --- | --- | --- |
| Re-runnable workflows | `scripts/reproduce_*.py` and `outputs/` | Does the public implementation execute the declared forward or fixed-seed inference workflow? |
| Software-verification reference | `reference/expected_results.json` | Does the current implementation remain numerically consistent with compact admitted values? |
| Retained synthetic evidence | `evidence/retained_v1/` | What synthetic camera arrays, conditional refits and fitted summaries support the selected published comparisons? |

`check_reference.py` recomputes values. `check_evidence.py` only verifies the
identity and structure of frozen files; it does not rerun fits or turn the
bundle into independent physical validation.

## Coverage

| Result or workflow | Public status | What is covered |
| --- | --- | --- |
| Configuration and contract validation | Reproduced | Both public drivers support `--validate-only` without writing output. |
| Dipolar Thomas–Fermi equilibrium endpoints | Reproduced | Deterministic, independently minimised `B_parallel_y` and `B_parallel_z` states and projected widths. |
| Optical and detector forward chain | Reproduced | Polarised 401 nm response, finite-aperture PCI mean raw counts and branch-resolved scattering at the reference probe setting. |
| Conditional thermodynamic sequence | Reproduced | Fixed-trapped-number, recoil-only update with complete re-equilibration assumed. |
| Default software reference | Reproduced | `python scripts/check_reference.py` checks the deterministic forward reference. |
| Fixed-seed PCI endpoint workflow | Reproduced as a workflow check | One generated pair, four starts and no generator truth supplied to the fit; checked with `--include-inference`. |
| Linked first/second-exposure PCI comparison | Bundled retained evidence | A synthetic four-role raw target, 22 fitted routes, 64 refits per route, status matrix and residual diagnostics. |
| Independent orientation PCI comparison | Bundled retained evidence | Two synthetic three-role raw targets, 22 fitted routes, endpoint diagnostics and 64 conditional refits per endpoint. |
| 17-point PCI ensemble | Not retained | No scan summary is bundled or reconstructed. |
| 17-point DGI ensemble | Not retained | DGI components remain in the library and tests, but no public DGI ensemble result is provided. |
| Profile-qualification presentation and thermal-halo stress | Not retained | Their dissertation claims are not reconstructed from figures, prose or aggregate tables. |
| Later-exposure camera fits | Not retained | No fifth-exposure simulated-camera ensemble or fit result is bundled. |
| Slope regressions | Not retained | No regression result should be inferred from endpoint or one-draw outputs. |
| Laboratory imaging data | Not part of this reproduction | No experimental image dataset is analysed; selected ORA values only anchor declared reference-state inputs. |

## Interpretation of the retained evidence

The retained targets are same-model synthetic camera arrays. They test whether
the declared inverse procedure extracts low-order morphology information from
data generated under the same forward assumptions. They do not test an
installed imaging arm, establish experimental magnetostriction or validate the
equilibrium and profile models independently.

The conditional refit ranges vary detector noise around retained fitted count
means while holding the forward model, support, optical transfer and projected
profile family fixed. They are not coverage intervals or total experimental
uncertainties. The two orientation targets are independent equilibrium
endpoints, not a field-rotation trajectory or repeatability sample.

The orientation bundle includes `conditional_refits.csv`, a public
row-preserving extraction from 64 admitted bootstrap records. No fits,
resampling, interval calculations or selections were performed when creating
that table. Its exact source-family and file identities are recorded in the
evidence manifest.

## The 30% depletion screen

The forward workflow uses a 30% condensate-depletion threshold as a reporting
screen. At and after the first threshold-crossing state, frames are marked as
outside the screened sequence. This is an analysis convention for deciding
which model-conditioned frames are reported; it is not an experimentally
measured validity boundary.

The public forward driver permits diagnostic continuation after the crossing.
A state4 value may therefore be calculated and recorded while remaining
outside the depletion screen. Its presence is not evidence that the
fixed-number, recoil-only and complete-re-equilibration assumptions remain
physically valid there.

## Evidence classes

Keep these categories separate when reporting results:

1. **Re-runnable software artefact:** a fresh local JSON generated under
   `outputs/`; useful for reproduction and debugging, but not immutable.
2. **Software-verification reference:** compact admitted values used by
   `check_reference.py` to detect implementation or configuration drift.
3. **Retained synthetic evidence:** immutable, hash-identified camera arrays,
   refits, tables and figures under `evidence/retained_v1/`.
4. **External source data:** the unbundled Oxford ORA archive. Selected values
   are transcribed into configuration with provenance; this repository is not a
   copy or experimental reanalysis of that archive.
5. **Non-retained dissertation analysis:** scans, stresses, later-exposure fits
   and regressions whose numerical surfaces are absent. These must not be
   reconstructed from PDF figures or prose.
