# Evidence and result identity

## Two kinds of public output

The release separates reproducible examples from retained numerical evidence.

- **Recomputed example:** a short deterministic or fixed-seed run that checks
  the released model chain. Its output is local and may be regenerated.
- **Admitted evidence:** a completed calculation family whose inputs, outputs
  and interpretation were checked before it was used by the dissertation. The
  complete family is bundled with an immutable artifact manifest.

A successful process is not automatically scientific evidence. The one-fit
public example is therefore not added to `results/` and does not replace any
admitted ensemble.

## Bundled families

| Family | Dissertation role |
| --- | --- |
| `chapter_4_three_state_four_method_static_v3` | Static three-state readouts and representative noisy camera images |
| `chapter_4_three_state_four_method_snr_400_v2` | Four-method whole-image SNR through 400 microseconds |
| `chapter_4_repeated_bec_dgi_dpfi_v1` | Repeated-BEC DGI/DPFI SNR and representative DPFI frames |
| `chapter_5_bec_multiframe_eta_ry_dpfi_v1` | DPFI recovery of BEC population and axial size |
| `chapter_5_three_state_density_recovery_v2` | DGI/DPFI recovery of SSP and ID physical quantities |
| `chapter_5_snr_recovery_relation_v1` | Observable-specific comparison of SNR with central-68-percent recovery span |

Together these directories contain 183 manifested artifacts totalling
67,080,996 bytes. Run:

```text
python scripts/verify_bundled_evidence.py
```

The verifier checks that:

1. each root manifest has the release-pinned SHA-256 identity;
2. the family is marked `admitted_immutable`;
3. every recorded artifact exists with the recorded byte count and hash;
4. no unlisted file has entered the evidence tree; and
5. the artifact count and total bytes agree with the manifest.

One unchanged shard-inventory record contains the original absolute worker
directories used to assemble the Section 5.2 ensemble. Those strings are
historical provenance, not paths used by the released code. They are retained
byte-for-byte because changing them would break the admitted manifest identity.

## Figures

The release also contains the 17 exact PDFs consumed by Chapters 2–6. Their
hashes identify the manuscript presentation, but the figure file alone is not
the numerical authority. The figure crosswalk in [`FIGURES.md`](FIGURES.md)
points to the relevant admitted family or analytic renderer.

Three worked/presentation figures were produced from internal diagnostic
payloads during dissertation editing. Their internal renderers are deliberately
excluded because copying them would reintroduce scratch dependencies. The
exact PDFs are retained as explanations with fixed hashes; quantitative claims
come from the admitted result families listed above.

## Claim boundary

The evidence supports results within the released synthetic density, optical,
detector and fitting models. It does not establish:

- installed Oxford apparatus performance;
- experimental state-classification or fit-success probabilities;
- a universal mapping from whole-image SNR to physical resolution;
- blind reconstruction of an unknown experimental density; or
- a time-dependent BEC–SSP–ID transition.

Those distinctions should remain attached to any reuse of the bundled tables
or figures.
