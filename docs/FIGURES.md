# Dissertation figure crosswalk

The release bundles the exact 17 PDFs used by Chapters 2–6. Run

```text
python scripts/render_public_figures.py --check-only
```

to verify their hashes, or omit `--check-only` to collect them into a new
directory with a machine-readable `index.json`.

## Density and atom–light explanation

| Figure | Public source | Role |
| --- | --- | --- |
| `figure_2_2_equilibrium_density_profiles.pdf` | `configs/three_state_target_trap_profiles_v4.json`; `scripts/render_chapter_2_equilibrium_profiles.py` | BEC0, SSP and ID column-density comparison |
| `figure_3_1_atom_light_interaction.pdf` | `scripts/chapter_3_interaction_overview/atom_light_interaction_overview.tex` | Common phase and Faraday rotation in one transmitted probe |
| `figure_3_2a_pci_optical_readout.pdf` | `scripts/optical_readout_schematics/pci.tex` | PCI concept |
| `figure_3_2b_dgi_optical_readout.pdf` | `scripts/optical_readout_schematics/dgi.tex` | DGI concept |
| `figure_3_3a_dffi_optical_readout.pdf` | `scripts/optical_readout_schematics/dffi.tex` | DFFI concept |
| `figure_3_3b_dpfi_optical_readout.pdf` | `scripts/optical_readout_schematics/dpfi.tex` | DPFI concept |
| `figure_3_4_phase_wrap_response.pdf` | `scripts/render_chapter_3_phase_wrap.py`; active density and optical configs | Phase-only response versus column density |

These are analytic or conceptual explanations. They do not carry recovery
statistics.

## Camera response and SNR

| Figure | Numerical authority | Role |
| --- | --- | --- |
| `figure_4_2_four_method_noisy_200us.pdf` | `results/chapter_4_three_state_four_method_static_v3/` | One simulated noisy acquisition for the three states and four readouts |
| `figure_4_3_four_method_snr.pdf` | `results/chapter_4_three_state_four_method_snr_400_v2/` | Whole-image SNR from 25 to 400 microseconds |
| `figure_4_4_target_tfbec_repeated_images.pdf` | `results/chapter_4_repeated_bec_dgi_dpfi_v1/` plus the exact bundled manuscript presentation | Repeated-BEC SNR and one DPFI image sequence |

Figure 4.4 is presentation material, not evidence of a time-resolved
trajectory. Each pulse-duration row begins from a separately prepared BEC0,
and image `q` is an equilibrium endpoint after `q-1` earlier pulses.

## Physical recovery

| Figure | Numerical authority | Role |
| --- | --- | --- |
| `figure_5_1_reference_forward_fit.pdf` | Exact bundled worked explanation | How a trial BEC becomes a camera prediction and is selected by the raw-frame residual |
| `figure_5_2_dpfi_eta_rho_repeated_exposures_presentation_v2.pdf` | `results/chapter_5_bec_multiframe_eta_ry_dpfi_v1/` | Recovered population and axial-size ratios over the repeated BEC sequence |
| `figure_5_3_dpfi_recovery_spread_vs_first_image_loss.pdf` | Same admitted BEC recovery family | First-image recovery span compared with pulse-induced condensate change |
| `figure_5_4_three_state_fit_examples_v1.pdf` | Exact bundled worked explanation | Representative DGI/DPFI SSP and ID fits within the prescribed density description |
| `figure_5_5_three_state_density_recovery.pdf` | `results/chapter_5_three_state_density_recovery_v2/` | Recovered visible peak spacing and valley-to-peak ratio |
| `figure_5_6_snr_vs_recovery_width.pdf` | `results/chapter_5_snr_recovery_relation_v1/`; `configs/public_chapter_5_snr_recovery_relation_v1.json` | Observable-specific relation between whole-image SNR and recovery span |

Figures 5.1 and 5.4 explain the fitting logic. Their original diagnostic
renderers are not distributed, and the figures are not used as retained
precision evidence. Figure 5.6 can be regenerated solely from admitted tables;
the regenerated PDF is byte-identical to the manuscript version.

## Experimental strategy

| Figure | Public source | Role |
| --- | --- | --- |
| `figure_6_1_rai_anchored_strategy.pdf` | `scripts/plot_chapter_6_rai_strategy.py` | Simulation-informed dispersive measurement with RAI cross-checks |

This is a proposed logic for an Oxford implementation, not a model of the
complete installed optical path and not a report of experimental data.
