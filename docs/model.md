# Model and limits

## The two calculations

The repository follows the same deliberate separation as the dissertation.

1. **Repeated smooth BECs.** An Oxford-informed finite-temperature equilibrium
   supplies BEC0. A probe pulse deposits recoil energy, and the equilibrium
   model supplies the later BEC endpoint before the next image. The fitted
   quantities are the remaining condensate-population ratio `eta` and the
   axial-radius ratio `rho_y`.
2. **Static density comparison.** The same BEC0 is used as the smooth density
   object. SSP and ID profiles are constructed from it using the three-peak
   morphology and representative spatial scale reported for an erbium
   supersolid/droplet example. The fitted physical outputs are visible peak
   spacing and valley-to-peak density ratio.

The SSP and ID are additional density targets, not later endpoints of the
repeated-BEC calculation. No released calculation simulates the motion from
BEC0 to the SSP or ID, or the effect of the imaging pulse on that transition.

## Shared imaging chain

All four readouts begin from the same atom–light response. The linearly
polarised probe is written in circular components; their average phase gives
the common dispersive phase and their phase difference gives Faraday rotation.
The numerical model retains the complete complex circular transmission,
including the declared scattering attenuation.

The readout optics then convert the transmitted field into ideal camera-plane
intensities:

- PCI shifts the uniform reference at the Fourier plane and records its
  interference with the spatially changed field;
- DGI attenuates the uniform reference with a Fourier-plane stop;
- DFFI records the orthogonal Faraday port;
- DPFI records two ±45-degree ports and uses their signed difference.

The objective transfer is applied to the complex field before the intensity is
formed. Camera pixels integrate that intensity over their physical area.
Photon counts are Poisson distributed and each raw frame receives independent
Gaussian read noise. DPFI's H and V ports belong to one exposure but have
independently sampled detector fluctuations.

## What the public example does

`scripts/run_public_example.py` uses the compact helpers in
`src/non_destructive_image/public_inference.py` to recompute:

- the BEC0, SSP and ID column densities;
- PCI, DGI, DFFI and DPFI camera responses for each density;
- one fixed-seed DPFI raw-frame acquisition of BEC0;
- one fit of `eta`, `rho_y` and the axial centre, together with the same camera
  calibration nuisance quantities used in the dissertation estimator.

That fit is a software workflow check. One draw cannot measure bias, precision,
coverage or a support probability. Those conclusions use the admitted
ensembles in `results/`.

The public example reports repeated-BEC `q=1` and `q=15` values from the
authenticated ensemble summary. It does not silently replace the
finite-temperature calculation with a quicker approximation. The maintained
finite-temperature implementation and its inputs are included for inspection,
while the full retained outcome is supplied as evidence.

## Main limitations

- The atom is represented by the declared isolated 401-nm transition and fixed
  polarisation geometry, not a calibrated complete multilevel apparatus.
- The phase dot and dark-ground stop act on the ideal uniform reference; their
  finite size and overlap with low spatial frequencies are not modelled.
- The objective transfer is a measured-testbed surrogate, not a measured
  complex pupil for the installed Oxford imaging arm.
- The repeated-BEC route assumes fixed total atom number, recoil heating and
  complete re-equilibration between images. It contains no intermediate
  condensate dynamics, evaporation, technical heating or thermal-halo image.
- The SSP/ID recovery is conditional on the prescribed three-Gaussian density
  description. It is not blind reconstruction of an unknown cloud.
- DPFI and the other dispersive responses are periodic at sufficiently high
  column density. A strong whole-image SNR therefore does not guarantee a
  unique density interpretation.

These limits are why the dissertation's experimental proposal combines
installed-system calibration, observable-specific image analysis and RAI
cross-checks instead of transferring one simulated pulse duration or one
method ranking directly to the apparatus.
