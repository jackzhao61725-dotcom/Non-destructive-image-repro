# Model boundary

The state model is a three-radius parabolic Thomas-Fermi variational condensate
with contact and dipolar interactions. The two field orientations are minimised
as independent equilibrium endpoints. The model omits kinetic pressure, thermal
mean-field coupling and a kinetic-inclusive extended-GPE correction.

The optical model uses the selected perpendicular linear eigenpolarisation of
the ideal isolated 401 nm manifold. Coherent phase and absorption are assembled
from branch amplitudes; spontaneous scattering is assembled from non-negative
branch rates with one shared saturation denominator. Nearby levels, optical
pumping, spatial spin texture and apparatus calibration are outside the model.

The repeated-exposure update assumes fixed trapped number, recoil-only deposited
energy and complete re-equilibration between frames. The Oxford non-saturation
relation is used as an equilibrium constitutive surface, not as a measured
fixed-number dynamical trajectory. Population-support and temperature guards
belong to that closure. A separate 30% condensate-depletion threshold is only a
reporting screen: a declared diagnostic can continue beyond it, but the
continued state does not thereby become an accepted sequence state.

The Oxford archive exposes the fitted Fig. 2c line in beta-corrected plotting
coordinates, while the selected atom-number repetition is stored in measured
coordinates. Supplemental Eq. S5 therefore gives the single conversion used at
runtime:

```text
C_measured = beta C_Fig2c        S_measured = beta^(3/5) S_Fig2c
X_measured = beta^(2/5) X_Fig2c  Y_measured = beta Y_Fig2c
```

The measured-coordinate coefficients are derived in code and are not stored as
a second independent authority.

PCI is the primary readout. DGI components are included because they are tested
by the code, but this snapshot does not turn the methods into a general
instrument comparison.
