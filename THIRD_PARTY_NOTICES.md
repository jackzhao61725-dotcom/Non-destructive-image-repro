# Third-party material and attribution

The BSD 3-Clause License in `LICENSE` applies to this repository's code. It
does not relicense the publications, source data, local group documents or
software dependencies listed below.

## Oxford equilibrium source

The Oxford-informed BEC equilibrium is tied to:

- M. Krstajić, J. Kučera, L. R. Hofer, G. Lamb, P. Juhász and R. P. Smith,
  “Interaction shift of the Bose-Einstein condensation temperature in a
  dipolar gas,” *Physical Review A* **111**, L051303 (2025),
  [doi:10.1103/PhysRevA.111.L051303](https://doi.org/10.1103/PhysRevA.111.L051303).
- R. P. Smith, *Data Associated with the Publication “Interaction Shift of the
  Bose-Einstein Condensation Temperature in a Dipolar Gas”*, University of
  Oxford Research Archive (2025),
  [doi:10.5287/ora-m8gpvdr2y](https://doi.org/10.5287/ora-m8gpvdr2y).

The original ORA archive is not bundled. Configuration files contain only the
values and provenance records needed by the simulation. Reuse of the original
archive remains subject to its own record-level rights statement and the
[Oxford Research Archive terms](https://ora.ox.ac.uk/terms_of_use).

## SSP and ID morphology source

The constructed three-peak SSP and ID profiles use the qualitative density
progression and representative axial scale discussed by L. Chomaz *et al.*,
“Long-lived and transient supersolid behaviors in dipolar quantum gases,”
*Physical Review X* **9**, 021012 (2019),
[doi:10.1103/PhysRevX.9.021012](https://doi.org/10.1103/PhysRevX.9.021012).
The paper is cited as scientific evidence; no article figure is redistributed.
The public code constructs new analytic comparison profiles and does not claim
that they are equilibrium solutions for the Oxford trap.

## Imaging methods and detector information

The model and discussion draw on published PCI, dark-ground and Faraday
imaging methods, including:

- M. Gajdacz *et al.*, *Review of Scientific Instruments* **84**, 083105
  (2013), [doi:10.1063/1.4818913](https://doi.org/10.1063/1.4818913);
- F. Kaminski *et al.*, *The European Physical Journal D* **66**, 227 (2012),
  [doi:10.1140/epjd/e2012-30038-0](https://doi.org/10.1140/epjd/e2012-30038-0);
- M. Pappa *et al.*, *New Journal of Physics* **13**, 115012 (2011),
  [doi:10.1088/1367-2630/13/11/115012](https://doi.org/10.1088/1367-2630/13/11/115012).

Camera parameters are derived from the Hamamatsu ORCA-Fusion C14440-20UP
technical information. Manufacturer documentation is not bundled and retains
its own terms.

The objective transfer used by the active simulation is based on a local
Oxford group report, *A&L12: Optimised Imaging of Ultracold Quantum Gases*
(candidate 1073949, 24 April 2026). That report is not redistributed. The
configuration records the values used and distinguishes testbed measurements
from installed-apparatus calibration.

## Python dependencies

NumPy, SciPy, Matplotlib and pytest are installed from their upstream
distributions and are not vendored. They remain under their respective
upstream licences.
