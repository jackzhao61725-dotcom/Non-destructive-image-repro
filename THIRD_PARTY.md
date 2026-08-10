# Third-party material and attribution

The BSD 3-Clause License in `LICENSE` applies to this repository's code. It does
not relicense the sources, data or dependencies listed below.

## Oxford research dataset

The original dataset is **not bundled** in this repository:

- Robert P. Smith, *Data Associated with the Publication "Interaction Shift of
  the Bose-Einstein Condensation Temperature in a Dipolar Gas"*, University of
  Oxford (2025), DOI
  [10.5287/ora-m8gpvdr2y](https://doi.org/10.5287/ora-m8gpvdr2y).

The public configuration transcribes only the numerical values needed by the
drivers and records the archive/member identity and SHA-256 digests needed to
trace those values. Those transcribed source values are not relicensed by the
repository's BSD licence. Anyone downloading or reusing the archive must follow
the [Oxford University Research Archive Terms of Use](https://ora.ox.ac.uk/terms_of_use)
and any rights or licence statement attached to the individual record or file.
ORA is the repository platform rather than the copyright owner.

No claim is made here that the dataset itself is licensed under CC BY 4.0.

## Associated article and Supplemental Material

The scientific source associated with the dataset is:

- M. Krstajić, J. Kučera, L. R. Hofer, G. Lamb, P. Juhász and R. P. Smith,
  "Interaction shift of the Bose-Einstein condensation temperature in a dipolar
  gas," *Physical Review A* **111**, L051303 (2025), DOI
  [10.1103/PhysRevA.111.L051303](https://doi.org/10.1103/PhysRevA.111.L051303).

The APS article and its linked Supplemental Material are published under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
Reuse must preserve attribution to the authors and the published article's
title, journal citation and DOI. This publication licence does not by itself
change the separate terms attached to the ORA dataset.

## Python dependencies

NumPy, SciPy and pytest are installed from their upstream distributions; their
source trees are not vendored here. They remain subject to their respective
upstream licences. The exact versions used by this release are declared in
`pyproject.toml`.
