# Chapter 3 optical-readout schematics

This directory is the maintained source for the four Chapter 3 optical-readout schematics. The figures use one PSTricks and `pst-optexp` visual language. Yellow denotes the unaffected probe, and cyan denotes the complete atom-modified field before any method-specific phase dot, Fourier stop or PBS. Colours separate polarisation projections only after a PBS: green for the DFFI orthogonal port and distinct magenta and violet fields for the two DPFI ports. DPFI uses a half-wave plate set at $22.5^\circ$ before a fixed-axis PBS to realise the $\pm45^\circ$ projections. Purple denotes the PCI reference after the phase dot. Transmission coefficients appear only as factors multiplying a field amplitude. Camera thumbnails are greyscale because they represent intensity rather than a field component. The figures carry no standalone colour legend; the manuscript text and captions define the field meanings in context.

## Build

The build requires `latex`, `dvips`, and `ps2pdf` on `PATH`.

From this directory, run:

```powershell
.\build.ps1
```

The default command writes the four PDFs to the local `build/` directory and does not modify the dissertation figures:

- `figure_3_2a_pci_optical_readout.pdf`
- `figure_3_2b_dgi_optical_readout.pdf`
- `figure_3_3a_dffi_optical_readout.pdf`
- `figure_3_3b_dpfi_optical_readout.pdf`

After reviewing those outputs, install all four manuscript figures only with the explicit flag:

```powershell
.\build.ps1 -InstallDissertationFigures
```

The install step overwrites the four matching files under `dissertation/figures/`; it does not touch any other manuscript file.

## Source layout

- `common_style.tex` owns the shared palette and reusable optical glyphs.
- `pci.tex`, `dgi.tex`, `dffi.tex`, and `dpfi.tex` own the four method-specific layouts.
- `build.ps1` owns the output-name mapping and the opt-in manuscript installation step.

The camera thumbnails are schematic readouts, not simulated data. Their contrast follows the sign convention used in Chapter 3: PCI is locally bright for the stated positive detuning and phase-dot convention; the DPFI `H` and `V` ports are complementary.
