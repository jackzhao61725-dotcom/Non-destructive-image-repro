"""Fourier-plane pupil propagation helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def propagate_scattered_field(scattered_field: ArrayLike, pupil: ArrayLike) -> NDArray[np.complexfloating]:
    """Propagate a two-dimensional scattered field through a sampled pupil.

    Only the scattered part of the field, for example
    ``exp(1j * phase_map) - 1``, is transformed using
    ``ifft2(fft2(scattered_field) * pupil)``. This helper preserves that exact
    convention: grid centring is encoded by the caller, and no implicit shift,
    padding, or alternative Fourier normalisation is applied.
    """

    field = np.asarray(scattered_field)
    pupil_array = np.asarray(pupil)
    if field.ndim != 2:
        raise ValueError("scattered_field must be a 2D array")
    if pupil_array.shape != field.shape:
        raise ValueError("pupil must have the same 2D shape as scattered_field")

    return np.fft.ifft2(np.fft.fft2(field) * pupil_array)
