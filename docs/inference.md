# Inference contract

Each field orientation owns an independent atom, bright-reference and dark raw
frame. The generator constructs those frames on a 1024 by 1024 physical grid;
the estimator evaluates a 512 by 512 object grid with the same camera sampling,
pupil and scalar optical response.

The fitted object is a five-parameter projected compact profile:

1. peak column density;
2. object-plane y and z centroids;
3. free y and z radii.

Positive scale parameters use logarithmic fit coordinates. Four declared starts,
fixed physical bounds and separate incident-count and dark nuisances are used for
each endpoint. The terminal with the smallest finite weighted chi-square is
reported. Generator widths are consumed only after fitting, when errors are
scored.

The default public inference is one deterministic detector-noise draw. It checks
that the published parameterisation and likelihood can be rerun; it does not
estimate sampling coverage, profile-family uncertainty, thermal-halo
contamination or state-model sensitivity. Those are distinct qualifications,
not extra components of the detector-noise interval.
