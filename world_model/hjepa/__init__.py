"""Issue #123: H-JEPA (kevinghst/H-JEPA @ 97054ec, MIT; vendored in ``vendor/``) on NovPhy.

``vendor/`` holds the verbatim official modules (only their top-level imports made package-relative;
per-file upstream and vendored sha256 in ``vendor/VENDOR.json``). ``novphy`` adapts them to NovPhy's
inputs, clip sampler, launch encoding, readouts and decision cost without changing the H-JEPA objective.
"""
