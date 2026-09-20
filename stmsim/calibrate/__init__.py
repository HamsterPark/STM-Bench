"""Calibration scripts — every number the simulator is not allowed to invent (DESIGN.md §4.5).

Each module is a CLI that reads the real corpus (read-only) and writes one JSON table under
``stmsim.paths.calib_dir()``; nothing here is imported by the physics at run time except the
table readers (``junction.load_ldos_template`` reads ``sts_templates.json``).

=====================  ====================================  ===============================
module                 input                                 output (``$STM_BENCH_DATA/calib``)
=====================  ====================================  ===============================
``fit_creep``          glance_drift / runs_by_position       ``creep.json``
``hysteresis``         sxm_index_full fb_corr                 ``hysteresis.json``
``lambda_from_rowjump`` rowjump per session                   ``lambda.json``
``working_points``     sxm_index_full bias/setpoint/range     ``working_points.json``
``index_dat``          raw ``.dat`` tree                      ``index/dat_index.parquet``
``iz_templates``       dat_index + the ``.dat`` files         ``iz_phi.json``, ``sts_templates.json``
``poke_timing``        TipShapeWithReadback readback traces   ``poke_timing.json``
=====================  ====================================  ===============================
"""
