"""Replays of benchmark episodes for an audience that has never used an STM.

A replay is built after the episode from its ledger (``episode.json``, ``driver.json``,
``events.json``, ``tip_timeline.json``, ``session/``) and shows two things side by side:
what the model saw and did, and what only the audience sees — the tip, the true sample and
the answers. Nothing here runs during an episode, so it cannot change a result or leak to
the model::

    python -m stmbench.cli replay <run dir> [--out DIR] [--single-file] [--open]
"""
