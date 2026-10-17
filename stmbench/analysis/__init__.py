"""Analysis that turns measured data into the numbers a paper reports.

Everything here reads instrument data only (arrays an agent can obtain through the instrument
API or the saved files) and never the simulator's truth, so the same code can serve the
privileged reference solutions (``stmbench.reference``) and, later, an agent's measurement tools.
"""
