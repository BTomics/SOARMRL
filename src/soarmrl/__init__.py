"""Sim-to-real deployment for the SO-ARM101 follower (Weeks 5-6).

Modules (each carries its contract in the docstring; implementations TBD):
- bridge: exported reach policy -> real arm, ~30 Hz, safety-clamped
- grasp: scripted pick-up of a predetermined object (the demo finale)
- trajectory_log: sim-vs-real recording for the gap analysis
"""
