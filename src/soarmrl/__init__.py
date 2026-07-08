"""Sim-to-real deployment for the SO-ARM101 follower (Weeks 5-6).

Planned modules:
- bridge: load exported Isaac Lab policy, run inference on encoder observations,
  stream joint-position targets via the LeRobot API at ~30 Hz.
- trajectory_log: record real encoder trajectories vs. sim trajectories for the
  same commands -- the data behind the sim-to-real writeup.
"""
