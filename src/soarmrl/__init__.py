"""Sim-to-real deployment for the SO-ARM101 follower (Weeks 5-6).

Modules (each carries its contract in the docstring; implementations TBD):
- bridge: exported reach policy -> real arm, ~30 Hz, safety-clamped
- grasp_bridge: exported lift/PickPlace policy -> real arm (grasp-family peer
  of bridge; 28-dim obs, binary gripper, FK object-tracking after grasp)
- kinematics: SO-101 forward kinematics (joint angles -> EE xyz) for the above
- grasp: scripted pick-up of a predetermined object (the demo finale)
- trajectory_log: sim-vs-real recording for the gap analysis
"""
