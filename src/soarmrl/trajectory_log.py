"""Sim-vs-real trajectory recording — the data behind the writeup.

CONTRACT — every bridge tick appends one record; a run produces one file.

RECORD PER TICK (minimum):
  - t_wall            monotonic timestamp (time.perf_counter())
  - obs               the exact vector fed to the policy (post-conversion!)
  - action_raw        policy output before scaling
  - target_sent       joint targets actually sent (post clamp/slow-mode)
  - encoders          raw joint positions as read back
  - phase/label       free-text tag ("reach:target3", "grasp:descend", ...)

FORMAT: keep it boring — np.savez or a flat CSV per run, filename with
timestamp + git hash of the policy. Loadable in a notebook with zero
ceremony.

THE COMPARISON THAT MAKES THE WRITEUP:
  same target pose -> rollout in sim (isaac_so_arm101 play script can dump
  sim trajectories; or re-run the policy offline against sim states) vs.
  the real log. Plot per-joint position over time, sim and real overlaid.
  The gap you see IS the sim-to-real gap: quantify it (RMSE per joint,
  settling time, overshoot) instead of hand-waving.

WHY THIS FILE MATTERS MORE THAN IT LOOKS: in autumn, the DexHand has no
encoders — this exact logging discipline, minus the encoder column and plus
camera-based joint estimates, is the Phase B evaluation plan. Get the habit
and the plotting code right now, in easy mode.
"""
