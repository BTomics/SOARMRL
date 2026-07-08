"""Scripted pick-up of a PREDETERMINED object (bridge milestone M4).

CONTRACT — the demo finale: the reach policy gets the arm close, dumb code
does the grabbing. No learning, no perception; the object sits at a known,
measured position every run.

PHASES (a small state machine; each phase has an explicit success/timeout):

  1. APPROACH   reach-policy drives the end-effector to a PRE-GRASP pose:
                directly above the object, gripper open, at hover height.
                You define this pose by measuring where the object is —
                tape marks the spot on the table.
  2. DESCEND    NOT the policy: interpolate joint targets straight down to
                the grasp pose (policy was trained to reach, not to move
                precisely near obstacles — take over manually here).
  3. CLOSE      command the gripper joint until closed-enough / current
                rises. STS3215 has current feedback via LeRobot — a crude
                "grasped?" signal. Cardboard box or foam cube first: light,
                forgiving, doesn't dent.
  4. LIFT       interpolate back up to hover. If the object comes along,
                that's the money shot.
  5. (stretch)  PLACE somewhere else / RELEASE.

DESIGN DECISIONS LEFT TO YOU:
  - how to represent poses here: joint-space targets are simplest (record
    them by hand-posing the arm and reading encoders — no IK needed!)
  - phase transitions: time-based is fine for v1, feedback-based is better
  - where APPROACH ends: when the end-effector has been within tolerance
    of the pre-grasp target for N ticks

REUSES from bridge.py: the same clamping/slow-mode/e-stop machinery must
wrap these motions too — scripted does not mean safe.
"""
