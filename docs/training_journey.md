# SO-ARM101 PickPlace — the training journey

Raw material for a writeup or LinkedIn post. Every number here is measured, and
every wrong turn is included, because the wrong turns are the interesting part.

**Result:** pure-RL pick-and-**place** on an SO-ARM101 in Isaac Lab — pick a 3 cm
cube, carry it to a commanded spot on the table, set it down, release, withdraw.
**56.5% strict success** at 12 000 iterations. Prior art for this arm is
imitation/VLA only (NVIDIA's Sim-to-Real SO-101 Workshop uses teleop demos +
GR00T); pure-RL place appears not to exist for it.

"Strict" means: cube within 20 mm horizontally and 10 mm vertically of the goal,
linear velocity under 0.02 m/s, angular under 0.05 rad/s, gripper open, and the
end-effector withdrawn 50 mm. It's a harder bar than most demos report.

---

## The capability ladder — what it learned to do, in order

Nothing here arrived at once. Each new ability had to be **bought with a specific
reward term**, and each one first failed in its own legible way. The pattern
repeats so consistently it's arguably the real story:

> Add the ability to do X → the policy finds a way to score well *without* doing X
> → the arithmetic shows why that pays better → fix the balance → X appears.

| # | new capability | what bought it | what it did first instead |
|---|---|---|---|
| 1 | reach a commanded pose | goal-tracking reward | hunted and oscillated on hardware |
| 2 | close on the cube | reach + grasp shaping | parked 19 mm away, couldn't close |
| 3 | lift it clear of the table | `lifting_object` | slid it along the table |
| 4 | carry it toward a goal | latched goal tracking | farmed tracking without ever picking up |
| 5 | **set it down** | `object_at_target_on_table` | **hovered above the target forever** |
| 6 | come to rest, not drop | `object_at_rest` | dropped it from height |
| 7 | **let go** | `object_released` | held on after a perfect placement |
| 8 | withdraw and go home | `joint_deviation` | flailed into a contorted pose after release |
| 9 | grasp top-down, not sideways | `grasp_top_down` | carried the cube at ~65° off vertical |
| 10 | do it across a wider region | wider goal box + penalty ramp | lost the pick entirely, twice |

### Stage 3–4: pick and carry (the baseline)

The upstream lift task gave a working pick. After the workspace fix and a reset:
**cube off the table 80% of steps, gripper 2.0 cm from the cube, cube landing
~6.3 cm from the commanded point** at 4000 iterations.

But the goal was a weak knob — the policy had learned "lift the cube to roughly
this region," not "track this setpoint." That distinction mattered later.

**The anti-slide latch** was the structural piece here. Every place reward is gated
behind "has this cube *ever* been lifted clear this episode." Without it a policy
can shove the cube along the table into the goal and collect everything. Sliding
never sets the latch, so it earns nothing.

Worth noting the latch was validated by a *negative* result: goal tracking sits at
exactly 0.0 until iteration ~600 and only rises in step with the lift. If it were
leaking across episode boundaries it would have been paying from iteration zero.

### Stage 5: setting it down — the hardest single step

Moving the goal from mid-air onto the table is the whole task, and it's where
every earlier attempt had died. The reason is a contradiction that's easy to write
by accident:

**the reward for being near the goal was gated on the cube being airborne.** So the
reward switched off exactly as the cube arrived. The policy was being punished for
finishing.

Fixing that gate wasn't enough on its own, because of the hovering economics
(27.2/step vs 21.0 — see below). It took both: a placement reward that stays alive
*through* the descent, with a tight vertical kernel so the last few centimetres are
the steepest part of the gradient, **and** decaying the lift bonus once the pick
was reliably learned.

Result: cube distance to goal went from ~9 cm to ~4 cm, and the arm started
committing to the descent instead of hovering.

### Stage 6–7: rest, then release

Placement immediately produced a new failure that looked like success. The arm
carried the cube to the goal and **dropped it** — from a few centimetres up, and
sometimes near the goal rather than on it.

Nothing objected, because **a dropped cube and a placed cube score identically once
both come to rest.** The reward couldn't tell them apart. Lift duty fell 70% → 13%
and the gripper drifted 2.0 → 4.7 cm from the cube; the policy had discovered that
letting go early was cheaper than carrying all the way down.

Two terms fixed it. One rewards the cube being genuinely *at rest* — low linear
**and** angular velocity, gated on being at the target, so a still cube in the
wrong place scores nothing. The other rewards opening the gripper, but only once
the cube is already placed and settled, so an early drop earns nothing.

There was a subtler coupling too: an always-on reach reward pays the arm to stay
near the cube it is supposed to *release*, and directly contradicts the success
condition's requirement to withdraw 5 cm. Gating the reach term off after the pick
removed that contradiction.

### Stage 8–9: posture, and the flail after release

With placement working, what remained was ugly rather than wrong: the arm carried
the cube in a contorted, sideways grip and then flailed back into a strange pose
after letting go.

Measured, the carry posture was worse than a *random* arm configuration — the
policy had an active preference for a sideways grip, not a missing gradient. And
the reset pose turns out to be 6° from straight down, so pulling toward home was
pulling toward *good* posture, not away from it.

This stage cost the most runs for the least gain, and produced the "ship one term
per run" lesson the hard way.

### Stage 10: doing it over a wider region

The final increment pushed the goal region 5 cm further out and away from the base.
The success rate went **up** — 56.5% against 55.3% — on a region where the measured
availability of a top-down grasp is meaningfully worse (0.672 vs 0.766).

Getting there took three failed runs, all of them about the penalty curriculum
rather than the box, and all of them covered below.

### Where the numbers landed

| stage | place success |
|---|---|
| placement wired, first working version | — (cube ~4 cm from goal) |
| release + rest added | first non-zero |
| posture term, 4000 iterations | 10.1% |
| same config, 12 000 iterations | 55.3% |
| wider goal box + penalty ramp, 12 000 | **56.5%** |

The jump from 10.1% to 55.3% is worth dwelling on: **it's the same configuration,
just trained three times longer.** A run that looks like a mediocre result at 4000
iterations was nowhere near converged. Several earlier increments were probably
judged too early.

---

## Part 1 — Teaching it to pick (the 11-run losing streak)

### The arithmetic that broke the streak

Eleven consecutive training runs where the arm never picked the cube up. The fix
wasn't more iterations or a bigger network — it was **doing the reward arithmetic
at the desk before training**.

Compute what each competing behaviour pays per step, and check the one you want
actually wins. It usually doesn't:

| behaviour | reward/step |
|---|---|
| hover over the target holding the cube | **27.2** |
| set it down and let go | **21.0** |

The policy wasn't broken. **Finishing the task was a net loss, and PPO correctly
refused.** Lift paid a flat 15/step for as long as the cube stayed airborne, and
the tracking terms measured 3D distance — so a cube held 8 cm above the target
already scored 11.8 out of 16. Letting go meant giving up the annuity.

Rebalanced to 15.2 vs 33.0 and the behaviour appeared. Every earlier failure had
been a farmable plateau the policy found before I did.

### Reading a reward number as a physical distance

The single most reusable trick from the whole project.

`Episode_Reward/*` in Isaac Lab is the time-averaged **weighted** term. Divide by
the weight, then by any gate's duty cycle, then invert the kernel. A `1 - tanh(d/std)`
reward becomes a distance in millimetres:

| `reaching_object` | EE-to-cube |
|---|---|
| 0.85 | 7.6 mm — grasps |
| 0.64 | 19 mm — **cannot close on a 30 mm cube** |
| 0.30 | 43 mm — not even close |

For eleven runs "reaching is around 0.6" meant nothing. Once it read "the gripper
parks 19 mm away and physically cannot close," the diagnosis was obvious and it
was never about the grasp reward at all — it was the approach.

**It's also the most error-prone step.** The weight division got skipped six times
by hand in one session, twice producing values above 1.0 — arithmetically
impossible, since the quantity is a product of two kernels each bounded by 1.
That's what `scripts/tools/read_terms.py` exists for.

### The goal box that was 30% impossible

The task inherited a goal region from an SO-100 template: `x[-0.1, 0.1],
y[-0.3, -0.1]`. The SO-100 convention is −y-forward; the SO-101 is +x-forward.
Nobody re-derived it.

Sampling 6 million arm configurations off the URDF showed that box is only
**68–72% reachable at any height**. Roughly a third of commanded goals were
physically impossible, and the policy had sensibly learned to ignore the goal
input — which had been logged for weeks as "the target pose is a weak knob."

It wasn't a weak knob. It was frequently an impossible one.

**Reachability alone hides the problem, though.** A point reachable by exactly one
contorted configuration counts as "reachable." *Configuration density* — how many
arm poses put the fingertip there — is what separates comfortable from cramped,
and it's what actually predicts whether a policy can work there.

### A sign error that made the intended behaviour impossible

A term rewarding top-down grasps used the gripper's approach axis. The axis had
been negated at some point to "fix" a posture problem.

Measured off the URDF: with the negated sign, the maximum achievable score at cube
height was **0.427, with 0% of poses above 0.7** — a top-down grasp was
*geometrically unreachable*. The term peaked only at 18–25 cm above the table with
the gripper **inverted**.

So the reward was paying for holding the cube overhead, gripper pointing up. **The
behaviour the sign flip was meant to eliminate was the one it created.** Restoring
the raw sign gave +0.994 at the arm's own home pose — 6° from straight down.

Both of these were invisible in code review. Both took ten minutes to find by
sampling the kinematics.

### A measurement that inverted when I changed the question

Earlier work raised the goal box's near edge because near targets were
"config-starved" — 1.95% of configurations at x≈0.10 versus 6.20% at x≈0.25.

That measurement was taken with the goal box **airborne** (z 0.06–0.20). When the
task changed to placing on the table and I re-measured at z=0.015, **the
relationship inverted**: density is now *highest* near the base, 74.6% of peak at
x=0.10 falling to 34.1% at x=0.30. The genuinely cramped region turned out to be
the outer *corners*, not the near edge.

I nearly repeated the old number as justification for a new design decision.
**A workspace measurement is only valid at the height you measured it.**

### The units trap that cost two runs

`modify_reward_weight(num_steps=...)` counts **environment steps**, not training
iterations. At 24 steps per iteration, `num_steps=12000` fires at iteration **500**,
not 12000 — a third of the way into a 1500-iteration run instead of never.

Two reward decays were blamed for "destroying the pick" when the real story was
that they fired long before the pick existed. The lesson isn't "never decay," it's
**date the behaviour from a previous run's logs, then decay well after it.**

### Shipping two changes at once cost a whole run

One increment added a posture term and a joint-regularization term together,
reasoning that they act in disjoint phases so a failure would stay attributable.

The phases were disjoint. The effects were not. Placement metrics fell 20–25% and
the run couldn't say which term did it. Disentangling took an extra full run, and
the answer was ~9% and ~13%.

**One term per run.** The temptation to bundle is strongest exactly when you're
most confident, which is exactly when it's most expensive.

---

## Part 2 — The curriculum that had no safe landing spot

The best single finding, and it cost three runs.

A penalty on jerky motion ramps from −1e-4 to −1e-1 at a chosen iteration. A 1000×
step. It failed at **both** ends:

**Landing early (~iteration 420):** the pick never formed. Five reward terms spiked
briefly and then read *exactly* zero forever. Exploration noise had been *climbing*
— 1.0 up through a peak of 2.75 — then crashed vertically to 0.32 and never
recovered.

The arithmetic: positive reward at that moment was ~0.32, against penalties of −0.9
and −0.45. **The penalty was four times the entire positive signal.** The cheapest
available gradient wasn't "find the cube," it was "stop moving." The policy took
that deal and never came back.

**Landing late (~iteration 2500):** the opposite, and worse. The pick formed fine —
11.8% lift duty, 5.3% success — and then the step destroyed it. With actions
effectively free for 2500 iterations, the fastest route to reward is fast, jerky
motion, so raw action rate had grown to **~500 against a converged 4.2**. The
penalty arrived at roughly −50 per step and invalidated everything the policy had
learned at once.

**The generalization: the cost of a discontinuity scales with how much behaviour it
disrupts.** Moving a large step *later* makes it strictly more dangerous, not less.

The fix was a geometric ramp — 10× per stage instead of 1000× once — so no single
step exceeds a fraction of the positive reward, and mild early pressure keeps the
action rate from ever running away.

There's a quieter lesson in it too: at iteration ~420 the original config was never
really a curriculum. It was "penalties on from the start" with a grace period, and
the policy grew up under the constraint instead of meeting it later. **That's why
it worked, and nobody had noticed.**

---

## Part 3 — The collapse that looked like reward hacking

Mid-run, success fell from 36.6% to 0.2% while every other metric stayed healthy.
The obvious reading: the policy found an exploit — hold the cube at the target
forever, farm the dense placement rewards, never release.

It's a satisfying story. It was wrong.

The release reward routes through the same at-target kernel as everything else, so
releasing pays strictly *more*:

| | holding | released |
|---|---|---|
| total per step | 21 + 17K | **22 + 25K** |

Releasing wins by `8K + 1`. Holding buys nothing. Summing the actual logged values,
the policy had moved **~2 reward per step downhill**.

**A policy moving downhill is never optimising.** That reframes the whole
investigation: don't look at the rewards, look at the optimiser.

The real cause was entropy collapse. The gripper-open action is near-discrete — as
the action distribution narrowed, the policy simply stopped *sampling* it, so the
gradient for releasing vanished. Reach, carry and hold are continuous behaviours
reinforced every step; they were untouched. Only the sharpest, most recently
acquired skill disappeared.

PPO's entropy bonus noticed the reward drop, pushed exploration back up, and the
release returned on its own — 48.5%, then 52.8%, then 56.5%.

**If I'd "fixed" the reward function to close the exploit I'd have broken a task
that was working.**

### The diagnosis that was wrong first

I blamed the resume. The collapse happened shortly after restarting from a
checkpoint, and there's a real known hazard there (the curriculum counter lives on
the environment, so resuming silently rewinds every scheduled weight).

Two things killed that theory. Plotting the learning rate across the boundary
showed both legs behaving identically. And the collapse landed at **iteration
~4900**, not at the 4000 restart. A resume artefact would have hit at 4000.

**Point samples lie; plot the curve.** I'd also called the run plateaued off two
noisy samples ~1000 iterations apart, right as it was climbing through a
penalty-absorption window. Don't judge convergence in the ~1000 iterations after a
curriculum event — that window is structurally flat for reasons that have nothing
to do with the policy.

---

## Part 4 — Everything that broke between sim and hardware

### The oscillation that came from a helpful-looking observation

The first real hardware loop went into a sustained limit cycle — ±0.5 rad, ~1.6 s
period, never damping.

Suspect one was the safety throttle. Loosening it made things **wilder**, which
falsified that cleanly: the throttle was *damping* the oscillation, not causing it.

The actual culprit was the velocity observation. In simulation it's true physics
velocity with zero latency. On hardware it was a finite difference of noisy
encoder readings, delayed by USB latency — a mis-phased derivative term feeding
positive feedback.

**Zeroing that block of the observation killed the oscillation entirely.** A
reference rollout in simulation converged cleanly with the same target, which is
what proved the policy was fine and the *input* was the problem.

### A calibration error invisible to every software check

The shoulder rotation direction was inverted in the calibration table. Forward
kinematics was self-consistent. The closed loop was self-consistent. Every unit
test passed.

**A symmetric sign error cancels in the round trip.** It only showed up when
someone commanded "move +x" and the arm physically went left.

Some classes of bug are only observable in the physical world. Budget for that.

### The power supply that browned out under load

The gripper stall-held the cube for the entire lift, and the 5 V supply couldn't
sustain it — brownouts and latched servo overload faults, which then require a
physical power-cycle to clear.

Fixed in two parts: back the grip command off to a *gentle hold* once the grasp has
latched (the cube blocks the jaws anyway, so the encoder reads the same and the
observation is unchanged — but the push force drops enormously), and write an
explicit torque limit to the gripper servo at connect.

**Sustained force is a power-budget problem, not a control problem.** Nothing in
simulation hints at it.

### The home pose the real arm cannot reach

The recurring root cause behind several unrelated-looking symptoms.

The trained policy's home posture wants one wrist joint at 1.57 rad. The real servo
tops out around 1.195. Because the action is an *offset from the home pose*, that
gap corrupts both the observation and every command.

It forced a 2× action-scale hack to stop the arm undershooting the cube, and it
caused a residual shake at the target. Two workarounds for one root cause — which
is usually the signal that you're treating symptoms.

Then, this week, sampling 2000 observations through the actual conversion code:

| joint | out-of-range commands |
|---|---|
| wrist_flex | **48.5%** |
| shoulder_pan | **22.1%** |
| wrist_roll | 4.9% |

The joint limits are enforced in **radians**, taken from the URDF — but the URDF's
range is *wider than the servos' actual travel*. So the clamp passes, the
conversion produces an out-of-range command, and the servo saturates **silently**.
Nearly half of all wrist commands.

**A limit check in the wrong unit is worse than no limit check**, because it looks
like it's working.

### How accurate does perception actually need to be?

The policy reads the cube's position from the simulator. On hardware that's a
camera, a detector, and a calibration — and it's the single largest untested gap.

Rather than guess, I perturbed that one input and replayed the trained policy:

| perturbation | magnitude | result |
|---|---|---|
| noise, resampled every control step | ±10 mm | **no visible degradation** |
| constant per-episode bias | ±5 mm | clean |
| constant per-episode bias | ±10 mm | struggles, still completes |

**Those aren't the same test, and the difference is the useful part.** Per-step
noise is resampled 50 times a second, so across a 250-step episode a closed-loop
controller averages it away almost for free. A miscalibrated camera is wrong in
the *same direction* every frame — nothing to average.

So the spec that came out isn't "buy a precise camera," it's **repeatability
matters far more than per-frame precision.** A noisy but unbiased sensor is fine.
A quiet but miscalibrated mount is not. That's a genuinely useful thing to know
*before* buying anything.

### Two policies composed on real hardware

A side result worth its own post: the grasp policy and a separately-trained reach
policy were composed in a single hardware run — grasp policy picks the cube up
(contact-rich, the part it's good at), hands off on a *confirmed* grasp, reach
policy carries it to a commanded pose (controllable end-effector servo, the part
*it's* good at), then release.

This also solved a problem rather than just demonstrating a technique: the lift
policy's goal input was a weak knob, while the reach policy's is a real setpoint.
Composition beat retraining.

---

## The transferable lessons

1. **Every new capability gets bought, and first fails in a specific way.** The
   pattern held at all ten rungs: add the ability to do X, watch the policy find a
   way to score well *without* doing X, work out why that pays better, rebalance.
   Budget for the second step — it isn't a setback, it's the mechanism.
2. **Do the reward arithmetic before training.** Compute what each competing
   behaviour pays per step and check the one you want wins. Most "the policy is
   broken" cases are the policy being right about a reward function that's wrong.
3. **Watch for rewards that switch off on success.** The place reward was gated on
   the cube being airborne, so it died exactly as the cube arrived — the policy was
   punished for finishing. Any gate that closes at the goal is this bug.
4. **Two behaviours that score identically will not be distinguished.** A dropped
   cube and a placed cube look the same once both are at rest. If you can't tell
   them apart in the reward, neither can the policy.
5. **Train longer before judging.** The same configuration read 10.1% at 4000
   iterations and 55.3% at 12 000. Several earlier increments were probably called
   on runs that hadn't converged.
6. **Convert reward numbers into physical units.** A number you can't state in
   millimetres isn't a diagnosis.
7. **Measure the workspace instead of inheriting it.** Two real bugs came from
   sampling kinematics; neither was visible in code review. And a workspace
   measurement is only valid at the height you took it.
8. **Ship one change per run.** Bundling is most tempting exactly when it's most
   expensive.
9. **Big discontinuities have no safe landing spot.** Ramp them. Cost scales with
   how much behaviour there is to disrupt.
10. **A policy moving downhill is never optimising.** Look at the optimiser, not the
   rewards.
11. **Plot the curve; point samples lie.** Especially near curriculum events.
12. **Write down falsifiable predictions in the config.** Several of mine were
   falsified, and being able to see that in a comment was worth more than being
   right would have been.
13. **Some bugs are only observable physically.** Symmetric sign errors cancel in
   every software check you'll write.
14. **Check limits in the unit they're enforced in.** A radian clamp doesn't protect
    a servo that thinks in normalized ticks.

---

## Candidate post angles

- **"Ten things a robot had to learn to put a block down — and how each one first
  went wrong."** The capability ladder as the spine. Strongest option for a general
  audience: it's a progression story rather than a debugging story, the failures
  are individually funny (hovered forever, dropped it from height, held on after a
  perfect placement), and the repeating pattern lands as a real insight. Carries a
  visual too — the ladder table, or before/after renders.
- **"My robot refused to finish the task, and it was right."** The hovering
  economics — 27.2 vs 21.0 per step. Single beat, concrete numbers, a
  counterintuitive punchline. Probably the most shareable.
- **"Three runs to learn that a 1000× step has nowhere safe to land."** Fails early
  *and* late, for opposite reasons, with a clean generalization.
- **"It looked like reward hacking. It was moving downhill."** Great for a
  technical audience — the satisfying story was wrong and the arithmetic said so.
- **"Everything that broke between sim and hardware."** Velocity observations,
  invisible sign errors, power brownouts, an unreachable home pose, and a
  perception spec that turned out to be about repeatability. Honest sim-to-real
  content is rarer than success demos.
- **"Two policies, one arm, one run."** Hierarchical composition on real hardware.

The strongest framing across all of them is that **the failures were legible**. Every
one had a number attached, and most were diagnosed at the desk rather than by
training longer.
