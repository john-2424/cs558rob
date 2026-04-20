# PM2 Q&A Prep

Topic-organized cheat-sheet for the live Q&A (classmates) and the viva
(professor). Answers are phrased so you can read them aloud and adapt.
Numbers match `results/m2/eval_results.json` exactly.

---

## 1. Foundations

**Q: What is PPO, and why did you choose it?**
PPO — Proximal Policy Optimization — is an on-policy actor-critic
algorithm. It maximises the expected return while clipping the ratio
between the new and old policy so that each update can't move the
policy too far. We chose it because it's the default for continuous
control, is robust to hyperparameter choices, and TorchRL ships a
well-tested implementation (`ClipPPOLoss`). SAC or TD3 would also have
worked but are off-policy and more sensitive to replay-buffer hygiene;
PPO was simpler to ship for a six-day iteration window.

**Q: What is residual reinforcement learning?**
Residual RL trains a learned policy whose output is *added* to a
non-learned nominal controller. The controller does the bulk of the
work; the policy only learns the correction. In our case the nominal
controller is PD+IK+RRT\*, and the residual is a bounded position
offset on the joint waypoint.

**Q: What is RRT\*, briefly?**
Rapidly-exploring Random Tree with the asymptotic-optimality extension.
It samples random joint configurations, extends a tree toward them
while rejecting collisions, and rewires edges when a shorter path is
found. Asymptotically it converges to the optimal path. We use it in
joint space (seven dimensions) with collision-checking against the
table and cube.

**Q: What's a PD controller and why velocity control?**
Proportional-Derivative feedback: commanded effort is proportional to
the position error plus the derivative error. We use joint-space PD
with PyBullet's `VELOCITY_CONTROL`, so our controller outputs joint
velocities (not torques). This hides low-level actuator dynamics and
lets the residual be expressed cleanly as a waypoint offset.

**Q: What's a Wilson confidence interval?**
The Wilson score interval is the correct way to put an interval around
a binomial proportion — more robust than the normal approximation
(Wald interval), especially near 0 or 1. With n=100 and p=1.0 the
Wilson upper bound is about 0.983, which is why our headline plot had
to clip error bars to non-negative values.

**Q: What does "hybrid control" mean here?**
Hybrid = classical planning + learned residual. The classical stack
generates joint waypoints; the learned PPO policy adds a bounded
correction; the PD controller tracks the combined target. It is not
"switching" between controllers — both are active at every step in the
relevant phases.

**Q: Why PPO instead of behavioural cloning or supervised learning?**
We don't have a ground-truth "correct residual" demonstration dataset.
Residual RL lets the policy discover corrections from its own
interactions with the environment, guided by the reward. BC would need
us to manually produce correction trajectories for every perturbation,
which defeats the purpose.

---

## 2. Problem framing

**Q: Why cube pose perturbation?**
It's the simplest perception-uncertainty proxy: in a real deployment a
camera-based pose estimator has 1–5 cm of error on a cube this size.
Perturbing the cube pose before execution mimics that without
requiring a camera pipeline.

**Q: Why bound the residual?**
Two reasons. First, safety — an unbounded residual could command
unsafe joint velocities. Second, reward-hacking mitigation — an
unbounded residual could trivially overshoot the PD target and make
the learned policy do the entire job, which defeats the "residual"
framing and wastes the classical backbone.

**Q: Why a position-target residual instead of a velocity residual?**
The original proposal specified a *torque* residual (`τ_total = τ_PD +
τ_RL`). We pivoted to velocity-additive in early implementation — the
PD controller was already wired against PyBullet's `VELOCITY_CONTROL`
mode and torque-level control would have cost another round of gain
tuning. Training on `qd_total = qd_pd + a·RES_MAX_VEL` was unstable:
the residual interfered with PD's tracking loop, gains wanted to fight
it. Switching to a position-target residual —
`q_target_corrected = q_target + a·RES_MAX_POS` — lets PD do its job
on the corrected target. Training stabilised immediately after that
pivot.

**Q: Why phase-gated?**
The residual is only useful in the phases where the cube pose actually
matters for control: pre-grasp and grasp-descend. During lift, the cube
is already attached and any residual motion could disturb the attach
constraint. During transfer/place/retreat, the cube target is defined
in world space, not cube space, so the residual has nothing to correct.
We tested both configurations; the phase-gated version trained faster
and was more stable.

**Q: Why XY perturbation primarily?**
XY dominates real pose-estimation noise for top-down tabletop grasping.
Z is largely determined by the known table height. Yaw matters less
because the cube is symmetric. We do include small Z and yaw
perturbations proportional to XY, but XY is the main robustness axis
we're studying.

---

## 3. Implementation

**Q: Walk me through the observation vector.**
Forty-one dimensions, fixed-scale component-wise normalized before the
network sees it. Seven joint positions and seven joint velocities give
the arm state. Three for end-effector position and three for Euler
angles — the "where the hand is." Three for cube position, three for
cube Euler. Three more for the vector from end-effector to cube,
explicitly — even though the policy could derive it from the previous
components, giving it directly accelerates learning. Seven for the PD
controller's nominal velocity command, so the policy knows what PD is
about to do. One for a phase indicator (integer-encoded). Three for
the current perturbation offset, which we added late in iteration
after finding an obs-dim bug. And one more for a waypoint-progress
scalar in `[0, 1]` that tells the policy how close the classical
scaffold is to its next target — added in a later tuning pass to help
the policy time its corrections relative to the classical trajectory.

**Q: What's the action, exactly?**
Seven continuous values, clipped to \[−1, 1\] by a tanh-normal policy
head, then scaled by `RESIDUAL_MAX_POS = 0.15 rad` per joint. That's
the offset we add to the PD controller's position target each step.

**Q: How many dimensions is the action space in rl\_only mode?**
Same seven dimensions, but the action is interpreted as full joint
*velocity* commands, scaled by the PD velocity limit. The PD
controller is bypassed. This is what makes it a "remove the backbone"
ablation.

**Q: What reward weights did you use?**
Proximity shaping and smoothness/residual penalties are kept small
relative to the terminal bonuses. The grasp bonus fires once per
episode on the geometric gate; the lift bonus is the largest single
term. We iterated on these weights heavily — early versions had
proximity dominating, which caused the "hover without grasping"
failure mode.

**Q: Why did the observation dimension go from 37 to 41?**
Two steps. First, we originally omitted the perturbation offset from
the observation, figuring the policy could infer it from cube pose
versus nominal cube pose. In practice the policy couldn't learn that
inference with the sample budget we had, so we added three explicit
perturbation-offset dimensions. That bumped obs from 37 to 40 and
exposed a subtle demo bug where the demo script was still passing 37
dimensions — caught and fixed in commit `3832e706`. Second, a later
tuning pass appended a scalar waypoint-progress channel in `[0, 1]`,
which brought the final observation to 41 and gave the policy an
explicit handle on the classical scaffold's own progress.

---

## 4. Training

**Q: Why 1M frames?**
That was our training budget: eight parallel workers for a few hours
on a CPU-only node. The reward curve largely plateaus before the full
budget is spent, but we let it run to give the linear LR decay and the
KL-early-stop room to finish out naturally.

**Q: Why eight parallel workers?**
Each PyBullet sim runs on a single CPU core. Eight workers saturated
the training host and cut wall-clock time roughly linearly, with
virtually no change in the final policy. We added a `--workers` CLI
flag so this scales to the available machine.

**Q: What went wrong during training and how did you fix it?**
Multiple things, in order. (1) Early runs oscillated between 60% and
95% success batch-to-batch. We lowered LR from the default down to
`5e-5` with a linear decay to zero, switched entropy from a fixed
coefficient to an annealed schedule (`0.015 → 0.005`), kept the
gradient clip tight at `0.5`, and enabled KL-based early stopping with
a target of `0.05` after a 100k-frame warmup. That killed the
oscillation. (2) Proximity reward was too large relative to terminals,
so the policy learned to hover near the cube — we replaced per-step
proximity with one-shot milestone bonuses at 0.08 / 0.05 / 0.03 m and
added one-shot geometric shaping plus an attempt bonus at the grasp
attempt moment. (3) The reward attractor was at the *nominal* cube
pose, so the policy learned to approach the wrong point — we
re-centred it to track the *post-perturbation* cube. (4) Best-model
selection was picking checkpoints that had one lucky batch; switched
to a composite smoothed metric combining grasp rate and reward.

**Q: What's "composite smoothed best-model selection"?**
Instead of saving the checkpoint with the best single-batch reward, we
track a running average of (grasp rate + reward) across recent batches,
nan-safe. Whenever that composite metric hits a new high, we save.
This stops a lucky reward spike on a narrow batch from being picked as
"final."

**Q: Did you use a curriculum?**
Not in the final run. We tried a short proximity-reward warm-up early
on but removed it. Full curriculum over perturbation magnitude is
explicitly scoped to Milestone 3.

**Q: How long did training take?**
On the order of a few hours for 1M frames with 8 workers. The user
ran training on a different machine from the eval host.

---

## 5. Evaluation

**Q: Why 100 episodes per level?**
With n=100 and observed success rates in the 0.2–0.85 range, Wilson
CIs are roughly ±0.08–0.10 wide — tight enough to distinguish hybrid
from planner at every perturbation level. We started at 50 episodes
and doubled it for the final report.

**Q: Why seven perturbation levels?**
Zero, 2, 4, 6, 8, 10, 12 cm. Zero is the nominal baseline. The 2–6 cm
range is where the planner starts collapsing. 8–12 cm is stress
territory where we expected residual RL to pay off the most. Seven
points is enough to draw a curve without being expensive.

**Q: Why stochastic actions at eval?**
To match the distribution the policy was trained on. A deterministic
(mean-only) eval would reward a policy that exploits the noise
schedule, which isn't what we actually care about. Early in iteration
we used deterministic eval and got misleadingly high numbers; fixing
this was commit `2339e83b`.

**Q: What does `mean_forced_wp_advances` measure?**
How often an episode had to force-advance to the next waypoint because
joint tolerance wasn't met in the step budget. High values indicate
the PD loop couldn't settle; we use it as a diagnostic for
phase-transition health.

**Q: Why didn't you run multiple seeds?**
Time budget. Each training run is a few hours; a multi-seed study of
all three methods was not feasible inside the PM2 window. It's on the
M3 plan.

---

## 6. Demos

**Q: Why exactly four demos?**
They cover the four quadrants of the design matrix that matter most:
planner at nominal (baseline works), planner at perturbed (baseline
fails), hybrid at perturbed (residual recovers), rl\_only at perturbed
(ablation fails). Together they tell the whole story in under four
minutes of video.

**Q: What does the residual overlay bar represent?**
For hybrid mode it's the magnitude of the commanded position offset
across all seven joints, in radians, normalised to the 0.15 rad budget.
For rl\_only mode it represents the full velocity command — the units
are rad/s, not rad, and the overlay was not re-normalised for that
mode, so the bar pins at 100%. That's a known cosmetic bug in the
demo overlay; it does not affect the eval numbers.

**Q: Why is the rl\_only demo's gripper still moving in a grasp-like way if the policy is 0% successful?**
The policy learned the nominal joint-space trajectory from the warmup
phase reward signal — the shape of "approach, descend, close" — but
not the Cartesian feedback needed to track the real cube. So the
motion looks right and then misses because the end-effector never
actually closes on the displaced target.

---

## 7. Analysis & interpretation

**Q: Why is hybrid at nominal lower than planner at nominal?**
The residual is always on, even at zero perturbation. At nominal the
plan is already correct, and any residual action adds noise that can
push the end-effector off a correct trajectory. A confidence-gated
residual that only activates when the cube pose disagrees with the
plan would fix this; it's an obvious M3 candidate.

**Q: Why is rl\_only exactly 0% at every level?**
Because without the PD+IK scaffold, the policy has to learn to both
(a) approximate a nominal joint trajectory and (b) close a Cartesian
feedback loop on the cube. With our sample budget it learned (a) — the
joint motion looks plausible — but never learned (b). The flat
end-effector-to-cube distance across perturbation levels (~0.10 m) is
the direct evidence: if the policy were tracking the cube, that
distance would decrease at low perturbation. It doesn't.

**Q: Why does the planner have a cliff from 100% at nominal to ~45% at 2cm?**
Because the gripper aperture is narrow. If the IK target is even 2 cm
off the cube, the fingers close on empty space or on one face instead
of opposing faces. This is not a smooth degradation — it's a hard
pass/fail gate at the grasp check.

**Q: Why does the planner then plateau at ~22–29% from 6 cm onward?**
Residual noise in our "lenient" grasp validation occasionally lucks
out — a finger accidentally hits the cube and the attach constraint
fires. That's a floor of grasp attempts that succeed by chance, not
design. It's also why tightening the grasp geometry mattered: a
looser check would have shown inflated planner numbers at high
perturbation.

**Q: "Residual is load-bearing" — what does that evidence look like?**
Three numbers together: (1) mean residual magnitude at 0.05 rad is
well above zero but below the 0.15 rad cap, so the policy is actively
using the budget. (2) Hybrid EE–cube distance stays near 2 cm while
planner drifts to 5.6 cm — the residual is cancelling the drift. (3)
Hybrid success rate is higher than planner's at every non-zero
perturbation level. If any of these three were absent, the "load-
bearing" claim would be weak; all three together make it strong.

---

## 8. Problems solved & iteration

**Q: Tell me about the biggest bug you found.**
The reward attractor. Our shaped proximity term was computed from the
*nominal* cube pose, not the *perturbed* cube pose. So the policy was
being rewarded for approaching the wrong point — exactly where the
classical planner already ended up. This single bug depressed every
result. Fixing it — commit `f1ce2afc` — re-centred the attractor each
episode after perturbation and was the single biggest step in final
performance.

**Q: How did you realise the reward attractor was wrong?**
We added per-step rollout traces (commit `728bc3af`) and looked at
what coordinates the policy was steering toward. The trajectories
converged on the nominal cube centre even when the real cube was 4 cm
away, and the proximity reward was near zero at episode end regardless.
That mismatch was the tell.

**Q: How did you tighten the grasp check?**
Three additions. (1) Below-top: both fingertips must be below the cube's
top AABB, so top-face contacts are rejected. (2) Finger-bracket: the
cube's centre must lie between the fingertips along the finger axis,
with a parametric `t ∈ [0.2, 0.8]`. (3) Opposite-face perpendicularity,
done geometrically rather than with contact normals: each fingertip,
expressed in the cube's local frame, has to sit near `±CUBE_HALF_SIZE`
along one horizontal axis (a face plane) and near `0` along the other
horizontal axis (centered on the face). We first tried keying on
contact-normal dot products but they were flaky across PyBullet's
contact sampler at our step rate; the cube-local geometric version was
stable. These were all additive — no existing branch was changed — and
each has a config threshold that's easy to retune.

**Q: The planner@nominal anomaly — what was happening?**
After tightening grasp geometry, planner success at zero perturbation
dropped below 100%. Root cause: joint-space waypoint tolerance (0.025
rad) compounds across seven DOF to roughly 1 cm of Cartesian error —
enough to make the stricter grasp check reject legitimate grasps. Fix:
after the planner reaches the joint waypoint, run a Cartesian settle
loop (IK-target the EE, step PD, until EE is within 5 mm) before
closing the gripper. That restored planner@nominal to 100%.

**Q: The finger-tip measurement bug?**
We were reading finger positions at the link base, not the physical
fingertip. That introduced a ~3–4 mm offset that skewed the grasp
geometry checks. Commit `86dc4b27` added an explicit offset from link
base to physical tip.

**Q: The Wilson CI plot crash?**
At nominal, hybrid and planner scored 1.00. Wilson's upper bound at
p=1.0 with n=100 is about 0.983, so `hi - sr` went slightly negative
and matplotlib rejected the `yerr`. Commit `27bf518b` clipped both
endpoints to non-negative.

**Q: What's the single lesson you'd generalise from the iteration?**
Instrument first, then fix. Every bug on the list was found by adding
a diagnostic log or plot — rollout traces, per-step reward
decomposition, grasp-geometry dumps — and *then* inspecting. Guessing
at fixes based on the final success rate was cheaper to start but
slower to converge.

---

## 9. Limitations & future (M3)

**Q: What are the honest limitations of this work?**
Single seed per policy. Single environment (one cube, one table, no
clutter). Grasp check is a geometric heuristic, not force closure. XY-
dominant perturbation, no orientation randomisation. No sim-to-real
transfer attempted. Hybrid at nominal is below planner at nominal.

**Q: What's the M3 plan?**
Four things. (1) Curriculum over perturbation magnitude to train a
single policy that's robust from 0 to 15 cm without over-fitting to a
single range. (2) Richer perturbation dimensions: full orientation
randomisation, distractor clutter, variable cube size. (3) Multi-seed
and multi-run aggregation so the CIs reflect training variance, not
just evaluation variance. (4) A refactor pass toward sim-to-real —
domain randomisation, delayed observations, commandable joint noise.

**Q: What would a learned grasp gate look like?**
A small classifier on contact/proximity features that outputs "grasp
now" vs "descend further." It would replace the current below-top +
bracket + perpendicularity heuristic. Training signal comes from
whether the subsequent lift phase succeeds. This moves the grasp
decision inside the policy rather than being a fixed outer gate.

**Q: How far is this from a real robot?**
Moderately far. The sim-to-real gap for Franka Panda is well studied,
and our policy's observation is low-dimensional state (not pixels),
so basic transfer is plausible. But three gaps remain: (1) our
perturbation model doesn't capture real pose-estimation *dynamics*
(latency, jitter); (2) our grasp model is a fixed constraint in sim,
not a real gripper; (3) domain randomisation on the PD gains and
dynamics would be needed before we would trust the policy on a real
arm.

---

## 10. Process & contributions

**Q: How did the team split the work?**
M1 was shared across planner, controller, and sim integration. For M2
the split was roughly: one member led the gym environment and reward
design, the other led the PPO training loop, TorchRL integration, and
evaluation pipeline. Plotting, debugging, and the iteration pass were
fully shared. (Update this answer with actual names/ratios for the
live talk.)

**Q: How long did M2 take?**
Six days of aggressive iteration, from April 13 to April 19. That
includes the full training infrastructure, iteration on reward and
architecture, the final eval of 2,100 episodes, eight plot types, and
four recorded demo videos.

**Q: What's the biggest lesson from this project?**
Residual RL only works if the backbone it's correcting is actually
correct. Every time we treated the backbone as a black box and tried
to fix things purely with reward tweaks, we failed. Every time we
went into the classical stack and asked "is the PD controller
actually reaching the target," we found the real bug. The learned
residual is a small, measurable correction on top of a working
classical system — not a rescue mission for a broken one.

**Q: If you had another week, what would you do?**
Multi-seed training and a curriculum-over-perturbation run, because
those two together would tighten our confidence in the headline result
without requiring any architectural change. Then the confidence-gated
residual, because it's the cleanest fix for the hybrid-at-nominal
regression.
