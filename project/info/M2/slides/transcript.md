# PM2 Presentation Transcript

One paragraph of spoken narration per slide. Target 45–60 s per content slide
(~130–150 spoken words per minute). Total ~10 min across slides 2–17.

Slides 1 and 18 are intentionally short. Appendix slides only get notes
because you may pull them up in Q&A.

---

## Slide 1 — Title (~15 s)

Hi everyone, I'm [Member A] and this is [Member B]. Our project is
"Planner-Guided Residual Reinforcement Learning for Robust Robotic
Manipulation." This is our CS558IRL Project Milestone 2 presentation.
Over the next ten minutes we will walk you through what we built in
Milestone 1, how Milestone 2 extends it with a residual RL policy, the
results we obtained, and what we plan to do for Milestone 3.

---

## Slide 2 — Motivation (~50 s)

Classical robotics gives us very reliable tools. A motion planner plus a
well-tuned controller can execute a pick-and-place flawlessly in
simulation. The catch is that every classical pipeline assumes the world
at runtime matches the world at plan time. In reality, perception is
noisy — even a few centimetres of error from a camera-based pose
estimator breaks that assumption. To make the problem concrete: our own
Milestone 1 pipeline scores a perfect one-hundred percent on nominal
pick-and-place. But when we perturb the cube by just twelve centimetres,
success drops to twenty-two percent. The natural question is whether a
small, bounded learned correction on top of the classical controller can
recover the lost robustness without throwing away the planner.

---

## Slide 3 — Project Vision (~40 s)

Our proposal was exactly that: planner-guided residual reinforcement
learning. Keep the classical backbone, add a bounded learned correction.
We divided the project into three milestones. In Milestone 1, which was
submission-ready on March twenty-third, we built the full classical
stack. In Milestone 2 — this presentation — we added a residual PPO
policy on top of that stack and evaluated it against two baselines.
Milestone 3 is future work: extended evaluation, training curriculum,
and research extensions like sim-to-real transfer. The narrative arc is
simple: brittle classical baseline, plus a bounded learned residual,
equals measurable robustness across the perturbation spectrum.

---

## Slide 4 — M1 Recap (~55 s)

Milestone 1 gave us the full classical skeleton. We simulated a Franka
Panda seven-degree-of-freedom arm in PyBullet. For motion planning we
implemented RRT-star in joint space with collision checking against the
table and the cube. For control we wrote a custom PD velocity tracker.
The full pick-and-place task is an eleven-phase state machine: home,
pre-grasp, grasp-descend, close gripper, validate, lift, transfer, place
descend, release, retreat, and return home. Grasp validation uses
contact and proximity checks, plus a Cartesian settle loop before we
close the gripper. With all of this in place, the system achieves
one-hundred percent success at nominal cube pose. That's our strong
starting point — everything we do in Milestone 2 is built on this
backbone.

---

## Slide 5 — M1 to M2: the stress test (~55 s)

The natural stress test is to perturb the cube pose just before
execution, so the planner's targets become stale. The numbers on this
slide are exactly that experiment. At zero perturbation the planner is
perfect. At two centimetres it's already dropped to forty-six percent.
At four centimetres, forty-five. At twelve centimetres it has collapsed
to twenty-two percent. The underlying reason is mechanical: the IK was
solved from the pre-perturbation cube pose, so the joint waypoints no
longer point at the real cube. The PD controller faithfully tracks
those stale targets, and there is no run-time perception loop to
correct them. This is precisely the gap that residual RL is designed
to fill.

---

## Slide 6 — M2 Architecture (~60 s)

Here is how we extend the M1 pipeline. RRT-star still produces a plan.
Inverse kinematics still solves for joint-space waypoints from the
nominal cube pose — that's unchanged. The new block, the red one, is a
PPO actor that produces a bounded residual. Its output, clipped to
minus-one to plus-one and scaled by zero-point-one-five radians, is
added to the joint target. The PD controller then tracks the corrected
target and sends velocity commands into PyBullet. Three design choices
matter here. First, we use a position-target residual, not a velocity
residual. This let PD keep doing the tracking and stabilised learning.
Second, the residual is bounded — the policy cannot commit large
deviations. Third, the residual is phase-gated: active only in
pre-grasp and grasp-descend, zeroed out in the lift and post-grasp
phases so it cannot disrupt a successful attach.

---

## Slide 7 — Observation, Action, Reward (~45 s)

The observation is forty-one-dimensional, with all channels normalized
component-wise by fixed scales. It includes the seven joint positions
and seven joint velocities, the end-effector pose, the cube pose
including orientation, the vector from end-effector to cube, the PD
controller's nominal velocity command, a phase indicator, the current
perturbation offset, and a waypoint-progress scalar that tells the
policy how close the classical scaffold is to its next target. The
action is seven-dimensional, clipped to minus-one to plus-one, then
scaled by the residual budget. The reward has eleven terms in total but
factors into three groups: a shaping group that combines a
delta-distance approach signal with one-shot milestone bonuses at
eight, five, and three centimetres, a one-shot geometric shaping term
and attempt bonus at the moment of grasp; a gate-and-terminal group —
the grasp-gate one-shot, the lift terminal, and a fall penalty; and
two small regularisers on the residual vector's L-one and L-two norms
plus a per-step time cost that applies only before attach.

---

## Slide 8 — Training Setup (~40 s)

Training uses TorchRL's PPO, headless PyBullet, with eight parallel
workers collecting one-million frames total. The actor is a two-layer
two-hundred-fifty-six unit MLP with a tanh nonlinearity and a
state-independent learnable log-standard-deviation; the critic is a
separate five-hundred-twelve to two-hundred-fifty-six MLP. Key
hyperparameters: learning rate of five times ten to the minus five,
linearly decayed to zero across training; PPO clip of
zero-point-two, discount factor zero-point-nine-nine, GAE lambda
zero-point-nine-five, gradient clipping at zero-point-five, critic
coefficient zero-point-five, four optimisation epochs per batch, and an
entropy schedule that linearly anneals from zero-point-zero-one-five
down to zero-point-zero-zero-five. We also early-stop the PPO epochs
when the batch-wise approximate KL exceeds zero-point-zero-five,
starting after a one-hundred-thousand-frame warmup. We log everything
to TensorBoard and use a composite smoothed best-model selection that
combines grasp rate and reward, so the best checkpoint is not a single
lucky batch.

---

## Slide 9 — Iteration Journey (~60 s)

Milestone 2 took six days of aggressive iteration. I want to call out
five real bugs we found and fixed, because each one changed the result
materially. First, the original design was a velocity-additive residual;
it was destabilising the PD loop, so we pivoted to a position-target
residual and training stabilised. Second, the proximity shaping term was
dominating reward, so the policy learned to hover near the cube without
ever attempting a grasp — we capped proximity and emphasised the
terminal bonuses. Third, and the biggest one, our reward attractor was
at the nominal cube pose, not the perturbed pose. The policy was being
incentivised to approach the wrong point. Fourth, the grasp gate was
accepting bad contacts, top-face touches and single-finger hits, so we
added below-top, finger-bracket, and opposite-face perpendicularity
checks. Fifth, we found a planner-at-nominal anomaly where joint-space
waypoint tolerance compounded to about one centimetre of Cartesian
error; we added a Cartesian settle loop before the gripper close.

---

## Slide 10 — Evaluation Protocol (~40 s)

Our evaluation is two-thousand-one-hundred episodes: three methods,
seven perturbation levels, one-hundred episodes per level. The methods
are planner-only, hybrid — meaning PD plus residual — and rl-only with
no PD backbone. Perturbation goes from zero to twelve centimetres in XY,
with Z and yaw scaled proportionally. For each configuration we report
success rate with a Wilson ninety-five-percent confidence interval, mean
reward, mean residual magnitude in radians, mean end-effector to cube
distance, episode length, and the phase distribution at termination.
Actions are stochastic during eval to match the training distribution.
A single seed per policy is a noted limitation.

---

## Slide 11 — Headline Result (~60 s)

This is the headline plot. The x-axis is perturbation magnitude in XY;
the y-axis is success rate with Wilson confidence intervals. The blue
line is the planner-only baseline — it collapses from one-hundred
percent at nominal down to twenty-two percent at twelve centimetres.
The orange line is the hybrid: sixty-five to eighty-five percent across
every perturbation level we tested. Flat. The grey line is the rl-only
ablation — zero percent at every level. The gap between hybrid and
planner at the four-centimetre perturbation and beyond is consistently
thirty-four to fifty-nine percentage points, peaking at fifty-nine at
eight centimetres. I do want to flag one thing
honestly: hybrid at nominal is seventy-eight percent, which is below
planner's one-hundred percent at nominal. The residual is always on and
adds some noise when the waypoint was already correct. We come back to
this in the discussion.

---

## Slide 12 — Policy is Actively Correcting (~50 s)

A natural follow-up question is whether the residual is actually doing
work or just coasting. This slide answers that two ways. On the left,
mean residual magnitude: the policy uses about zero-point-zero-five
radians on average — that's roughly thirty-three to thirty-seven
percent of the zero-point-one-five-radian budget. Not zero, not
saturated. On the right side of the slide, look at end-effector to cube
distance at the grasp attempt. Hybrid stays under about two
centimetres across every perturbation level. Planner drifts from two
centimetres at nominal up to five-point-six centimetres at twelve. So
the residual is measurably cancelling the planner's drift. It is
load-bearing, not cosmetic.

---

## Slide 13 — Ablation: RL-only Fails (~55 s)

The rl-only baseline strips away the PD plus IK backbone entirely. The
policy has to command joint targets from scratch. Two things tell us it
fails. First, the phase breakdown: between ninety-six and one-hundred percent
of rl-only episodes per level terminate in the grasp-descend phase —
the arm cannot even reach the cube cleanly. Lift success is zero percent at every
perturbation level. Second, look at the end-effector to cube distance —
it is flat at about ten centimetres across every perturbation level. If
the policy were tracking the cube, that distance would vary with
perturbation. It doesn't. So rl-only learned a rough joint-space
scaffold similar to the nominal plan, but it did not learn Cartesian
feedback on the real cube. That validates our design choice:
the PD plus IK backbone is load-bearing, and residual RL is correcting
around it.

---

## Slide 14 — Demo 1: Planner fails, Hybrid recovers (~50 s + video)

Now I will show you two demos at the same perturbation — XY of
four centimetres, Z of five millimetres, yaw of zero-point-one radians.
On the left, the planner-only controller. Watch the gripper close on
empty space because the IK target was pointing at the pre-perturbation
cube. On the right, the hybrid controller at exactly the same
perturbation. Watch the residual overlay bar grow as the arm descends —
that's the PPO policy actively steering the end-effector onto the
displaced cube. The grasp lands, the lift succeeds. These two videos
are exactly the same perturbation condition, exactly the same arm,
exactly the same planner; the only difference is whether the residual
is wired in.

*[Play planner_perturbed.mp4, then hybrid_perturbed.mp4 from the ZIP.]*

---

## Slide 15 — Demo 2: Nominal baseline and RL-only ablation (~45 s + video)

Two more demos. On the left, the planner-only controller at zero
perturbation. This is the clean eleven-phase pick-and-place from
Milestone 1 — the baseline that was perfect at nominal but brittle
everywhere else. On the right, the rl-only baseline at the same
four-centimetre perturbation. Watch the end-effector never actually
reach the cube. The policy is moving the joints in a plausible descend
motion, but because it doesn't have the PD plus IK Cartesian feedback
loop, it cannot close the gap on the displaced target. That is the
whole story of the ablation in one video.

*[Play planner_nominal.mp4, then rl_only_perturbed.mp4 from the ZIP.]*

---

## Slide 16 — Discussion and Limitations (~50 s)

We want to be honest about four limitations. First, hybrid at nominal is
lower than planner at nominal. The residual is always on, and at zero
perturbation its action noise occasionally nudges the end-effector off
a correct plan. A confidence-gated residual that only activates when
the cube disagrees with the plan is the obvious next step. Second, we
only trained with a single seed per policy; multi-seed evaluation is
deferred to Milestone 3. Third, our grasp-readiness check is still a
geometric heuristic — below-top, bracket, and opposite-face
perpendicularity — it catches obvious failure modes but does not test
real force closure. Fourth, our perturbation is XY-dominant; richer
orientation and clutter perturbations are Milestone 3 work.

---

## Slide 17 — Future, Remaining, Contributions (~45 s)

For Milestone 3 we plan a curriculum over perturbation magnitude,
richer perturbation dimensions including full orientation and
distractor clutter, multi-seed and multi-run aggregation, and a
refactor pass towards sim-to-real with domain randomisation and
delayed observations. Stretch goals are a learned grasp gate, a
confidence-gated residual, and real-robot transfer on an actual Panda.
For this Milestone 2 submission, two items remain: recording the
ten-minute Zoom presentation and packaging the deliverables ZIP with
the README, the demo videos, and the code. On contributions:
[Member A] led [role], [Member B] led [role], with shared
responsibility on training, evaluation, and the final plots.

---

## Slide 18 — Thanks + Q&A (~15 s)

Thank you for your time. The repository and contact information are on
the slide. We're happy to take any questions.

---

## Appendix notes

**Mean reward vs. perturbation.** Shows the same story as the success
plot through a different metric: hybrid's per-episode reward is
consistently higher than the planner's from the second perturbation
level onward.

**Episode length vs. perturbation.** Hybrid episodes are longer on
average because the residual keeps the arm working on the grasp rather
than giving up. Planner episodes terminate early once the gripper
closes on empty space.

**Diagnostic metrics.** Shows grasp-attempted rate, grasp-attached
rate, cube-fallen rate, and cube-lifted rate broken down by method.
Useful if a reviewer asks *when* an episode fails, not just whether.

**Summary table.** All seven perturbation levels by three methods, with
success rate, mean reward, mean steps, mean residual, and EE–cube
distance. This is the numeric source of truth behind every plot.

**Grasp analysis.** Distribution of grasp outcomes per method and
level. Useful for explaining why the rl-only ablation has a nonzero
grasp-attempted rate despite a zero success rate.
