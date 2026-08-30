# Communications — draft team message + advisor email

Edit freely. Written to sound like Dennis wrote them.

---

## A) Message for the team (WhatsApp / Slack)

---

guys, big update on the capstone. i've reworked the whole thing based on sir's review and the
implementation is basically done. sharing the gist so we're all on the same page.

**what changed**

sir said 3 things: pivot to an actual AI/ML method (RL ideally), the ML has to be the main
contribution not just a reality check, and we need one clear selling point + one novel point.

so we're keeping our simulator but changing the question. instead of "here's a crime ABM", it's
now "what happens when a reinforcement learning police-patrol policy is put in a feedback loop
with our simulator". the crime ABM becomes the environment. the paper reads as an RL + fairness
paper that uses policing as the testbed.

**the idea in plain terms**

predictive policing sends patrols where crime was recorded, then learns from the crime those
patrols record. but patrolling an area means you catch more of the crime there, so the area looks
worse, so you send more patrols... that's the runaway feedback loop. it's known to hit poor
neighbourhoods hardest.

our angle: our simulator is unseeded (no real crime injected), so we actually KNOW the true crime
in every district. nobody studying this loop has had ground truth before - they use real data
where the true rate is unknown. so we can measure exactly how far the learned policy's patrols
drift from where crime actually is, and prove the drift lands on the poorest districts. then we
show a fairness-constrained RL objective fixes it.

**what we found (3 training seeds x 10 held-out eval seeds)**

- the naive RL rule posts the best RECORDED-crime coverage of any method (looks great on the
  dashboard) but the worst REAL-crime coverage of the targeted policies - it's optimising what it
  can see, not what's happening
- it does this by concentrating way harder than the heuristic (enforcement Gini 0.46 vs 0.28) and
  it literally learns a positive "tilt toward high-hardship districts" parameter
- regression: it over-polices hardship ~3x harder than the standard PredPol heuristic, controlling
  for true crime (coef +0.017, p basically 0)
- in an efficiency-vs-equity plot it's Pareto-dominated - every baseline beats it on BOTH axes
- a swept fairness penalty removes the tilt along a clean monotone frontier
- since the sim is unseeded, none of this can be blamed on biased data - it's the rule

**status**

code is done, tests pass, calibration passes. the full multi-seed results run is going right now
(few hours). paper draft is written (LaTeX, targeting JAAMAS 2027).

**what's in the repo** (folder: CrimeML)

- `HANDOFF.md` - read this first, full explanation of everything
- `RUNNING.md` - step by step to run it in vscode
- `EXPLAINER.md` - deep dive if you want to really understand it
- `paper/main.tex` - the draft

can everyone go through HANDOFF.md and RUNNING.md, run the quicktest (~15 min, instructions in
RUNNING.md), and then let's split the remaining work - i put a suggested division at the bottom of
HANDOFF.md. want to meet before we message sir.

---

## B) Email to advisor

---

**Subject:** Capstone Phase 2 — revised direction and implementation update

Respected Sir,

Thank you for the review last week. Based on your feedback we have reworked the project and I
wanted to update you before our next meeting.

**The three points you raised.** You suggested we (1) move toward an AI/ML method, ideally
reinforcement learning, so the work reads as method research rather than a crime-domain paper,
(2) make the ML the central contribution and not a validation step, and (3) settle on one clear
selling point and one novel point. The revised project addresses all three.

**What we changed, and the continuity.** We keep the same simulator, the same Chicago data, and
the same theoretical grounding (SDT and RAT). What changes is the research question. Our
simulator is unseeded — no historical crime is injected — which in Phase 2 we used to argue
against circular validation. That same property now becomes the enabling condition for a new
experiment: because no crime is injected, the true (latent) crime process in the simulator is
known exactly. We use the simulator as the *environment* for a reinforcement-learning study of
police patrol allocation, and the RandomForest from Phase 2 becomes one of the baselines the RL
agent is compared against. For the panel, the progression is: Phase 1–2 built the unseeded
simulator; Phase 2 (revised) uses it to study something that a seeded model cannot, because there
the ground truth is unknown.

**The problem.** Predictive-policing systems allocate patrols to where crime was recorded and
then retrain on the crime those patrols record. Because a patrolled district has more of its
crime discovered, patrolling raises its recorded crime even when the true rate is unchanged,
which reinforces the original forecast. This runaway feedback loop is documented (Ensign et al.,
2018; Lum and Isaac, 2016) and is known to fall hardest on disadvantaged neighbourhoods.

**Our contribution.** Prior studies of the loop use abstract models or real data where the true
crime rate is unknown; recent RL work on patrol allocation optimises coverage of a fixed
historical signal and never closes the loop. We put a sequential RL policy in a closed loop with
our unseeded simulator, where ground truth is known. This lets us (a) separate genuine crime
reduction from the feedback loop, (b) measure whether the policy over-polices disadvantaged
districts beyond what their true crime justifies, and (c) show that a fairness-constrained RL
objective (a Lagrangian formulation with a demographic-parity cost on enforcement per capita)
breaks the loop, traced as an efficiency–equity Pareto frontier.

**Selling point.** A reinforcement-learning patrol allocator, trained only on the crime it
observes, learns to over-police the poorest districts — improving its apparent performance while
missing more of the actual violence — and a fairness-constrained objective removes most of that
distortion at a small cost in crime prevented.

**Novel point.** To our knowledge this is the first closed-loop study of predictive-policing
feedback built on a theory-grounded, unseeded emergent simulator with known ground-truth crime,
and the first to show constrained RL mitigating the loop dynamics themselves rather than static
coverage disparity.

**Implementation status.** The full pipeline is implemented, tested, and has produced multi-seed
results (three training seeds by ten held-out evaluation seeds). The simulator is a vectorised
port of our Phase-2 model that passes a fidelity gate (it reproduces the hardship elasticity, the
night-crime effect, and emergent spatial clustering). The reinforcement-learning policy learns
the parameters of an allocation rule; trained on the deployable objective it posts the highest
recorded-crime coverage of any method while its coverage of actual crime is the lowest among the
targeted policies, and it over-polices district hardship, after controlling for latent crime, at
about three times the rate of a PredPol-style heuristic. In an efficiency-versus-equity plot it
is Pareto-dominated. A swept fairness penalty removes the effect along a monotone frontier. A
robustness sweep, including a perfect-observation control that switches the loop off, is running.

**Plan.** We are targeting the Journal of Autonomous Agents and Multi-Agent Systems (JAAMAS) for
2027, with AIES and FAccT workshops as parallel venues. A draft is written. We would be grateful
for your feedback on the framing and the choice of venue, and would like to schedule a meeting at
your convenience.

Thank you,
Dennis Philip
(on behalf of Ashmith, Dhrithi, Diya)
