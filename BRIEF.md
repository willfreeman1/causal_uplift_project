# Causal Uplift + Budget Allocation — Project Brief

Hand this to the coding assistant as the starting context. This is a **portfolio
project**, not a take-home — build it clean, on public data, with an evaluation
the reader can verify.

---

## Context — why this project exists

Portfolio project for a data scientist (career-changer, ~3 years in DS)
job-searching for deeper-ML roles, especially **marketing DS / product DS /
decision science**. The rest of the portfolio is "train a model that predicts."
A large share of those roles is instead "measure the effect of an intervention
and decide what to do" — and the resume has A/B testing + bandits but not the
modern causal / uplift toolkit, which is often the senior-level differentiator.
This project fills that gap and demonstrates the full path: **estimate
heterogeneous treatment effects → turn them into a budget-constrained targeting
policy → evaluate that policy honestly.**

**Prior work (blueprint, do not reuse directly):** the candidate recently did a
very similar exercise as a screening take-home for "In Tandem" — multi-offer
retention budget allocation on a *synthetic* dataset. It cannot be showcased
because (a) it's the company's proprietary exercise and data, (b) the grader's
ground truth is held back so its headline numbers are unverifiable, and (c) its
policy evaluation was **circular** — it graded the allocation using the same
model's own uplift predictions (`Σ predicted_uplift × value`), which inflates
results and is exactly what a causal interviewer will catch. This project
rebuilds the *approach* on public data with a **proper policy-value evaluation**.

---

## The task

You have an experiment where each user was randomly assigned one of several
"offers" (treatment arms), each with a cost. For a fresh set of users, decide
**which offer (if any) to give each one, under a total budget**, to maximize
**net incremental value** = (value of the outcome you cause) − (cost of offers
sent). Predicting who converts and deciding who to treat are different problems,
and the users differ: some are persuadable by a cheap offer, some only by an
expensive one, some are **"sleeping dogs"** who churn *more* when contacted.

---

## Data

### Primary — Hillstrom "MineThatData" email dataset
- ~64,000 customers who purchased in the prior 12 months, **randomly** assigned
  1/3 each to: **Men's email**, **Women's email**, **No email (control)**.
- Features: `recency` (months since last purchase), `history_segment`,
  `history` ($ spent last year), `mens`/`womens` (bought that category last year),
  `zip_code` (urban/suburban/rural), `newbie`, `channel` (phone/web/multichannel).
- Outcomes over the next two weeks: `visit` (0/1), `conversion` (0/1),
  `spend` ($). Use **conversion** (or **spend**) as the primary outcome.
- Known propensities: exactly 1/3 per arm — this makes unbiased policy evaluation
  clean.
- It's a canonical uplift dataset, widely mirrored as a single CSV.

**Impose the economics yourself** (the raw data has no costs/budget): assign a
send cost per arm — e.g. make Men's and Women's emails different costs, or add a
notional "incentive" cost — and a **total budget** (or equivalently "you may
contact at most K% of the list"). This recreates the allocation tension: the
budget will not stretch to give everyone the best offer.

### Secondary — a simulation appendix (do this, it's high value)
Generate a synthetic dataset with a **known** data-generating process: known
per-arm CATEs (including a sleeping-dog segment with negative effect), known
costs, known optimal allocation. Because you know the truth, you can compute the
**perfect-foresight optimum** and report your policy's **exact value-capture %** —
the number that was unverifiable in the take-home. Use it to prove the whole
pipeline (CATE estimation → allocation → evaluation) recovers the right answer.

---

## Method

### 1. Look before modeling
Conversion/spend by arm overall (differences will be small — they hide who each
offer helps vs. hurts). Confirm randomization (feature means ≈ equal across arms).

### 2. Naive baseline (the floor to beat)
Predict conversion, treat the highest-propensity users with the best offer that
fits budget. Measure its policy value (see §5). This should be roughly break-even
or worse — it funds people who'd convert anyway and pays sleeping dogs.

### 3. Per-arm uplift / CATE
Estimate the effect of *each* arm vs. control, per user. Methods ladder, with an
honest comparison:
- **S-learner** (one model, arm as a feature)
- **T-learner** (one model per arm) — note: usually the *least* sample-efficient;
  if it "wins by 2x" your evaluation is probably still circular.
- **X-learner**
- **DR-learner / Causal Forest** (EconML `CausalForestDML` or `DRLearner`,
  or CausalML) — gives doubly-robust CATEs and confidence intervals.
- Use `sklift` / `causalml` / `econml`. Calibrate base learners (uplift is a
  difference of two probabilities — distortion compounds).

### 4. The allocation (name it what it is)
For each user pick `arm* = argmax_arm [ CATE(arm) × value − cost(arm) ]`, then
spend the budget where each dollar buys the most incremental value. With each
user getting ≤ 1 offer and a single budget, sorting user–arm pairs by
**value-per-dollar** and taking them greedily is exact for fractional knapsack
and a strong heuristic for this all-or-nothing version — state that clearly.
Never treat users with negative CATE (sleeping dogs).
Optionally also show the exact ILP/knapsack solution (OR-Tools) and confirm greedy
matches it.

### 5. Policy-value evaluation — THE core of the project
Do **not** grade the policy with the model's own CATE predictions. On the
randomized **holdout**, with known propensity `p(a) = 1/3`:

- **IPW estimator:** `V(π) = (1/N) Σ_i [ 1{A_i = π(x_i)} / p(A_i) ] · ( Y_i·value − cost(A_i) )`
- **Doubly-robust (AIPW):** combine IPW with an outcome model to cut variance —
  report this as the headline.
- Bootstrap the estimator for confidence intervals.
- Compare policies: `V(uplift policy)` vs `V(naive churn policy)` vs `V(random)`
  vs `V(treat-all-cheapest)` vs `V(treat-nobody)`.

Also report:
- **Qini / uplift curves** (per arm and overall) on the holdout — `sklift` or
  `causalml`.
- **Sleeping-dogs validation:** on the holdout "do-not-treat" segment, actual
  outcome under treatment vs. control — should confirm treatment hurts them.
- **Is the heterogeneity real?** permutation test on the Qini, or check whether
  the causal-forest CIs for CATE exclude zero for a meaningful share of users.
- **Budget sensitivity:** re-run the allocation at half budget; report the arm
  mix shift (don't just reason about it — run it).

### 6. Money chart
Bar chart of **DR-estimated policy value (net incremental $)** for each policy on
the randomized holdout, with CIs — uplift policy clearly above the naive and
random baselines. Plus the simulation appendix's **exact value-capture %** vs.
perfect foresight.

---

## Business framing (keep brief — 1 paragraph each)
- **Finance / ROI:** net incremental value and return on the offer budget.
- **Product:** the actionable segments — who gets no offer / cheap offer /
  expensive offer, and why, with descriptive profiles (SHAP or simple group
  means).

## Production transfer (keep brief — a short list)
Batch scoring on a warehouse; refresh cadence; monitor effect decay (the CATEs
are from one experiment cohort); retraining trigger; hand-off to an activation
tool. One paragraph, not a chapter.

---

## Scope guardrails
- One primary outcome (conversion or spend), 2–3 arms. Don't sprawl.
- The deliverable is the **honest policy evaluation** + the decision layer, not a
  leaderboard CATE model. A clean S/T/X + DR comparison with correct evaluation
  beats five exotic learners graded circularly.
- Simulation appendix is in scope (it's what makes the numbers verifiable).
- Non-LLM. No deep learning needed — gradient-boosted base learners are fine.

## First steps / de-risking (~½ day)
1. Load Hillstrom, confirm the 3 arms are ~1/3 each and features look balanced.
2. Pick the outcome (conversion), impose costs + a budget that creates real
   scarcity (budget << cost of treating everyone with the best arm).
3. Implement the IPW policy-value estimator and sanity-check it: `V(treat-nobody)`
   should ≈ 0 incremental; `V(random policy)` should be finite and plausible
   given the average effects and costs; `V(always men's email)`
   should ≈ the experiment's measured ATE for that arm × value − cost.
4. If those sanity checks pass, the evaluation backbone is sound — proceed.

## Definition of done
- Naive baseline + S/T/X + one DR method (Causal Forest or DR-learner), calibrated.
- Budget-constrained allocation (greedy value-per-dollar; optional ILP check).
- **DR/IPW policy-value evaluation on the randomized holdout**, with CIs, across
  ≥4 policies. Qini curves. Sleeping-dogs holdout check. Heterogeneity
  significance check. Budget-halved re-run.
- Simulation appendix with exact value-capture vs. perfect foresight.
- Short writeup: iteration log (v1 naive → measured → v2 uplift → v3 DR + proper
  eval), what each version scored on the *unbiased* metric, Finance + Product
  framing, brief production note.
- Clean repo, runnable from scratch, `README`.

## Links
- Hillstrom MineThatData dataset — https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html
  (CSV mirrored widely, e.g. in the `causalml` and `sklift` example data)
- CausalML (Uber) — https://github.com/uber/causalml
- EconML (PyWhy / Microsoft) — https://github.com/py-why/EconML
- scikit-uplift (`sklift`) — https://www.uplift-modeling.com/
- DoWhy (refutation tests) — https://github.com/py-why/dowhy
- Background: Künzel et al. 2019 "Metalearners for estimating heterogeneous
  treatment effects" (S/T/X-learner); Athey & Wager "Generalized Random Forests".
