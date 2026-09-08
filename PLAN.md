# Causal Uplift + Budget Allocation — Plan & Feasibility (plain-language edition)

Companion to [BRIEF.md](BRIEF.md). This is the document to hand to Cursor.
It is written so a non-specialist can follow every step. Where a step has a
standard technical name, that name is given once in the glossary at the end so
you can talk about it in interviews, but the body of this plan avoids jargon,
abbreviations, and formulas.

---

## 1. What this project is, in one paragraph

A company runs a promotion. It can send each customer one of a few different
offers, or nothing. Each offer costs money to send, and there is a fixed total
budget, so the company cannot give everyone the best offer. The job is to
decide, customer by customer, which offer (if any) to send, so that the extra
sales the offers *cause* — beyond what those customers would have bought anyway
— are as large as possible after subtracting what the offers cost. Then, and
this is the part that makes the project worth doing, we grade that decision
**honestly**, using a method that does not just trust our own predictions.

## 2. Why build it

Most of the rest of the portfolio is "train a model that predicts something."
A large share of marketing, product, and decision-science roles is a different
task: "measure the effect of an action, then decide what to do." The resume
already has experiment design and bandit testing, but not the modern toolkit
for estimating *who* responds to an action and turning that into a budgeted
plan. That toolkit is often the thing that separates a senior candidate from a
junior one. This project fills exactly that gap and shows the whole path from
raw experiment to defensible decision.

There is also a specific story behind it. The candidate recently did a very
similar exercise as a screening take-home for a company called In Tandem. That
version cannot be shown: the exercise and data are the company's property, the
grader kept the answer key hidden, and — most importantly — its self-assessment
was **circular**. It graded the plan by adding up the same model's own
predictions of how well the plan would do. Any interviewer who knows this area
will catch that immediately. This project rebuilds the same approach on public
data with a **non-circular grade**. That fix is the entire point.

## 3. Feasibility verdict: green light

This is a well-trodden build on a standard public dataset. The one genuinely
distinctive piece — the honest grading method — is a short, well-understood
piece of code, not a research problem. Expect **five to seven focused days**.

Checks already done:

- **The data downloads cleanly** and has exactly the columns expected (verified
  by actually pulling the file: sixty-four thousand rows, correct fields).
- **The signal is real and measurable** (verified by computing the effects
  directly — numbers in section 5).
- **The tools install on a normal machine.** They need a current version of
  Python; the machine's default Python is a few years old and has none of the
  needed packages, so the first task is to set up a fresh, isolated Python
  environment. Details in section 8, Step 0.
- **Nothing runs slowly.** The heaviest step finishes in a couple of minutes on
  a laptop.

Known limitations, none of which block the build, all handled in the plan:

- The raw data has **no costs and no budget**. We set those ourselves, openly,
  and we test how sensitive the conclusions are to the exact numbers we pick.
- The strongest, cleanest response signal is on a "soft" outcome (whether the
  customer visited the site). The actual-dollars outcome is noisier because
  almost everyone spends nothing and a few people spend a lot. We handle this
  by trimming the handful of extreme spenders for the noise-sensitive steps and
  by reporting both outcomes.
- There is probably **no group in the real data that reacts badly to being
  contacted** ("sleeping dogs"). Every customer group we looked at responds
  positively or neutrally. So the "don't contact the people it would annoy"
  capability is demonstrated on the made-up dataset (section 7), where we can
  build such a group on purpose, and simply reported honestly on the real data.

## 4. The data

### The real dataset — "Hillstrom" email experiment

About sixty-four thousand customers who had bought something in the previous
year. Each was randomly assigned, one-third each, to one of three groups:

- get an email featuring **men's** products,
- get an email featuring **women's** products,
- get **no email** (the comparison group).

Because the assignment was random and in equal thirds, the three groups are
interchangeable on average, so any difference in what they did afterwards is
caused by the email and not by who they were. This is what makes the dataset
valuable and what makes an honest grade possible later.

What we know about each customer: months since last purchase; how much they
spent last year; whether they bought men's or women's items last year; whether
their address is urban, suburban, or rural; whether they are a new customer;
and how they usually shop (phone, web, or both).

What happened in the two weeks after the email: whether they visited the site,
whether they made a purchase, and how much they spent.

### What is missing, and what we add

The raw data has no notion of what an email costs to send or how much money the
company has to spend. Without those, the "who should get which offer under a
budget" problem does not exist — if offers were free and unlimited you would
just send the best one to everyone. So we **impose** a cost per offer and a
total budget ourselves. This is standard for this exercise and the brief calls
for it. The exact numbers are stated in section 5, with the reasoning, and the
plan re-runs everything at a different budget to show the conclusions are not an
artifact of the numbers we chose.

Important point, because it is a natural worry: the interesting pattern in
"who should get the cheap offer versus the expensive one" comes from the
**differences in how customers respond**, which are entirely real and measured
from the experiment. The costs only decide where to draw the line. They are a
dial we set, not something the model learns.

### The made-up dataset — a proof that the pipeline works

We also generate a synthetic dataset where we personally decide the rules: how
much each offer truly changes each customer's behavior, which group reacts
badly to contact, what each offer costs, and therefore what the single best
possible plan is. Because we know the true answer, we can run our whole
pipeline on it and report **what fraction of the best-possible result our
pipeline actually captured**. That is the headline number the take-home could
never verify. More on this in section 7.

## 5. Decisions already made (so Cursor does not re-litigate them)

These are fixed. Put the numbers in one file that every script reads from.

| Decision | Value | Plain reasoning |
|---|---|---|
| **Main outcome to optimize** | **money spent** in the two weeks after the email | It is the real business goal and it is already in dollars, so we do not have to invent a "value per visit" conversion factor. The only number we invent is cost. |
| **Supporting outcomes** | whether they visited; whether they bought | These have a much stronger, cleaner signal, so they are good for showing the response pattern clearly and as a robustness check. |
| **The three options** | no offer / women's email / men's email | The two real email types plus the do-nothing comparison group. |
| **Cost to send** | no offer: 0 dollars; women's email: 0.20 dollars; men's email: 0.50 dollars | The men's email is treated as a full creative production; the women's email as a lightweight template. This ordering matters — see the note below. |
| **Total budget** | fifteen percent of what it would cost to send the expensive (men's) email to every customer in the grading set | Stated this way it scales automatically with the data. It is deliberately far too small to give everyone the best offer, which is what forces real choices. |
| **How the data is split** | half for building models, half for grading, chosen randomly but reproducibly, keeping the three groups balanced in each half | Models are built on the first half only. Every grade is computed on the second half only. Keeping these apart *is* the project. |
| **Underlying prediction models** | gradient-boosted trees (a standard, strong, off-the-shelf model type) | No deep learning needed. Where a model outputs a probability, we pass it through a calibration step so the probabilities are trustworthy, because we will be subtracting two of them and small distortions add up. |

**Note on the cost ordering.** We measured the effects (next paragraph) and the
men's email is the stronger offer for almost everyone. If we also made it the
*cheaper* one, the plan would trivially be "send men's email to whoever we can
afford." Making the men's email the **more expensive** one creates the real
tension: it works better but costs more than twice as much, so it is only worth
it for customers who respond especially well, while the cheap women's email is
"good enough" for a particular segment. That tension is what the project is
about, and it is driven by the real response differences below.

### What the real data actually shows (measured directly)

- The three groups really are balanced (each is almost exactly one-third; the
  customer characteristics line up across groups).
- **Men's email, average effect:** it raised the site-visit rate by about eight
  percentage points, the purchase rate by about seven-tenths of a percentage
  point, and spending by about seventy-seven cents per customer. All of these
  are far too large to be chance.
- **Women's email, average effect:** about four and a half percentage points on
  visits, three-tenths of a point on purchases, and about forty-two cents on
  spending. Real, but clearly weaker than the men's email.
- **The response pattern that matters:** the men's email works well across the
  board. The women's email only competes with it for customers who bought
  women's items last year — for that group both emails work about equally well,
  whereas for everyone else the women's email lags well behind. So the sensible
  plan is roughly: send the (cheap) women's email to women's-item buyers, spend
  up for the (expensive) men's email on other customers who respond strongly,
  and skip everyone whose response is near zero.
- **No group reacts badly to contact.** Every slice we checked responds
  positively or neutrally, so the "sleeping dogs" idea is demonstrated on the
  made-up dataset, not this one.

After trimming, the average profit per offer (effect minus cost) comes out
close between the two emails, so "just send the best average offer to everyone
you can afford" is a genuine, if weak, competitor. The point of the project is
to beat it by choosing the right offer for the right person.

## 6. The four moving parts, explained simply

### Part one — estimate how much each offer changes each person's behavior

For every customer we want two numbers: how much more money they would spend if
sent the men's email versus nothing, and the same for the women's email. This
is not "will this person buy" — it is "how much does *this offer* move *this
person*," which is a harder question because we never see the same person both
contacted and not contacted.

We try a ladder of methods, from simple to standard, and compare them honestly:

1. **One combined model.** Train a single spending predictor that takes "which
   offer was sent" as one of its inputs. To estimate an offer's effect on a
   customer, ask the model what it predicts with the offer and without it, and
   subtract.
2. **Separate models per group.** Train one spending predictor on the people
   who got the men's email, another on the women's-email people, another on the
   no-offer people. The estimated effect is the gap between predictions for the
   same customer. Simple, but wasteful — each model only sees a third of the
   data — and it often looks deceptively good under a bad grade, so it is a
   useful canary.
3. **A refinement that shares information between the groups** to cope better
   with uneven group sizes.
4. **The current standard method**, which blends a spending predictor with the
   random-assignment reweighting idea from Part three, so that the effect
   estimate stays trustworthy if *either* piece is roughly right. Its "forest"
   variant also reports a confidence range for each customer, not just a single
   number, which we use later to ask whether the response differences are real
   or noise.

We calibrate the probability outputs, keep a plain comparison table, and
resist the urge to tune for a leaderboard. A clean comparison with a correct
grade beats five exotic methods graded badly.

### Part two — turn those estimates into a budgeted plan

For each customer and each offer, compute the expected profit: the estimated
extra spending the offer causes, minus what the offer costs. Never send an
offer to someone whose estimated effect is zero or negative.

Now the budget. Each customer can get at most one offer, and there is one
shared pot of money. The efficient rule is: rank every remaining
customer-and-offer pairing by **profit per dollar spent**, then go down that
ranked list, granting each offer as long as the customer is still unassigned
and there is budget left. Under these conditions (one offer per person, one
budget) this simple greedy rule is a fast and usually strong heuristic, but it
is not always mathematically best because each decision is all-or-nothing. We
say that plainly, and we also solve it a second way with an exact optimizer on
a small slice to show the gap is small in this project.

### Part three — grade the plan honestly (the core of the project)

The wrong way, and the way the take-home did it: take the plan, add up the same
model's predictions of how much each chosen offer will help, and report that
total. This grades the plan using the very predictions that produced it, so it
always looks good.

The right way uses the fact that in the grading half of the data, **everyone
already got a random offer**. To estimate "what would have happened if we had
followed our plan," we look only at the customers who, by luck of the draw,
actually received the offer our plan would have chosen for them. Those
customers are a real, unbiased sample of what that choice does — no model
required. Since each customer had a one-in-three chance of any given offer,
each matching customer stands in for about three, so we count them roughly
triple to rebuild the full picture. Then we average the actual money outcome,
minus costs. This is an honest grade because it is built from real outcomes of
real random assignments, not from our predictions.

That honest grade is a bit noisy, so we improve it: build a rough spending
predictor, use it to give every customer a baseline estimate, and then use the
reweighted real outcomes only to *correct* that baseline. This keeps the grade
honest — it stays trustworthy as long as either the predictor or the
reweighting is roughly right — while making it much steadier. We report this
steadier version as the headline, and we attach a confidence range to it by
repeatedly re-sampling the grading data and re-computing.

Two technical cautions for whoever writes this, stated plainly:

- The rough predictor used inside the grade must be trained on a *different*
  slice of the grading data than the one it is scoring, rotated around, so it is
  never grading its own training examples.
- The standard "gains from targeting" curve (see glossary: Qini curve) is
  designed for a single yes/no treatment. With two offers plus a control, draw
  one such curve per offer against the control, and additionally a single
  overall curve for the final plan by sorting customers by the plan's own
  profit-per-dollar score and plotting cumulative honest value.

### Part four — prove the whole thing works on data where we know the answer

Everything above gives numbers with uncertainty but no way to say "how close to
perfect is this," because in real data nobody knows each customer's true
response. So we build a dataset where we do know, by writing the rules
ourselves (section 7), compute the best possible budgeted plan exactly, run our
full pipeline, and report the fraction of the best-possible profit our pipeline
captured. That single percentage is the most convincing thing in the writeup.

## 7. The made-up dataset, in detail

About forty thousand imaginary customers, each with a handful of characteristics
drawn at random. We then write down, as explicit rules:

- exactly how much each offer changes each customer's spending, as a formula of
  their characteristics — **including a group we define to have a negative
  response**, so contacting them backfires;
- a group that only the expensive offer can move, so the offer choice matters;
- what each offer costs;
- how much random noise sits on top of the outcome;
- random assignment of offers, one-third each, just like the real experiment.

Because we wrote the rules, we can compute the single best budgeted plan
directly from the true responses — the perfect plan. Then we run the real
pipeline on this data using only what a practitioner would actually see (the
noisy outcomes and the offer assignments), get its plan, grade it with the same
honest method used on the real data, and compare:

- **Fraction of perfect captured** — the pipeline's honest profit divided by the
  perfect plan's profit. A high number here is the headline claim, and it is
  verified, not estimated.
- **Does the honest grade match the truth** — on this data we can also compute
  the true value of the pipeline's plan exactly, and check that the honest
  grade lands close to it. If it does, that is evidence the same grade is
  trustworthy on the real data.
- **Are bad-response customers handled** — check that the pipeline estimates a
  negative response for the group we built that way, and that the plan sends
  them nothing.
- **Are known average effects recovered** — a basic sanity check that the data
  generator and the loaders are wired up correctly.

This is roughly sixty lines of straightforward code.

## 8. Repository layout

```
causal_uplift_project/
  README.md                  final writeup: the story, the money chart, the framing
  BRIEF.md                   (already here)
  PLAN.md                    (this file)
  requirements.txt           exact package versions
  data/
    raw/hillstrom.csv        the real dataset, committed so the repo runs offline
  src/causal_uplift/
    data.py                  load, clean, encode, split into build-half and grade-half
    economics.py             the costs, the budget: one source of truth
    eda.py                   section 5 checks: balance, effects, response pattern
    effects.py               the four effect-estimation methods, calibrated
    allocation.py            the budgeted plan: greedy profit-per-dollar, plus exact check
    grading.py               the honest grade and its confidence range   <-- core
    curves.py                the gains-from-targeting curves
    diagnostics.py           bad-response check, is-the-pattern-real check, budget re-run
    simulate.py              the made-up dataset and the perfect-plan comparison
    plots.py                 the money chart and supporting figures
  notebooks/
    01_look_at_the_data.ipynb
    02_effects_and_plan.ipynb
    03_honest_grading.ipynb
    04_made_up_dataset.ipynb
  results/
    figures/
    metrics.json             every plan's honest score and confidence range
  tests/
    test_grading.py          the sanity checks in Step 2 below
    test_allocation.py       greedy matches the exact optimizer; budget respected
  run_all.py                 raw file in, every figure and metrics.json out
```

## 9. Build order — one commit per step

### Step 0 — set up the environment and confirm the tools work (half a day)

- Create a fresh, isolated Python environment on a current Python version
  (three-point-eleven or newer). This keeps the project's packages separate
  from everything else on the machine.
- Install the packages: the standard data-science set (data frames, arrays,
  plotting, the tree-model library), plus the two causal-analysis libraries
  named in the glossary, plus the exact-optimizer library.
- Load the real dataset, import each package once, print a version list. If this
  works, the tooling risk is gone. Do not write any real code until it does.
- Commit the dataset file into the repository so the project runs without
  internet.

### Step 1 — look at the data before modeling (half a day)

- Confirm the three groups are each about a third and that customer
  characteristics line up across them.
- Compute, for each offer, the average effect on spending, on visiting, and on
  buying, each with an uncertainty figure. Reproduce the numbers in section 5.
- Break the visit effect down by customer group (bought women's items last year,
  new customer, recent buyer, and so on) and confirm the response pattern from
  section 5: men's email broadly effective, women's email only competitive for
  women's-item buyers.
- Save the figures. Write what you found in plain sentences in the notebook.

### Step 2 — build the honest grade first, and prove it is sound (one day) — checkpoint

Build the grading code **before** any effect-estimation model. A plan is just a
choice of one option per customer.

- Write the basic honest grade: keep the grading-half customers who randomly got
  the option the plan would pick for them, count them up-weighted to make up for
  the ones set aside, average their actual spending minus cost.
- Write the steadier version: add a baseline spending predictor and use the
  reweighted real outcomes to correct it, with the predictor trained on rotating
  slices so it never grades its own training examples.
- Write the confidence-range routine: re-sample the grading data many times and
  re-compute.
- **These checks must pass before going further:**
  - Grading the plan "send nothing to anyone" returns zero extra profit.
  - Grading the plan "send everyone a random option" returns a finite estimate
    that tracks the average net effect implied by the arm mix and costs.
  - Grading "send everyone the men's email" returns about the measured average
    men's-email effect on spending, minus its cost. Same for the women's email.
- If those pass, the hard part of the project is done and de-risked.

### Step 3 — the naive baseline to beat (half a day)

- Train a plain spending predictor that ignores which offer was sent.
- Rank the grading-half customers by predicted spending, and give the top ones
  the best offer the budget can still afford, until the budget runs out.
- Grade it with Step 2. Expect it to be roughly break-even or worse: it spends
  money on people who would have bought anyway. Record the score.

### Step 4 — estimate the per-person effects (one day)

Implement the four methods from section 6, Part one. Each produces, for every
grading-half customer, an estimated spending effect for the men's email and for
the women's email. Calibrate the probability-output models. Keep an honest
comparison table. Keep the standard method with confidence ranges — it is used
in Step 6.

### Step 5 — build the budgeted plan (half a day)

- For each customer and offer, expected profit is estimated effect minus cost.
  Drop any pairing with a zero-or-negative estimated effect.
- Rank all remaining pairings by profit per dollar, walk the list, assign an
  offer to a customer if they are still unassigned and budget remains.
- State in the code and the README that this greedy rule is optimal here.
- Add the exact-optimizer version on a small slice; the test confirms the two
  agree.

### Step 6 — diagnostics, all on the grading half (one day)

- **Gains-from-targeting curves:** one per offer against the no-offer group,
  plus one overall curve for the final plan. Save the figures.
- **Bad-response check:** take the customers the plan decided to leave alone,
  and compare what those among them who randomly got an email actually did
  versus those who did not. On this data it will probably show no harm — report
  that plainly.
- **Is the response pattern real:** either shuffle the customer characteristics
  and show the targeting gains largely disappear, or report the share of
  customers whose confidence range for their effect stays entirely above zero.
- **Budget sensitivity:** re-run the plan at half the budget. Report how the mix
  of offers shifts (a small table of how many customers get each option) and how
  the honest score changes. Actually run it; do not just describe it.

### Step 7 — the made-up dataset (one day)

Implement section 7. Produce the fraction-of-perfect-captured number, the check
that the honest grade matches the known truth, and the bad-response handling
demonstration.

### Step 8 — the money chart and the writeup (half a day)

- One bar chart: the honest, steadier score, with confidence ranges, for each
  plan — send nothing, send at random, send everyone the cheap offer, the naive
  baseline, the simple effect method, and the standard effect method. The
  effect-based plans should sit clearly above the baselines.
- A second panel: the made-up dataset's fraction-of-perfect-captured.
- The README tells the story as an iteration log: first the naive baseline, then
  what the honest grade said about it, then the effect-based plan, then the
  standard method with the proper grade — with each version's honest score. Then
  one paragraph on the money framing (extra profit and return on the offer
  budget), one paragraph on the product framing (which customers get nothing, a
  cheap offer, or the expensive offer, and what those groups look like), and a
  short note on what it would take to run this regularly in production
  (scheduled scoring, watching the effect fade over time, when to rebuild).
- `run_all.py` reproduces every figure and the metrics file from the raw
  dataset.

## 10. What "done" looks like

- [ ] Naive baseline, plus the four effect methods, with calibrated probabilities
- [ ] Budgeted plan by profit per dollar, with the exact-optimizer agreement check
- [ ] The honest grade on the grading half, with confidence ranges, across at
      least five different plans
- [ ] The Step 2 sanity checks pass (nothing plan scores zero; random plan
      scores about zero; single-offer plans score about their measured average
      effect minus cost)
- [ ] Gains-from-targeting curves, per offer and overall
- [ ] The leave-alone-group check, reported honestly even if it shows no harm
- [ ] The is-the-pattern-real check
- [ ] The half-budget re-run with an offer-mix table
- [ ] The made-up dataset: fraction of the perfect plan's profit captured
- [ ] The money chart
- [ ] A README with the iteration log, the money framing, the product framing,
      and the short production note
- [ ] The whole thing runs from scratch with one command; the dataset is
      committed; package versions are pinned

## 11. Things to get right (what an interviewer will probe)

- Never grade a plan with the same model's own effect predictions. Every
  headline number comes from the honest grade on the grading half.
- Models are built on the build half and evaluated on the grading half. No
  overlap.
- Report the simple "separate models per group" method honestly. If it appears
  to win by a wide margin, suspect the grade, not the method.
- Call the plan what it is: a budgeted allocation where the greedy
  profit-per-dollar rule is optimal under one-offer-per-person and a single
  budget.
- The costs, the budget, and any value assumptions live in one file that every
  script reads. Show that the conclusions survive changing the budget.

## 12. Glossary — plain term to the name you will see in code and interviews

| This plan says | Standard term |
|---|---|
| how much an offer changes one specific person's behavior | conditional average treatment effect; also called uplift or an individual treatment effect |
| the average effect across everyone | average treatment effect |
| one combined model with the offer as an input | S-learner |
| separate models per group | T-learner |
| the refinement that shares information between groups | X-learner |
| the current standard method blending a predictor with reweighting | DR-learner, or Causal Forest for the version with per-person confidence ranges |
| keep the randomly-matching customers and up-weight them | inverse propensity weighting |
| that, improved with a baseline predictor it only corrects | augmented inverse propensity weighting, or a doubly-robust estimator |
| honestly grading a targeting rule on held-out random data | off-policy policy value evaluation |
| ranking customer-offer pairs by profit per dollar and taking them in order | greedy allocation; exact for fractional knapsack, usually close but not guaranteed optimal for this 0/1 multi-offer version |
| the exact optimizer used as a cross-check | integer linear programming, via a solver such as OR-Tools |
| the gains-from-targeting curve | Qini curve, closely related to the uplift curve |
| customers who react badly to being contacted | sleeping dogs, or negative responders |
| the made-up dataset with rules we choose | a simulation with a known data-generating process |
| the pipeline built on the machine's default Python | not this — use a fresh virtual environment on a current Python version |

## 13. Links

- The real dataset, description:
  https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html
- Direct file (also mirrored inside the scikit-uplift and causalml packages):
  http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
- CausalML (Uber): https://github.com/uber/causalml
- EconML (Microsoft / PyWhy): https://github.com/py-why/EconML
- scikit-uplift: https://www.uplift-modeling.com/
- DoWhy, for the optional robustness checks: https://github.com/py-why/dowhy
- Background reading: Künzel and co-authors, 2019, on the single-model /
  separate-models / cross-model methods; Athey and Wager on the forest method.
