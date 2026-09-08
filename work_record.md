# Work record — Causal Uplift + Budget Allocation

A catch-up log in plain language. Technical decisions and the full build order live in [PLAN.md](PLAN.md). The original assignment is [BRIEF.md](BRIEF.md).

**Last updated:** 7 September 2026  
**Status:** Steps 0–8 of 8 are done. The hiring-manager story is [README.md](README.md). The made-up dataset stays in this log and in `notebooks/04_made_up_dataset.ipynb`; it is **not** in the README.

---

## What this project is (30 seconds)

A company can send each customer a men’s-product email, a women’s-product email, or nothing. Each email costs money, and there is not enough budget to give everyone the expensive one. The job is:

1. Figure out **who actually responds** to which email (not just who was going to buy anyway).
2. **Spend the budget** where each dollar causes the most extra sales.
3. **Grade that plan fairly**, using real experimental outcomes, not the model’s own guesses.

This is a portfolio piece for marketing / product / decision-science roles. The resume already has “train a model that predicts.” This one shows “measure the effect of an action, then decide what to do.”

A similar take-home (In Tandem) cannot be shown: the data is theirs, and it graded its plan by adding up the same model’s predictions. Interviewers catch that. This rebuilds the approach on **public data** with a **fair grade**.

---

## Locked choices (do not reopen these)

These live in `src/causal_uplift/economics.py` so every script uses the same numbers.

| Choice | What we picked | Why, in one line |
|---|---|---|
| What to maximize | Extra **dollars spent** after the email | Already in dollars; we only invent the cost |
| The three options | Nothing / women’s email / men’s email | What the real experiment ran |
| Send cost | $0 / **$0.20** / **$0.50** | Men’s is the stronger email; making it *cheaper* would make the plan trivial |
| Budget | **15%** of “men’s email to everyone” | Too small to treat everyone well, so we have to choose |
| Data split | Half to **build** models, half to **grade**, random but fixed | Keeping these apart *is* the project |
| Models | Gradient-boosted trees, with probabilities calibrated | No deep learning needed |

---

## How to pick this back up

```powershell
cd C:\Users\willf\OneDrive\Documents\causal_uplift_project
.\.venv\Scripts\Activate.ps1
python -m pytest
```

If the virtual environment is missing, recreate it from the README. Python **3.11** (not 3.13, not Anaconda). Git is initialized locally; there is **no GitHub remote yet**. That is fine until you want it public.

When Cursor asks which Python to use for notebooks, pick **`.venv` (3.11.9)**. The “previous config / new custom / ignore current root” popup was a **Git name-and-email** prompt, not the Python environment.

---

## What we have already done

### Step 0 — Tools and data (done)

Created a clean Python 3.11 environment and installed the libraries. Confirmed the Hillstrom file loads: **64,000 customers**, three email groups of about one-third each, and it runs offline from `data/raw/hillstrom.csv`.

If this ever breaks, run `python -m causal_uplift.smoke`.

### Step 1 — Look at the data before any model (done)

**In plain terms:** we checked that the experiment is fair, and we measured the average effect of each email.

Because people were assigned at random, the three groups start out the same (same recency, past spend, and so on). They did. So differences afterward are caused by the email, not by who they already were.

Average effects (vs sending nothing):

- **Men’s email:** about **8 extra visits per 100 people**, and about **77 cents** extra spend per customer.
- **Women’s email:** about **4–5 extra visits per 100**, and about **42 cents** extra spend. Real, but weaker.

Those averages hide the useful pattern:

- The **men’s email works for almost everyone**.
- The **women’s email mainly works for people who already bought women’s items last year**. For everyone else, it barely moves visits.

No group we checked got *worse* after being emailed. “Don’t contact people it would annoy” is a real idea, but it does not show up in this dataset. We demonstrated it on **made-up data** (Step 7 in this log only — not in the README).

That is why a smart plan is roughly: cheap women’s email to women’s-item buyers, expensive men’s email on other strong responders, skip people whose response is near zero.

Figures: `results/figures/`. Notebook: `notebooks/01_look_at_the_data.ipynb`.

### Step 2 — A fair report card, built *before* any targeting model (done)

**In plain terms:** a “plan” is just a rule — men’s, women’s, or nothing for each person. The hard question is: if we had actually followed that plan, how much **extra profit** would we have made? Extra profit = extra spending the email *caused*, minus send cost.

**The dishonest shortcut** (what the take-home did): ask your model “how much do you *think* this plan helps?” and add those guesses. That always looks good, because the plan was built from those same guesses. You are grading your own homework.

**The honest version:** half the customers were held out and **already got a random email**. We look only at the people who, by luck, already received the email our plan would have chosen. They are a fair sample of what that choice actually does. We use their **real spending**, minus **real cost** — no targeting model required. That estimate is a bit noisy, so we also make a rough spending guess and use the real random outcomes only to *correct* it. We still are not trusting a targeting model’s own “uplift” scores. We repeat the calculation many times on reshuffled copies of the holdout to get a **range**, not a fake-precise single number.

To prove the report card works, we graded dumb plans whose answers we already know (on the holdout slice):

| Dumb plan | What should happen | What we got |
|---|---|---|
| Email nobody | Extra profit **$0** | Exactly **$0** |
| Pick an email at random | A weighted average of each email's net effect, so it can be above or below **$0** depending on effects and costs | About **$0.20**, and the range includes $0 |
| Everyone gets the men’s email | Typical men’s-email lift, minus **$0.50** | **$0.12** (matches). The lift is real, but 50 cents is a high cost, so leftover profit is small and the range is wide |
| Everyone gets the women’s email | Typical women’s-email lift, minus **$0.20** | **$0.33** (matches). Cheaper email keeps more of the lift |

Those checks passing means: when we later grade a smart targeting plan, the number is a real-world estimate, not the model cheering for itself.

Code: `src/causal_uplift/grading.py`. Notebook: `notebooks/03_honest_grading.ipynb`.

### Step 3 — A naive plan to beat (done)

**In plain terms:** ignore which email was sent. Predict who is likely to spend anyway. Blast the expensive men’s email at those people until the budget runs out.

That spent the full **$2,400** budget on **4,800** people (15% of the holdout). The fair report card said about **14–15 cents** extra profit per person, and the range **includes zero**. Roughly break-even. It is paying to email people who were going to buy anyway.

That is the floor. The uplift plan has to beat it.

Code: `src/causal_uplift/baselines.py`.

---

## Step 4 — who responds to which email? (done)

This is the first real model. It still does **not** spend the budget (that is Step 5) and it still does **not** use a model’s own guesses as the headline score.

### What we estimated

For every person in the **grade** half, two numbers: extra dollars from the **men’s** email vs nothing, and extra dollars from the **women’s** email vs nothing. That is the **change** caused by the email, not “will they buy anyway.” Train on the **build** half only.

### The four methods

Same LightGBM settings as the naive baseline. No leaderboard tuning.

1. **S-learner** — one spending model that is also told which email was sent.
2. **T-learner** — one model per group. Each model only sees a third of the data.
3. **X-learner** — starts like the separate models, then a second step that learns the *gaps*.
4. **Causal forest** — spending model plus random-assignment reweighting, with a **confidence range per person**.

### Results (honest recovery check, not a leaderboard)

On this grade split the experiment’s measured extra spend was about **$0.53** (women’s) and **$0.62** (men’s).

| Method | Predicted extra, women’s | Predicted extra, men’s |
|---|---|---|
| S-learner | $0.27 | $0.83 |
| T-learner | $0.28 | $0.90 |
| X-learner | $0.29 | $0.91 |
| Causal forest | $0.25 | $1.00 |

In English: all four get the **direction** right (men’s is stronger) and the **neighborhood** right (tens of cents to about a dollar, not $20). They do not hit the measured averages on the nose because spend is mostly $0 with a few big orders. Women’s extra comes in a bit low; men’s a bit high.

A chunk of people get a *predicted* negative extra even though Step 1 found no real “annoyed by email” group. That is estimation noise, not sleeping dogs. The forest is calmer on men’s email (~1% predicted negative). About **23%** of people have a men’s-email range entirely above zero; almost nobody does for women’s — weaker, noisier extra, which matches Step 1.

**Visit (supporting, cleaner signal).** S/T/X predicted about **+7.6 to +7.9 points** for men’s email vs measured **+7.5**; women’s about **+3.8 to +4.0** vs measured **+5.2**. Same story as Step 1, recovered cleanly.

Which method makes a **better send list** is Step 5, below.

Code: `src/causal_uplift/effects.py`. Notebook: `notebooks/02_effects_and_plan.ipynb`.

---

## Step 5 — spend the budget (done)

**In plain terms:** we now have, for each person, a guess of extra dollars from each email. Subtract the send cost. If that leftover is zero or negative, skip that email for that person (not worth sending). Rank every remaining “this person + this email” pair by **how much leftover profit you get per dollar spent**. Walk down that list. Each person gets at most one email. Stop buying an email when it no longer fits; leftover cents can still buy the cheaper one further down.

That ranking is a fast, sensible rule when each person gets one thing and there is one shared pot of money, but it is not guaranteed to be the absolute best in every 0/1 case. An exact optimizer on a small test slice agrees when only one email is in play, and is only a hair better when both emails compete (a couple percent). We use the simple ranking.

All plans below spend essentially the full **$2,400** on the 32,000-person holdout. Extra profit is **per person**, from the fair report card (real outcomes, not the model’s own guesses). The range is wide because almost everyone spends $0.

| Plan | Women’s emails | Men’s emails | Left alone | Fair extra profit / person |
|---|---|---|---|---|
| Email nobody | 0 | 0 | 32,000 | **$0.00** (exactly) |
| Naive (email likely spenders with the expensive one) | 0 | 4,800 | 27,200 | **$0.15** (range includes $0) |
| S-learner plan | 2,140 | 3,944 | 25,916 | **$0.07** (range includes $0) |
| T-learner plan | 5,972 | 2,411 | 23,617 | **$0.25** (range about $0.03 to $0.50) |
| X-learner plan | 5,890 | 2,444 | 23,666 | **$0.10** (range includes $0) |
| Causal forest plan | 2,697 | 3,721 | 25,582 | **$0.15** (range includes $0) |

What that means:

- The naive plan **never uses the cheap email**. It blows the whole budget on men’s emails to people it thinks will spend anyway. About break-even.
- Every uplift plan **uses both emails**. That is the point of the cost split: women’s is weaker but less than half the price, so it is worth it for some people.
- The **T-learner plan** is the only one whose fair range sits entirely above zero on this split. It sends the most cheap emails (~6,000) and the fewest expensive ones. Extra profit about **25 cents a person**, versus **15 cents** for naive. Not a blowout — spend is noisy — but it is the first plan that looks like it actually beats “do nothing” after you stop trusting the model’s own guesses.
- S, X, and the forest look similar to naive once you grade them honestly. Similar *counts* of emails (X vs T) does not mean the *same people*: X and T send about the same mix but T’s list scores better on real outcomes.
- T-learner is also the noisiest method (each spending model only saw a third of the build data). The project plan warned: if T “wins by 2×,” suspect a circular grade. Here the grade is **not** circular, and the edge is about 1.7× naive, with a wide range. Treat it as “this mix is promising,” not “T is the champion.”

Of the T-learner women’s emails, about **two-thirds** went to last-year women’s-item buyers (who are about half the list). That lines up with Step 1: the cheap email is mainly for that group.

### How to read “extra profit / person”

**Compared with doing nothing, not compared with naive.** Emailing nobody is $0.00. Naive is about $0.15 above nobody. T-learner is about $0.25 above nobody, so only about **10 cents above naive**. The models are not “+$0.25 better than the dumb plan.”

**Averaged over the whole list, including people we did not email.** Those people contribute about $0. So 25 cents across 32,000 people is about **$8,000** extra profit on the holdout after paying for emails. Naive is about **$4,800**. Same $2,400 budget.

**Why that is not a “big” impact**

The experiment itself is small. Even emailing *everyone* the men’s email only caused about 60–80 cents extra spend, and 50 cents of that is the send cost. Women’s was about 40–50 cents extra spend minus 20 cents cost. Targeting cannot invent a bigger effect than exists. It can only spend the small budget on people who are a bit more movable, and use the cheaper email where it is “good enough.”

The $2,400 only covers a slice of the list (about 4,800 expensive emails, or more people if we mix in cheap ones). T-learner contacted about **8,400** people; naive contacted **4,800**. A lot of T’s extra total dollars is simply **more contacts for the same money**, not magic identification of a secret buyer type.

And the ranges overlap. T’s fair range is roughly $0.03–$0.50; naive’s includes $0 and goes up to about $0.35. We cannot honestly say T beat naive for sure. Spend is almost all zeros, so the report card is wobbly.

**What the models are actually trying to learn**

They are *trying* to learn “this kind of person moves on men’s, that kind on women’s, this kind on nothing.” On **visits**, that pattern is real and fairly clear (Step 1). On **dollars**, almost nobody buys, so the models often learn noise: they rank people as “high extra spend” who were just standing next to a $200 order in the training data. When you grade with real dollars, those fancy rankings collapse toward “about as good as emailing likely buyers” (S, X, forest) or “a bit better, maybe, mostly by using the cheap email more” (T).

**X vs T, same mix, different people**

Think of two managers told: “Send about 6,000 cheap emails and 2,400 expensive ones.” Same headcount, different names on the list. Only the people who, by luck of the old experiment, actually got that email count toward the fair score. T’s names happened to match people who spent more than X’s names. That can be skill or luck; the wide ranges mean we should not pretend we know which.

Code: `src/causal_uplift/allocation.py`.

---

## Step 6 — stress-test the T-learner plan on the holdout (done)

**Gains from targeting (visits).** Ranking people by predicted extra *spend* and asking whether that ranking finds extra *visits*: men’s email shows a small real bump (Qini above almost all shuffled rankings). Women’s is about zero / slightly negative — ranking by dollars is not the same as ranking by visits. Spend Qini is not used: the usual Qini tool only works for yes/no outcomes.

**People we left alone.** We skipped ~23,600 people. Among them, those who randomly still got an email **visited more**, not less (about +5 points for women’s, +7 for men’s). Men’s email also shows extra spend in that leftover group. So we are not protecting sleeping dogs. We are leaving extra visits on the table because **the budget ran out**. That matches Step 1.

**Is the pattern real?** Causal forest: about **23%** of people have a men’s-email extra whose range stays entirely above zero; about **1%** for women’s. Weak, not fake, and much clearer for men’s.

**Half budget ($1,200).** Still uses both emails (3,259 cheap, 1,096 expensive). Honest extra profit falls to about **$0.03** per person, and the range includes $0. Tighter money, noisier leftover. The mix does not collapse to “only men’s” or “only women’s.”

Figures: `qini_visit.png`, `cumulative_ipw.png`, `mix_full_vs_half.png`.

---

## Step 7 — made-up data with a known answer (done; not in the README)

This is a **code check**, not a store result. We wrote easy rules for 40,000 fake customers (step functions of features a tree can split): a group that gets **worse** if emailed; a group only the expensive email can move; women’s-item buyers who respond to the cheap one. Same random 1/3 assignment as the real experiment. Then we compared the T-learner + greedy plan to the **perfect** plan computed from the hidden extras.

| Check | Result |
|---|---|
| Fraction of perfect extra profit captured | **99.9%** |
| Honest grade vs true extra of our plan | **$0.62 vs $0.64** (the range covers the truth) |
| Sleeping dogs left alone | **100%** |
| Mix | Both emails, close to the perfect mix |

**Do not put 99.9% on the README.** The extras are large, clean, and written in the same columns the model sees. Close agreement of the honest grade with the true extra only shows the scoring code is not obviously broken. It says **nothing** about how much profit this pipeline captures on Hillstrom.

Code: `src/causal_uplift/simulate.py`. Notebook: `notebooks/04_made_up_dataset.ipynb`. `run_all.py` does **not** call this.

---

## Step 8 — money chart and hiring-manager story (done)

- Chart: `results/figures/money_chart.png` — honest extra profit vs nobody, with ranges, for nobody / random / cheap-to-everyone / naive / S / T / X / forest. **No fake-data panel.**
- Numbers: `results/metrics.json` (includes T-learner product profile and half-budget).
- README: iteration log, money, who gets which email, production note. No 99.9%.
- Reproduce store figures: `python run_all.py` (look at data → score lists → holdout checks → tests).

Headline from the chart (same numbers as Step 5): T-learner ~**$0.25**/person (range ~$0.03–$0.50); naive ~**$0.15** (range includes $0). Cheap-to-everyone is unconstrained (~$6,400 spend). Do not crown T.

**When you next sit down:** the build is complete. Activate `.venv`, run `python -m pytest` if you change code (25 tests). Git is local only until you want a public remote.
