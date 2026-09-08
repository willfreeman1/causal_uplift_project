# Causal Uplift + Budget Allocation

A marketing team has a **limited budget**. The classic question is not “who is likely to buy?” It is “who should we contact so that each dollar of send cost causes the most leftover profit?” People who were going to buy anyway waste the budget. People who only buy because of the message are the ones worth paying for. That difference — extra sales *caused* by the contact — is **causal uplift**. This project estimates that extra, turns it into a send list under a hard budget, and grades the list with real experimental outcomes instead of the model’s own guesses.

The specific setting is Kevin Hillstrom’s MineThatData email experiment: about 64,000 customers randomly assigned, one-third each, to a **men’s-product email**, a **women’s-product email**, or **nothing**. We impose send costs the raw file does not have: **$0.50** for the men’s email, **$0.20** for the women’s email. On a 32,000-person holdout, the budget is **$2,400** — 15% of what it would cost to send the expensive email to everyone — so the plan has to choose. Details: [METHODS.md](METHODS.md).

---

## The money chart

Each bar answers the same question, under the same **$2,400** cap: **if we had followed this send list on the 32,000-person holdout, how much extra leftover profit would we have made per person, compared with emailing nobody?** Leftover profit is extra dollars spent minus send cost. People we skip count as $0, so the average is over the whole list. Whiskers are a 95% range. Almost everyone spends $0 after the campaign, so the ranges are wide.

![Honest extra profit by send list](results/figures/money_chart.png)

Every plan except “send nobody” spends the full $2,400. The first two rivals do not use a model:

- **Random emails, $2,400** — walk the list in random order; send men’s or women’s at random if it still fits. About 3,400 of each. Slightly **below** nobody on this split; the range includes $0.
- **Cheap email, $2,400** — women’s email to the 12,000 people the budget can buy, no targeting. About **13 cents** a person; the range includes $0.

The rest try to pick *who* gets which email. The tallest of those is the separate-models (T) list, about **25 cents** a person. That is the same budget as blasting 12,000 cheap emails, not a bigger pot.

---

## Iteration log

**Dumb plans that still spend $2,400.** Random mix: slightly negative. Cheap blast: 12,000 women’s emails, **$0.13**. Naive: ignore which email was sent, predict who will spend anyway, blast the expensive men’s email until the money runs out — **4,800** contacts, **$0.15**. The range includes $0. Roughly break-even: paying to email people who were going to buy anyway.

**Effect-based plans.** Fit models on the other half of the data that estimate extra dollars from each email vs nothing. Rank person–email pairs by leftover profit per dollar of send cost. Skip emails whose leftover is zero or negative. Each person gets at most one email.

| Plan | Women’s | Men’s | Left alone | Extra profit / person | 95% range |
|---|---:|---:|---:|---:|---|
| Send nobody | 0 | 0 | 32,000 | $0.00 | — |
| Random emails, $2,400 | 3,377 | 3,449 | 25,174 | −$0.08 | includes $0 |
| Cheap email, $2,400 | 12,000 | 0 | 20,000 | $0.13 | includes $0 |
| Naive | 0 | 4,800 | 27,200 | $0.15 | −$0.03 to $0.35 |
| Combined model (S) | 2,140 | 3,944 | 25,916 | $0.07 | includes $0 |
| Separate models (T) | 5,972 | 2,411 | 23,617 | $0.25 | $0.03 to $0.50 |
| Cross model (X) | 5,890 | 2,444 | 23,666 | $0.10 | includes $0 |
| Causal forest | 2,697 | 3,721 | 25,582 | $0.15 | includes $0 |

**What the honest grade said.** T is the only plan whose range sits entirely above zero on this split. It is about **25 cents** above nobody, versus **15 cents** for naive and **13 cents** for the cheap blast — only about **10 cents** above the dumb plans, and the ranges overlap. S, X, and the forest look like naive once you stop trusting the models’ own scores. Do not crown a winner.

The T edge is mostly **using both emails on purpose**: ~8,400 contacts for the same $2,400, versus 4,800 expensive emails (naive) or 12,000 untargeted cheap ones. Targeting cannot invent a bigger effect than the experiment has (men’s email is only about 60–80 cents extra spend before the 50-cent cost).

Exact numbers: [results/metrics.json](results/metrics.json).

---

## Money

On the 32k holdout, 25 cents per person is about **$8,000** extra profit after paying for emails; naive is about **$4,700**. Same **$2,400** budget. That is leftover sales, not a multiple of spend. Half the budget ($1,200) drops the T-learner score to about **3 cents** a person, and the range includes $0 — there is not a large, stable pie to slice.

---

## Who gets which email

Under the T-learner list:

- **Left alone (~23.6k):** typical last-year spend (~$223), about half already bought women’s items. We skip them because the budget ran out, not because email would hurt them. People we skipped who still got a random email in the old experiment visited *more*, not less.
- **Women’s email (~6.0k):** more last-year women’s-item buyers (~66%). That matches the raw experiment: the cheap email mainly moves that group.
- **Men’s email (~2.4k):** higher last-year spend (~$398) and more recent buyers. The expensive email is the one that works for almost everyone; the budget only stretches to the people where leftover profit per dollar looks highest.

No “do not contact, it will annoy them” group shows up in this store file.

---

## Running this regularly

Score a fresh list with models trained on the last randomized send. Rebuild when the mix of responders drifts (season, creative, list source) or when a new experiment is cheaper than trusting old extras. Watch the honest score on a held-out slice of each campaign, not the model’s predicted extra. If you cannot randomize a slice, you cannot grade the plan the way this repo does.

---

## Setup (Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python run_all.py
```

The CSV is already in `data/raw/hillstrom.csv` (offline). `run_all.py` rebuilds the figures, `results/metrics.json`, and runs tests. It does **not** generate made-up customers.

Notebooks: `notebooks/01_look_at_the_data.ipynb`, `02_effects_and_plan.ipynb`, `03_honest_grading.ipynb`. `04_made_up_dataset.ipynb` is a code check on fake customers with a known answer, not a store result.
