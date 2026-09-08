# Causal Uplift + Budget Allocation

A marketing team has a limited budget. The question is not “who is likely to buy?” It is “who should we contact, and with which message, to make the most profit?” People who were going to buy anyway waste the money. People who buy only because of the message are worth paying for. That extra sales caused by the contact is called **causal uplift**.

This project estimates that extra, builds a send list under a hard budget, and grades the list with real experimental outcomes rather than the model’s own guesses.

The data is Kevin Hillstrom’s MineThatData email experiment: about 64,000 customers, randomly split into three equal groups. One group got a **men’s-product email**, one got a **women’s-product email**, and one got **nothing**. The raw file has no send costs, so we set them: **$0.50** for the men’s email and **$0.20** for the women’s email. We hold out 32,000 people to grade the plans. Their budget is **$2,400**, which is 15% of the cost of sending the expensive email to everyone, so the plan has to choose. How we score lists is in [METHODS.md](METHODS.md).

---

## The money chart

Each bar answers the same question under the same **$2,400** cap: if we had used this send list on the 32,000 held-out people, how much profit would we have made per person, compared with emailing nobody? Profit is extra dollars spent minus the cost of the emails. People we do not email count as $0, so the average covers the whole list.

Sales are rare. Almost everyone spent $0; a few people spent a lot. That makes any “average profit per person” shaky. The whiskers are a 95% range for that average. Wide whiskers mean the bar could easily have come out much higher or much lower — including at $0.

![Honest extra profit by send list](results/figures/money_chart.png)

Every plan except “send nobody” spends the full $2,400. The first two rivals do not use a model.

- **Random emails, $2,400.** We shuffle the 32,000 people and, for each one, send either the men’s email or the women’s email at random, as long as money remains. The list used about 3,400 men’s emails and 3,400 women’s emails. On this split the bar sits slightly below “send nobody.”
- **Cheap email, $2,400.** We send the women’s email to the 12,000 people the budget can buy and do not try to pick who. That scores about **13 cents** a person.

The other bars try to choose who gets which email. The tallest of those is the separate-models (T) list, about **25 cents** a person. T spent the same $2,400 as the cheap blast; it did not get a larger budget. T’s bar is higher than the naive plan, but the ranges overlap, so this split does not settle a winner.

---

## How the lists were built

We first try three simple plans that still spend $2,400. The random mix scores slightly below sending nobody. The cheap blast sends 12,000 women’s emails and scores about **13 cents** a person. The naive plan ignores which email was sent, predicts who will spend anyway, and sends the expensive men’s email until the money runs out. That reaches 4,800 people and scores about **15 cents** a person. It is roughly break-even: we are paying to email people who were going to buy anyway.

The other plans come from four causal uplift models: a combined model (S-learner), separate models for each email group (T-learner), a cross model (X-learner), and a causal forest. Each is trained on one half of the customers. On the other half — people the model has not seen — it predicts how much extra each person would spend after each email, compared with nothing. We subtract the send cost. If the profit is zero or negative, we skip that email for that person. We rank the rest by profit per dollar of send cost, give each person at most one email, and stop when the $2,400 is gone.

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

A bad way to score a list is to add up the same model’s own guesses of extra spend. That always looks good, because the list was built from those guesses. We do not do that. The numbers above come from real spending by held-out people who, by luck, already got the email the list would have sent.

The T-learner is about **25 cents** a person, versus 15 cents for naive and 13 cents for the cheap blast. The S-learner, X-learner, and causal forest land near the naive plan: 7, 10, and 15 cents. T’s bar is the highest, but the ranges overlap the simple plans, so this split does not pick a winner.

Most of T’s extra comes from mixing the two emails instead of betting on one. Naive spends the whole $2,400 on 4,800 expensive men’s emails. The cheap blast spends it on 12,000 women’s emails and does not choose who. T spends the same $2,400 on about 6,000 cheap emails and 2,400 expensive ones, so it reaches about 8,400 people and tries to give each the email that looks worth the cost.

Even a perfect list cannot create a large effect if the emails themselves are weak. In the original experiment, the men’s email added only about 60 to 80 cents of extra spend.

Exact numbers: [results/metrics.json](results/metrics.json).

---

## What 25 cents means in dollars

On the 32,000-person holdout, 25 cents a person is about **$8,000** extra profit after paying for emails. Naive is about **$4,700**. Both spend **$2,400**. That is profit after paying for emails, not a return multiple on the budget.

If we cut the budget in half, to $1,200, T’s score falls to about **3 cents** a person and the range includes $0. There is not a large, stable pie to slice.

---

## Who gets which email

Under the T list:

- **Left alone (about 23,600 people).** Their last-year spend was typical, about $223, and about half had already bought women’s items. We skip them because the budget ran out, not because email would hurt them. Among those we skipped, people who still got a random email in the original experiment visited more, not less.
- **Women’s email (about 6,000 people).** About 66% had bought women’s items last year. That matches the raw experiment: the cheap email mainly moves that group.
- **Men’s email (about 2,400 people).** They spent more last year, about $398, and had bought more recently. The expensive email works for almost everyone; the budget only stretches to the people where profit per dollar looks highest.

This store file does not show a group that should be left alone because contact would annoy them.

---

## Running this regularly

Score a fresh list with models trained on the last randomized send. Rebuild when the mix of responders changes — season, creative, or list source — or when a new experiment is cheaper than trusting old extras. Watch the honest score on a held-out slice of each campaign, not the model’s predicted extra. If you cannot randomize a slice, you cannot grade the plan the way this repo does.

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

The CSV is already in `data/raw/hillstrom.csv` and runs offline. `run_all.py` rebuilds the figures and `results/metrics.json`, then runs the tests. It does not generate made-up customers.

Walkthroughs: `notebooks/01_look_at_the_data.ipynb`, `02_effects_and_plan.ipynb`, and `03_honest_grading.ipynb`. `04_made_up_dataset.ipynb` checks the code on fake customers with a known answer. It is not a store result.
