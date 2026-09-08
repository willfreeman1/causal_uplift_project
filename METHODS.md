# Methods

The story and numbers live in the [README](README.md). This page records the choices a reviewer needs in order to reproduce or challenge the result. Costs, budget, split, and outcome are also in `src/causal_uplift/economics.py` so every script uses the same numbers.

## Data

Kevin Hillstrom’s MineThatData email experiment: about 64,000 customers who bought in the prior year, randomly assigned in equal thirds to a men’s-product email, a women’s-product email, or no email. Features are last-year spend and recency, category purchased, zip type, newbie status, and channel. Outcomes over the next two weeks are visit, purchase, and dollars spent.

The raw file has no send costs and no budget. Those are imposed here so “who gets which email under a scarce pot of money” is a real problem.

Source: [Hillstrom, 2008](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html). The CSV is committed at `data/raw/hillstrom.csv`.

## Locked choices

| Choice | Value | Why |
|---|---|---|
| Outcome to optimize | Dollars spent after the email | Already in dollars. The only invented number is send cost. |
| Supporting outcomes | Visit and purchase | Cleaner signal for the response pattern; not the headline score. |
| Send cost | $0 / $0.20 / $0.50 | Men’s email is the stronger offer for almost everyone. Making it the expensive one creates a real trade-off. If it were cheaper, the plan would be “send men’s until the money runs out.” |
| Budget | 15% of “men’s email to everyone” on the grade set | Scales with list size. On the 32k holdout that is **$2,400**. Too small to treat everyone well, so the plan has to choose. |
| Split | Half to build models, half to grade, stratified by email group, seed 42 | Models never see the people they are graded on. |
| Assignment probabilities used in the grade | Known design shares: 1/3 each | The experiment assigned emails at random in equal thirds. Using those known chances, not the slightly uneven realized counts in one split. |
| Base models | Gradient-boosted trees, same settings across methods | A clean comparison of how effects are estimated, not a leaderboard of tuned models. |

## What the experiment shows before any model

On the full file, versus no email:

- Men’s email: about +8 visits per 100 people, +0.7 purchase points, **+$0.77** spend.
- Women’s email: about +4.5 visits per 100, +0.3 purchase points, **+$0.42** spend.

Men’s email works across the board. Women’s email mainly competes among people who bought women’s items last year. No slice we checked spent *less* after being emailed, so “do not contact, it will annoy them” is not a finding on this store file.

## How a send list is built

1. On the **build** half, estimate extra spend from each email vs nothing (S-learner, T-learner, X-learner, causal forest).
2. Expected profit = estimated extra spend minus send cost. Skip any person–email pair whose profit is zero or negative.
3. Rank remaining pairs by profit **per dollar of send cost**. Walk the list. Each person gets at most one email. Stop buying an email when it no longer fits.

That ranking is the exact rule if you could send a fraction of an email. Here each send is all-or-nothing, so the ranking is a fast heuristic. An exact solver on a small slice matches when only one email is eligible, and is only a couple percent better when both compete. Reported lists use the ranking.

Simple lists that also spend the same $2,400, so they are fair rivals:

- **Cheap until the money runs out** — women’s email to as many people as $2,400 buys (12,000), no targeting.
- **Random until the money runs out** — walk the list in random order and pick men’s or women’s at random if it still fits.
- **Naive** — ignore which email was sent, predict who will spend anyway, blast the expensive email until the budget is gone.

The model lists have to beat those, not an uncapped blast of the cheap email to the whole file.

## How a send list is graded

Never by adding up the same model’s own extra-spend guesses. On the **grade** half, people already received a random email.

- Keep the people whose random email matches the list’s choice for them.
- Weight them by the inverse of the known 1/3 chance so they stand in for the full list.
- Average real spend minus real send cost, then subtract what would have happened under “send nobody.” Sending nobody therefore scores exactly $0.

A spending predictor trained on rotating slices of the grade set is used only to *correct* that estimate (doubly robust). It is not a targeting model grading its own list. Ranges come from resampling the grade set.

Sanity checks on the grader (not the money chart): nobody scores $0; always men’s / always women’s recover the measured average extra spend minus cost. Those “always” plans are uncapped on purpose — they check the scoring math, not the budget contest.

## What is not claimed

- The T-learner list is the only budgeted plan whose range sits above zero **on this split**. Ranges overlap the naive list. Do not crown a winner.
- Half budget ($1,200) drops that score to about 3 cents a person, and the range includes $0. There is not a large, stable pie to slice.
- Feature columns are aligned from the build half onto the grade half so a missing zip or channel category cannot change the model’s input shape.

## Optional code check (not a store result)

`notebooks/04_made_up_dataset.ipynb` and `src/causal_uplift/simulate.py` generate customers with known extras, including a group that is hurt by contact. That run checks that the pipeline recovers a known answer and leaves that group alone. `run_all.py` does not call it. Those capture percentages are not store results and are not in the README chart.
