# Task 2 -- controlled design study: expectation (write BEFORE launching `task2/run_experiments.py`)

The assignment asks you to state what you expect increasing alignment pressure to do *before*
interpreting the study's results. Write it in your own words, commit this file, then launch the runs.
The target results from the study are analysis-only and must not be used to change any setting.

Chosen study (pass to `--study`): [ ] `dan` -- lambda_MMD in {0.1, 1, 10}     [ ] `dann` -- max GRL strength in {0.25, 0.5, 1}

| Quantity | Expected effect of stronger alignment (and why) |
|---|---|
| Source-validation performance (per domain, mean) | |
| Domain separability (source-val vs. target features; 50% = chance) | |
| Target (Sketch) recognition | |

Also state, for the main comparison, what you expect for DAN vs. DANN vs. CDAN (marginal vs. class-conditional alignment):

