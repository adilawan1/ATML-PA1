# Task 2 -- controlled design study: expectation (write BEFORE you read the results)

The assignment asks you to state what you expect increasing alignment pressure to do _before_
interpreting the study's results. The runs may already be going (`--blind` prints nothing); write this in your
own words and commit it before opening any result -- the results-commit cell refuses to push until you have.
The target results from the study are analysis-only and must not be used to change any setting.

Chosen study (pass to `--study`): [x] `dan` -- lambda_MMD in {0.1, 1, 10} [ ] `dann` -- max GRL strength in {0.25, 0.5, 1}

| Quantity                                                           | Expected effect of stronger alignment (and why)                                                                                               |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Source-validation performance (per domain, mean)                   | I expect the source accuracy to start dropping because MMD term starts to dominate classification loss more strongly.                         |
| Domain separability (source-val vs. target features; 50% = chance) | I expect the domain separability to go down as lambda is responsible for shrinking source/target distance so the distance should go down too. |
| Target (Sketch) recognition                                        | I expect it to be a sweet spot between 0.1 and 10 where the accuracy is at its best since both extremes can hurt the accuracy in my opinion.  |
