# Task 3 -- controlled design study: expectation (write BEFORE you read the results)

State what you expect before interpreting results, in your own words; commit this file before opening any
result (runs launched with `--blind` print nothing).
Task 2's Sketch results must not be used to choose or change any Task 3 setting.

Chosen study (pass to `--study`): [] `dan_dg` -- lambda_DG in {0.1, 1, 10} [x] `sam` -- rho in {0.01, 0.05, 0.1}

| Quantity                                                                                        | Expected effect (and why)                                                      |
| ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Source-validation performance (mean / worst source)                                             | I expect the source accuracy to dip as ρ increases because of a higher minima. |
| The method-specific diagnostic (source-domain separability for DAN-DG; sharpness proxy for SAM) | Sharpness Proxy decreases as ρ increases because SAM directly targets this     |
| Sketch performance                                                                              | It has a sweet spot i.e. it shall be best somewhere in the middle values       |

The main comparison must keep lambda_DG = 1 and rho = 0.05; do not replace them with a post-hoc winner.
