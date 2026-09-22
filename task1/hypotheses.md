# Task 1 -- design choices: hypotheses and metrics

The assignment requires a stated hypothesis and an appropriate metric for each of your
experimental-design choices **before** interpreting the corresponding result: the dataset,
the additional color intervention, the cue-conflict class pairs/style strength, and the
representation-visualization settings. Metrics are filled in; the hypotheses are yours to
write in your own words. Commit this file before you read the corresponding result (runs use
`--blind`) and before results are committed -- the commit cells refuse to push results until
this file is complete and committed, so the git history shows the order.

## Required before you commit Task 1 results

| Design choice | Setting | Metric(s) | Hypothesis (write before results) |
|---|---|---|---|
| Dataset | STL-10 | -- | |
| Additional color intervention | Hue rotation by 180 degrees (`hue_shift: 0.5` in `configs/task1.yaml`) | accuracy change vs. clean; prediction consistency vs. clean; per-class accuracy | |
| Cue-conflict class pairs | (fill in once chosen) | shape bias %, coverage %, shape/texture/other counts | |
| Cue-conflict style strength | (fill in once chosen) | same as above; also visual acceptance rate | |
| Representation visualization | t-SNE, perplexity 30, seed 6304, PCA init, one joint fit per (backbone, condition) | cosine stability I_T; qualitative cluster mixing between clean and transformed points | |

## Optional -- not required to commit, but worth predicting for a stronger report

Translation and patch structure aren't "design choices" in the assignment's sense (their
deltas/grid are fixed by the spec, not chosen by you), so these aren't required before you
interpret the results. Fill them in whenever you're ready; leaving them as-is doesn't block anything.

| Intervention | Setting | Metric(s) | Hypothesis (optional) |
|---|---|---|---|
| Translation | 0 / 8 / 16 / 32 px, four cardinal directions, reflection padding | accuracy and prediction consistency vs. displacement | (optional -- not required to commit) |
| Patch structure | 4x4 pixel-space grid, one non-identity permutation per image, seed 6304 | accuracy drop; prediction consistency; mean confidence after shuffling | (optional -- not required to commit) |
