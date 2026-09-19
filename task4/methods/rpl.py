from __future__ import annotations

"""Optional extension (Task 4, Step 5): Reciprocal Point Learning, following Chen et al. (2020).
Only attempt this after Vanilla, GCSC, and PROSER are complete and evaluated -- it is
explicitly optional and the assignment's schedule does not budget time for it by default.

If implemented, must clearly identify (see the assignment): how reciprocal points are
represented/learned, how distance to a reciprocal point yields a known-class score, how the
open-space regularization term constrains the feature space, and which score is used to
reject unknowns at test time. Train on CIFAR-10 only; no CIFAR-100 image may influence
training, checkpoint selection, or hyperparameters.
"""


def train_rpl(*args, **kwargs):
    raise NotImplementedError("Optional extension -- only implement if time remains after required Task 4 methods.")
