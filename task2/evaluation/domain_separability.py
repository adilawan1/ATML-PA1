from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SPLIT_SEED = 6304


def domain_separability_score(source_features, target_features) -> float:
    """Held-out accuracy of a balanced source-vs-target logistic-regression classifier.

    50% indicates chance performance (domains indistinguishable in feature space); this is
    evidence about residual domain information, not proof that class information survived --
    interpret it alongside target recognition, per the assignment's warning.
    """
    x = np.concatenate([source_features, target_features], axis=0)
    y = np.concatenate([np.zeros(len(source_features)), np.ones(len(target_features))])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=SPLIT_SEED, stratify=y
    )
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
    clf.fit(x_train, y_train)
    return float(clf.score(x_test, y_test))
