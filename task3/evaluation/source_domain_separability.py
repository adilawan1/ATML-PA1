from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SPLIT_SEED = 6304


def source_domain_separability_score(photo_features, art_features, cartoon_features) -> float:
    """Held-out accuracy of a 3-way (Photo/Art/Cartoon) logistic-regression classifier on
    source-validation features. Chance performance is 33.3%; lower indicates stronger
    invariance across the OBSERVED sources -- it says nothing by itself about class
    information or unseen-domain (Sketch) performance, per the assignment's warning.
    """
    x = np.concatenate([photo_features, art_features, cartoon_features], axis=0)
    y = np.concatenate(
        [
            np.zeros(len(photo_features)),
            np.ones(len(art_features)),
            np.full(len(cartoon_features), 2),
        ]
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=SPLIT_SEED, stratify=y
    )
    clf = LogisticRegression(C=1.0, max_iter=1000)  # lbfgs is multinomial for >2 classes by default
    clf.fit(x_train, y_train)
    return float(clf.score(x_test, y_test))
