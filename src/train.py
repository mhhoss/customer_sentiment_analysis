from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from .data_split import RANDOM_STATE
LR_MAX_ITER = 2000


def train_lr_baseline(x_train_tfidf, y_train: pd.Series) -> LogisticRegression:
    model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=LR_MAX_ITER,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(x_train_tfidf, y_train)
    return model
