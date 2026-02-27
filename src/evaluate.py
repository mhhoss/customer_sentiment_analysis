from __future__ import annotations

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score



def evaluate_split(model, x_tfidf, y_true: pd.Series, split_name: str) -> dict:
    """
    evaluate a trained classifier on one split and print detailed metrics.
    returns a metrics dict.
    """
    y_pred = model.predict(x_tfidf)
    
    # compare true vs predicted labels using macro-averaged F1 across classes
    # macro f1 is important here because classes are imbalanced (especially neutral)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n[{split_name}] Macro F1: {macro_f1:.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    print(confusion_matrix(y_true, y_pred))

    return {
        "split": split_name,
        "macro_f1": float(macro_f1),
    }
