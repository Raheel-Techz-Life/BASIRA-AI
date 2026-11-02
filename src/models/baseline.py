"""Trivial baseline: always predict the majority class.

Required comparison point so a trained classifier's performance can be shown
to beat doing nothing informative (master prompt Section 8).
"""
from collections import Counter

import numpy as np


class MajorityClassBaseline:
    def __init__(self):
        self.majority_label_ = None

    def fit(self, X, y):
        counts = Counter(y)
        self.majority_label_ = counts.most_common(1)[0][0]
        return self

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.array([self.majority_label_] * n)

    def predict_proba(self, X, classes):
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        proba = np.zeros((n, len(classes)))
        idx = list(classes).index(self.majority_label_)
        proba[:, idx] = 1.0
        return proba
