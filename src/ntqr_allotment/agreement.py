"""Reusable exact observed-frequency views, scoped to one panel evaluation.

The upstream helper functions receive the observed pattern table explicitly.
Normalization and marginal frequencies are prepared once, not once per pair.
"""

from __future__ import annotations
from ntqr.r2.datasketches import classifier_label_votes, classifiers_labels_votes


class AgreementMoments:
    def __init__(self, observed):
        self.frequencies = observed.to_frequencies_exact()
        self.patterns = tuple(observed.vote_counts)
        # Caches belong to this view and die with it; no process-global panel data.
        self._marginals = {}
        self._pairs = {}

    def marginal(self, classifier, label):
        key = (classifier, label)
        if key in self._marginals:
            return self._marginals[key]
        matched = classifier_label_votes(classifier, label, self.patterns)
        value = float(sum((self.frequencies[pattern] for pattern in matched), start=0))
        self._marginals[key] = value
        return value

    def pair(self, pair, label):
        key = (pair, label)
        if key in self._pairs:
            return self._pairs[key]
        matched = classifiers_labels_votes(pair, (label, label), self.patterns)
        value = float(sum((self.frequencies[pattern] for pattern in matched), start=0))
        self._pairs[key] = value
        return value

    def centered_pair(self, pair, label):
        first, second = pair
        return self.pair(pair, label) - self.marginal(first, label) * self.marginal(
            second, label
        )
