from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Metrics:
    trials: int
    tp: int
    tn: int
    fp: int
    fn: int
    accuracy: float
    far: float
    frr: float
    precision: float
    recall: float
    f1: float


def compute_metrics(results: list[dict]) -> Metrics:
    tp = sum(1 for row in results if row["expected_accept"] and row["accepted"])
    tn = sum(1 for row in results if not row["expected_accept"] and not row["accepted"])
    fp = sum(1 for row in results if not row["expected_accept"] and row["accepted"])
    fn = sum(1 for row in results if row["expected_accept"] and not row["accepted"])

    total = len(results)
    positives = tp + fn
    negatives = tn + fp
    precision_den = tp + fp

    precision = tp / precision_den if precision_den else 0.0
    recall = tp / positives if positives else 0.0
    return Metrics(
        trials=total,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        accuracy=(tp + tn) / total if total else 0.0,
        far=fp / negatives if negatives else 0.0,
        frr=fn / positives if positives else 0.0,
        precision=precision,
        recall=recall,
        f1=(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
    )
