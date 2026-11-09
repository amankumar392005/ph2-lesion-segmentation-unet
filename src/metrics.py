# ============================================================
# metrics.py
# Purpose:
#   Calculate segmentation metrics for the lesion class.
#
# Metrics:
#   Precision
#   Recall
#   F1 Score
#   IoU
# ============================================================


def calculate_metrics(predictions, targets):

    # predictions and targets should contain 0/1 values

    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)

    true_positive = (
        (predictions == 1)
        & (targets == 1)
    ).sum().item()

    false_positive = (
        (predictions == 1)
        & (targets == 0)
    ).sum().item()

    false_negative = (
        (predictions == 0)
        & (targets == 1)
    ).sum().item()

    # Precision
    precision_denominator = (
        true_positive + false_positive
    )

    if precision_denominator == 0:
        precision = 0.0
    else:
        precision = (
            true_positive
            / precision_denominator
        )

    # Recall
    recall_denominator = (
        true_positive + false_negative
    )

    if recall_denominator == 0:
        recall = 0.0
    else:
        recall = (
            true_positive
            / recall_denominator
        )

    # F1
    f1_denominator = (
        precision + recall
    )

    if f1_denominator == 0:
        f1 = 0.0
    else:
        f1 = (
            2 * precision * recall
            / f1_denominator
        )

    # IoU
    iou_denominator = (
        true_positive
        + false_positive
        + false_negative
    )

    if iou_denominator == 0:
        iou = 0.0
    else:
        iou = (
            true_positive
            / iou_denominator
        )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou
    }