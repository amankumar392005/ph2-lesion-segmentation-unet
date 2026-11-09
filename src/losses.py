# ============================================================
# losses.py
# Purpose:
#   Loss functions for lesion segmentation.
#
# Baseline:
#   Cross-Entropy
#
# Proposed:
#   Cross-Entropy + Soft Dice
# ============================================================

import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        super().__init__()

        self.smooth = smooth

    def forward(self, logits, targets):

        # Convert logits to probabilities
        probabilities = torch.softmax(
            logits,
            dim=1
        )

        # Lesion probability
        lesion_prob = probabilities[:, 1, :, :]

        # Ground truth lesion mask
        lesion_target = (
            targets == 1
        ).float()

        # Flatten
        lesion_prob = lesion_prob.reshape(
            lesion_prob.size(0),
            -1
        )

        lesion_target = lesion_target.reshape(
            lesion_target.size(0),
            -1
        )

        # Intersection
        intersection = (
            lesion_prob * lesion_target
        ).sum(dim=1)

        # Dice coefficient
        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            lesion_prob.sum(dim=1)
            + lesion_target.sum(dim=1)
            + self.smooth
        )

        # Dice loss
        loss = 1.0 - dice

        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Proposed compound loss:

        L = L_CE + lambda * L_Dice
    """

    def __init__(self, dice_weight=1.0):
        super().__init__()

        self.ce = nn.CrossEntropyLoss()

        self.dice = SoftDiceLoss()

        self.dice_weight = dice_weight

    def forward(self, logits, targets):

        ce_loss = self.ce(
            logits,
            targets
        )

        dice_loss = self.dice(
            logits,
            targets
        )

        total_loss = (
            ce_loss
            + self.dice_weight * dice_loss
        )

        return total_loss