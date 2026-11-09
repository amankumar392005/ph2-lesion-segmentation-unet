# ============================================================
# dataset.py
# Purpose:
#   Load the preprocessed PH2 images and segmentation masks.
#
# Input:
#   dataset/processed/{train,val,test}/images
#   dataset/processed/{train,val,test}/masks
#
# Output:
#   Image tensor  -> [3, 256, 256]
#   Mask tensor   -> [256, 256]
# ============================================================

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PH2Dataset(Dataset):
    """
    Dataset class for the preprocessed PH2 lesion segmentation dataset.
    """

    def __init__(self, root_dir, split):
        self.root_dir = Path(root_dir)
        self.split = split

        self.image_dir = self.root_dir / split / "images"
        self.mask_dir = self.root_dir / split / "masks"

        self.image_paths = sorted(self.image_dir.glob("*.png"))

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        # Corresponding mask has the same filename
        mask_path = self.mask_dir / image_path.name

        # Read image
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        # Convert BGR -> RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Read grayscale mask
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            raise RuntimeError(
                f"Could not read mask: {mask_path}"
            )

        # Convert image:
        # [H,W,C] -> [C,H,W]
        image = image.astype(np.float32) / 255.0
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # Convert mask from 0/255 -> 0/1
        mask = (mask > 127).astype(np.int64)

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        mask = torch.tensor(
            mask,
            dtype=torch.long
        )

        return image, mask