"""A small convolutional network for the educational existing-tumor
classification task — an optional, clearly separate comparison point
against the traditional-ML baseline (src/models/train.py).

Per the master prompt's own phasing (Section 6): neural networks are only
introduced after a traditional-ML baseline exists and has been evaluated.
That baseline is the Random Forest in experiments/results/test_results.json
(83.3% accuracy / 0.828 macro-F1). This CNN is built to compare against it
honestly, not to replace it by default — the Streamlit app's live "Try it"
demo still uses the Random Forest unless this model is shown to clearly
outperform it (see docs/evaluation.md for the actual comparison).
"""
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Three conv blocks + two FC layers, operating on 1x128x128 grayscale
    MRI slices (the same preprocessed input as the traditional pipeline's
    feature extraction — see src/data/preprocessing.py).
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8 x 64 x 64
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 x 32 x 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 x 16 x 16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
