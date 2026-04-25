#!/usr/bin/env python
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


# Number of classes (based on your CSV auto-fill)
N_MATERIAL = 3        # printed, manuscript, unknown
N_SCRIPT = 5          # Devanagari, Grantha, Tamil-Grantha, Sharada, unknown
N_MANUSCRIPT_FLAG = 3 # yes, no, unknown


class VisualFingerprintNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Backbone: EfficientNet-B0 (lightweight baseline)
        self.backbone = EfficientNet.from_pretrained("efficientnet-b0")

        # Output feature size of B0
        feat_dim = self.backbone._fc.in_features

        # Remove built-in classifier head
        self.backbone._fc = nn.Identity()

        # Classification heads
        self.head_material = nn.Linear(feat_dim, N_MATERIAL)
        self.head_script = nn.Linear(feat_dim, N_SCRIPT)
        self.head_manu_flag = nn.Linear(feat_dim, N_MANUSCRIPT_FLAG)

    def forward(self, x):
        feat = self.backbone(x)  # shape: (batch, feat_dim)

        out_material = self.head_material(feat)
        out_script = self.head_script(feat)
        out_manu = self.head_manu_flag(feat)

        return {
            "material_type": out_material,
            "script_family": out_script,
            "is_manuscript": out_manu,
        }


if __name__ == "__main__":
    # Test model
    model = VisualFingerprintNet()
    x = torch.randn(2, 3, 224, 224)  # dummy input
    out = model(x)
    for k, v in out.items():
        print(k, v.shape)
