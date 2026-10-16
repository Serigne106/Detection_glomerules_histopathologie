"""Modèles : U-Net résiduel entraîné de zéro, et U-Net à encodeur pré-entraîné (comparaison)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """2 x (conv 3x3 -> BN) + raccourci : y = ReLU(F(x) + x). stride=2 divise la taille par 2."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity() if (in_ch == out_ch and stride == 1) else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.shortcut(x), inplace=True)


class UNet(nn.Module):
    """U-Net résiduel léger (≈ 2 M de paramètres pour 16 filtres / 5 niveaux), poids aléatoires."""

    def __init__(self, base_channels=16, depth=5, dropout=0.2, fg_prior=0.05):
        super().__init__()
        ch = [base_channels * 2 ** i for i in range(depth)]
        self.stem = ResidualBlock(3, ch[0])
        self.encoder = nn.ModuleList(ResidualBlock(ch[i - 1], ch[i], stride=2) for i in range(1, depth))
        self.dropout = nn.Dropout2d(dropout)
        self.decoder = nn.ModuleList(ResidualBlock(ch[i] + ch[i - 1], ch[i - 1]) for i in range(depth - 1, 0, -1))
        self.head = nn.Conv2d(ch[0], 1, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # le réseau part d'une prédiction "≈ 5 % de glomérule" (proportion réelle)
        nn.init.constant_(self.head.bias, float(np.log(fg_prior / (1 - fg_prior))))

    def forward(self, x):
        skips = [self.stem(x)]
        for block in self.encoder:
            skips.append(block(skips[-1]))
        y = self.dropout(skips.pop())
        for block in self.decoder:
            skip = skips.pop()
            y = F.interpolate(y, size=skip.shape[2:], mode="bilinear", align_corners=False)
            y = block(torch.cat([y, skip], dim=1))
        return self.head(y)


def build_model(model_cfg: dict, pretrained: bool = True) -> nn.Module:
    """``pretrained=False`` au rechargement d'un checkpoint : inutile de retélécharger l'encodeur."""
    arch = model_cfg["arch"]
    if arch == "unet_scratch":
        return UNet(model_cfg["base_channels"], model_cfg["depth"], model_cfg["dropout"])
    if arch == "smp_unet":
        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install segmentation-models-pytorch") from e
        return smp.Unet(encoder_name=model_cfg["encoder"],
                        encoder_weights=model_cfg["encoder_weights"] if pretrained else None,
                        in_channels=3, classes=1)
    raise ValueError(f"Architecture inconnue : {arch}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
