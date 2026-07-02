"""
model.py — 唇語辨識模型
架構：Front-end (3D Conv + SE-ResNet18) + Back-end (Bi-GRU + MS-TCN)
輸入：(B, T, C, H, W)  — dataset stack 後的格式
輸出：(B, num_classes), (B, T, 256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1.  SE-Res Block
# ============================================================
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.gap(x).flatten(1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w.view(w.size(0), w.size(1), 1, 1)


class SEResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.se    = SEBlock(out_ch, reduction)
        self.downsample = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                          nn.BatchNorm2d(out_ch))
            if stride != 1 or in_ch != out_ch else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(out + residual)


def make_se_res_layer(in_ch, out_ch, blocks=2, stride=1):
    layers = [SEResBlock(in_ch, out_ch, stride=stride)]
    for _ in range(1, blocks):
        layers.append(SEResBlock(out_ch, out_ch))
    return nn.Sequential(*layers)


# ============================================================
# 2.  Front-end：3D Conv stem + SE-ResNet18
# ============================================================
class FrontEnd(nn.Module):
    """
    輸入：(B, T, C, H, W)
    輸出：(B, T, 512)
    """
    def __init__(self, in_channels: int = 1):
        super().__init__()
        # 3D Conv（時序+空間卷積，TFLite 原生支援 Conv3D）
        self.conv3d = nn.Conv3d(in_channels, 64,
                                kernel_size=(5, 7, 7),
                                stride=(1, 2, 2),
                                padding=(2, 3, 3),
                                bias=False)
        self.bn   = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        # 注意：原本用 nn.MaxPool3d(kernel=(1,3,3)) 只對空間維度做池化，
        # 時間維度 kernel=1/stride=1 完全不動，數學上等價於逐幀 MaxPool2d。
        # 改寫成 Reshape + MaxPool2d 是因為 MaxPool3D 在 ONNX→TFLite 轉換時
        # 會被 onnx2tf 標記為 Flex op（需要 TensorFlow Select Ops 才能跑），
        # 在手機/嵌入式裝置上不可用。拆成標準 2D MaxPool 可避免這個問題，
        # 不影響任何可學習參數，可直接沿用舊權重。
        self.pool_kernel  = 3
        self.pool_stride  = 2
        self.pool_padding = 1

        # SE-Res1~4：64 → 64 → 128 → 256 → 512
        self.layer1 = make_se_res_layer(64,  64,  blocks=2, stride=1)
        self.layer2 = make_se_res_layer(64,  128, blocks=2, stride=2)
        self.layer3 = make_se_res_layer(128, 256, blocks=2, stride=2)
        self.layer4 = make_se_res_layer(256, 512, blocks=2, stride=2)
        self.gap    = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) → Conv3d 需要 (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.relu(x)                            # (B, 64, T, H, W)

        # 逐幀 2D MaxPool（取代 MaxPool3d，TFLite 友善）
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, 64, H, W)
        x = x.view(B * T, C, H, W)                 # (B*T, 64, H, W)
        x = F.max_pool2d(x, kernel_size=self.pool_kernel,
                         stride=self.pool_stride,
                         padding=self.pool_padding)  # (B*T, 64, H', W')

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gap(x).flatten(1)                  # (B*T, 512)
        x = x.view(B, T, -1)                        # (B, T, 512)
        return x


# ============================================================
# 3.  Back-end：Bi-GRU + MS-TCN
# ============================================================
class TemporalConvBlock(nn.Module):
    """dilated causal Conv1d + residual"""
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.shortcut = (nn.Conv1d(in_ch, out_ch, 1)
                         if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)[..., :x.size(-1)]         # 截掉 causal padding 多餘部分
        return F.relu(out + self.shortcut(x))


class MSTCN(nn.Module):
    """
    TCN-1: 1024 → 512  (dilation=1)
    TCN-2:  512 → 512  (dilation=2)
    TCN-3:  512 → 256  (dilation=4)
    """
    def __init__(self, in_ch: int = 1024, dropout: float = 0.2):
        super().__init__()
        self.tcn1 = TemporalConvBlock(in_ch, 512, dilation=1, dropout=dropout)
        self.tcn2 = TemporalConvBlock(512,   512, dilation=2, dropout=dropout)
        self.tcn3 = TemporalConvBlock(512,   256, dilation=4, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = self.tcn3(x)
        return x                                    # (B, 256, T)


class BackEnd(nn.Module):
    """
    輸入：(B, T, 512)
    輸出：(B, T, 256)
    """
    def __init__(self, input_size: int = 512, hidden_size: int = 512,
                 dropout: float = 0.3):
        super().__init__()
        # Bi-GRU: 512 → 1024
        self.bigru = nn.GRU(input_size, hidden_size,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.gru_drop = nn.Dropout(dropout)
        # MS-TCN: 1024 → 512 → 512 → 256
        self.mstcn = MSTCN(in_ch=hidden_size * 2, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.bigru(x)                        # (B, T, 1024)
        x = self.gru_drop(x)
        x = x.permute(0, 2, 1)                     # (B, 1024, T)
        x = self.mstcn(x)                           # (B, 256, T)
        x = x.permute(0, 2, 1)                     # (B, T, 256)
        return x


# ============================================================
# 4.  FullModel
# ============================================================
class FullModel(nn.Module):
    """
    Args:
        num_classes: 輸出類別數（163）
        in_channels: 輸入影像 channel（灰階=1）
        dropout:     Dropout 比例

    Returns:
        logits:   (B, num_classes)  — 用於 CrossEntropyLoss
        features: (B, T, 256)       — 序列特徵（備用）
    """
    def __init__(self, num_classes: int = 187,
                 in_channels: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        self.frontend   = FrontEnd(in_channels=in_channels)
        self.backend    = BackEnd(input_size=512, hidden_size=512, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, C, H, W)  — DataLoader stack 後的格式
        """
        x = x.float()
        feat = self.frontend(x)                     # (B, T, 512)
        seq  = self.backend(feat)                   # (B, T, 256)
        logits = self.classifier(seq.mean(dim=1))   # (B, num_classes)
        return logits, seq


# ============================================================
# 5.  快速測試
# ============================================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = FullModel(num_classes=187, in_channels=1).to(device)

    # dataset 輸出格式：(B, T, C, H, W)
    dummy = torch.randn(2, 40, 1, 88, 88).to(device)
    logits, seq = model(dummy)

    print(f"logits : {logits.shape}")    # → (2, 163)
    print(f"seq    : {seq.shape}")       # → (2, 40, 256)

    total = sum(p.numel() for p in model.parameters())
    print(f"params : {total:,}")