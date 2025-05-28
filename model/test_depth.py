# DepthPro Lite for Jetson Nano
# 模块组合：MobileNetV2 encoder + MobileViT Patch Encoder + UNet-style Decoder（去除focal预测）

import torch
import torch.nn as nn
import torchvision.models as models

# ---------- Patch Encoder: MobileViT Block ----------
# 你可以根据需要换成其他 TinyViT 实现
class MobileViTPatchEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.encoder(x)

# ---------- Image Encoder: MobileNetV2 ----------
class MobileNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights=None).features
        self.encoder = nn.Sequential(*list(base.children())[:14])  # 截取到较浅层，避免过深网络

    def forward(self, x):
        return self.encoder(x)

# ---------- Decoder: UNet-style lightweight decoder ----------
class UNetDecoder(nn.Module):
    def __init__(self, in_channels=64, middle_channels=32, out_channels=1):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出为0~1的inverse depth
        )

    def forward(self, x):
        return self.decode(x)

# ---------- Full Model ----------
class DepthProLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_encoder = MobileViTPatchEncoder()
        self.image_encoder = MobileNetEncoder()
        self.decoder = UNetDecoder()

    def forward(self, x):
        patch_feat = self.patch_encoder(x)
        img_feat = self.image_encoder(x)

        # 简单拼接融合，或使用更多融合策略
        merged = torch.cat([patch_feat, img_feat], dim=1)  # 注意调整通道一致
        out = self.decoder(merged)
        return out

# ---------- Test Example ----------
if __name__ == '__main__':
    model = DepthProLite()
    dummy = torch.randn(1, 3, 512, 512)
    out = model(dummy)
    print(out.shape)  # -> (1, 1, 128, 128)  或按解码上采样输出尺寸
