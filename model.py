import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(groups, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(groups, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)


class ResolutionBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        return x.flatten(1)


class CrossResolutionAttentionFusion(nn.Module):
    def __init__(self, dim=1024, hidden=512):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(dim, 512)

    def forward(self, low_feat, high_feat):
        fused = torch.cat([low_feat, high_feat], dim=1)
        weights = self.att(fused)
        fused = fused * weights
        fused = self.proj(fused)
        return fused


class SliceLevelAttentionAggregation(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, slice_features):
        scores = self.att(slice_features)
        weights = torch.softmax(scores, dim=1)
        subject_feature = torch.sum(slice_features * weights, dim=1)
        return subject_feature, weights


class GraphReasoning(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.fc_q = nn.Linear(dim, dim)
        self.fc_k = nn.Linear(dim, dim)
        self.fc_v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)

        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (x.size(-1) ** 0.5), dim=-1)
        out = torch.matmul(att, v)
        out = self.out(out)
        return out.mean(dim=1)


class SMRNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.low_branch = ResolutionBranch()
        self.high_branch = ResolutionBranch()
        self.fusion = CrossResolutionAttentionFusion()
        self.slaa = SliceLevelAttentionAggregation()

        self.graph_reasoning = GraphReasoning(dim=512)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        x shape: [B, S, 1, H, W]
        B = batch size
        S = number of MRI slices
        """

        B, S, C, H, W = x.shape
        slice_features = []

        for i in range(S):
            slice_img = x[:, i]

            high_img = slice_img
            low_img = F.interpolate(slice_img, size=(64, 64), mode="bilinear", align_corners=False)

            low_feat = self.low_branch(low_img)
            high_feat = self.high_branch(high_img)

            fused_feat = self.fusion(low_feat, high_feat)
            slice_features.append(fused_feat)

        slice_features = torch.stack(slice_features, dim=1)

        subject_feature, attention_weights = self.slaa(slice_features)

        graph_feature = self.graph_reasoning(slice_features)

        final_feature = torch.cat([subject_feature, graph_feature], dim=1)

        output = self.classifier(final_feature)

        return output, attention_weights
