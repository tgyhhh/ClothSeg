import torch
from torch import nn, Tensor
from typing import Tuple
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)        # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))

class MyFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, local_feat, global_feat):
        global_feat_norm = torch.norm(global_feat, p=2, dim=1).unsqueeze(dim=0).permute(1, 0, 2, 3)
        projection = torch.mul(global_feat, local_feat)
        projection = torch.mul(global_feat, projection)
        global_feat_norm = global_feat_norm * global_feat_norm
        projection = projection / global_feat_norm

        # orthogonal_comp = local_feat - projection
        # 1
        feat = projection + global_feat
        # 2
        # feat = torch.cat((global_feat, projection), dim=1)

        return feat

class FPF(nn.Module):
    def __init__(self, dims: list, embed_dim: int = 256, num_classes: int = 19):
        super().__init__()
        for i, dim in enumerate(dims):
            self.add_module(f"linear_c{i+1}", MLP(dim, embed_dim))

        self.linear_fuse = ConvModule(embed_dim*4, embed_dim)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

        self.my_fuse = MyFusion()

    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        B, _, H, W = features[0].shape
        outs = [self.linear_c1(features[0]).permute(0, 2, 1).reshape(B, -1, *features[0].shape[-2:])]

        for i, feature in enumerate(features[1:]):
            cf = eval(f"self.linear_c{i+2}")(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            outs.append(F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False))
        out = outs[-1]
        outs2 = []
        for item in outs[-2::-1]:
            new_feature = self.my_fuse(item, out)
            outs2.append(new_feature)
            out = item
        outs2.append(outs[0])
        seg = self.linear_fuse(torch.cat(outs2, dim=1))
        seg = self.linear_pred(self.dropout(seg))
        return seg
