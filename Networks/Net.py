import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from timm.models.vision_transformer import Block as TimmBlock
from timm.models.vision_transformer import Attention as TimmAttention
from Networks.transformer import Transformer
import torchvision
class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()

        self.v1 = nn.Sequential(
            # nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.v2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.v3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(258, 128, 3, padding=1, dilation=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(258, 128, 3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(258, 128, 3, padding=3, dilation=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(258, 384, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.res = nn.Sequential(
            nn.Conv2d(384, 64, 3, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU()
        )

        self.init_param()

    def forward(self,x1,x2):
        x1=self.v1(x1)
        x2 = self.v2(x2)
        x = x1+x2
        coord_features= self.compute_coordinates(x)
        x = torch.cat([coord_features, x], dim=1)
        y1 = self.stage1(x)
        y2 = self.stage2(x)
        y3 = self.stage3(x)
        y4 = self.stage4(x)
        y = torch.cat((y1,y2,y3), dim=1) + y4
        y = self.res(y)
        return y
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = -1.0 + 2.0 * torch.arange(h, device=x.device) / (h - 1)
        x_loc = -1.0 + 2.0 * torch.arange(w, device=x.device) / (w - 1)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Netmodule(nn.Module):
    def __init__(self):
        super(Netmodule, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.detr = Transformer(d_model=512, return_intermediate_dec=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, 64))
        self.v1 = nn.Sequential(
            # nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
            nn.Conv2d(1024,512, 3, padding=1, dilation=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.v2 = nn.Sequential(
            # nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
            nn.Conv2d(2048,512, 3, padding=1, dilation=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.regression = Regression()
        
    def forward(self, x):
        x = self.resnet50(x)
        B,C,H,W = x[3].shape
        x[2]=self.v1(x[2])
        x[3] = self.v2(x[3])
        out= self.detr(x[3],None,x[2].flatten(2).permute(2, 0, 1),self.pos_embed)
        mu = self.regression(x[1],out)
        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed