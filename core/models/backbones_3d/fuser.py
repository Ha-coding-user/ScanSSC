from mmdet3d.models.builder import BACKBONES
import torch
from mmcv.runner import BaseModule
from mmdet3d.models import builder
import torch.nn as nn
import torch.nn.functional as F

@BACKBONES.register_module()
class Fuser(BaseModule):
    def __init__(
        self,
        embed_dims=128,
        spatial_mixing_net=None
    ):
        super().__init__()
        self.depth_mixer = builder.build_backbone(spatial_mixing_net)
        self.height_mixer = builder.build_backbone(spatial_mixing_net)
        self.width_mixer = builder.build_backbone(spatial_mixing_net)

        self.combine_coeff = nn.Sequential(
            nn.Conv3d(embed_dims * 3, 3, kernel_size=1, bias=False),
            nn.Softmax(dim=1)
        )
    
    def forward(self, depth_feats, height_feats, width_feats):
        depth_feats = self.depth_mixer(depth_feats)
        height_feats = self.height_mixer(height_feats)
        width_feats = self.width_mixer(width_feats)

        weights = self.combine_coeff(torch.cat((depth_feats, height_feats, width_feats), dim=1))
        out_feats = depth_feats * weights[:, 0:1, ...] + height_feats * weights[:, 1:2, ...] + width_feats * weights[:, 2:3, ...]

        return out_feats