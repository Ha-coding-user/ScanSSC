import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.ops import MultiScaleDeformableAttention

@HEADS.register_module()
class VoxFormerHead(nn.Module):
    def __init__(
        self,
        *args,
        volume_h,
        volume_w,
        volume_z,
        data_config,
        point_cloud_range,
        embed_dims,
        cross_transformer,
        self_transformer,
        positional_encoding,
        mlp_prior=False,
        **kwargs
    ):
        super().__init__()
        num_heads = 8
        num_points = 4
        
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z
        self.embed_dims = embed_dims
        
        self.data_config = data_config
        self.point_cloud_range = point_cloud_range
        
        # initial voxel(CGVT)
        self.volume_embed = nn.Embedding((self.volume_h) * (self.volume_w) * (self.volume_z), self.embed_dims)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.cross_transformer = build_transformer(cross_transformer)
        self.self_transformer = build_transformer(self_transformer)
        
        if mlp_prior:
            self.mlp_prior = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims//2),
                nn.LayerNorm(self.embed_dims//2),
                nn.LeakyReLU(),
                nn.Linear(self.embed_dims//2, self.embed_dims)
            )
        else:
            self.mlp_prior = None
            self.mask_embed = nn.Embedding(1, self.embed_dims)

        self.depth_block = ScanBlock(embed_dims, target_axis='depth')
        self.height_block = ScanBlock(embed_dims, target_axis='height')
        self.width_block = ScanBlock(embed_dims, target_axis='width')
        self.axis_positional_encoding = build_positional_encoding(positional_encoding)

        image_grid = self.create_grid()
        self.register_buffer('image_grid', image_grid)
        vox_coords, ref_3d = self.get_voxel_indices()
        self.register_buffer('vox_coords', vox_coords)
        self.register_buffer('ref_3d', ref_3d)

        
    def get_voxel_indices(self):
        xv, yv, zv = torch.meshgrid(
            torch.arange(self.volume_h), torch.arange(self.volume_w),torch.arange(self.volume_z), 
            indexing='ij')
        
        idx = torch.arange(self.volume_h * self.volume_w * self.volume_z)
        vox_coords = torch.cat([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1), idx.reshape(-1, 1)], dim=-1)

        ref_3d = torch.cat(
            [(xv.reshape(-1, 1) + 0.5) / self.volume_h, 
             (yv.reshape(-1, 1) + 0.5) / self.volume_w, 
             (zv.reshape(-1, 1) + 0.5) / self.volume_z], dim=-1
        )

        return vox_coords, ref_3d

    def create_grid(self):
        # make grid in image plane
        ogfH, ogfW = self.data_config['input_size']
        xs = torch.linspace(0, ogfW - 1, ogfW, dtype=torch.float).view(1, 1, ogfW).expand(1, ogfH, ogfW)
        ys = torch.linspace(0, ogfH - 1, ogfH, dtype=torch.float).view(1, ogfH, 1).expand(1, ogfH, ogfW)

        grid = torch.stack((xs, ys), 1)
        return nn.Parameter(grid, requires_grad=False)
    
    def forward(self, mlvl_feats, proposal, cam_params, lss_volume=None, img_metas=None,  **kwargs):
        """ Forward funtion.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            img_metas: Meta information
            depth: Pre-estimated depth map, (B, 1, H_d, W_d)
            cam_params: Transformation matrix, (rots, trans, intrins, post_rots, post_trans, bda)
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype, device = mlvl_feats[0].dtype, mlvl_feats[0].device        

        volume_queries = self.volume_embed.weight.to(dtype)
        if lss_volume is not None:
            # Todo: support batch size > 1
            assert lss_volume.shape[0] == 1
            # lss_volume = self.aspp(lss_volume)
            lss_volume_flatten = lss_volume.flatten(2).squeeze(0).permute(1, 0)
            volume_queries = volume_queries + lss_volume_flatten

        if proposal.sum() < 2:
            proposal = torch.ones_like(proposal)
        # Generate bev postional embeddings for cross and self attention
        bev_pos_cross_attn = self.positional_encoding(torch.zeros((bs, 512, 512), device=volume_queries.device).to(dtype)).to(dtype) # [1, dim, 128*4, 128*4]
        bev_pos_self_attn = self.positional_encoding(torch.zeros((bs, 512, 512), device=volume_queries.device).to(dtype)).to(dtype) # [1, dim, 128*4, 128*4]

        vox_coords, ref_3d = self.vox_coords.clone(), self.ref_3d.clone()
        # proposal = torch.zeros([bs, self.volume_h, self.volume_w, self.volume_z])
        # proposal[unq[:, 0], unq[:, 1], unq[:, 2], unq[:, 3]] = 1
        unmasked_idx = torch.nonzero(proposal.reshape(-1) > 0).view(-1)
        masked_idx = torch.nonzero(proposal.reshape(-1) == 0).view(-1)
        
        # --
        # Compute seed features of query proposals by deformable cross attention
        seed_feats = self.cross_transformer.get_vox_features(
            mlvl_feats,
            volume_queries,
            self.volume_h,
            self.volume_w,
            ref_3d=ref_3d,
            vox_coords=vox_coords,
            unmasked_idx=unmasked_idx,
            grid_length=None,
            bev_pos=bev_pos_cross_attn,
            img_metas=img_metas,
            prev_bev=None,
            cam_params=cam_params,
            **kwargs
        )
        
        vox_feats = torch.empty((self.volume_h, self.volume_w, self.volume_z, self.embed_dims), device=volume_queries.device)
        vox_feats_flatten = vox_feats.reshape(-1, self.embed_dims)
        vox_feats_flatten[vox_coords[unmasked_idx, 3], :] = seed_feats[0]
        
        if self.mlp_prior is None:
            vox_feats_flatten[vox_coords[masked_idx, 3], :] = self.mask_embed.weight.view(1, self.embed_dims).expand(masked_idx.shape[0], self.embed_dims).to(dtype)
        else:
            vox_feats_flatten[vox_coords[masked_idx, 3], :] = self.mlp_prior(lss_volume_flatten[masked_idx, :])
        
        vox_feats_diff = self.self_transformer.diffuse_vox_features(
            mlvl_feats,
            vox_feats_flatten,
            512,
            512,
            ref_3d=ref_3d,
            vox_coords=vox_coords,
            unmasked_idx=unmasked_idx,
            grid_length=None,
            bev_pos=bev_pos_self_attn,
            img_metas=img_metas,
            prev_bev=None,
            cam_params=cam_params,
            **kwargs
        )
        
        vox_feats_diff = vox_feats_diff.reshape(self.volume_h, self.volume_w, self.volume_z, self.embed_dims)
        
        depth_pos = self.axis_positional_encoding(torch.zeros((self.volume_h, self.volume_w, self.volume_z), device=vox_feats_diff.device).to(dtype)).to(dtype)
        height_pos = self.axis_positional_encoding(torch.zeros((self.volume_h, self.volume_w, self.volume_z), device=vox_feats_diff.device).to(dtype)).to(dtype)
        width_pos = self.axis_positional_encoding(torch.zeros((self.volume_h, self.volume_w, self.volume_z), device=vox_feats_diff.device).to(dtype)).to(dtype)
        depth_feats = self.depth_block(vox_feats_diff, depth_pos)
        height_feats = self.height_block(vox_feats_diff, height_pos)
        width_feats = self.width_block(vox_feats_diff, width_pos)
        
        depth_feats = depth_feats.permute(3, 0, 1, 2).unsqueeze(0)
        height_feats = height_feats.permute(3, 0, 1, 2).unsqueeze(0)
        width_feats = width_feats.permute(3, 0, 1, 2).unsqueeze(0)
        
        return depth_feats, height_feats, width_feats
    
class DeformableTransformerLayer2D(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 num_levels=3,
                 num_points=4,
                 mlp_ratio=4,
                 attn_layer=MultiScaleDeformableAttention,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        self.norm1 = norm_layer(embed_dims)
        self.attn = attn_layer(
            embed_dims, num_heads, num_levels, num_points, batch_first=True, **kwargs)

        if mlp_ratio == 0:
            return
        self.norm2 = norm_layer(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dims * mlp_ratio, embed_dims),
        )

    def forward(self,
                query,
                value=None,
                query_pos=None,
                ref_pts=None,
                spatial_shapes=None,
                level_start_index=None):
        query = query + self.attn(
            self.norm1(query),
            value=value,
            query_pos=query_pos,
            reference_points=ref_pts,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)
        if not hasattr(self, 'ffn'):
            return query
        query = query + self.ffn(self.norm2(query))
        return query
    
class ScanBlock(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 target_axis='depth'):
        super(ScanBlock, self).__init__()
        
        self.embed_dims = embed_dims
        
        assert target_axis in ['depth', 'width', 'height']
        self.target_axis = target_axis
        
        self.scanblocklayer = ScanBlockLayer(
            embed_dims=embed_dims,
            dim_feedforward=embed_dims*2,
            num_heads=num_heads,
            target_axis=target_axis,
            dropout=0.1
        )
        
    def forward(self, scene_embed, pos_embed=None):
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        if self.target_axis == 'depth':
            scene_embed = scene_embed.permute(1, 2, 0, 3)
            if pos_embed is not None:
                pos_embed = pos_embed.permute(1, 2, 0, 3)
        elif self.target_axis == 'width':
            scene_embed = scene_embed.permute(0, 2, 1, 3)
            if pos_embed is not None:
                pos_embed = pos_embed.permute(0, 2, 1, 3)
            
        axis_1, axis_2, target_axis, dim = scene_embed.shape
        scene_embed = scene_embed.flatten(0, 1)
        pos_embed = pos_embed.flatten(0, 1)
        scene_embed = self.scanblocklayer(scene_embed, pos_embed)
        
        scene_embed = scene_embed.reshape(axis_1, axis_2, target_axis, dim)
        
        if self.target_axis == 'depth':
            scene_embed = scene_embed.permute(2, 0, 1, 3)
        elif self.target_axis == 'width':
            scene_embed = scene_embed.permute(0, 2, 1, 3)
            
        return scene_embed  # h, w, z, c
        

class ScanBlockLayer(nn.Module):
    def __init__(self,
                 embed_dims,
                 dim_feedforward,
                 num_heads,
                 target_axis='depth',
                 dropout=0.):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.target_axis = target_axis
        
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        
        # ffn
        self.linear1 = nn.Linear(embed_dims, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dims)
        self.activation = nn.ReLU()
        self.drop2 = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        
    def forward(self, x, pos_embed=None):
        B, L, C = x.shape
            
        attn_mask = None
        if self.target_axis == 'depth':
            attn_mask = torch.triu(torch.ones(L, L), diagonal=1) == 1
            attn_mask[:,:L//2] = False
            attn_mask = attn_mask.to(x.device)
        elif self.target_axis == 'height':
            attn_mask = torch.tril(torch.ones(L, L), diagonal=-1) == 1
            attn_mask = attn_mask.to(x.device)
        elif self.target_axis == 'width':
            half_L = L // 2
            attn_quarter_mask = torch.tril(torch.ones(half_L, half_L), diagonal=-1) == 1
            attn_half_mask = torch.cat((attn_quarter_mask, attn_quarter_mask.flip(dims=[-1])), dim=-1)
            attn_mask = torch.cat((attn_half_mask, attn_half_mask.flip(dims=[-2])), dim=0)
            attn_mask[:,L//4:-(L//4)] = False
            attn_mask = attn_mask.to(x.device)
                        
        x_norm = self.norm1(x)
        x = x + self.drop1(self.attn(x_norm + pos_embed if pos_embed is not None else x_norm,
                                     x_norm + pos_embed if pos_embed is not None else x_norm,
                                     x_norm,
                                     attn_mask = attn_mask)[0])
        
        # ffn
        x1 = self.norm2(x)
        x = x + self.drop2(self.linear2(self.activation(self.linear1(x1))))
        
        return x

def generate_grid(grid_shape, value=None, offset=0, normalize=False):
    """
    Args:
        grid_shape: The (scaled) shape of grid.
        value: The (unscaled) value the grid represents.
    Returns:
        Grid coordinates of shape [len(grid_shape), *grid_shape]
    """
    if value is None:
        value = grid_shape
    grid = []
    for i, (s, val) in enumerate(zip(grid_shape, value)):
        g = torch.linspace(offset, val - 1 + offset, s, dtype=torch.float)
        if normalize:
            g /= s - 1
        shape_ = [1 for _ in grid_shape]
        shape_[i] = s
        g = g.reshape(1, *shape_).expand(1, *grid_shape)
        grid.append(g)
    return torch.cat(grid, dim=0)

def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.
    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.
    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
        tuple: The [H, W] shape.
    """
    return x.flatten(2).transpose(1, 2).contiguous()

def flatten_multi_scale_feats(feats):
    feat_flatten = torch.cat([nchw_to_nlc(feat) for feat in feats], dim=1)
    shapes = torch.stack([torch.tensor(feat.shape[2:]) for feat in feats]).to(feat_flatten.device)
    return feat_flatten, shapes

def get_level_start_index(shapes):
    return torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))