import torch
import numpy as np
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from einops import rearrange
from .torch_utils import apply_affine, init_points2d_torch
from .corr_lookup import coords_grid


def compute_init_correspondences(pred_Ms, tem_mask, size=(16,16)):
    B, H, W = tem_mask.shape
    assert H == W
    patch_size = H // size[0]
    tem_mask = F.interpolate(tem_mask.unsqueeze(1), size=size)
    
    grid_points = init_points2d_torch(H, patch_size).repeat(B,1,1).to(pred_Ms.device)
    pred_pts = apply_affine(pred_Ms, grid_points) / patch_size
    pred_pts = rearrange(pred_pts, "b (w h) c -> b c h w", h=size[0])

    xx = torch.arange(0, size[0], device=pred_pts.device)
    yy = torch.arange(0, size[1], device=pred_pts.device)
    grid = coords_grid(B, xx, yy)

    init_flow = pred_pts.float()*tem_mask-grid
    init_certainty = tem_mask
    return init_flow, init_certainty

def compute_stage3_correspondences(pred_flow, pred_certainty, threshold=0.5):
    B, _, H, W = pred_flow.shape
    xx = torch.arange(0, W, device=pred_flow.device)
    yy = torch.arange(0, H, device=pred_flow.device)
    grid = coords_grid(B, xx, yy).permute(0, 2, 3, 1)

    tar_pts_ = pred_flow.permute(0,2,3,1) + grid

    outside0 = torch.logical_and(tar_pts_[..., 0] > 0, tar_pts_[..., 1] > 0)
    outside1 = torch.logical_and(tar_pts_[..., 0] < H-1, tar_pts_[..., 1] < W-1)
    mask = pred_certainty.squeeze(1).sigmoid() > threshold

    pred_src_mask = torch.logical_and(mask, torch.logical_and(outside0, outside1))

    B, H, W = pred_src_mask.shape
    device = pred_src_mask.device
    src_pts_ = torch.nonzero(pred_src_mask)
    b, h, w = (
        src_pts_[:, 0],
        src_pts_[:, 1],
        src_pts_[:, 2],
    )
    src_pts = torch.full((B, H, W, 2), -1, dtype=torch.long, device=device)
    src_pts[b, h, w] = src_pts_[:, [2, 1]]  # swap x, y
    src_pts = rearrange(src_pts, "b h w c -> b (w h) c")

    tar_pts = torch.full((B, H, W, 2), -1, dtype=torch.long, device=device)
    tar_pts[b, h, w] = tar_pts_[b, h, w].long()
    # src_pts = src_pts[..., [1, 0]]  # swap x, y
    tar_pts = rearrange(tar_pts, "b h w c -> b (w h) c")

    return tar_pts, src_pts
