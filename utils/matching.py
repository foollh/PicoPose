import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def matching_features_similarity(src_feat, tar_feat, src_mask, tar_mask):
    B = src_feat.shape[0]
    feat_size = (src_feat.shape[2], src_feat.shape[3])

    assert feat_size[0] == feat_size[1]
    num_patches = feat_size[0]

    tar_feat = F.normalize(tar_feat, dim=1)
    tar_feat = rearrange(tar_feat, "b c h w -> b c (h w)")

    src_mask = F.interpolate(src_mask.unsqueeze(1), size=feat_size)
    src_mask = rearrange(src_mask, "b 1 h w -> b (h w)")
    src_feat = F.normalize(src_feat, dim=1)
    src_feat = rearrange(src_feat, "b c h w -> b c (h w)")

    # Step 1: Find nearest neighbor for each patch of query image
    sim = torch.einsum("b c t, b c s -> b t s", tar_feat, src_feat)
    sim *= src_mask[:, None, :]
    sim[sim < 0] = 0
    sim = rearrange(sim, "b (w h) c -> b c h w", h=feat_size[0])
    return sim


def matching_templates(
    src_feats, tar_feat, src_masks, tar_mask, topk=5
):
    B, N, C, H, W = src_feats.shape
    device = tar_mask.device

    assert H == W
    num_patches = H

    tar_mask = F.interpolate(tar_mask.unsqueeze(1), size=(H, W))
    tar_mask = rearrange(tar_mask, "b 1 h w -> b (h w)")
    tar_feat = F.normalize(tar_feat, dim=1)
    tar_feat = rearrange(tar_feat, "b c h w -> b c (h w)")

    src_feats = F.normalize(src_feats, dim=2)
    src_feats = rearrange(src_feats, "b n c h w -> b n c (h w)")

    # Step 1: Find nearest neighbor for each patch of query image
    sim = torch.einsum("b c t, b n c s -> b n t s", tar_feat, src_feats)
    sim *= tar_mask[:, None, :, None]

    score_tar2src, idx_tar2src = torch.max(sim, dim=3)  # b x n x t
    score_src2tar, idx_src2tar = torch.max(sim, dim=2)  # b x n x s

    # Find valid patches has mask in both source and target
    tar_masks = repeat(tar_mask, "b t -> b n t", n=N)

    mask_all = (
        tar_masks  # mask of query = 0
        * (idx_src2tar != 0)  # sim = 0
        * (idx_tar2src != 0)  # sim = 0
    )
    
    # Step 2: Find best template for each target
    mask = mask_all.sum(dim=2) > 0
    sim_avg = torch.zeros(B, N, device=device)
    sim_avg[mask] = torch.sum(score_tar2src * mask_all, dim=2)[mask] / (
        num_patches**2
    )
    pred_score_src, pred_id_src = torch.topk(sim_avg, topk, dim=1)
    return pred_score_src, pred_id_src

