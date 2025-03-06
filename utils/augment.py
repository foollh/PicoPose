import torch
import numpy as np

from .torch_utils import affine_torch, cosSin, get_relative_M

def aug_M_noise(gt_Ms, 
    std_scales=[0.01, 0.05, 0.1, 0.15, 0.2], min_scales=0.5, max_scales=1.5,
    std_rots=[1, 2, 5, 10, 15], max_rot=45,
    std_trans=[2, 5, 10, 15, 20], max_trans=56,
):
    B = gt_Ms.size(0)
    device = gt_Ms.device
    gt_scales = torch.norm(gt_Ms[:, 0, :2], dim=1)
    gt_rots = torch.acos(gt_Ms[:, 0, 0] / gt_scales)
    gt_trans = gt_Ms[:, :2, 2]

    std_scale = np.random.choice(std_scales)
    rand_scales = torch.normal(
        mean=torch.ones([B, ]).to(device),
        std=torch.tensor(std_scale, device=device),
    )
    rand_scales = rand_scales.clamp(min=-min_scales, max=max_scales)
    noise_scales = gt_scales * rand_scales

    std_rot = np.random.choice(std_rots)
    rand_rots = torch.normal(mean=0, std=std_rot, size=(B,)).to(device=device)
    rand_rots = rand_rots.clamp(min=-max_rot, max=max_rot)
    noise_rots = gt_rots + (rand_rots/180) * torch.pi

    std_tran = np.random.choice(std_trans)
    rand_trans = torch.normal(
        mean=torch.zeros([B, 2]).to(device),
        std=torch.tensor([std_tran, std_tran], device=device).view(1, 2),
    )
    rand_trans = torch.clamp(rand_trans, min=-max_trans, max=max_trans)
    noise_trans = gt_trans + rand_trans

    gt_inplane = cosSin((noise_rots + 2 * torch.pi) % (2 * torch.pi))
    cos_theta, sin_theta = gt_inplane[:, 0], gt_inplane[:, 1]
    R = torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], dim=1)
    R = R.reshape(-1, 2, 2)
    noise_Ms = affine_torch(rotation=R, scale=noise_scales, translation=noise_trans)

    return noise_Ms.detach()

def aug_gtM_noise(end_points):
    gt_Ms = get_relative_M(
        src_K=end_points['tem_K'],
        tar_K=end_points['real_K'],
        src_pose=end_points['tem_pose'],
        tar_pose=end_points['real_pose'],
        src_M=end_points['tem_M'],
        tar_M=end_points['real_M'],
    )
    pred_Ms = aug_M_noise(gt_Ms)
    return pred_Ms