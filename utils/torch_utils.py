import torch
import random
import numpy as np
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from einops import repeat, rearrange


def set_seed(random_seed):
  np.random.seed(random_seed)
  random.seed(random_seed)
  torch.manual_seed(random_seed)
  torch.cuda.manual_seed_all(random_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def calc_gt_trans_scale_inplane(end_points):
    # compute gt relScale, relInplane template -> real
    gt_relScale, gt_relInplane = get_relative_scale_inplane(
        src_K=end_points['tem_K'],
        tar_K=end_points['real_K'],
        src_pose=end_points['tem_pose'],
        tar_pose=end_points['real_pose'],
        src_M=end_points['tem_M'],
        tar_M=end_points['real_M'],
    )
    # compute 2D translation
    real_translation = rearrange(end_points['real_pose'][:, :3, 3], "b c -> b c 1")
    real_center2d = torch.matmul(end_points['real_K'], real_translation)
    real_center2d = real_center2d / real_center2d[:, 2].unsqueeze(2)
    real_center2d_M = torch.matmul(end_points['real_M'], real_center2d)
    temp_translation = rearrange(end_points['tem_pose'][:, :3, 3], "b c -> b c 1")
    temp_center2d = torch.matmul(end_points['tem_K'], temp_translation)
    temp_center2d = temp_center2d / temp_center2d[:, 2].unsqueeze(2)
    temp_center2d_M = torch.matmul(end_points['tem_M'], temp_center2d)
    gt_2d_translation = (real_center2d_M-temp_center2d_M)[:, :2].squeeze(-1)
    return gt_2d_translation, gt_relScale, gt_relInplane

def calc_pred_Ms(pred_scale, pred_inplane, pred_translation, tem_pose, tem_K, tem_M, trans_scale=14):
    cos_theta, sin_theta = pred_inplane[:, 0], pred_inplane[:, 1]
    R = torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], dim=1)
    R = R.reshape(-1, 2, 2)
    temp_translation = rearrange(tem_pose[:, :3, 3], "b c -> b c 1")
    temp_center2d = torch.matmul(tem_K, temp_translation)
    temp_center2d = temp_center2d / temp_center2d[:, 2].unsqueeze(2)
    temp_center2d_M = torch.matmul(tem_M, temp_center2d)
    pred_Ms = affine_torch(scale=pred_scale, rotation=R)  # (N, 3, 3)
    aff_src_pts = apply_affine(pred_Ms, temp_center2d_M[:, :2, 0])
    real_center2d_M = temp_center2d_M[:, :2, 0] + pred_translation * trans_scale
    pred_Ms[:, :2, 2] = real_center2d_M - aff_src_pts
    return pred_Ms

def affine_torch(rotation, scale=None, translation=None):
    if len(rotation.shape) == 2:
        """
        Create 2D affine transformation matrix
        """
        M = torch.eye(3, device=scale.device, dtype=scale.dtype)
        M[:2, :2] = rotation
        if scale is not None:
            M[:2, :2] *= scale
        if translation is not None:
            M[:2, 2] = translation
        return M
    else:
        Ms = torch.eye(3, device=scale.device, dtype=scale.dtype)
        Ms = Ms.unsqueeze(0).repeat(rotation.shape[0], 1, 1)
        Ms[:, :2, :2] = rotation
        if scale is not None:
            Ms[:, :2, :2] *= scale.unsqueeze(1).unsqueeze(1)
        if translation is not None:
            Ms[:, :2, 2] = translation
        return Ms


def homogenuous(pixel_points):
    """
    Convert pixel coordinates to homogenuous coordinates
    """
    device = pixel_points.device
    if len(pixel_points.shape) == 2:
        one_vector = torch.ones(pixel_points.shape[0], 1).to(device)
        return torch.cat([pixel_points, one_vector], dim=1)
    elif len(pixel_points.shape) == 3:
        one_vector = torch.ones(pixel_points.shape[0], pixel_points.shape[1], 1).to(
            device
        )
        return torch.cat([pixel_points, one_vector], dim=2)
    else:
        raise NotImplementedError


def inverse_affine(M):
    """
    Inverse 2D affine transformation matrix of cropping
    """
    if len(M.shape) == 2:
        M = M.unsqueeze(0)
    if len(M.shape) == 3:
        assert (M[:, 1, 0] == 0).all() and (M[:, 0, 1] == 0).all()
        assert (M[:, 0, 0] == M[:, 1, 1]).all(), f"M: {M}"

        scale = M[:, 0, 0]
        M_inv = torch.eye(3, device=M.device, dtype=M.dtype)
        M_inv = M_inv.unsqueeze(0).repeat(M.shape[0], 1, 1)
        M_inv[:, 0, 0] = 1 / scale  # scale
        M_inv[:, 1, 1] = 1 / scale  # scale
        M_inv[:, :2, 2] = -M[:, :2, 2] / scale.unsqueeze(1)  # translation
    else:
        raise ValueError("M must be 2D or 3D")
    return M_inv


def apply_affine(M, points):
    """
    M: (N, 3, 3)
    points: (N, 2)
    """
    if len(points.shape) == 2:
        transformed_points = torch.einsum(
            "bhc,bc->bh",
            M,
            homogenuous(points),
        )  # (N, 3)
        transformed_points = transformed_points[:, :2] / transformed_points[:, 2:]
    elif len(points.shape) == 3:
        transformed_points = torch.einsum(
            "bhc,bnc->bnh",
            M,
            homogenuous(points),
        )
        transformed_points = transformed_points[:, :, :2] / transformed_points[:, :, 2:]
    else:
        raise NotImplementedError
    return transformed_points


def unproject_points(points2d, K, depth):
    """
    Unproject points from 2D to 3D
    """

    idx = torch.arange(points2d.shape[0])[:, None].repeat(1, points2d.shape[1])
    points2d[:, :, 1] = torch.clamp(points2d[:, :, 1], 0, depth.shape[1] - 1)
    points2d[:, :, 0] = torch.clamp(points2d[:, :, 0], 0, depth.shape[2] - 1)
    depth1d = depth[idx, points2d[:, :, 1].long(), points2d[:, :, 0].long()]
    points3d = homogenuous(points2d).float()
    K_inv = torch.inverse(K).float()
    points3d = torch.matmul(K_inv, points3d.permute(0, 2, 1)).permute(0, 2, 1)
    points3d = points3d * depth1d.unsqueeze(-1)
    return points3d


def project_points(points3d, K):
    """
    Project points from 3D to 2D
    points_3d: (N, 3)
    """
    points2d = torch.matmul(K, points3d.permute(0, 2, 1)).permute(0, 2, 1)
    points2d = points2d[:, :, :2] / points2d[:, :, 2:]
    return points2d


def cosSin(angle):
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)


def get_relative_scale_inplane(src_K, tar_K, src_pose, tar_pose, src_M, tar_M):
    """
    scale(source->target) = (ref_z / query_z) * (scale(query_M) / scale(ref_M)) * (ref_f / query_f)
    """
    relZ = src_pose[:, 2, 3] / tar_pose[:, 2, 3]
    relCrop = torch.norm(tar_M[:, :2, 0], dim=1) / torch.norm(src_M[:, :2, 0], dim=1)
    rel_focal = src_K[:, 0, 0] / tar_K[:, 0, 0]
    relScale = relZ * relCrop / rel_focal

    relR = torch.matmul(tar_pose[:, :3, :3], src_pose[:, :3, :3].transpose(1, 2))
    if relR.device == torch.device("cpu"):
        relativeR = Rotation.from_matrix(relR.numpy()).as_euler("zxy")
    else:
        relativeR = Rotation.from_matrix(relR.cpu().numpy()).as_euler("zxy")
    relInplane = torch.from_numpy(relativeR[:, 0]).float().to(relR.device)
    return relScale, (relInplane + 2 * torch.pi) % (2 * torch.pi)

def get_relative_outplane(src_pose, tar_pose):
    relR = torch.matmul(tar_pose[:, :3, :3], src_pose[:, :3, :3].transpose(1, 2))
    if relR.device == torch.device("cpu"):
        relativeR = Rotation.from_matrix(relR.numpy()).as_euler("zxy")
    else:
        relativeR = Rotation.from_matrix(relR.cpu().numpy()).as_euler("zxy")
    relX = torch.from_numpy(relativeR[:, 1]).float().to(relR.device)
    relY = torch.from_numpy(relativeR[:, 2]).float().to(relR.device)
    return [(relX + 2 * torch.pi) % (2 * torch.pi), (relY + 2 * torch.pi) % (2 * torch.pi)]

def get_relative_M(src_K, tar_K, src_pose, tar_pose, src_M, tar_M):
    """
    scale(source->target) = (ref_z / query_z) * (scale(query_M) / scale(ref_M)) * (ref_f / query_f)
    """
    relZ = src_pose[:, 2, 3] / tar_pose[:, 2, 3]
    relCrop = torch.norm(tar_M[:, :2, 0], dim=1) / torch.norm(src_M[:, :2, 0], dim=1)
    rel_focal = src_K[:, 0, 0] / tar_K[:, 0, 0]
    relScale = relZ * relCrop / rel_focal

    relR = torch.matmul(tar_pose[:, :3, :3], src_pose[:, :3, :3].transpose(1, 2))
    if relR.device == torch.device("cpu"):
        relativeR = Rotation.from_matrix(relR.numpy()).as_euler("zxy")
    else:
        relativeR = Rotation.from_matrix(relR.cpu().numpy()).as_euler("zxy")
    relInplane = torch.from_numpy(relativeR[:, 0]).float().to(relR.device)

    gt_inplane = cosSin((relInplane + 2 * torch.pi) % (2 * torch.pi))
    cos_theta, sin_theta = gt_inplane[:, 0], gt_inplane[:, 1]
    R = torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], dim=1)
    R = R.reshape(-1, 2, 2)
    real_translation = rearrange(tar_pose[:, :3, 3], "b c -> b c 1")
    real_center2d = torch.matmul(tar_K, real_translation)
    real_center2d = real_center2d / real_center2d[:, 2].unsqueeze(2)
    real_center2d_M = torch.matmul(tar_M, real_center2d)
    temp_translation = rearrange(src_pose[:, :3, 3], "b c -> b c 1")
    temp_center2d = torch.matmul(src_K, temp_translation)
    temp_center2d = temp_center2d / temp_center2d[:, 2].unsqueeze(2)
    temp_center2d_M = torch.matmul(src_M, temp_center2d)
    M_candidates = affine_torch(scale=relScale, rotation=R)  # (N, 3, 3)
    aff_src_pts = apply_affine(M_candidates, temp_center2d_M[:, :2, 0])
    M_candidates[:, :2, 2] = real_center2d_M[:, :2, 0] - aff_src_pts
    return M_candidates

def normalize_affine_transform(transforms):
    """
    Input: Affine transformation
    Output: Normalized affine transformation
    """
    norm_transforms = torch.zeros_like(transforms)
    norm_transforms[:, 2, 2] = 1

    scale = torch.norm(transforms[:, :2, 0], dim=1)
    scale = repeat(scale, "b -> b h w", h=2, w=2)

    norm_transforms[:, :2, :2] = transforms[:, :2, :2] / scale
    return norm_transforms


def geodesic_distance(pred_cosSin, gt_cosSin, normalize=False):
    if normalize:
        pred_cosSin = F.normalize(pred_cosSin, dim=1)
        gt_cosSin = F.normalize(gt_cosSin, dim=1)
    pred_cos = pred_cosSin[:, 0]
    pred_sin = pred_cosSin[:, 1]
    gt_cos = gt_cosSin[:, 0]
    gt_sin = gt_cosSin[:, 1]
    cos_diff = pred_cos * gt_cos + pred_sin * gt_sin
    cos_diff = torch.clamp(cos_diff, -1, 1)
    loss = torch.acos(cos_diff).mean()
    return loss


def gather(features, index_patches):
    """
    Args:
    - features: (B, C, H, W)
    - index_patches: (B, N, 2) where N is the number of patches, and 2 is the (x, y) index of the patch
    Output:
    - selected_features: (BxN, C) where index_patches!= -1
    """
    B, C, H, W = features.shape
    features = rearrange(features, "b c h w -> b (h w) c")

    index_patches = index_patches.clone()
    x, y = index_patches[:, :, 0], index_patches[:, :, 1]
    mask = torch.logical_and(x != -1, y != -1)
    index_patches[index_patches == -1] = H - 1  # dirty fix so that gather does not fail

    # Combine index_x and index_y into a single index tensor
    index = y * W + x

    # Gather features based on index tensor
    flatten_features = torch.gather(
        features, dim=1, index=index.unsqueeze(-1).repeat(1, 1, C)
    )

    # reshape to (BxN, C)
    flatten_features = rearrange(flatten_features, "b n c -> (b n) c")
    mask = rearrange(mask, "b n -> (b n)")
    return flatten_features[mask]


def init_points2d_numpy(tar_size, patch_size):
    x = np.arange(0, tar_size, patch_size, dtype=np.float32)
    x += patch_size / 2
    y = np.arange(0, tar_size, patch_size, dtype=np.float32)
    y += patch_size / 2

    yy, xx = np.meshgrid(y, x, indexing='ij')
    grid_points = np.stack([yy, xx], axis=2)
    return grid_points

def init_points2d_torch(tar_size, patch_size):
    x = torch.arange(0, tar_size, patch_size).float()
    x += patch_size / 2
    y = torch.arange(0, tar_size, patch_size).float()
    y += patch_size / 2

    yy, xx = torch.meshgrid(y, x)
    grid_points = torch.stack([yy.flatten(), xx.flatten()], dim=1)
    return grid_points