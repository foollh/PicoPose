import cv2
import torch
import numpy as np
from einops import rearrange, repeat
from .torch_utils import inverse_affine, normalize_affine_transform, gather



def pose_recovery_2d_prediction(
    query_M,
    query_K,
    pred_Ms,
    template_K,
    template_Ms,
    template_poses,
):
    """
    Recover 6D pose from 2D predictions
    1. Rotation = Inplane (Kabsch) rotation + Viewpoint rotation
    2. 2D translation = 2D (Kabsch) transform + Crop transform of query and template
    3. Z (on the ray) scale = 2D scale + Focal length ratio
    Input:
        query_M: (B, 3, 3)
        query_K: (B, 3, 3)

        pred_Ms: (B, 3, 3)

        template_K: (B, 3, 3)
        template_Ms: (B, 3, 3)
        template_poses: (B, 4, 4)
    """
    # Step 1: Rotation = Azmith, Elevation + Inplane (Kabsch) rotation
    pred_poses = template_poses.clone()
    pred_R_inplane = normalize_affine_transform(pred_Ms)

    pred_poses[:, :3, :3] = torch.matmul(
        pred_R_inplane, pred_poses[:, :3, :3]
    )

    # Step 2: 2D translation
    temp_z = pred_poses[:, 2, 3].clone()
    temp_translation = rearrange(pred_poses[:, :3, 3], "b c -> b c 1")
    temp_center2d = torch.matmul(template_K, temp_translation)
    temp_center2d = temp_center2d / temp_center2d[:, 2].unsqueeze(1)

    # fully 2D affine transform from template to query
    inv_query_M = inverse_affine(query_M)
    affine2d = torch.matmul(torch.matmul(inv_query_M, pred_Ms), template_Ms)

    # recover 2D center of query
    query_center2d = torch.matmul(affine2d, temp_center2d)
    inv_query_K = torch.inverse(query_K)

    # recover Z_query = (Z_temp / scale_2d) * (focal length ratio)
    scale2d = torch.norm(affine2d[:, :2, 0], dim=1)
    focal_ratio = query_K[:, 0, 0] / template_K[:, 0, 0]
    query_z = (temp_z / scale2d) * focal_ratio

    # combine 2D translation and Z scale
    query_translation = torch.matmul(inv_query_K, query_center2d).squeeze(-1)
    query_translation_z = query_translation[:, 2].unsqueeze(-1).clone()
    query_translation /= query_translation_z
    pred_poses[:, :3, 3] = query_translation * query_z.unsqueeze(-1)

    return pred_poses


def pose_recovery_ransac_pnp(
    tar_pts_2d, 
    src_pts_3d, 
    K, 
    tem_pose, 
    tar_pts, 
    src_pts, 
):  
    coord_2d = gather(tar_pts_2d[None], tar_pts[None].long())
    coord_3d = gather(src_pts_3d[None], src_pts[None].long())

    coord_2d = coord_2d.detach().cpu().numpy()
    coord_3d = coord_3d.detach().cpu().numpy()
    tem_pose = tem_pose.detach().cpu().numpy()
    intrinsic_matrix = K.detach().cpu().numpy()

    coord_3d = (coord_3d - tem_pose[:3, 3][None]) @ tem_pose[:3, :3]

    success = True
    coord_2d = np.ascontiguousarray(coord_2d.astype(np.float32))
    coord_3d = np.ascontiguousarray(coord_3d.astype(np.float32))
    intrinsic_matrix = np.ascontiguousarray(intrinsic_matrix)

    try:
        # ransac+epnp
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(coord_3d.astype(np.float32),
                                            coord_2d.astype(np.float32), intrinsic_matrix, distCoeffs=None,
                                            reprojectionError=2, iterationsCount=150, flags=cv2.SOLVEPNP_EPNP)
        rot, _ = cv2.Rodrigues(rvecs, jacobian=None)
        inliers_ratio = len(inliers) / coord_2d.shape[0]

    except:
        rot = np.eye(3)
        tvecs = np.array([[0], [0], [1]])
        inliers_ratio = 0.
        success = False

    return rot, tvecs, inliers_ratio, success

