import os
import argparse
import numpy as np
from tqdm import tqdm
import time
from functools import partial
import multiprocessing
from omegaconf import OmegaConf
import glob
from pathlib import Path
import json

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(BASE_DIR, '..', 'src'))

from rendering.src.lib3d.template_transform import get_obj_poses_from_template_level
from rendering.src.utils.logging import get_logger
from utils.trimesh import get_obj_diameter

logger = get_logger(__name__)


def call_render(
    idx_obj,
    list_cad_path,
    list_output_dir,
    obj_pose_paths,
    disable_output,
    num_gpus,
    use_blenderProc,
):
    output_dir = list_output_dir[idx_obj]
    cad_path = list_cad_path[idx_obj]
    obj_pose_path = obj_pose_paths[idx_obj]
    if os.path.exists(output_dir):
        os.system("rm -r {}".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gpus_device = idx_obj % int(num_gpus)
    os.makedirs(output_dir, exist_ok=True)
    if use_blenderProc: # TODO: remove blenderProc
        command = f"blenderproc run ./src/lib3d/blenderproc.py --cad_path {cad_path} --obj_pose {obj_pose_path} --output_dir {output_dir} --gpus_devices {gpus_device}"
    else: # TODO: understand why this is not working for tless and itodd
        command = f"python -m src.custom_megapose.call_panda3d --cad_path {cad_path} --obj_pose {obj_pose_path} --output_dir {output_dir} --gpus_devices {gpus_device}"

    if disable_output:
        command += " --disable_output true"
    else:
        command += " --disable_output false"
    command += " --scale_translation true"  # scale translation to meter
    os.system(command)

    # make sure the number of rendered images is correct
    num_images = len(glob.glob(f"{output_dir}/*.png"))
    if num_images == len(np.load(obj_pose_path)) * 2:
        return True
    else:
        logger.info(f"Found only {num_images} for  {cad_path} {obj_pose_path}")
        return False


def render(cfg) -> None:
    num_gpus = cfg.devices
    disable_output = False

    num_workers = int(cfg.num_workers)

    root_dir = Path(cfg.bop_dataset_dir)
    save_dir = Path(cfg.save_dir)
    root_save_dir = save_dir / "templates"
    return_cam = False
    template_poses = get_obj_poses_from_template_level(level=1, pose_distribution="all", return_cam=return_cam)

    for dataset_name in [
        "lmo",
        "tless",
        "tudl",
        "icbin",
        "itodd",
        "hb",
        "ycbv",
    ]:
        dataset_save_dir = root_save_dir / f"{dataset_name}_normalize_2"
        logger.info(f"Rendering templates for {dataset_name}")
        os.makedirs(dataset_save_dir, exist_ok=True)
        obj_pose_dir = dataset_save_dir / "object_poses"
        os.makedirs(obj_pose_dir, exist_ok=True)

        if dataset_name in ["tless"]:
            cad_name = "models_cad"
        else:
            cad_name = "models"
        cad_dir = root_dir / dataset_name / cad_name

        cad_paths = cad_dir.glob("*.ply")
        cad_paths = list(cad_paths)
        logger.info(f"Found {len(list(cad_paths))} objects")

        output_dirs = []
        obj_pose_paths = []
        for cad_path in cad_paths:
            object_id = int(os.path.basename(cad_path).split(".")[0][4:])
            output_dir = dataset_save_dir / f"{object_id:06d}"
            output_dirs.append(output_dir)

            diameter = get_obj_diameter(cad_path)

            obj_pose_path = os.path.join(obj_pose_dir, f"{object_id:06d}.npy")
            obj_poses = template_poses.copy()
            obj_poses[:, :3, 3] = np.array([0, 0, diameter])[None].repeat(len(template_poses),axis=0) 
            obj_pose_paths.append(obj_pose_path)
            np.save(obj_pose_path, obj_poses)

        os.makedirs(dataset_save_dir, exist_ok=True)

        pool = multiprocessing.Pool(processes=int(num_workers))

        logger.info("Start rendering for {} objects".format(len(cad_paths)))
        start_time = time.time()
        pool = multiprocessing.Pool(processes=num_workers)
        call_render_ = partial(
            call_render,
            list_cad_path=cad_paths,
            list_output_dir=output_dirs,
            obj_pose_paths=obj_pose_paths,
            disable_output=disable_output,
            num_gpus=num_gpus,
            use_blenderProc=True if dataset_name in ["tless", "itodd"] else False,
        )
        values = list(
            tqdm(
                pool.imap_unordered(call_render_, range(len(cad_paths))),
                total=len(cad_paths),
            )
        )
        correct_values = [val for val in values]
        logger.info(f"Finished for {len(correct_values)}/{len(cad_paths)} objects")
        finish_time = time.time()
        logger.info(f"Total time {len(cad_paths)}: {finish_time - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default=1, nargs="?")
    parser.add_argument("--num_workers", default=10, nargs="?")
    parser.add_argument("--bop_dataset_dir", default='data/BOP_Datasets', nargs="?")
    parser.add_argument("--save_dir", default='data/BOP_Datasets', nargs="?")
    args = parser.parse_args()
    render(args)