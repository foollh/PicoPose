
'''
Modified from https://github.com/rasmushaugaard/surfemb/blob/master/surfemb/data/obj.py
'''

import os
import glob
import json
import numpy as np
import trimesh
from tqdm import tqdm


class Obj:
    def __init__(
            self, obj_id,
            mesh: trimesh.Trimesh,
            model_points,
            diameter: float,
            symmetry_flag: int,
    ):
        self.obj_id = obj_id
        self.mesh = mesh
        self.model_points = model_points
        self.diameter = diameter
        self.symmetry_flag = symmetry_flag

    def get_item(self, return_color=False, sample_num=2048):
        if return_color:
            model_points, _, model_colors = trimesh.sample.sample_surface(self.mesh, sample_num, sample_color=True)
            model_points = model_points.astype(np.float32) / 1000.0
            return (model_points, model_colors, self.symmetry_flag)
        else:
            return (self.model_points, self.symmetry_flag)


def load_obj(
        model_path, obj_id: int, sample_num: int,
):
    models_info = json.load(open(os.path.join(model_path, 'models_info.json')))
    if 'lmoWonder3d' in model_path:
        mesh = trimesh.load_mesh(os.path.join(model_path, f'obj_{obj_id:06d}.obj'))
    else:
        mesh = trimesh.load_mesh(os.path.join(model_path, f'obj_{obj_id:06d}.ply'))
    model_points = mesh.sample(sample_num).astype(np.float32) / 1000.0
    diameter = models_info[str(obj_id)]['diameter'] / 1000.0
    if 'symmetries_continuous' in models_info[str(obj_id)]:
        symmetry_flag = 1
    elif 'symmetries_discrete' in models_info[str(obj_id)]:
        symmetry_flag = 1
    else:
        symmetry_flag = 0
    return Obj(
        obj_id, mesh, model_points, diameter, symmetry_flag
    )


def load_objs(
        model_path='models',
        sample_num=512,
        show_progressbar=True
):
    objs = []
    if 'lmoWonder3d' in model_path:
        obj_ids = sorted([int(p.split('/')[-1][4:10]) for p in glob.glob(os.path.join(model_path, '*.obj'))])
    else:
        obj_ids = sorted([int(p.split('/')[-1][4:10]) for p in glob.glob(os.path.join(model_path, '*.ply'))])

    cnt = 0
    for obj_id in tqdm(obj_ids, 'loading objects') if show_progressbar else obj_ids:
        objs.append(
            load_obj(model_path, obj_id, sample_num)
        )
        cnt+=1
    return objs, obj_ids


