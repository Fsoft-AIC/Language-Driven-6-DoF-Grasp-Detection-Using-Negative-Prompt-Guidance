"""
Sample code for point cloud and grasps visualization using trimesh.
"""


import pickle
import trimesh
import numpy as np
from utils.test_utils import create_gripper_marker
 
 
id = "007d90628c0eb878feeb4c44c1a7b717cec5e4cf10c160689198bd908a1e9d04"
pc = np.load(f"/cm/shared/toannt28/grasp-anything/pc/{id}.npy")
pc = trimesh.points.PointCloud(vertices=pc[:, :3], colors=pc[:, 3:])
 
with open(f"/cm/shared/toannt28/grasp-anything/grasp/{id}_0", "rb") as file:
    grasp = pickle.load(file)
   
grasp = [create_gripper_marker(w/0.14).apply_transform(Rt) for (Rt, w) in zip(grasp[0], grasp[1])]
scene = trimesh.scene.Scene([pc, grasp])
scene.show(line_settings={'point_size':20})
 