import torch
from pytorchse3.se3 import se3_log_map, se3_exp_map
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import trimesh

MAX_WIDTH = 0.202
NORMAL_WIDTH = 0.140


def create_gripper_marker(width_scale=1.0, color=[0, 255, 0], tube_radius=0.005, sections=6):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        width_scale (float, optional): Scale of the grasp with w.r.t. the normal width of 140mm, i.e., 0.14. 
        color (list, optional): RGB values of marker.
        tube_radius (float, optional): Radius of cylinders.
        sections (int, optional): Number of sections of each cylinder.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [7.10000000e-02*width_scale, -7.27595772e-12, 1.154999996e-01],
            [7.10000000e-02*width_scale, -7.27595772e-12, 1.959999998e-01],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [-7.100000e-02*width_scale, -7.27595772e-12, 1.154999996e-01],
            [-7.100000e-02*width_scale, -7.27595772e-12, 1.959999998e-01],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=tube_radius, sections=sections, segment=[[0, 0, 0], [0, 0, 1.154999996e-01]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[[-7.100000e-02*width_scale, 0, 1.154999996e-01], [7.100000e-02*width_scale, 0, 1.154999996e-01]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    return tmp


def earth_movers_distance(train_grasps, gen_grasps):
    """
    Compute Earth Mover's Distance between two sets of vectors.
    """
    # Ensure the input sets have the same dimensionality
    assert train_grasps.shape[1] == gen_grasps.shape[1]
    if np.isnan(train_grasps).any() or np.isnan(gen_grasps).any():
        raise ValueError("NaN values exist!")
    # Calculate pairwise distances between vectors in the sets
    distances = np.linalg.norm(train_grasps[:, np.newaxis] - gen_grasps, axis=-1)
    # Solve the linear sum assignment problem
    _, assignment = linear_sum_assignment(distances)
    # Compute the total Earth Mover's Distance
    emd = distances[np.arange(len(assignment)), assignment].mean()
    return emd


def coverage_rate(train_grasps, gen_grasps):
    """
    Function to compute the coverage rate metric.
    """
    assert train_grasps.shape[1] == gen_grasps.shape[1]
    if np.isnan(train_grasps).any() or np.isnan(gen_grasps).any():
        raise ValueError("NaN values exist!")
    dist = cdist(train_grasps, gen_grasps)
    rate = np.sum(np.any(dist <= 0.4, axis=1)) / train_grasps.shape[0]
    return rate


def collision_check(pc, gripper):
    """
    Function to check collision between a point cloud and a gripper.
    """
    return np.sum(gripper.contains(pc)) > 0


def collision_rate(pc, grasps):
    """
    Function to compute the collision rate metric.
    pc' size: N x 6
    """
    if np.isnan(grasps).any():
        return None
    pc = pc[:, :3]  # use only the coordinates of the point cloud
    Rts, ws = se3_exp_map(torch.from_numpy(grasps[:, :-1])).numpy(), (grasps[:, -1] + 1.0)*MAX_WIDTH/2
    grippers = [create_gripper_marker(width_scale=w/NORMAL_WIDTH).apply_transform(Rt) for w, Rt in zip(ws, Rts)]
    collision_rate = np.mean(np.array([collision_check(pc, gripper) for gripper in grippers]))
    return collision_rate