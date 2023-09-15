# MIT License
#
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, University of Washington. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import numpy as np
import time
from tqdm.auto import tqdm, trange
import os
import sys
import h5py
import random

from robofin.robots import FrankaRobot, FrankaGripper
from robofin.bullet import Bullet, BulletController

from pathlib import Path
from geometrout.primitive import Cuboid, Cylinder
from geometrout.transform import SE3

import pickle
from dataclasses import dataclass, field
from typing import List, Union, Optional, Dict
import argparse

import torch
import open3d as o3d
from robofin.pointcloud.torch import FrankaSampler
# from mpinets.model import MotionPolicyNetwork

sys.path.append('/home/jayaram/research/research_tracks/table_top_rearragement/global_classifier_guidance_for_7DOF_manipulator/mpinet_environment')
# sys.path.append('/home/vishal/Volume_E/Active/Undergrad_research/CoRL2023/codes/Vishal/Structured_Bullet/diffusion/')

from mpinets.geometry import construct_mixed_point_cloud
from mpinets.utils import normalize_franka_joints, unnormalize_franka_joints
from mpinets.metrics import Evaluator
from mpinets.types import PlanningProblem, ProblemSet
from mpinets.model import MotionPolicyNetwork

import trimesh
import meshcat
import urchin
from robot import Robot

from  mpinet_environment.diffusion.Models.temporalunet import TemporalUNet
from mpinet_environment.guidance.collision_checker import (
    NNSceneCollisionChecker,
    NNSelfCollisionChecker,
)

END_EFFECTOR_FRAME = "right_gripper"
NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128
MAX_ROLLOUT_LENGTH = 150

from mpinet_environment.diffusion.infer_diffusion import infer

device = "cuda" if torch.cuda.is_available() else "cpu"

def make_point_cloud_from_problem(
    q0: torch.Tensor,
    target: SE3,
    obstacle_points: np.ndarray,
    fk_sampler: FrankaSampler,
) -> torch.Tensor:
    robot_points = fk_sampler.sample(q0, NUM_ROBOT_POINTS)

    target_points = fk_sampler.sample_end_effector(
        torch.as_tensor(target.matrix).type_as(robot_points).unsqueeze(0),
        num_points=NUM_TARGET_POINTS,
    )
    xyz = torch.cat(
        (
            torch.zeros(NUM_ROBOT_POINTS, 4),
            torch.ones(NUM_OBSTACLE_POINTS, 4),
            2 * torch.ones(NUM_TARGET_POINTS, 4),
        ),
        dim=0,
    )
    xyz[:NUM_ROBOT_POINTS, :3] = robot_points.float()
    random_obstacle_indices = np.random.choice(
        len(obstacle_points), size=NUM_OBSTACLE_POINTS, replace=False
    )
    xyz[
        NUM_ROBOT_POINTS : NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS,
        :3,
    ] = torch.as_tensor(obstacle_points[random_obstacle_indices, :3]).float()
    xyz[
        NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS :,
        :3,
    ] = target_points.float()
    return xyz


def make_env_point_cloud_from_primitives_with_robot(
    q0, 
    obstacles: List[Union[Cuboid, Cylinder]],
    fk_sampler,
) -> torch.Tensor:
    """
    Creates the pointcloud of the scene, including the target and the robot. When performing
    a rollout, the robot points will be replaced based on the model's prediction

    :param q0 torch.Tensor: The starting configuration (dimensions [1 x 7])
    :param target SE3: The target pose in the `right_gripper` frame
    :param obstacles List[Union[Cuboid, Cylinder]]: The obstacles in the scene
    :param fk_sampler FrankaSampler: A sampler that produces points on the robot's surface
    :rtype torch.Tensor: The pointcloud (dimensions
                         [1 x NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS + NUM_TARGET_POINTS x 4])
    """
    robot_points = fk_sampler.sample(q0, NUM_ROBOT_POINTS)

    xyz = torch.cat(
        (
            torch.zeros(NUM_ROBOT_POINTS, 4),
            torch.ones(NUM_OBSTACLE_POINTS, 4),
        ),
        dim=0,
    )
    obstacle_points = construct_mixed_point_cloud(obstacles, NUM_OBSTACLE_POINTS)

    xyz[:NUM_ROBOT_POINTS, :3] = robot_points.float()
    xyz[NUM_ROBOT_POINTS: NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS,
        :3,
    ] = torch.as_tensor(obstacle_points[:, :3]).float()
    return xyz

def make_env_point_cloud_from_primitives(
    obstacles: List[Union[Cuboid, Cylinder]],
) -> torch.Tensor:
    """
    Creates the pointcloud of the scene, including the target and the robot. When performing
    a rollout, the robot points will be replaced based on the model's prediction

    :param q0 torch.Tensor: The starting configuration (dimensions [1 x 7])
    :param target SE3: The target pose in the `right_gripper` frame
    :param obstacles List[Union[Cuboid, Cylinder]]: The obstacles in the scene
    :param fk_sampler FrankaSampler: A sampler that produces points on the robot's surface
    :rtype torch.Tensor: The pointcloud (dimensions
                         [1 x NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS + NUM_TARGET_POINTS x 4])
    """
    xyz = torch.cat(
        (
            torch.ones(NUM_OBSTACLE_POINTS, 4),
        ),
        dim=0,
    )
    obstacle_points = construct_mixed_point_cloud(obstacles, NUM_OBSTACLE_POINTS)

    xyz[:NUM_OBSTACLE_POINTS,
        :3,
    ] = torch.as_tensor(obstacle_points[:, :3]).float()
    return xyz



def rollout_until_success(
    mdl: MotionPolicyNetwork,
    q0: np.ndarray,
    target: SE3,
    point_cloud: torch.Tensor,
    fk_sampler: FrankaSampler,
) -> np.ndarray:
    """
    Rolls out the policy until the success criteria are met. The criteria are that the
    end effector is within 1cm and 15 degrees of the target. Gives up after 150 prediction
    steps.

    :param mdl MotionPolicyNetwork: The policy
    :param q0 np.ndarray: The starting configuration (dimension [7])
    :param target SE3: The target in the `right_gripper` frame
    :param point_cloud torch.Tensor: The point cloud to be fed into the model. Should have
                                     dimensions [1 x NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS + NUM_TARGET_POINTS x 4]
                                     and consist of the constituent points stacked in
                                     this order (robot, obstacle, target).
    :param fk_sampler FrankaSampler: A sampler that produces points on the robot's surface
    :rtype np.ndarray: The trajectory
    """
    q = torch.as_tensor(q0).unsqueeze(0).float().cuda()
    assert q.ndim == 2
    # This block is to adapt for the case where we only want to roll out a
    # single trajectory
    trajectory = [q]
    q_norm = normalize_franka_joints(q)
    assert isinstance(q_norm, torch.Tensor)
    success = False

    def sampler(config):
        return fk_sampler.sample(config, NUM_ROBOT_POINTS)

    for i in range(MAX_ROLLOUT_LENGTH):
        q_norm = torch.clamp(q_norm + mdl(point_cloud, q_norm), min=-1, max=1)
        qt = unnormalize_franka_joints(q_norm)
        assert isinstance(qt, torch.Tensor)
        trajectory.append(qt)
        eff_pose = FrankaRobot.fk(
            qt.squeeze().detach().cpu().numpy(), eff_frame="right_gripper"
        )
        # Stop when the robot gets within 1cm and 15 degrees of the target
        if (
            np.linalg.norm(eff_pose._xyz - target._xyz) < 0.01
            and np.abs(
                np.degrees((eff_pose.so3._quat * target.so3._quat.conjugate).radians)
            )
            < 15
        ):
            break
        samples = sampler(qt).type_as(point_cloud)
        point_cloud[:, : samples.shape[1], :3] = samples

    return np.asarray([t.squeeze().detach().cpu().numpy() for t in trajectory])

def make_point_cloud_from_primitives(
    q0: torch.Tensor,
    target: SE3,
    obstacles: List[Union[Cuboid, Cylinder]],
    fk_sampler: FrankaSampler,
) -> torch.Tensor:
    """
    Creates the pointcloud of the scene, including the target and the robot. When performing
    a rollout, the robot points will be replaced based on the model's prediction

    :param q0 torch.Tensor: The starting configuration (dimensions [1 x 7])
    :param target SE3: The target pose in the `right_gripper` frame
    :param obstacles List[Union[Cuboid, Cylinder]]: The obstacles in the scene
    :param fk_sampler FrankaSampler: A sampler that produces points on the robot's surface
    :rtype torch.Tensor: The pointcloud (dimensions
                         [1 x NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS + NUM_TARGET_POINTS x 4])
    """
    obstacle_points = construct_mixed_point_cloud(obstacles, NUM_OBSTACLE_POINTS)
    robot_points = fk_sampler.sample(q0, NUM_ROBOT_POINTS)

    target_points = fk_sampler.sample_end_effector(
        torch.as_tensor(target.matrix).type_as(robot_points).unsqueeze(0),
        num_points=NUM_TARGET_POINTS,
    )
    xyz = torch.cat(
        (
            torch.zeros(NUM_ROBOT_POINTS, 4),
            torch.ones(NUM_OBSTACLE_POINTS, 4),
            2 * torch.ones(NUM_TARGET_POINTS, 4),
        ),
        dim=0,
    )
    xyz[:NUM_ROBOT_POINTS, :3] = robot_points.float()
    xyz[
        NUM_ROBOT_POINTS : NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS,
        :3,
    ] = torch.as_tensor(obstacle_points[:, :3]).float()
    xyz[
        NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS :,
        :3,
    ] = target_points.float()
    return xyz


def convert_primitive_problems_to_depth(problems: ProblemSet):
    """
    Converts the planning problems in place from primitive-based to point-cloud-based.
    This used PyBullet to create the scene and sample a depth image. That depth image is
    then turned into a point cloud with ray casting.

    :param problems ProblemSet: The list of problems to convert
    :raises NotImplementedError: Raises an error if the environment type is not supported
    """
    print("Converting primitive problems to depth")
    sim = Bullet()
    franka = sim.load_robot(FrankaRobot)
    # These are the camera views used for evaluations in Motion Policy Networks
    # Count the problems
    total_problems = 0
    for scene_sets in problems.values():
        for problem_set in scene_sets.values():
            total_problems += len(problem_set)
    with tqdm(total=total_problems) as pbar:
        for environment_type, scene_sets in problems.items():
            if "dresser" in environment_type:
                camera = SE3(
                    xyz=[0.08307640315968651, 1.986952324350807, 0.9996085854670145],
                    quaternion=[
                        -0.10162310189063647,
                        -0.06726290364234049,
                        0.5478233048853433,
                        0.8276702686337273,
                    ],
                ).inverse
            elif "cubby" in environment_type:
                camera = SE3(
                    xyz=[0.08307640315968651, 1.986952324350807, 0.9996085854670145],
                    quaternion=[
                        -0.10162310189063647,
                        -0.06726290364234049,
                        0.5478233048853433,
                        0.8276702686337273,
                    ],
                ).inverse
            elif "tabletop" in environment_type:
                camera = SE3(
                    xyz=[1.5031788593125708, -1.817341016921562, 1.278088299149147],
                    quaternion=[
                        0.8687241016192855,
                        0.4180885960330695,
                        0.11516106409944685,
                        0.23928704613569252,
                    ],
                ).inverse
            else:
                raise NotImplementedError(
                    f"Camera angle is not implemented for environment type: {environment_type}"
                )
            for problem_set in scene_sets.values():
                for p in problem_set:
                    franka.marionette(p.q0)
                    sim.load_primitives(p.obstacles)
                    p.obstacle_point_cloud = sim.get_pointcloud_from_camera(
                        camera,
                        remove_robot=franka,
                    )
                    sim.clear_all_obstacles()
                    pbar.update(1)

def render_traj_in_env(point_cloud, traj, obstacles = None):    #this is in pybullet
    sim = BulletController(hz=12, substeps=20, gui=True)
    eval = Evaluator()
    # Load the meshcat visualizer to visualize point cloud (Pybullet is bad at point clouds)
    viz = meshcat.Visualizer()

    # Load the FK module
    urdf = urchin.URDF.load(FrankaRobot.urdf)
    # Preload the robot meshes in meshcat at a neutral position
    for idx, (k, v) in enumerate(urdf.visual_trimesh_fk(np.zeros(8)).items()):
        viz[f"robot/{idx}"].set_object(
            meshcat.geometry.TriangularMeshGeometry(k.vertices, k.faces),
            meshcat.geometry.MeshLambertMaterial(color=0xEEDD22, wireframe=False),
        )
        viz[f"robot/{idx}"].set_transform(v)

    franka = sim.load_robot(FrankaRobot)
    gripper = sim.load_robot(FrankaGripper, collision_free=True)
    point_cloud_colors = np.zeros(
                    (3, NUM_OBSTACLE_POINTS + NUM_TARGET_POINTS)
                )
    point_cloud_colors[1, :NUM_OBSTACLE_POINTS] = 1
    point_cloud_colors[0, NUM_OBSTACLE_POINTS:] = 1
    viz["point_cloud"].set_object(
        # Don't visualize robot points
        meshcat.geometry.PointCloud(
            position=point_cloud[NUM_ROBOT_POINTS:, :3].numpy().T,
            color=point_cloud_colors,
            size=0.005,
        )
    )
    if obstacles is not None:
        sim.load_primitives(obstacles, visual_only=True)

    # gripper.marionette(problem.target)
    franka.marionette(traj[0])
    time.sleep(5.2)

    for q in traj:
        franka.control_position(q)
        sim.step()
        sim_config, _ = franka.get_joint_states()
        # Move meshes in meshcat to match PyBullet
        for idx, (k, v) in enumerate(
            urdf.visual_trimesh_fk(sim_config[:8]).items()
        ):
            viz[f"robot/{idx}"].set_transform(v)
        time.sleep(0.08)

    # Assuming the robot model is loaded into the 'franka' variable

    # Get the number of joints/links in the robot
    # num_joints = p.getNumJoints(franka)

    # # Iterate over each link
    # for link_index in range(num_joints):
    #     link_info = p.getJointInfo(franka, link_index)
    #     link_name = link_info[1].decode('UTF-8')  # Convert bytes to string
    #     print(f"Link {link_index}: {link_name}")
        
    for _ in range(20):
        sim.step()
        sim_config, _ = franka.get_joint_states()
        # Move meshes in meshcat to match PyBullet
        for idx, (k, v) in enumerate(
            urdf.visual_trimesh_fk(sim_config[:8]).items()
        ):
            viz[f"robot/{idx}"].set_transform(v)
        time.sleep(0.08)
    sim.clear_all_obstacles()

def create_obstacles(cuboids_info, cylinders_info):
    obstacles = []
    n_cuboids = cuboids_info[0].shape[0]
    n_cylinders = cylinders_info[0].shape[0]
    n_obstacles = n_cuboids + n_cylinders
    obstacles_tr = []

    #create cuboid objects 
    for i in range(n_cuboids):
        cub_obj = Cuboid(cuboids_info[0][i], cuboids_info[1][i], cuboids_info[2][i])
        has_zero = any(num == 0.0 for num in cuboids_info[1][i])
        if has_zero:
            continue
        obstacles.append(cub_obj)
        obstacles_tr.append((cuboids_info[0][i], cuboids_info[2][i]))   #(centre, quaternion)
    #create cylinder objects
    for i in range(n_cylinders):
        cyl_obj = Cylinder(cylinders_info[0][i], cylinders_info[1][i][0], cylinders_info[2][i][0], cylinders_info[3][i])
        has_zero = any(num == 0.0 for num in [cylinders_info[1][i], cylinders_info[2][i]])
        if has_zero:
            continue
        obstacles.append(cyl_obj)
        obstacles_tr.append((cylinders_info[0][i], cylinders_info[2][i]))   #(centre, quaternion)

    return obstacles, obstacles_tr

def generate_rectangle_corners(center_x, center_y, width, height):
    half_width = width / 2
    half_height = height / 2
    
    top_left = (center_x - half_width, center_y - half_height)
    top_right = (center_x + half_width, center_y - half_height)
    bottom_left = (center_x - half_width, center_y + half_height)
    bottom_right = (center_x + half_width, center_y + half_height)
    
    return top_left, top_right, bottom_left, bottom_right

def closest_point(points, target):
    distances = np.linalg.norm(points - target, axis=1)
    closest_index = np.argmin(distances)
    closest = points[closest_index]
    return closest

def create_obstacles_colliding_with_traj(cuboids_info, cylinders_info, traj, fk_sampler):
    obstacles = []
    n_cuboids = cuboids_info[0].shape[0]
    n_cylinders = cylinders_info[0].shape[0]
    n_obstacles = n_cuboids + n_cylinders
    n_conf = traj.shape[0]

    #AT present, we are identifying bases based on quaternions and they are observed to be at the beginning in the list of obstacles
    #create bases based on quaternions (which are cuboids)
    bases_mask = (cuboids_info[2][:, 0] == 1.0) & (cuboids_info[2][:, 1] == 0.0) & (cuboids_info[2][:, 2] == 0.0) & (cuboids_info[2][:, 3] == 0.0)
    #bases_mask is indices of all rows in cuboids
    n_bases = bases_mask.sum()
    bases_hts = []
    #create bases
    bases_corners = []
    bases_centres = []
    for i in range(n_bases):
        cub_obj = Cuboid(cuboids_info[0][i], cuboids_info[1][i], cuboids_info[2][i])   #(centres, dims, quats)
        has_zero = any(num == 0.0 for num in cuboids_info[1][i])   
        if has_zero:
            continue
        obstacles.append(cub_obj)
        bases_hts.append(cuboids_info[1][i][2])   #only one will be base of arm (with less ht), remaining bases should have same ht
        #get base rectangle corners
        base_corners = generate_rectangle_corners(cuboids_info[0][i][0], cuboids_info[0][i][1], cuboids_info[1][i][0], cuboids_info[1][i][1])
        bases_corners.append(base_corners)

        #get base centres as well
        bases_centres.append(cuboids_info[0][i])

    #select random conf in traj
    random_row_index = np.random.randint(0, traj.shape[0])       
    random_conf = traj[random_row_index]
    #get point randomly sampled on robot.
    robot_points = fk_sampler.sample(random_conf, NUM_ROBOT_POINTS)
    robot_points = robot_points.squeeze(0)
    random_point_index = np.random.randint(0, robot_points.shape[0])       
    random_point = robot_points[random_point_index]
    #find the base which is right down the sampled point and add obstacle (either cuboid or cylinder).
    closes_base_centre = closest_point(np.array(bases_centres), random_point)

    #create cuboid objects 
    for i in range(n_bases, n_cuboids):
        cub_obj = Cuboid(cuboids_info[0][i], cuboids_info[1][i], cuboids_info[2][i])
        has_zero = any(num == 0.0 for num in cuboids_info[1][i])  #dims cant be 0
        
        if has_zero:
            continue
        obstacles.append(cub_obj)

    #create cylinder objects
    for i in range(n_cylinders):
        cyl_obj = Cylinder(cylinders_info[0][i], cylinders_info[1][i][0], cylinders_info[2][i][0], cylinders_info[3][i])
        has_zero = any(num == 0.0 for num in [cylinders_info[1][i], cylinders_info[2][i]])
        if has_zero:
            continue
        obstacles.append(cyl_obj)

    return obstacles

def create_env_pc(cuboids_info, cylinders_info, q, cpu_fk_sampler):   
    #cuboids info: [centres, dims, quats], cylinders_info: [centres, radii, hts, quats]   
    obstacles, obstacles_tr = create_obstacles(cuboids_info, cylinders_info)

    #create pc from primitives
    # point_cloud = make_env_point_cloud_from_primitives_with_robot(
    #                         q, 
    #                         obstacles,
    #                         cpu_fk_sampler
    #                     )
    point_cloud = make_env_point_cloud_from_primitives(
                            obstacles,
                        )
    return point_cloud, obstacles, obstacles_tr

def create_colliding_env_pc(cuboids_info, cylinders_info, traj, cpu_fk_sampler):   
    #cuboids info: [centres, dims, quats], cylinders_info: [centres, radii, hts, quats]
    obstacles = create_obstacles_colliding_with_traj(cuboids_info, cylinders_info, traj, cpu_fk_sampler)

    #create pc from primitives
    # point_cloud = make_env_point_cloud_from_primitives_with_robot(
    #                         q, 
    #                         obstacles,
    #                         cpu_fk_sampler
    #                     )
    point_cloud = make_env_point_cloud_from_primitives(
                            obstacles,
                        )
    return point_cloud


# Check collisions between the robot and optionally the object in hand with the scene
# for a batch of rollouts
def _check_collisions_and_get_collision_gradients(rollouts, robot, self_coll_nn, scene_coll_nn, scene_pc, obstacles_tr, check_obj=False):
    collision_steps = 7
    #rollouts shape: [b, 50, 7]
    num_path = rollouts.shape[0]
    horizon = rollouts.shape[1]
    alpha = (     #(1,1,7,1)
        torch.linspace(0, 1, collision_steps)
        .reshape([1, 1, -1, 1])
        .to(device)
    )
    waypoints = (
        alpha * rollouts[:, 1:, None]
        + (1.0 - alpha) * rollouts[:, :-1, None]
    ).reshape(-1, 7) #robot.dof)

    self_collision_checker = (
            NNSelfCollisionChecker(     #this is simple MLP network from (num_joints i.e 8 to 1))
                self_coll_nn, device
            )
    )
    scene_collision_checker = (
            NNSceneCollisionChecker(    #(robot link features are also extracted in this fn  -- done only once)
                scene_coll_nn,
                robot,
                device,
                use_knn=False,
            )
    )
    # if isinstance(self_collision_checker, FCLSelfCollisionChecker):
    #     coll_mask = np.zeros(len(waypoints), dtype=np.bool)
    #     for i, q in enumerate(waypoints):
    #         coll_mask[i] = self_collision_checker(q)
    # else:

    #compute scene_pc features
    model = scene_collision_checker.model   #SceneCollisionNet
    # model.bounds = [b.to(device) for b in model.bounds]
    model.vox_size = model.vox_size.to(device)
    model.num_voxels = model.num_voxels.to(device)

    # Clip points to model bounds and feed in for features
    in_bounds = (
        scene_pc[..., :3] > model.bounds[0] + 1e-5
    ).all(dim=-1)
    in_bounds &= (
        scene_pc[..., :3] < model.bounds[1] - 1e-5
    ).all(dim=-1)

    scene_features = scene_collision_checker.model.get_scene_features(
        scene_pc[:, :3][in_bounds].unsqueeze(0)
    ).squeeze(0)

    zeros_column = torch.zeros(waypoints.shape[0], 1).to(device)
    waypoints = torch.cat((waypoints, zeros_column), dim=1)
    waypoints = waypoints.float()
    coll_mask = self_collision_checker(waypoints)

    # coll_mask |= self.scene_collision_checker(waypoints, threshold=0.45)
    tmp_coll_mask = scene_collision_checker(waypoints, threshold=0.45)
    coll_mask |= tmp_coll_mask[0]

    # if check_obj:
    #     obj_trs = torch.cat(
    #         (
    #             self.robot.ee_pose[:, :3, 3]
    #             - torch.from_numpy(self.ee_offset).float().to(self.device),
    #             torch.ones(len(self.robot.ee_pose), 1, device=self.device),
    #         ),
    #         dim=1,
    #     )
    #     model_obj_trs = (
    #         scene_collision_checker.robot_to_model @ obj_trs.T
    #     )
    #     obj_coll = self.scene_collision_checker.check_object_collisions(
    #         model_obj_trs[:3].T, threshold=0.45
    #     )
    #     coll_mask |= obj_coll.reshape(coll_mask.shape)
    return coll_mask.reshape(
        num_path, horizon - 1, collision_steps
    )
    
def read_mpinet_dataset(args):
    mdl = MotionPolicyNetwork.load_from_checkpoint(args.mdl_path).cuda()
    mdl.eval()
    eval = Evaluator()
    cpu_fk_sampler = FrankaSampler("cpu", use_cache=True)
    gpu_fk_sampler = FrankaSampler("cuda:0", use_cache=True)
    robot = Robot(    #create Robot arm instance  -- from scene collision net
        "data/panda/panda.urdf",
        "right_gripper",
    )

    with h5py.File(args.dataset, "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        print("Keys: %s" % f.keys())  #['cuboid_centers', 'cuboid_dims', 'cuboid_quaternions', 'cylinder_centers', 'cylinder_heights', 'cylinder_quaternions', 'cylinder_radii', 'global_solutions', 'hybrid_solutions']

        # Every env has  a global and hybrid expert solution
        print(f['global_solutions'].shape)  #(3.27m, 50, 7)  
        print(f['hybrid_solutions'].shape)  #(3.27m, 50, 7)
        print(f['cuboid_centers'].shape)    #(3.27m, 40, 3)
        # path_to_save = "/scratch/jayaram.reddy/mpinet_pc_data"
        # # Check if the directory exists
        # if not os.path.exists(path_to_save):
        #     os.makedirs(path_to_save)
        #     print("Directory created successfully.")
        # else:
        #     print("Directory already exists.")

        n_scenes = f['global_solutions'].shape[0]
        n_collisions = 0
        for scene in range(n_scenes):
            scene_no = np.random.randint(low=0, high=n_scenes)
            scene_no = 1210809
            # print()
            #create env point cloud
            cuboids_info = [f['cuboid_centers'][scene_no], f['cuboid_dims'][scene_no], f['cuboid_quaternions'][scene_no]]
            cylinders_info = [f['cylinder_centers'][scene_no], f['cylinder_radii'][scene_no], f['cylinder_heights'][scene_no], f['cylinder_quaternions'][scene_no]]
            # time_strt = time.time()

            joint_conf_waypoint = 35
            q = f['global_solutions'][scene_no][joint_conf_waypoint]  #35th conf in the traj

            # TODO: verify if world frame is at base of robot
            pc, obstacles, obstacles_tr = create_env_pc(cuboids_info, cylinders_info, torch.as_tensor(q), cpu_fk_sampler)
            # obstacles_tr : list , each el: (obs_centre, obs_quat)

            traj_no = scene_no # np.random.randint(low=0, high=n_scenes)
            # traj = f['global_solutions'][traj_no] # [scene_no]
            traj = f['hybrid_solutions'][traj_no]
            #generate traj passing thru a waypoint from initial to final conf
            

            # in_collision = eval.in_collision(traj, obstacles)
            # if in_collision:
            #     n_collisions += 1
                # render_traj_in_env(pc[:, :3], f['global_solutions'][scene_no], obstacles)
            print(f"Scene Number: {scene_no}\tTraj number: {scene_no}\tNumber of collisions: {n_collisions}")
            # render_traj_in_env(pc[:, :3], traj, obstacles) # (pc[:, :3], f['hybrid_solutions'][scene_no], obstacles)
            # print(f"\rScene no: {scene_no}\tTraj no: {traj_no}\tCOllision Status: {in_collision}\tNUmber of collisions: {n_collisions}\tPercent of collisions: {(100*n_collisions)/(scene_no+1)}%", end="")
            # colliding_pc = create_colliding_env_pc(cuboids_info, cylinders_info, torch.as_tensor(traj), cpu_fk_sampler)
            # time_end = time.time()
            # print('time taken to create rnv pc: {}'.format(time_end - time_strt))
            
            # Create an Open3D point cloud object
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pc[:, :3])

            # o3d.io.write_point_cloud("env_point_cloud_" + str(scene_no + 1) + ".ply", pcd)

            # # Visualize the point cloud
            # o3d.visualization.draw_geometries([pcd])
            print("scene_" + str(scene_no + 1) + " completed")

            # TODO scenes for (diffuser+scenecollisionnet): 1147692,  1210809(colliding)
            # diffusion without guidance
            # joint_7_samples = np.linspace(-2.8973, 2.8973, 100)
            # for joint_ang in joint_7_samples:
            #     ik_solutions = FrankaRobot.ik(problem.target, joint_ang, 'right_gripper')
            #     if len(ik_solutions)!=0:
            #         break
            # trajectory = infer(denoiser, problem.q0, ik_solutions[0])

            # load diffusion model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            traj_len=50
            T = 255
            model_name = "./diffuser_ckpts_7dof_mpinets/7dof/" + "TemporalUNetModel" + str(T) + "_N" + str(traj_len)
            if not os.path.exists(model_name):
                print("Model does not exist for these parameters. Train a model first.")
                _ = input("Press anything to exit")
                exit()
            denoiser = TemporalUNet(model_name = model_name, input_dim = 7, time_dim = 32, dims=(32, 64, 128, 256, 512, 512))
            _ = denoiser.to(device)
            gt_traj = f['global_solutions'][scene_no]
            trajectory = infer(denoiser, gt_traj[0], gt_traj[-1])
            
            print(f"Scene Number: {scene_no}\tTraj number: {scene_no}\tNumber of collisions: {n_collisions}")
            # render_traj_in_env(pc[:, :3], trajectory, obstacles) # (pc[:, :3], f['hybrid_solutions'][scene_no], obstacles)    
            # render_traj_in_env(pc[:, :3], traj, obstacles)        
            # render_traj_in_env(pc[:, :3], f['hybrid_solutions'][scene_no], obstacles)

            eff_pose = FrankaRobot.fk( np.squeeze(gt_traj[-1]), eff_frame="right_gripper" )

            mpinet_trajectory = rollout_until_success(
                    mdl,
                    gt_traj[0], #problem.q0,   (7, )
                    eff_pose,  #problem.target,
                    pc.unsqueeze(0).cuda(),
                    gpu_fk_sampler,
                )
            render_traj_in_env(pc[:, :3], mpinet_trajectory, obstacles)
            #final task: transfer scenecollisionnet gradient computation to above infer path
            #Note: We have transformation of each obstacle wrt world frame (figure it out in scenecollision net)
                        #now check scene collision for remaining initial grasp poses, after filtering o remain from n
            trajectory = trajectory[np.newaxis, :]
            # Convert NumPy array to PyTorch tensor
            trajectory = torch.from_numpy(trajectory).to(device)
            pc = pc.to(device)

            # collisions = _check_collisions_and_get_collision_gradients(trajectory, robot, args.self_coll_nn, args.scene_coll_nn, pc, obstacles_tr, check_obj=False)            

            # _jacobian = env._gym.acquire_jacobian_tensor(env._sim, "franka")
            # print(_jacobian)

        #render first traj in env
        # obstacles = create_obstacles(cuboids_info, cylinders_info)


    # with open(args.problems, "rb") as f:
    #     problems = pickle.load(f)
    # env_type = args.environment_type.replace("-", "_")
    # problem_type = args.problem_type.replace("-", "_")
    # if env_type != "all":
    #     problems = {env_type: problems[env_type]}
    # if problem_type != "all":
    #     for k in problems.keys():
    #         problems[k] = {problem_type: problems[k][problem_type]}

    #get point clouds
    # cnt = 0
    # for scene_type, scene_sets in problems.items():
    #     for problem_type, problem_set in scene_sets.items():
    #         for problem in tqdm(problem_set, leave=False):
    #             eval.create_new_group(f"{scene_type}, {problem_type}")
    #             if problem.obstacle_point_cloud is None:
    #                 point_cloud = make_point_cloud_from_primitives(
    #                     torch.as_tensor(problem.q0).unsqueeze(0),
    #                     problem.target,
    #                     problem.obstacles,
    #                     cpu_fk_sampler,
    #                 )
    #             else:
    #                 point_cloud = make_point_cloud_from_problem(
    #                     torch.as_tensor(problem.q0).unsqueeze(0),
    #                     problem.target,
    #                     problem.obstacle_point_cloud,
    #                     cpu_fk_sampler,
    #                 )

    #             cnt = cnt + 1

    # print('cnt : {}'.format(cnt))    #only 600, why?

    # Create an Open3D point cloud object
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc[:, :3])

    # o3d.io.write_point_cloud("env1_point_cloud.ply", pcd)
    # # Visualize the point cloud
    # o3d.visualization.draw_geometries([pcd])
    # print('...............................')
        # get first object name/key; may or may NOT be a group
        # a_group_key = list(f.keys())[0]

        # # get the object type for a_group_key: usually group or dataset
        # print(type(f[a_group_key])) 

        # # If a_group_key is a group name, 
        # # this gets the object names in the group and returns as a list
        # data = list(f[a_group_key])

        # # If a_group_key is a dataset name, 
        # # this gets the dataset values and returns as a list
        # data = list(f[a_group_key])
        # # preferred methods to get dataset values:
        # ds_obj = f[a_group_key]      # returns as a h5py dataset object
        # ds_arr = f[a_group_key][()]  # returns as a numpy array
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mdl_path",
        type=str,
        default="/home/jayaram/research/research_tracks/table_top_rearragement/global_classifier_guidance_for_7DOF_manipulator/mpinets_hybrid_expert.ckpt",
        help="checkpoint file from training MotionPolicyNetwork",
    )

    parser.add_argument(
        "-mpinets_data_path",
        "--dataset",
        type=str,
        default="/home/jayaram/research/research_tracks/table_top_rearragement/global_classifier_guidance_for_7DOF_manipulator/datasets/mpinets_hybrid_training_data/train/train.hdf5",
        help="Path to mpinets path",
    )

    parser.add_argument(
        "--environment_type",
        choices=["tabletop", "cubby", "merged-cubby", "dresser", "all"],
        default = "tabletop",
        help="The environment class",
    )

    parser.add_argument(
        "--use-depth",
        action="store_true",
        help=(
            "If set, uses a partial view pointcloud rendered in Pybullet. If not set,"
            " uses pointclouds sampled from every side of the primitives in the scene"
        ),
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help=(
            "If set, will not show visuals and will only display metrics. This will be"
            " much faster because the trajectories are not displayed"
        ),
    )

    parser.add_argument(
        "--problems",
        type=str,
        default="/home/jayaram/research/research_tracks/table_top_rearragement/global_classifier_guidance_for_7DOF_manipulator/datasets/both_solvable_problems.pkl",
        help="A pickle file of sample problems that follow the PlanningProblem format",
    )

    parser.add_argument(
        "--problem_type",
        choices=["task-oriented", "neutral-start", "neutral-goal", "all"],
        default = "all",
        help="The type of planning problem",
    )

    parser.add_argument(
        "--scene_coll_nn",
        default = "weights/scene_coll_nn",
        help="checkpoint for scene collision",
        type=str
    )
    parser.add_argument(
        "--self_coll_nn",
        default = "weights/self_coll_nn",
        help="checkpoint for self collision",
        type=str
    )
    # from time import sleep
    # print("sleeping")

    # # sleep(15)
    # print("sleeping done!")

    args = parser.parse_args()
    # with open(args.problems, "rb") as f:
    #     problems = pickle.load(f)
    # env_type = args.environment_type.replace("-", "_")
    # problem_type = args.problem_type.replace("-", "_")
    # if env_type != "all":
    #     problems = {env_type: problems[env_type]}
    # if problem_type != "all":
    #     for k in problems.keys():
    #         problems[k] = {problem_type: problems[k][problem_type]}
    # if args.use_depth:
    #     convert_primitive_problems_to_depth(problems)
    # if args.skip_visuals:
    #     pass
    #     # calculate_metrics(args.mdl_path, problems)
    # else:
    #     visualize_results(args.mdl_path, problems)

    read_mpinet_dataset(args)    # env_type = args.environment_type.replace("-", "_")
    # problem_type = args.problem_type.replace("-", "_")
    # if env_type != "all":
    #     problems = {env_type: problems[env_type]}
    # if problem_type != "all":
    #     for k in problems.keys():
    #         problems[k] = {problem_type: problems[k][problem_type]}
    # if args.use_depth:
    #     convert_primitive_problems_to_depth(problems)
    # if args.skip_visuals:
    #     pass
    #     # calculate_metrics(args.mdl_path, problems)
    # else:
    #     visualize_results(args.mdl_path, problems)

    
