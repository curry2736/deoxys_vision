import time
import numpy as np
import imageio
from scipy.spatial.transform import Rotation as R
from robosuite.utils.input_utils import *
import cpgen_envs

from robosuite.utils.transform_utils import quat2mat, mat2quat
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix

def convert_opencv_to_opengl(pose_matrix: np.ndarray) -> np.ndarray:
    """Converts a pose matrix from OpenCV to OpenGL coordinate convention."""
    pose_matrix[:3, 1] *= -1
    pose_matrix[:3, 2] *= -1
    return pose_matrix


def setup_env(env_name: str = "SquareRealBetterReal3pvCameraTableAlign", robot: str = "Panda_PandaUmiGripper"):
    """Initializes and resets the environment."""

    env = suite.make(
        env_name=env_name,
        robots=robot,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
        base_types=None,
        camera_heights=720,
        camera_widths=1280,
        renderer="mujoco",
    )
    env.reset()
    action = np.zeros(7)
    action[-2:] = -1
    for _ in range(10):
        env.step(action)
    return env


def update_robot_joints(env, joint_pos: np.ndarray):
    """Updates the robot joint positions and camera pose."""
    robot = env.robots[0]
    robot.set_robot_joint_positions(joint_pos)
    env.sim.forward()
    return


    # env.robots[0].robot_model.set_base_xpos([-0.63, 0, 0.82])

    env.sim.model.body("robot0_base").pos = np.array([-0.63, 0, 0.82])
    # Get robot base pose
    robot_pos = env.sim.model.body("robot0_base").pos
    robot_quat_wxyz = env.sim.model.body("robot0_base").quat
    robot_quat_xyzw = robot_quat_wxyz[[1, 2, 3, 0]]
    robot_mat = quat2mat(robot_quat_xyzw)

    robot_pose = np.eye(4)
    robot_pose[:3, :3] = robot_mat
    robot_pose[:3, 3] = robot_pos

    camera_pose_robot = np.eye(4)
    new_pos = np.array([1.15, -0.042, 0.55])
    camera_pose_robot[:3, 3] = new_pos
    new_euler_angles = np.array([-135, 0, 90])
    rotation_matrix = R.from_euler("xyz", new_euler_angles, degrees=True).as_matrix()
    camera_pose_robot[:3, :3] = rotation_matrix

    camera_pose_robot = convert_opencv_to_opengl(camera_pose_robot)

    # Compute global camera pose
    camera_pose_global = robot_pose @ camera_pose_robot
    camera_pos_global = camera_pose_global[:3, 3]
    camera_quat_xyzw_global = mat2quat(camera_pose_global[:3, :3])
    camera_quat_wxyz_global = camera_quat_xyzw_global[[3, 0, 1, 2]]

    # Update camera parameters
    camera_name = "agentview"
    camera_id = env.sim.model.camera_name2id(camera_name)
    env.sim.model.cam_pos[camera_id] = camera_pos_global
    env.sim.model.cam_quat[camera_id] = camera_quat_wxyz_global
    # import ipdb;ipdb.set_trace()
    env.sim.forward()
    print(f"Updated camera pose: {camera_pos_global}, {camera_quat_wxyz_global}")
    return camera_id


def render_obs(env, camera_name: str = "agentview"):
    """Renders the observation from the specified camera and saves the image."""
    print(f"rendering from {camera_name=}")
    print(f"intrinsic_matrix: {get_camera_intrinsic_matrix(env.sim, camera_name, 720, 1280)}")
    obs = env._get_observations(force_update=True)
    if f"{camera_name}_image" in obs:
        img = obs[f"{camera_name}_image"][::-1]
        return img
    

if __name__ == "__main__":
    env = setup_env()
    joint_positions = np.array([.087, -.1625, -.015, -2.472, -.013, 2.299, 0.849])
    reset_joint_positions = [
        0.09162008114028396,
        -0.19826458111314524,
        -0.01990020486871322,
        -2.4732269941140346,
        -0.01307073642274261,
        2.30396583422025,
        0.8480939705504309,
    ]
    update_robot_joints(env, joint_positions)
    res = render_obs(env)
