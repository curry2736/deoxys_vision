"""This is a script to record joints by moving robot around, and save it to a list"""
import os
import time
import numpy as np
import simplejson as json

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.input_utils import input2action

config_folder = os.path.join(os.path.expanduser("~/"), ".deoxys_vision/calibration_configuration")
os.makedirs(os.path.join(os.path.expanduser("~/"), config_folder), exist_ok=True)

# TODO: REMOVE
import argparse
import json
import os
import struct
import time

import cv2
import init_path
import numpy as np
import redis
from easydict import EasyDict


from deoxys_vision.networking.camera_redis_interface import CameraRedisPubInterface
from deoxys_vision.camera.rs_interface import RSInterface
from deoxys_vision.utils.img_utils import preprocess_color, preprocess_depth, save_depth
from deoxys_vision.utils.camera_utils import assert_camera_ref_convention, get_camera_info

import sys
sys.path.append('/home/yifengz/robot_workspace')

from PIL import Image
def setup_camera_interface(
    camera_ref="rs_1", 
    host="172.16.0.1", 
    port=6379, 
    use_rgb=True, 
    use_depth=True, 
    use_rec=False, 
    rgb_convention="rgb"
):
    """
    Set up camera interface based on provided arguments.

    Args:
        camera_ref (str): Camera reference.
        host (str): Host for Redis connection.
        port (int): Port for Redis connection.
        use_rgb (bool): Whether to use RGB images.
        use_depth (bool): Whether to use depth images.
        use_rec (bool): Whether to rectify images.
        rgb_convention (str): RGB convention to use.

    Returns:
        camera_interface: Instantiated camera interface object.
        camera2redis_pub_interface: Instantiated CameraRedisPubInterface object.
        node_config: EasyDict containing node configuration options.
    """
    assert_camera_ref_convention(camera_ref)
    camera_info = get_camera_info(camera_ref)

    # Print camera information
    print(f"This node runs with the camera {camera_info.camera_type} with id {camera_info.camera_id}")

    # Create camera configuration
    camera_config = EasyDict(
        camera_type=camera_info.camera_type,
        camera_id=camera_info.camera_id,
        use_rgb=use_rgb,
        use_depth=use_depth,
        use_rec=use_rec,
        rgb_convention=rgb_convention,
    )

    # Print data publication information
    print("The node will publish the following data:")
    if use_rgb:
        print("- Color image")
    if use_depth:
        print("- Depth image")
    if use_rec:
        print("Note that Images are rectified with undistortion")

    # Create node configuration
    node_config = EasyDict(use_color=True, use_depth=True)
    if not use_rgb:
        node_config.use_color = False

    if not use_depth:
        node_config.use_depth = False

    if camera_info.camera_type == "rs":
        import pyrealsense2 as rs

        color_cfg = EasyDict(
            enabled=node_config.use_color, img_w=1280, img_h=720, img_format=rs.format.bgr8, fps=30
        )

        depth_cfg = EasyDict(
            enabled=node_config.use_depth, img_w=1280, img_h=720, img_format=rs.format.z16, fps=30
        )
        pc_cfg = EasyDict(enabled=False)
        camera_interface = RSInterface(
            device_id=camera_info.camera_id, color_cfg=color_cfg, depth_cfg=depth_cfg, pc_cfg=pc_cfg
        )

    # Create CameraRedisPubInterface
    camera2redis_pub_interface = CameraRedisPubInterface(
        camera_info=camera_info,
        redis_host=host, redis_port=port, 
    )

    return camera_interface, camera2redis_pub_interface, node_config, camera_config

def main(overlay_image):
    camera_interface, camera2redis_pub_interface, node_config, camera_config = setup_camera_interface()
    camera_interface.start()

    overlay_image = np.array(Image.open(overlay_image))
    pixel_ranges_to_show = [(600, 900), (0, 150)] #(x_low, x_high), (y_low, y_high)
    # pixel_ranges_to_show = [(0, 1280), (0, 720)] #(x_low, x_high), (y_low, y_high)
    scale = 4  # or whatever integer factor you like
    overlay_image = overlay_image[pixel_ranges_to_show[1][0]:pixel_ranges_to_show[1][1], pixel_ranges_to_show[0][0]:pixel_ranges_to_show[0][1], :]

    # print(config_root)
    robot_interface = FrankaInterface(config_root + "/charmander.yml", use_visualizer=False)
    controller_cfg = YamlConfig(config_root + "/compliant-joint-impedance-controller.yml").as_easydict()
    controller_type = "JOINT_IMPEDANCE"

    # # Make it low impedance so that we can easily move the arm around
    # controller_cfg["Kp"]["translation"] = 50
    # controller_cfg["Kp"]["rotation"] = 50

    joints = []
    joint = False
    time.sleep(1.)    
    while True:
        # spacemouse_action, grasp = input2action(
        #     device=device,
        #     controller_type="OSC_POSE",
        # )

        # if spacemouse_action is None:
        #     break
        
        # if len(robot_interface._state_buffer) > 0:
        #     print(spacemouse_action[-1])
        #     if spacemouse_action[-1] > 0 and not recorded_joint:
        #         joints.append(robot_interface._state_buffer[-1].q)
        #         print(len(robot_interface._state_buffer[-1].q))
        #         recorded_joint = True
        #         for _ in range(5):
        #             spacemouse_action, grasp = input2action(
        #                 device=device,
        #                 controller_type=controller_type,
        #             )
        #     elif spacemouse_action[-1] < 0:
        #         recorded_joint = False
        # else:
        #     print('here')
        #     continue
            
        capture = camera_interface.get_last_obs()
        if capture is not None:
            color_img = preprocess_color(capture["color"], flip_channel=camera_config.rgb_convention == "rgb")
            color_img = color_img[pixel_ranges_to_show[1][0]:pixel_ranges_to_show[1][1], pixel_ranges_to_show[0][0]:pixel_ranges_to_show[0][1], :]

            # Blend images 50-50
            blended_array = (overlay_image * 0.5 + color_img * 0.5).astype(np.uint8)

            h, w = blended_array.shape[:2]
            big = cv2.resize(
                blended_array,
                (w * scale, h * scale),
                interpolation=cv2.INTER_NEAREST
            )

            cv2.imshow("test", big[..., ::-1])
            cv2.waitKey(10)
        
        # action = list(robot_interface._state_buffer[-1].q) + [-1]
        # robot_interface.control(
        #     controller_type=controller_type, action=action, controller_cfg=controller_cfg
        # )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the main process, optionally with an overlay image."
    )
    parser.add_argument(
        "overlay_image",
        type=str,
        help="Path to an image file to overlay",
    )
    args = parser.parse_args()
    main(overlay_image=args.overlay_image)
