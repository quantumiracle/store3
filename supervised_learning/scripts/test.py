#! /usr/bin/env python

# Copyright (c) 2016-2018, Rethink Robotics Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rospy
import argparse
import numpy as np
from intera_motion_interface import (
    MotionTrajectory,
    MotionWaypoint,
    MotionWaypointOptions
)
from intera_motion_msgs.msg import TrajectoryOptions
from geometry_msgs.msg import PoseStamped
import PyKDL
from tf_conversions import posemath
from intera_interface import Limb
from move import Move
from real_follow import load_policy, initialize_camera, close_camera, get_state, get_action
from deform_visualize import plot_list_new_real_arrow
import time

def main():
    """
    Move the robot arm to the specified configuration.
    Call using:
    $ rosrun intera_examples go_to_cartesian_pose.py  [arguments: see below]

    -p 0.4 -0.3 0.18 -o 0.0 1.0 0.0 0.0 -t right_hand
    --> Go to position: x=0.4, y=-0.3, z=0.18 meters
    --> with quaternion orientation (0, 1, 0, 0) and tip name right_hand
    --> The current position or orientation will be used if only one is provided.

    -q 0.0 -0.9 0.0 1.8 0.0 -0.9 0.0
    --> Go to joint angles: 0.0 -0.9 0.0 1.8 0.0 -0.9 0.0 using default settings
    --> If a Cartesian pose is not provided, Forward kinematics will be used
    --> If a Cartesian pose is provided, the joint angles will be used to bias the nullspace

    -R 0.01 0.02 0.03 0.1 0.2 0.3 -T
    -> Jog arm with Relative Pose (in tip frame)
    -> x=0.01, y=0.02, z=0.03 meters, roll=0.1, pitch=0.2, yaw=0.3 radians
    -> The fixed position and orientation paramters will be ignored if provided

    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__)
    parser.add_argument(
        "-p", "--position", type=float,
        nargs='+',
        help="Desired end position: X, Y, Z")
    parser.add_argument(
        "-o", "--orientation", type=float,
        nargs='+',
        help="Orientation as a quaternion (x, y, z, w)")
    parser.add_argument(
        "-R", "--relative_pose", type=float,
        nargs='+',
        help="Jog pose by a relative amount in the base frame: X, Y, Z, roll, pitch, yaw")
    parser.add_argument(
        "-T", "--in_tip_frame", action='store_true',
        help="For relative jogs, job in tip frame (default is base frame)")
    parser.add_argument(
        "-q", "--joint_angles", type=float,
        nargs='+', default=[],
        help="A list of joint angles, one for each of the 7 joints, J0...J6")
    parser.add_argument(
        "-t",  "--tip_name", default='right_hand',
        help="The tip name used by the Cartesian pose")
    parser.add_argument(
        "--linear_speed", type=float, default=0.6,
        help="The max linear speed of the endpoint (m/s)")
    parser.add_argument(
        "--linear_accel", type=float, default=0.6,
        help="The max linear acceleration of the endpoint (m/s/s)")
    parser.add_argument(
        "--rotational_speed", type=float, default=1.57,
        help="The max rotational speed of the endpoint (rad/s)")
    parser.add_argument(
        "--rotational_accel", type=float, default=1.57,
        help="The max rotational acceleration of the endpoint (rad/s/s)")
    parser.add_argument(
        "--timeout", type=float, default=None,
        help="Max time in seconds to complete motion goal before returning. None is interpreted as an infinite timeout.")
    args = parser.parse_args(rospy.myargv()[1:])
    move_controller = Move()
    ini_pos=initialize_position(move_controller)
    
    # move with policy
    camera=initialize_camera()
    print('Finished Initializing Camera!')
    policy=load_policy()
    print('Finished Loading Policy!')
    episode_length=40
    action_range=0.05
    #command_rate = rospy.Rate(5)
    control_rate = rospy.Rate(100)
    action_set=[]
    curr_pos = np.array(ini_pos)
    tap_height=0.02
    for _ in range(5):
        get_state(camera, 0)
    s, xy=get_state(camera, 0)
    for i in range(1, episode_length):
        a=get_action(policy, s)
        action_set.append(a)
        # if a[0]>a[1] and a[0]>1:
        #     a=np.array([1,0])
        # elif a[0]<a[1] and a[0]<0:
        #     a=np.array([-1,0])
        # elif a[0]<a[1] and a[1]>0:
        #     a=np.array([0,1])
        # elif a[0]>a[1] and a[1]<0:
        #     a=np.array([0,-1])
        # a=a*0.2
        a=a*0.05
        print(i, a)
        plot_list_new_real_arrow(xy, idx=i, action=1000*a)
        a=np.clip(a, -action_range, action_range)
        target_pos=curr_pos+np.concatenate((a, [0]))
        # lift up and move in the air
        move_controller.move(curr_pos+np.array([0,0,tap_height]))
        move_controller.move(target_pos+np.array([0,0,tap_height]))
        move_controller.move(target_pos)
        time.sleep(0.3)
        s, xy=get_state(camera, i)  # need to make sure the image is captured when tapping
        curr_pos=target_pos
        #command_rate.sleep()
    np.save('actions', action_set)
    close_camera(camera)

def initialize_position(move_controller):
    tap_height=0.02
    ini_position=np.array([0.6, 0.02, 0.0055])    # [0.6, 0.02, -0.0005] for small box
    move_controller.move(ini_position+np.array([0,0,tap_height]))    
    move_controller.move(ini_position)
    return ini_position





if __name__ == '__main__':
	
    main()
