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


def move(position, orientation=[0.0, 1.0, 0.0, 0.0], relative_pose=None, joint_angles=[], in_tip_frame=False, tip_name='right_hand', linear_speed=0.6, linear_accel=0.6, rotational_speed=1.57, rotational_accel=1.57, timeout=None  ):
    """
    Move the robot arm to the specified configuration.
    """
    try:
        rospy.init_node('go_to_cartesian_pose_py')
        limb = Limb()

        traj_options = TrajectoryOptions()
        traj_options.interpolation_type = TrajectoryOptions.CARTESIAN
        traj = MotionTrajectory(trajectory_options = traj_options, limb = limb)

        wpt_opts = MotionWaypointOptions(max_linear_speed=linear_speed,
                                         max_linear_accel=linear_accel,
                                         max_rotational_speed=rotational_speed,
                                         max_rotational_accel=rotational_accel,
                                         max_joint_speed_ratio=1.0)
        waypoint = MotionWaypoint(options = wpt_opts.to_msg(), limb = limb)

        joint_names = limb.joint_names()
	print(position)
        if joint_angles and len(joint_angles) != len(joint_names):
            rospy.logerr('len(joint_angles) does not match len(joint_names!)')
            return None

        if (position is None and orientation is None
            and relative_pose is None):
            if joint_angles:
                # does Forward Kinematics
                waypoint.set_joint_angles(joint_angles, tip_name, joint_names)
            else:
                rospy.loginfo("No Cartesian pose or joint angles given. Using default")
                waypoint.set_joint_angles(joint_angles=None, active_endpoint=tip_name)
        else:
            endpoint_state = limb.tip_state(tip_name)
            if endpoint_state is None:
                rospy.logerr('Endpoint state not found with tip name %s', tip_name)
                return None
            pose = endpoint_state.pose

            if relative_pose is not None:
                if len(relative_pose) != 6:
                    rospy.logerr('Relative pose needs to have 6 elements (x,y,z,roll,pitch,yaw)')
                    return None
                # create kdl frame from relative pose
                rot = PyKDL.Rotation.RPY(relative_pose[3],
                                         relative_pose[4],
                                         relative_pose[5])
                trans = PyKDL.Vector(relative_pose[0],
                                     relative_pose[1],
                                     relative_pose[2])
                f2 = PyKDL.Frame(rot, trans)
                # and convert the result back to a pose message
                if in_tip_frame:
                  # end effector frame
                  pose = posemath.toMsg(posemath.fromMsg(pose) * f2)
                else:
                  # base frame
                  pose = posemath.toMsg(f2 * posemath.fromMsg(pose))
            else:
                if position is not None and len(position) == 3:
                    pose.position.x = position[0]
                    pose.position.y = position[1]
                    pose.position.z = position[2]
                if orientation is not None and len(orientation) == 4:
                    pose.orientation.x = orientation[0]
                    pose.orientation.y = orientation[1]
                    pose.orientation.z = orientation[2]
                    pose.orientation.w = orientation[3]
            poseStamped = PoseStamped()
            poseStamped.pose = pose

            if not joint_angles:
                # using current joint angles for nullspace bais if not provided
                joint_angles = limb.joint_ordered_angles()
                waypoint.set_cartesian_pose(poseStamped, tip_name, joint_angles)
            else:
                waypoint.set_cartesian_pose(poseStamped, tip_name, joint_angles)



        rospy.loginfo('Sending waypoint: \n%s', waypoint.to_string())

        traj.append_waypoint(waypoint.to_msg())

        result = traj.send_trajectory(timeout=timeout)
        if result is None:
            rospy.logerr('Trajectory FAILED to send')
            return

        if result.result:
            rospy.loginfo('Motion controller successfully finished the trajectory!')
        else:
            rospy.logerr('Motion controller failed to complete the trajectory with error %s',
                         result.errorId)
    except rospy.ROSInterruptException:
        rospy.logerr('Keyboard interrupt detected from the user. Exiting before trajectory completion.')
	
