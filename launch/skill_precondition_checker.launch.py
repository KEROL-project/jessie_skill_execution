#!/usr/bin/env python3
# Author: Alex Mitrevski

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    precondition_config_file_path = os.path.join(
        get_package_share_directory('jessie_skill_execution'),
        'config',
        'skill_preconditions.yaml'
    ) 
    vlm_name = 'llava'

    return LaunchDescription([
        Node(
            package='jessie_skill_execution',
            executable='skill_precondition_checker',
            name='skill_precondition_checker',
            parameters=[{'precondition_file_path': precondition_config_file_path,
                         'vlm_name': vlm_name}]
        )
    ])
