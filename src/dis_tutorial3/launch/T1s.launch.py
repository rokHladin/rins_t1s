from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    pkg_dis_tutorial3 = get_package_share_directory('dis_tutorial3')

    # Arguments
    arguments = [
        DeclareLaunchArgument('rviz', default_value='true', description='Launch RViz2'),
        DeclareLaunchArgument('world', default_value='demo3', description='Simulation world'),
        DeclareLaunchArgument('model', default_value='standard', description='Turtlebot4 model'),
        DeclareLaunchArgument('map', default_value=PathJoinSubstitution([pkg_dis_tutorial3, 'maps', 'map.yaml']), description='Map YAML')
    ]

    # Step 0: Base simulation and nav2
    sim_base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_dis_tutorial3, 'launch', 'sim_turtlebot_nav.launch.py'])
        )
    )

    # Step 1: Launch planner.py (generates /inspection_markers)
    planner_node = TimerAction(
        period=5.0,  # Wait for sim + nav2 + map to be ready
        actions=[Node(
            package='dis_tutorial3',
            executable='planner.py',
            name='inspection_planner',
            output='screen'
        )]
    )

    # Step 2: detect_people.py
    detect_people_node = TimerAction(
        period=7.0,
        actions=[Node(
            package='dis_tutorial3',
            executable='detect_people.py',
            name='detect_people',
            output='screen'
        )]
    )

    # Step 3: detect_rings.py
    detect_rings_node = TimerAction(
        period=9.0,
        actions=[Node(
            package='dis_tutorial3',
            executable='detect_rings.py',
            name='detect_rings',
            output='screen'
        )]
    )

    # Step 4: face_search.py
    face_search_node = TimerAction(
        period=11.0,
        actions=[Node(
            package='dis_tutorial3',
            executable='face_search.py',
            name='face_search',
            output='screen'
        )]
    )

    return LaunchDescription(arguments + [
        sim_base,
        planner_node,
        detect_people_node,
        detect_rings_node,
        face_search_node
    ])

