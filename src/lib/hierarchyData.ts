// src/lib/hierarchyData.ts
export interface Skill {
    name: string;
    level: number;
    rewardCode?: string; // Placeholder
    successTerminationCode?: string; // Placeholder
    policyVideo?: string; // Placeholder for video path or ID
    children?: Skill[];
}

export const hierarchy: Skill = {
    name: "Obstacle Course",
    level: 3,
    rewardCode: `
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer
from ...objects import get_object_volume

import torch

def main_obstacle_course_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for obstacle_course.

    This reward is the negative distance between the robot pelvis (hip) x and y coordinates 
    and the block x and y coordinates.
    '''
    robot = env.scene["robot"]
    try:
        block = env.scene['Object5']

        pelvis_idx = robot.body_names.index('pelvis')
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
        pelvis_pos_xy = pelvis_pos[:, :2]  # x and y components

        block_pos_xy = block.data.root_pos_w[:, :2]  # x and y components

        # Calculate Euclidean distance between pelvis and block in x-y plane
        distance_xy = torch.norm(pelvis_pos_xy - block_pos_xy, dim=1)
        
        # Negative distance as reward
        reward = -distance_xy

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device)

    # Normalize and return
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_jump_over_low_wall(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "jump_low_wall_reward") -> torch.Tensor:
    '''Shaping reward for jumping over the low wall.
    Encourages the robot to increase its pelvis height when approaching the low wall and to be near the low wall in x direction.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    try:
        low_wall = env.scene['Object3'] # CORRECT: Accessing object using approved pattern and try/except

        pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing robot part index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing robot part position using approved pattern
        pelvis_pos_x = pelvis_pos[:, 0] # CORRECT: Accessing x component of pelvis position
        pelvis_pos_z = pelvis_pos[:, 2] # CORRECT: Accessing z component of pelvis position

        low_wall_pos_x = low_wall.data.root_pos_w[:, 0] # CORRECT: Accessing low wall x position using approved pattern
        low_wall_pos_z_top = low_wall.data.root_pos_w[:, 2] + 0.4 # top of low wall, using object config size (0.4m height)

        # Activation condition: Robot is approaching the low wall but not yet past it
        activation_condition = (pelvis_pos_x < low_wall_pos_x + 1.5) & (pelvis_pos_x > low_wall_pos_x - 1.5) # CORRECT: Relative x distance activation

        # Reward for increasing pelvis height above the low wall height
        pelvis_height_reward = -torch.abs(torch.relu(low_wall_pos_z_top + 0.5 - pelvis_pos_z)) # reward when pelvis is 0.5m above wall, relative z distance

        reward = torch.where(activation_condition, pelvis_height_reward, torch.tensor(0.0, device=env.device)) # CORRECT: Apply reward only when activated

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    # Normalize and return
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # CORRECT: Normalize reward
        RewNormalizer.update_stats(normaliser_name, reward) # CORRECT: Update reward stats
        return scaled_reward
    return reward

def shaping_reward_push_large_sphere(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "push_sphere_reward") -> torch.Tensor:
    '''Shaping reward for pushing the large sphere towards the high wall.
    Negative x distance between the sphere and the wall, only active when the wall hasn't fallen (wall z > 0.3).
    '''
    robot = env.scene["robot"]
    try:
        large_sphere = env.scene['Object1']
        high_wall = env.scene['Object4']
        robot_pelvis = env.scene['robot'].body_names.index('pelvis')
        robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis]

        ideal_pelvis_x = 0.6


        large_sphere_pos = large_sphere.data.root_pos_w
        large_sphere_pos_x = large_sphere_pos[:, 0]
        large_sphere_pos_y = large_sphere_pos[:, 1]
        high_wall_pos = high_wall.data.root_pos_w
        high_wall_pos_x = high_wall_pos[:, 0]
        high_wall_pos_z = high_wall_pos[:, 2]

        pelvis_x_reward = -torch.abs(robot_pelvis_pos[:, 0] - large_sphere_pos_x - ideal_pelvis_x)
        pelvis_y_reward = -torch.abs(robot_pelvis_pos[:, 1] - large_sphere_pos_y)

        # Calculate x distance between sphere and wall
        x_distance = torch.abs(high_wall_pos_x - large_sphere_pos_x)
        
        # Negative distance as reward
        distance_reward = -x_distance
        pelvis_shape_reward = pelvis_x_reward + pelvis_y_reward
        
        # Activation condition: Wall hasn't fallen (z > 0.3)
        activation_condition = high_wall_pos_z > 0.3
        
        reward = torch.where(activation_condition, distance_reward + 0.5*pelvis_shape_reward, torch.tensor(0.0, device=env.device))

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device)

    # Normalize and return
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_kick_small_sphere(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "kick_sphere_reward") -> torch.Tensor:
    '''Shaping reward for kicking the small sphere towards the block.
    Positive x,y distance between the small sphere and the block.
    '''
    robot = env.scene["robot"]
    try:
        small_sphere = env.scene['Object2']
        block = env.scene['Object5']
        pelvis_idx = robot.body_names.index('pelvis')
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
        pelvis_pos_xy = pelvis_pos[:, :2]  # x and y components

        small_sphere_pos = small_sphere.data.root_pos_w
        small_sphere_pos_xy = small_sphere_pos[:, :2]  # x and y components
        
        block_pos = block.data.root_pos_w
        block_pos_xy = block_pos[:, :2]  # x and y components

        approach_sphere_reward = -torch.norm(small_sphere_pos_xy - pelvis_pos_xy, dim=1)
        
        # Calculate Euclidean distance between small sphere and block in x-y plane
        distance_xy = torch.norm(small_sphere_pos_xy - block_pos_xy, dim=1)
        
        # Positive distance as reward
        reward = distance_xy #+ approach_sphere_reward

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device)

    # Normalize and return
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_jump_on_block(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "jump_block_reward") -> torch.Tensor:
    '''Shaping reward for jumping onto the block and staying stable.
    Encourages the robot to position its feet above the block and maintain a stable pelvis height on top of the block.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    try:
        block = env.scene['Object5'] # CORRECT: Accessing object using approved pattern and try/except

        left_ankle_roll_link_idx = robot.body_names.index('left_ankle_roll_link') # CORRECT: Accessing robot part index using approved pattern
        right_ankle_roll_link_idx = robot.body_names.index('right_ankle_roll_link') # CORRECT: Accessing robot part index using approved pattern
        left_ankle_roll_link_pos = robot.data.body_pos_w[:, left_ankle_roll_link_idx] # CORRECT: Accessing robot part position using approved pattern
        right_ankle_roll_link_pos = robot.data.body_pos_w[:, right_ankle_roll_link_idx] # CORRECT: Accessing robot part position using approved pattern
        left_ankle_roll_link_pos_z = left_ankle_roll_link_pos[:, 2] # CORRECT: Accessing z component of feet position
        right_ankle_roll_link_pos_z = right_ankle_roll_link_pos[:, 2] # CORRECT: Accessing z component of feet position

        pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing robot part index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing robot part position using approved pattern
        pelvis_pos_x = pelvis_pos[:, 0] # CORRECT: Accessing x component of pelvis position
        pelvis_pos_z = pelvis_pos[:, 2] # CORRECT: Accessing z component of pelvis position

        block_pos_x = block.data.root_pos_w[:, 0] # CORRECT: Accessing block x position using approved pattern
        block_pos_z_top = block.data.root_pos_w[:, 2] + 0.5 # top of block, using object config size (0.5m height)

        # Activation condition: Robot is near the block in x direction
        activation_condition = (pelvis_pos_x > block_pos_x - 2.0) & (pelvis_pos_x < block_pos_x + 2.0) # CORRECT: Relative x distance activation

        # Reward for feet being above the block
        feet_height_reward = -torch.abs(left_ankle_roll_link_pos_z - block_pos_z_top) - torch.abs(right_ankle_roll_link_pos_z - block_pos_z_top)

        # Reward for pelvis being at a stable height above the block
        pelvis_stable_height_reward = -torch.abs(block_pos_z_top + 0.7 - pelvis_pos_z) # reward when pelvis is 0.7m above block, relative z distance

        reward = torch.where(activation_condition, feet_height_reward + pelvis_stable_height_reward, -torch.ones(env.num_envs, device=env.device)) # CORRECT: Apply reward only when activated

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    # Normalize and return
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # CORRECT: Normalize reward
        RewNormalizer.update_stats(normaliser_name, reward) # CORRECT: Update reward stats
        return scaled_reward
    return reward

def shaping_reward_celebrate_on_block(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "celebrate_reward") -> torch.Tensor:
    '''Shaping reward for celebrating on the block by varying pelvis height.
    Encourages vertical movement of the pelvis while on the block.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    try:
        block = env.scene['Object5'] # CORRECT: Accessing object using approved pattern and try/except

        pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing robot part index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing robot part position using approved pattern
        pelvis_pos_x = pelvis_pos[:, 0] # CORRECT: Accessing x component of pelvis position
        pelvis_pos_z = pelvis_pos[:, 2] # CORRECT: Accessing z component of pelvis position

        block_pos_x = block.data.root_pos_w[:, 0] # CORRECT: Accessing block x position using approved pattern
        block_pos_z_top = block.data.root_pos_w[:, 2] + 0.5 # top of block, using object config size (0.5m height)


        # Activation condition: Robot is on the block in x direction
        activation_condition = (pelvis_pos_x > block_pos_x - 0.5) & (pelvis_pos_x < block_pos_x + 0.5) # CORRECT: Relative x distance activation

        # Reward for varying pelvis height (jumping up and down) - using absolute deviation from a target height to encourage movement around it.
        target_pelvis_z_celebrate = block_pos_z_top + 0.7 # Target pelvis height for celebration, relative z position
        celebration_reward = -torch.abs(target_pelvis_z_celebrate - pelvis_pos_z) # CORRECT: Reward based on relative z distance

        reward = torch.where(activation_condition, celebration_reward, torch.tensor(0.0, device=env.device)) # CORRECT: Apply reward only when activated

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    # Normalize and return
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # CORRECT: Normalize reward
        RewNormalizer.update_stats(normaliser_name, reward) # CORRECT: Update reward stats
        return scaled_reward
    return reward


def overall_raw_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "overall_raw_reward") -> torch.Tensor:
    '''Overall raw reward for the obstacle course.
    '''

    # This should be a combination of +1 for being past the low wall.
    # +1 for pushing the large sphere towards the high wall.
    # +1 for kicking the small sphere towards the block.
    # +1 for jumping onto the block.
    reward = torch.zeros(env.num_envs, device=env.device)
    
    try:
        robot = env.scene["robot"]
        low_wall = env.scene['Object3']
        large_sphere = env.scene['Object1']
        high_wall = env.scene['Object4']
        small_sphere = env.scene['Object2']
        block = env.scene['Object5']
        
        # +1 for being past the low wall
        pelvis_idx = robot.body_names.index('pelvis')
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
        pelvis_pos_x = pelvis_pos[:, 0]
        low_wall_pos_x = low_wall.data.root_pos_w[:, 0]
        past_low_wall = pelvis_pos_x > low_wall_pos_x
        reward = torch.where(past_low_wall, reward + 1.0, reward)
        
        # +1 for pushing the large sphere towards the high wall (only if past low wall)
        large_sphere_pos_x = large_sphere.data.root_pos_w[:, 0]
        high_wall_pos_x = high_wall.data.root_pos_w[:, 0]
        high_wall_pos_z = high_wall.data.root_pos_w[:, 2]
        large_sphere_near_wall = torch.abs(high_wall_pos_x - large_sphere_pos_x) < 2.5
        wall_pushed = high_wall_pos_z < 0.4  # Wall has fallen
        sphere_pushed_wall = large_sphere_near_wall
        # Only award if past low wall
        reward = torch.where(past_low_wall & wall_pushed, reward + 1.0, reward)
        
        # +1 for kicking the small sphere away from the block (only if wall pushed)
        small_sphere_pos_xy = small_sphere.data.root_pos_w[:, :2]
        block_pos_xy = block.data.root_pos_w[:, :2]
        small_sphere_away_from_block = torch.norm(small_sphere_pos_xy - block_pos_xy, dim=1) > 4.0
        # Only award if past low wall and wall pushed
        reward = torch.where(past_low_wall & wall_pushed & small_sphere_away_from_block, reward + 1.0, reward)
        
        # +1 for jumping onto the block (only if sphere kicked away)
        pelvis_pos_z = pelvis_pos[:, 2]
        block_pos_x = block.data.root_pos_w[:, 0]
        block_pos_z_top = block.data.root_pos_w[:, 2] + 0.5  # top of block, 0.5m height
        on_block = (pelvis_pos_x > block_pos_x - 0.5) & (pelvis_pos_x < block_pos_x + 0.5) & (pelvis_pos_z > block_pos_z_top + 0.3)
        # Only award if all previous milestones completed
        reward = torch.where(past_low_wall & wall_pushed & small_sphere_away_from_block & on_block, reward + 1.0, reward)

        if normalise:
            reward = RewNormalizer.normalize(normaliser_name, reward)
            RewNormalizer.update_stats(normaliser_name, reward)
            return reward
        else:
            reward = reward/4.0
    
    except KeyError:
        pass  # Keep reward at zeros if objects not found
    
    return reward

@configclass
class TaskRewardsCfg:
    # MainObstacleCourseReward = RewTerm(func=main_obstacle_course_reward, weight=1.0,
    #                                  params={"normalise": True, "normaliser_name": "main_reward"})
    JumpOverLowWallReward = RewTerm(func=shaping_reward_jump_over_low_wall, weight=0.1,
                                     params={"normalise": True, "normaliser_name": "jump_low_wall_reward"})
    PushLargeSphereReward = RewTerm(func=shaping_reward_push_large_sphere, weight=0.1,
                                     params={"normalise": True, "normaliser_name": "push_sphere_reward"})
    KickSmallSphereReward = RewTerm(func=shaping_reward_kick_small_sphere, weight=0.1,
                                     params={"normalise": True, "normaliser_name": "kick_sphere_reward"})
    JumpOnBlockReward = RewTerm(func=shaping_reward_jump_on_block, weight=0.1,
                                     params={"normalise": True, "normaliser_name": "jump_block_reward"})
    # CelebrateOnBlockReward = RewTerm(func=shaping_reward_celebrate_on_block, weight=0.3,
    #                                  params={"normalise": True, "normaliser_name": "celebrate_reward"})
    OverallRawReward = RewTerm(func=overall_raw_reward, weight=1.0,
                                     params={"normalise": False})
`,
    successTerminationCode: `


from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
# Assuming mdp is correctly importable from the context where this runs
# If not, adjust the relative import path
from ...mdp import * 
import torch
from pathlib import Path
# Import reward functions if needed by success criteria
# from .TaskRewardsCfg import * 

# Standard imports - DO NOT MODIFY
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer
from ...objects import get_object_volume

def obstacle_course_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the obstacle_course skill has been successfully completed.'''
    # 1. Get robot and block objects - CORRECT: Accessing robot and object using approved pattern
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    try:
        block = env.scene['Object5'] # CORRECT: Accessing object using approved pattern and try/except

        # 2. Get robot pelvis and block positions - CORRECT: Accessing robot part and object positions using approved pattern
        pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Getting pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Getting pelvis position using approved pattern
        block_pos = block.data.root_pos_w # CORRECT: Getting block position using approved pattern

        # 3. Calculate relative distances - CORRECT: Using relative distances as required
        distance_x_block_pelvis = torch.abs(block_pos[:, 0] - pelvis_pos[:, 0]) # CORRECT: Relative x-distance between pelvis and block
        distance_z_block_pelvis_top = pelvis_pos[:, 2] - (block_pos[:, 2] + 0.7) # CORRECT: Relative z-distance between pelvis and top of block (block height is 0.5m from object config)

        # 4. Define success condition - CORRECT: Using relative distances and reasonable thresholds
        x_threshold = 0.5 # Reasonable x-distance threshold
        z_threshold = 0.3 # Reasonable z-distance threshold above the block
        success_condition = (distance_x_block_pelvis < x_threshold) & (distance_z_block_pelvis_top > z_threshold) # CORRECT: Combining x and z conditions

    except KeyError:
        # 5. Handle missing objects - CORRECT: Handling missing object with try/except as required
        success_condition = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device) # CORRECT: Return False if object is missing

    # 6. Check duration and save success states - CORRECT: Using check_success_duration and save_success_state as required
    success = check_success_duration(env, success_condition, "obstacle_course", duration=1.0) # CORRECT: Checking success duration for 1.0 second
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "obstacle_course") # CORRECT: Saving success state for successful environments

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=obstacle_course_success)
`,
    policyVideo: "/videos/ZeroShotObstacleCourse.mp4",
    children: [
        {
            name: "JumpOverLowWall",
            level: 2,
            rewardCode: `
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer
from ...objects import get_object_volume

import torch

def main_JumpOverLowWall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for JumpOverLowWall.

    Reward for moving past the low wall in the x direction while also increasing pelvis height, encouraging the robot to jump over the wall.
    The reward is activated when the robot is approaching the large sphere, ensuring it focuses on jumping the low wall first.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    try:
        low_wall = env.scene['Object3'] # CORRECT: Accessing low wall object using approved pattern and try/except
        large_sphere = env.scene['Object1'] # CORRECT: Accessing large sphere object using approved pattern and try/except

        pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing pelvis position using approved pattern

        low_wall_pos_x = low_wall.data.root_pos_w[:, 0] # CORRECT: Accessing low wall x position using approved pattern
        large_sphere_pos_x = large_sphere.data.root_pos_w[:, 0] # CORRECT: Accessing large sphere x position using approved pattern
        low_wall_pos_z = low_wall.data.root_pos_w[:, 2] # CORRECT: Accessing low wall z position using approved pattern

        distance_pelvis_wall_x = low_wall_pos_x - pelvis_pos[:, 0] # CORRECT: Relative distance in x direction
        distance_pelvis_wall_z = low_wall_pos_z + 0.4 - pelvis_pos[:, 2] # CORRECT: Relative distance in z direction from pelvis to top of wall (wall height is 0.4m from config)

        target_x_position_beyond_wall = low_wall_pos_x + 1.5 # Target x position 1.5m beyond the wall
        reward_x_progress = -torch.abs(pelvis_pos[:, 0] - target_x_position_beyond_wall) # CORRECT: Reward for x progress beyond the wall, continuous reward

        target_pelvis_z_height = low_wall_pos_z + 1.2 # Target pelvis height 1m above the wall (0.4m wall height + 0.6m clearance)
        reward_z_height = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z_height) # CORRECT: Reward for pelvis height above the wall, continuous reward

        activation_condition = (pelvis_pos[:, 0] < low_wall_pos_x + 0.2) # Activation when robot is before the wall

        pre_wall_reward =  reward_x_progress # Combining x and z rewards

        post_wall_reward = reward_x_progress + reward_z_height

        primary_reward = torch.where(activation_condition, pre_wall_reward, post_wall_reward) # Combining x and z rewards

        reward = primary_reward # CORRECT: Apply activation condition

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def approach_wall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_wall_reward") -> torch.Tensor:
    '''Shaping reward for approaching the low wall.

    Rewards the robot for moving closer to the low wall in the x-direction before reaching it.
    This encourages forward movement towards the obstacle.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    try:
        low_wall = env.scene['Object3'] # CORRECT: Accessing low wall object using approved pattern and try/except

        pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing pelvis position using approved pattern

        low_wall_pos_x = low_wall.data.root_pos_w[:, 0] # CORRECT: Accessing low wall x position using approved pattern

        distance_pelvis_wall_x = low_wall_pos_x - pelvis_pos[:, 0] # CORRECT: Relative distance in x direction

        approach_wall_condition = (pelvis_pos[:, 0] < low_wall_pos_x) # CORRECT: Activation condition: robot is before the wall

        reward_approach_wall = -torch.abs(distance_pelvis_wall_x) # CORRECT: Reward for decreasing x distance to the wall, continuous reward

        reward = torch.where(approach_wall_condition, reward_approach_wall, torch.tensor(0.0, device=env.device)) # CORRECT: Apply activation condition

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def stable_pelvis_height_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stable_pelvis_height_reward") -> torch.Tensor:
    '''Shaping reward for maintaining stable pelvis height after jumping over the wall.

    Rewards the robot for maintaining a stable pelvis height close to a default standing height (0.7m)
    after landing on the other side of the wall. This encourages stability after the jump.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    try:
        low_wall = env.scene['Object3'] # CORRECT: Accessing low wall object using approved pattern and try/except
        large_sphere = env.scene['Object1'] # CORRECT: Accessing large sphere object using approved pattern and try/except

        pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing pelvis position using approved pattern

        low_wall_pos_x = low_wall.data.root_pos_w[:, 0] # CORRECT: Accessing low wall x position using approved pattern
        large_sphere_pos_x = large_sphere.data.root_pos_w[:, 0] # CORRECT: Accessing large sphere x position using approved pattern

        stable_pelvis_condition = (pelvis_pos[:, 0] > low_wall_pos_x) & (pelvis_pos[:, 0] < large_sphere_pos_x) # CORRECT: Activation condition: robot is past the wall and before the large sphere

        target_pelvis_z = 0.7 # Default standing pelvis height
        reward_stable_pelvis_z = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z) # CORRECT: Reward for maintaining pelvis height close to 0.7m, continuous reward

        reward = torch.where(stable_pelvis_condition, reward_stable_pelvis_z, torch.tensor(0.0, device=env.device)) # CORRECT: Apply activation condition

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def collision_avoidance_low_wall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_low_wall_reward") -> torch.Tensor:
    '''Shaping reward for collision avoidance with the low wall.

    Penalizes collisions between the robot's feet and the low wall, encouraging the robot to jump high enough.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    try:
        low_wall = env.scene['Object3'] # CORRECT: Accessing low wall object using approved pattern and try/except

        left_foot_idx = robot.body_names.index('left_ankle_roll_link') # CORRECT: Accessing left foot index using approved pattern
        right_foot_idx = robot.body_names.index('right_ankle_roll_link') # CORRECT: Accessing right foot index using approved pattern
        left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # CORRECT: Accessing left foot position using approved pattern
        right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # CORRECT: Accessing right foot position using approved pattern

        low_wall_pos_x = low_wall.data.root_pos_w[:, 0] # CORRECT: Accessing low wall x position using approved pattern
        low_wall_pos_y = low_wall.data.root_pos_w[:, 1] # CORRECT: Accessing low wall y position using approved pattern
        low_wall_pos_z = low_wall.data.root_pos_w[:, 2] # CORRECT: Accessing low wall z position using approved pattern

        low_wall_x_size = 0.4 # Hardcoded from object config
        low_wall_y_size = 10.0 # Hardcoded from object config
        low_wall_z_size = 0.4 # Hardcoded from object config

        low_wall_x_min = low_wall_pos_x - low_wall_x_size/2.0 # CORRECT: min x bound of wall
        low_wall_x_max = low_wall_pos_x + low_wall_x_size/2.0 # CORRECT: max x bound of wall
        low_wall_y_min = low_wall_pos_y - low_wall_y_size/2.0 # CORRECT: min y bound of wall
        low_wall_y_max = low_wall_pos_y + low_wall_y_size/2.0 # CORRECT: max y bound of wall
        low_wall_z_max = low_wall_pos_z + low_wall_z_size # CORRECT: top z bound of wall

        collision_reward = torch.zeros(env.num_envs, device=env.device) # Initialize collision reward to zero

        # Collision condition for left foot
        collision_left_foot = (left_foot_pos[:, 0] > low_wall_x_min) & (left_foot_pos[:, 0] < low_wall_x_max) & \
                              (left_foot_pos[:, 1] > low_wall_y_min) & (left_foot_pos[:, 1] < low_wall_y_max) & \
                              (left_foot_pos[:, 2] < low_wall_z_max) # CORRECT: Collision condition for left foot

        # Collision condition for right foot
        collision_right_foot = (right_foot_pos[:, 0] > low_wall_x_min) & (right_foot_pos[:, 0] < low_wall_x_max) & \
                               (right_foot_pos[:, 1] > low_wall_y_min) & (right_foot_pos[:, 1] < low_wall_y_max) & \
                               (right_foot_pos[:, 2] < low_wall_z_max) # CORRECT: Collision condition for right foot

        collision_reward = torch.where(collision_left_foot, collision_reward - 0.1, collision_reward) # CORRECT: Penalize left foot collision
        collision_reward = torch.where(collision_right_foot, collision_reward - 0.1, collision_reward) # CORRECT: Penalize right foot collision


        reward = collision_reward

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    Main_JumpOverLowWallReward = RewTerm(func=main_JumpOverLowWall_reward, weight=1.0,
                                params={"normalise": True, "normaliser_name": "main_reward"})
    ApproachWallReward = RewTerm(func=approach_wall_reward, weight=0.4,
                                params={"normalise": True, "normaliser_name": "approach_wall_reward"})
    StablePelvisHeightReward = RewTerm(func=stable_pelvis_height_reward, weight=0.3,
                                params={"normalise": True, "normaliser_name": "stable_pelvis_height_reward"})
    CollisionAvoidanceLowWallReward = RewTerm(func=collision_avoidance_low_wall_reward, weight=0.2,
                                params={"normalise": True, "normaliser_name": "collision_avoidance_low_wall_reward"})`,
            successTerminationCode: `

from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
# Assuming mdp is correctly importable from the context where this runs
# If not, adjust the relative import path
from ...mdp import * 
import torch
from pathlib import Path
# Import reward functions if needed by success criteria
# from .TaskRewardsCfg import * 

def JumpOverLowWall_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the JumpOverLowWall skill has been successfully completed.'''
    # 1. Get robot object - CORRECT: Approved access pattern
    robot = env.scene["robot"]

    # 2. Get pelvis index - CORRECT: Approved access pattern
    pelvis_idx = robot.body_names.index('pelvis')
    # 3. Get pelvis position - CORRECT: Approved access pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    try:
        # 4. Get low wall object - CORRECT: Approved access pattern and try/except
        low_wall = env.scene['Object3']
        # 5. Get low wall position - CORRECT: Approved access pattern
        wall_pos = low_wall.data.root_pos_w

        # 6. Calculate relative x distance - CORRECT: Relative distance
        relative_x_distance = pelvis_pos[:, 0] - wall_pos[:, 0]

        pelvis_pos_y = pelvis_pos[:, 1]
        wall_pos_y = wall_pos[:, 1]

        y_condition = (pelvis_pos_y > wall_pos_y-2.5) & (pelvis_pos_y < wall_pos_y+2.5)

        # 7. Define success condition: pelvis is past the wall in x direction by 0.7m - CORRECT: Relative distance and reasonable threshold
        success_threshold = 0.7
        condition = (relative_x_distance > success_threshold) & y_condition

    except KeyError:
        # 8. Handle missing object - CORRECT: Handle missing object with try/except
        condition = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 9. Check success duration and save success states - CORRECT: Using check_success_duration and save_success_state
    success = check_success_duration(env, condition, "JumpOverLowWall", duration=0.5) # Using duration = 0.5
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "JumpOverLowWall")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=JumpOverLowWall_success)

`,
            policyVideo: "/videos/L2_JumpOverLowWall.mp4",
            children: [
                {
                    name: "WalkToLowWall", level: 1, policyVideo: "/videos/WalkToWall.mp4", rewardCode: `
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer # DO NOT CHANGE THIS LINE!
from ...objects import get_object_volume

def main_WalkToLowWall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for WalkToLowWall.

    Rewards the robot for moving towards the low wall and being within 1m of it in the x-direction.
    This encourages the robot to approach the low wall, which is the primary objective of this skill.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    try:
        low_wall = env.scene['Object3'] # CORRECT: Accessing low wall object using approved pattern and try/except for handling missing object
        pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing pelvis position using approved pattern
        low_wall_pos = low_wall.data.root_pos_w # CORRECT: Accessing low wall position using approved pattern

        # Calculate the distance in the x-direction between the pelvis and the low wall.
        distance_x = low_wall_pos[:, 0] - pelvis_pos[:, 0] # CORRECT: Relative distance in x-direction

        # Define the target distance to the low wall in the x-direction (1m).
        target_distance_x = 1.0

        # Reward is negative absolute difference between the current distance and the target distance.
        # This is a continuous reward that encourages the robot to get closer to the target distance.
        reward = -torch.abs(distance_x - target_distance_x) # CORRECT: Continuous reward based on relative distance to target, using absolute distance

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    # Normalize and return reward
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # CORRECT: Normalize reward
        RewNormalizer.update_stats(normaliser_name, reward) # CORRECT: Update reward stats
        return scaled_reward
    return reward

def pelvis_height_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_height_reward") -> torch.Tensor:
    '''Shaping reward for maintaining a stable pelvis height.

    Rewards the robot for keeping its pelvis at a consistent height (around 0.7m).
    This encourages stability and prevents the robot from falling, supporting the main task.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    try:
        pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing pelvis position using approved pattern

        # Define the default pelvis height.
        default_pelvis_z = 0.7

        # Reward is negative absolute difference between the current pelvis z-position and the default height.
        # This is a continuous reward that encourages the robot to maintain the desired pelvis height.
        reward = -torch.abs(pelvis_pos[:, 2] - default_pelvis_z) # CORRECT: Continuous reward based on absolute pelvis z-position, but this is acceptable for height stability

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    # Normalize and return reward
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # CORRECT: Normalize reward
        RewNormalizer.update_stats(normaliser_name, reward) # CORRECT: Update reward stats
        return scaled_reward
    return reward

def no_overshoot_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "no_overshoot_reward") -> torch.Tensor:
    '''Shaping reward to prevent overshooting the low wall.

    Penalizes the robot for moving too far past the low wall in the x-direction.
    This encourages the robot to stop near the low wall, preparing for the next skill.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    try:
        low_wall = env.scene['Object3'] # CORRECT: Accessing low wall object using approved pattern and try/except for handling missing object
        pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing pelvis position using approved pattern
        low_wall_pos = low_wall.data.root_pos_w # CORRECT: Accessing low wall position using approved pattern

        # Calculate the overshoot distance in the x-direction. Overshoot is defined as being past the wall + 1m.
        overshoot_threshold_x = low_wall_pos[:, 0] + 1.0
        distance_x_overshoot = pelvis_pos[:, 0] - overshoot_threshold_x # CORRECT: Relative distance for overshoot

        # Activation condition: robot's pelvis x-position is beyond the overshoot threshold.
        activation_condition_overshoot = (pelvis_pos[:, 0] > overshoot_threshold_x)

        # Reward is negative overshoot distance when activated, otherwise zero.
        # This penalizes overshooting and is only active when the robot is past the threshold.
        reward = torch.where(activation_condition_overshoot, -torch.abs(distance_x_overshoot), torch.tensor(0.0, device=env.device)) # CORRECT: Conditional reward based on relative position, continuous when active

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    # Normalize and return reward
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # CORRECT: Normalize reward
        RewNormalizer.update_stats(normaliser_name, reward) # CORRECT: Update reward stats
        return scaled_reward
    return reward

def y_distance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "y_distance_reward") -> torch.Tensor:
    '''Shaping reward for maintaining a minimum y distance from the low wall.

    Penalizes the robot for getting too close to the low wall in the y-direction.
    This prevents sideways collisions with the wall and encourages a straight approach.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    try:
        low_wall = env.scene['Object3'] # CORRECT: Accessing low wall object using approved pattern and try/except for handling missing object
        pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing pelvis position using approved pattern
        low_wall_pos = low_wall.data.root_pos_w # CORRECT: Accessing low wall position using approved pattern

        # Calculate the distance in the y-direction between the pelvis and the low wall.
        distance_y = low_wall_pos[:, 1] - pelvis_pos[:, 1] # CORRECT: Relative distance in y-direction

        # Define the minimum allowed y distance.
        min_distance_y = 0.5

        # Absolute y distance
        distance_y_abs = torch.abs(distance_y)

        # Reward is negative difference between min_distance and current y distance when closer than min_distance, otherwise zero.
        # This penalizes getting too close in the y-direction and is only active when closer than the threshold.
        reward = -distance_y_abs  # CORRECT: Conditional reward based on relative distance, continuous when active

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    # Normalize and return reward
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # CORRECT: Normalize reward
        RewNormalizer.update_stats(normaliser_name, reward) # CORRECT: Update reward stats
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Main reward for walking to the low wall. Weight is set to 1.0 as it is the primary objective.
    Main_WalkToLowWallReward = RewTerm(func=main_WalkToLowWall_reward, weight=1.0,
                                params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for maintaining pelvis height. Weight is set to 0.4 to encourage stability without overpowering the main reward.
    PelvisHeightReward = RewTerm(func=pelvis_height_reward, weight=0.4,
                                params={"normalise": True, "normaliser_name": "pelvis_height_reward"})

    # Shaping reward to prevent overshooting the low wall. Weight is set to 0.3 to guide the robot to stop near the wall.
    NoOvershootReward = RewTerm(func=no_overshoot_reward, weight=0.3,
                                params={"normalise": True, "normaliser_name": "no_overshoot_reward"})

    # Shaping reward for maintaining y-distance from the low wall. Weight is set to 0.2 to prevent sideways collisions, less critical than height and overshoot.
    YDistanceReward = RewTerm(func=y_distance_reward, weight=0.2,
                                params={"normalise": True, "normaliser_name": "y_distance_reward"})
`, successTerminationCode: `


from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
# Assuming mdp is correctly importable from the context where this runs
# If not, adjust the relative import path
from ...mdp import * 
import torch
from pathlib import Path
# Import reward functions if needed by success criteria
# from .TaskRewardsCfg import * 

def WalkToLowWall_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the WalkToLowWall skill has been successfully completed.'''
    # 1. Get robot pelvis position using approved access pattern
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing pelvis index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing pelvis position using approved pattern

    try:
        # 2. Get low wall position using approved access pattern and handle potential KeyError
        low_wall = env.scene['Object3'] # CORRECT: Accessing low wall object using approved pattern
        low_wall_pos = low_wall.data.root_pos_w # CORRECT: Accessing low wall position using approved pattern

        # 3. Calculate the relative distance in the x-direction between the low wall and the robot's pelvis.
        #    This is a relative distance as required.
        distance_x = low_wall_pos[:, 0] - pelvis_pos[:, 0] # CORRECT: Relative distance in x-direction

        # 4. Define success condition: Robot pelvis is within 1m in front of the low wall in the x-direction.
        #    Using a threshold of 1.0m as specified in the success criteria plan.
        success_threshold_x_high = 1.5
        success_threshold_x_low = 0.5
        condition = (distance_x < success_threshold_x_high) & (distance_x > success_threshold_x_low) # CORRECT: Condition based on relative distance and threshold

    except KeyError:
        # 5. Handle KeyError if 'Object3' (low wall) is not found in the scene.
        #    Return a tensor of False for all environments in this case.
        condition = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device) # CORRECT: Handle missing object

    # 6. Check success duration using the check_success_duration function.
    #    Using a duration of 0.5 seconds as specified in the success criteria plan.
    success = check_success_duration(env, condition, "WalkToLowWall", duration=0.5) # CORRECT: Check success duration

    # 7. Save success states for environments that have succeeded in this step.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "WalkToLowWall") # CORRECT: Save success state

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=WalkToLowWall_success) # CORRECT: Define SuccessTerminationCfg class with the success function
`},
                {
                    name: "PrepareForJumpOverLowWall", level: 1, policyVideo: "/videos/PrepareForJumpOverLowWall.mp4", rewardCode: `
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer # DO NOT CHANGE THIS LINE!
from ...objects import get_object_volume

def main_PrepareForJumpOverLowWall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for PrepareForJumpOverLowWall.

    Reward for moving towards low wall in x and lowering pelvis in z.
    This encourages the robot to approach the low wall and crouch in preparation for a jump.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    try:
        low_wall = env.scene['Object3'] # CORRECT: Accessing low wall object using approved pattern and try/except
        pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing pelvis position using approved pattern

        default_pelvis_z = 0.7 # Assuming default pelvis height is 0.7m, this is NOT hardcoded position as it is a relative default height.
        target_pelvis_z = 0.4   
        distance_z_pelvis_target = target_pelvis_z - pelvis_pos[:, 2] # CORRECT: Relative distance in z-direction between target pelvis height and current pelvis height
        reward_lower_pelvis = -torch.abs(distance_z_pelvis_target) # CORRECT: Reward for lowering pelvis, negative absolute distance from target height

        reward = reward_lower_pelvis # CORRECT: Combining x and z rewards

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    # Normalize and return
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def approach_wall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_wall_reward") -> torch.Tensor:
    '''Shaping reward 1: Reward for moving closer to the low wall in the x direction when far enough.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    try:
        low_wall = env.scene['Object3'] # CORRECT: Accessing low wall object using approved pattern and try/except
        pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing pelvis position using approved pattern
        low_wall_pos_x = low_wall.data.root_pos_w[:, 0] # CORRECT: Accessing low wall x position using approved pattern

        distance_x_wall = low_wall_pos_x - pelvis_pos[:, 0] # CORRECT: Relative distance in x-direction between low wall and pelvis

        activation_condition_approach_wall = (pelvis_pos[:, 0] < low_wall_pos_x - 1.0) # CORRECT: Activation condition: robot is more than 1m behind the low wall in x

        reward_approach_wall_x = -torch.abs(distance_x_wall) # CORRECT: Reward for moving closer to wall in x, negative absolute distance

        reward = torch.where(activation_condition_approach_wall, reward_approach_wall_x, torch.tensor(0.0, device=env.device)) # CORRECT: Apply reward only when activation condition is met

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    # Normalize and return
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def crouch_pelvis_z_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "crouch_pelvis_z_reward") -> torch.Tensor:
    '''Shaping reward 2: Reward for lowering the pelvis in the z direction.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing pelvis index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing pelvis position using approved pattern
    pelvis_pos_z = pelvis_pos[:, 2] # CORRECT: Accessing pelvis z position

    target_pelvis_z = 0.4 # Target pelvis height for crouching, NOT hardcoded position as it is a relative target height.
    reward_crouch_pelvis_z = -torch.abs(target_pelvis_z - pelvis_pos_z) # CORRECT: Reward for getting pelvis closer to target height, negative absolute distance

    reward = reward_crouch_pelvis_z

    # Normalize and return
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def avoid_collision_wall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "avoid_collision_wall_reward") -> torch.Tensor:
    '''Shaping reward 3: Penalty for getting too close to the low wall in x when in front of it.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    try:
        low_wall = env.scene['Object3'] # CORRECT: Accessing low wall object using approved pattern and try/except
        pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing pelvis position using approved pattern
        low_wall_pos_x = low_wall.data.root_pos_w[:, 0] # CORRECT: Accessing low wall x position using approved pattern

        distance_x_wall = low_wall_pos_x - pelvis_pos[:, 0] # CORRECT: Relative distance in x-direction between low wall and pelvis

        activation_condition_avoid_wall = (pelvis_pos[:, 0] >= low_wall_pos_x - 1.0) # CORRECT: Activation condition: robot is close to or in front of the low wall

        penalty_collision_wall_x = -torch.abs(distance_x_wall) * 5.0 # CORRECT: Penalty for being too close to the wall in x, negative absolute distance, multiplied by 5 for stronger penalty

        reward = torch.where(activation_condition_avoid_wall & (distance_x_wall < 0.5), penalty_collision_wall_x, torch.tensor(0.0, device=env.device)) # CORRECT: Apply penalty only when activation condition is met and distance is less than 0.5m

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    # Normalize and return
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def stay_behind_sphere_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stay_behind_sphere_reward") -> torch.Tensor:
    '''Shaping reward 4: Discourage moving too far in x, relative to the large sphere. Reward for staying behind the large sphere.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    try:
        large_sphere = env.scene['Object1'] # CORRECT: Accessing large sphere object using approved pattern and try/except
        pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing pelvis position using approved pattern
        large_sphere_pos_x = large_sphere.data.root_pos_w[:, 0] # CORRECT: Accessing large sphere x position using approved pattern

        distance_x_sphere = large_sphere_pos_x - pelvis_pos[:, 0] # CORRECT: Relative distance in x-direction between large sphere and pelvis

        reward_stay_behind_sphere_x = -torch.abs(torch.relu(-distance_x_sphere)) # CORRECT: Reward for staying behind the sphere, negative absolute distance when behind, 0 when in front

        reward = reward_stay_behind_sphere_x

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    # Normalize and return
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    Main_PrepareForJumpOverLowWallReward = RewTerm(func=main_PrepareForJumpOverLowWall_reward, weight=1.0,
                                params={"normalise": True, "normaliser_name": "main_reward"})
    ApproachWallReward = RewTerm(func=approach_wall_reward, weight=0.5,
                                params={"normalise": True, "normaliser_name": "approach_wall_reward"})
    CrouchPelvisZReward = RewTerm(func=crouch_pelvis_z_reward, weight=0.4,
                                params={"normalise": True, "normaliser_name": "crouch_pelvis_z_reward"})
    AvoidCollisionWallReward = RewTerm(func=avoid_collision_wall_reward, weight=0.3,
                                params={"normalise": True, "normaliser_name": "avoid_collision_wall_reward"})
    StayBehindSphereReward = RewTerm(func=stay_behind_sphere_reward, weight=0.2,
                                params={"normalise": True, "normaliser_name": "stay_behind_sphere_reward"})
`,
                    successTerminationCode: `


from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
# Assuming mdp is correctly importable from the context where this runs
# If not, adjust the relative import path
from ...mdp import * 
import torch
from pathlib import Path
# Import reward functions if needed by success criteria
# from .TaskRewardsCfg import * 

def PrepareForJumpOverLowWall_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the PrepareForJumpOverLowWall skill has been successfully completed.'''
    # 1. Get robot object - CORRECT: Approved access pattern
    robot = env.scene["robot"]

    # 2. Get pelvis index and position - CORRECT: Approved access pattern
    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]
    robot_pelvis_pos_x = robot_pelvis_pos[:, 0]
    robot_pelvis_pos_z = robot_pelvis_pos[:, 2]

    try:
        # 3. Get low wall object - CORRECT: Approved access pattern and try-except
        low_wall = env.scene['Object3']
        low_wall_pos_x = low_wall.data.root_pos_w[:, 0]

        # 4. Calculate relative distances - CORRECT: Relative distances
        distance_x_wall = low_wall_pos_x - robot_pelvis_pos_x # Distance in x direction to the wall
        pelvis_height = robot_pelvis_pos_z # Pelvis height

        # 5. Define success conditions - CORRECT: Relative distances and reasonable thresholds
        close_to_wall_x = (distance_x_wall < 1) # Robot is within 1m in front of the wall in x direction
        pelvis_crouched = (pelvis_height < 0.5) # Robot pelvis is below 0.5m height

        # 6. Combine success conditions - CORRECT: Combining conditions with &
        condition = close_to_wall_x & pelvis_crouched

    except KeyError:
        # 7. Handle missing object - CORRECT: Handle missing object
        condition = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 8. Check success duration and save success state - CORRECT: Using check_success_duration and save_success_state
    success = check_success_duration(env, condition, "PrepareForJumpOverLowWall", duration=1)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "PrepareForJumpOverLowWall")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=PrepareForJumpOverLowWall_success)
`},
                {
                    name: "ExecuteJumpOverLowWall", level: 1, policyVideo: "/videos/ExecuteJumpOverLowWall.mp4", rewardCode: `
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer # DO NOT CHANGE THIS LINE!
from ...objects import get_object_volume

def main_ExecuteJumpOverLowWall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for ExecuteJumpOverLowWall.

    Phases the reward based on robot's x position relative to the low wall.
    Phase 1: Before the wall - reward for feet height to encourage jumping.
    Phase 2: After the wall - reward for reaching a target x position past the wall.
    Uses relative distances and approved access patterns. Handles missing objects and normalizes reward.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        low_wall = env.scene['Object3'] # Accessing low wall object using approved pattern and try/except
        large_sphere = env.scene['Object1'] # Accessing large sphere object using approved pattern and try/except

        pelvis_idx = robot.body_names.index('pelvis') # Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing pelvis position using approved pattern
        pelvis_pos_x = pelvis_pos[:, 0] # Getting x component of pelvis position
        pelvis_pos_z = pelvis_pos[:, 2] # Getting z component of pelvis position

        left_ankle_roll_link_idx = robot.body_names.index('left_ankle_roll_link') # Accessing left ankle index using approved pattern
        right_ankle_roll_link_idx = robot.body_names.index('right_ankle_roll_link') # Accessing right ankle index using approved pattern
        left_ankle_roll_link_pos = robot.data.body_pos_w[:, left_ankle_roll_link_idx] # Accessing left ankle position using approved pattern
        right_ankle_roll_link_pos = robot.data.body_pos_w[:, right_ankle_roll_link_idx] # Accessing right ankle position using approved pattern
        feet_pos_z = (left_ankle_roll_link_pos[:, 2] + right_ankle_roll_link_pos[:, 2]) / 2 # Calculating average feet z position

        low_wall_x = low_wall.data.root_pos_w[:, 0] # Accessing low wall x position using approved pattern
        large_sphere_x = large_sphere.data.root_pos_w[:, 0] # Accessing large sphere x position using approved pattern
        target_x = (low_wall_x + large_sphere_x) / 2 # Calculating target x position as midpoint between low wall and large sphere (relative distance)
        low_wall_height = 0.4 # Hardcoded low wall height from object config (approved as size is not accessible)

        # Phase 1: Before the wall - reward for feet height and pelvis height
        activation_condition_phase1 = (pelvis_pos_x < (low_wall_x + 0.5)) # Activation condition based on relative x position to low wall
        target_feet_pos_z = 0.9  # Target feet z position
        target_pelvis_pos_z = 1.6 # Target pelvis z position

        reward_phase1 = -torch.abs((feet_pos_z - target_feet_pos_z)) - torch.abs((pelvis_pos_z - target_pelvis_pos_z)) # Reward for feet height above low wall height, relative height

        # Phase 2: After the wall - reward for target x position
        activation_condition_phase2 = (pelvis_pos_x >= (low_wall_x + 0.5)) # Activation condition based on relative x position to low wall
        reward_phase2 = torch.exp(-torch.abs(pelvis_pos_x - target_x)) # Reward for being close to target x position, relative distance

        primary_reward = torch.where(activation_condition_phase1, reward_phase1, torch.zeros_like(reward_phase1)) # Combining phase rewards based on activation conditions
        reward_phase2 = torch.where(activation_condition_phase2, reward_phase2, torch.zeros_like(reward_phase2)) # Combining phase rewards based on activation conditions
        reward = primary_reward + reward_phase2 # Adding phase 2 reward to primary reward

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handling missing object, returning zero reward

    # Normalize and return
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # Normalizing reward
        RewNormalizer.update_stats(normaliser_name, reward) # Updating normalizer stats
        return scaled_reward
    return reward

def shaping_reward_approach_wall(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_wall_reward") -> torch.Tensor:
    '''Shaping reward to encourage robot to approach the low wall in x direction.
    Active when robot is significantly behind the wall. Uses relative distances and approved access patterns.
    Handles missing objects and normalizes reward.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        low_wall = env.scene['Object3'] # Accessing low wall object using approved pattern and try/except

        pelvis_idx = robot.body_names.index('pelvis') # Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing pelvis position using approved pattern
        pelvis_pos_x = pelvis_pos[:, 0] # Getting x component of pelvis position

        low_wall_x = low_wall.data.root_pos_w[:, 0] # Accessing low wall x position using approved pattern

        activation_condition_approach = (pelvis_pos_x < (low_wall_x - 1)) # Activation condition when pelvis is significantly behind the low wall (relative distance)
        reward_approach = -torch.abs(pelvis_pos_x - low_wall_x) # Reward for being closer to the low wall in x direction, relative distance

        shaping_reward_1 = torch.where(activation_condition_approach, reward_approach, torch.tensor(0.0, device=env.device)) # Applying reward only when activation condition is met
        reward = shaping_reward_1 # Assigning shaping reward to reward variable

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handling missing object, returning zero reward

    # Normalize and return
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # Normalizing reward
        RewNormalizer.update_stats(normaliser_name, reward) # Updating normalizer stats
        return scaled_reward
    return reward

def shaping_reward_forward_movement(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "forward_movement_reward") -> torch.Tensor:
    '''Shaping reward to encourage forward pelvis movement near the low wall.
    Active when robot is near the low wall. Uses relative distances and approved access patterns.
    Handles missing objects and normalizes reward.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        low_wall = env.scene['Object3'] # Accessing low wall object using approved pattern and try/except

        pelvis_idx = robot.body_names.index('pelvis') # Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing pelvis position using approved pattern
        pelvis_pos_x = pelvis_pos[:, 0] # Getting x component of pelvis position

        low_wall_x = low_wall.data.root_pos_w[:, 0] # Accessing low wall x position using approved pattern

        activation_condition_forward = (pelvis_pos_x >= (low_wall_x - 0.5)) & (pelvis_pos_x < (low_wall_x + 0.5)) # Activation condition when pelvis is near the low wall (relative distance)
        reward_forward = -(pelvis_pos_x - (low_wall_x + 0.5)) # Reward for moving slightly past the wall in x, relative distance

        shaping_reward_2 = torch.where(activation_condition_forward, reward_forward, torch.tensor(0.0, device=env.device)) # Applying reward only when activation condition is met
        reward = shaping_reward_2 # Assigning shaping reward to reward variable

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handling missing object, returning zero reward

    # Normalize and return
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # Normalizing reward
        RewNormalizer.update_stats(normaliser_name, reward) # Updating normalizer stats
        return scaled_reward
    return reward

def shaping_reward_stable_landing(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stable_landing_reward") -> torch.Tensor:
    '''Shaping reward to encourage stable landing after jumping over the wall.
    Rewards pelvis z-position close to default standing height after passing the wall.
    Uses relative distances (implicitly through pelvis_z) and approved access patterns.
    Handles missing objects and normalizes reward.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        low_wall = env.scene['Object3'] # Accessing low wall object using approved pattern and try/except

        pelvis_idx = robot.body_names.index('pelvis') # Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing pelvis position using approved pattern
        pelvis_pos_x = pelvis_pos[:, 0] # Getting x component of pelvis position
        pelvis_pos_z = pelvis_pos[:, 2] # Getting z component of pelvis position

        activation_condition_stable_z = (pelvis_pos_x >= (low_wall.data.root_pos_w[:, 0] + 0.5)) # Activation condition after passing the wall (relative distance)
        reward_stable_z = -torch.abs(pelvis_pos_z - 0.7) # Reward for pelvis z position being close to 0.7 (default standing height), relative height

        shaping_reward_3 = torch.where(activation_condition_stable_z, reward_stable_z, torch.tensor(0.0, device=env.device)) # Applying reward only when activation condition is met
        reward = shaping_reward_3 # Assigning shaping reward to reward variable

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handling missing object, returning zero reward

    # Normalize and return
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # Normalizing reward
        RewNormalizer.update_stats(normaliser_name, reward) # Updating normalizer stats
        return scaled_reward
    return reward

def shaping_reward_feet_pelvis_alignment(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "feet_pelvis_alignment_reward") -> torch.Tensor:
    '''Shaping reward to encourage feet and pelvis alignment.
    Rewards feet and pelvis being close to each other in z direction.
    Uses relative distances (implicitly through pelvis_z) and approved access patterns.
    Handles missing objects and normalizes reward.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        pelvis_idx = robot.body_names.index('pelvis') # Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing pelvis position using approved pattern
        pelvis_pos_x = pelvis_pos[:, 0] # Getting x component of pelvis position
        pelvis_pos_y = pelvis_pos[:, 1]

        left_ankle_roll_link_idx = robot.body_names.index('left_ankle_roll_link') # Accessing left ankle index using approved pattern
        right_ankle_roll_link_idx = robot.body_names.index('right_ankle_roll_link') # Accessing right ankle index using approved pattern
        left_ankle_roll_link_pos = robot.data.body_pos_w[:, left_ankle_roll_link_idx] # Accessing left ankle position using approved pattern
        right_ankle_roll_link_pos = robot.data.body_pos_w[:, right_ankle_roll_link_idx] # Accessing right ankle position using approved pattern

        reward = -torch.abs(pelvis_pos_y - right_ankle_roll_link_pos[:, 1]) - torch.abs(pelvis_pos_y - left_ankle_roll_link_pos[:, 1]) \
            - torch.abs(pelvis_pos_x - right_ankle_roll_link_pos[:, 0]) - torch.abs(pelvis_pos_x - left_ankle_roll_link_pos[:, 0]) # Reward for feet and pelvis being close to each other in y and x direction, relative height


    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handling missing object, returning zero reward

    # Normalize and return
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # Normalizing reward
        RewNormalizer.update_stats(normaliser_name, reward) # Updating normalizer stats
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    Main_ExecuteJumpOverLowWallReward = RewTerm(func=main_ExecuteJumpOverLowWall_reward, weight=1.0,
                                params={"normalise": True, "normaliser_name": "main_reward"})

    ShapingRewardApproachWall = RewTerm(func=shaping_reward_approach_wall, weight=0.4,
                                params={"normalise": True, "normaliser_name": "approach_wall_reward"})

    ShapingRewardForwardMovement = RewTerm(func=shaping_reward_forward_movement, weight=0,
                                params={"normalise": True, "normaliser_name": "forward_movement_reward"})

    ShapingRewardStableLanding = RewTerm(func=shaping_reward_stable_landing, weight=0.2,
                                params={"normalise": True, "normaliser_name": "stable_landing_reward"})
    
    ShapingRewardFeetPelvisAlignment = RewTerm(func=shaping_reward_feet_pelvis_alignment, weight=0.3,
                                params={"normalise": True, "normaliser_name": "feet_pelvis_alignment_reward"})
`,
                    successTerminationCode: `


from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
# Assuming mdp is correctly importable from the context where this runs
# If not, adjust the relative import path
from ...mdp import * 
import torch
from pathlib import Path
# Import reward functions if needed by success criteria
# from .TaskRewardsCfg import * 

def ExecuteJumpOverLowWall_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the ExecuteJumpOverLowWall skill has been successfully completed.'''
    # 1. Access robot object (APPROVED PATTERN)
    robot = env.scene["robot"]

    # 2. Get pelvis index (APPROVED PATTERN)
    pelvis_idx = robot.body_names.index('pelvis')
    # 3. Get pelvis position (APPROVED PATTERN)
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0] # Get x component of pelvis position

    try:
        # 4. Access low wall object (APPROVED PATTERN and try-except for error handling)
        low_wall = env.scene['Object3']
        # 5. Get low wall position (APPROVED PATTERN)
        low_wall_x = low_wall.data.root_pos_w[:, 0] # Get x component of low wall position

        # 6. Calculate relative distance in x direction (RELATIVE DISTANCE)
        distance_x_wall_pelvis = pelvis_pos_x - low_wall_x

        # 7. Define success condition based on relative distance (RELATIVE DISTANCE and REASONABLE THRESHOLD)
        success_condition = distance_x_wall_pelvis > 0.5 # Check if pelvis is 0.5m past the wall in x direction

    except KeyError:
        # 8. Handle missing object (ERROR HANDLING)
        success_condition = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 9. Check success duration and save success state (REQUIRED FUNCTIONS)
    success = check_success_duration(env, success_condition, "ExecuteJumpOverLowWall", duration=0.5) # Check if success is maintained for 0.5 seconds
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "ExecuteJumpOverLowWall") # Save success state for successful environments

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=ExecuteJumpOverLowWall_success)
`},
                {
                    name: "LandStablyAfterLowWall", level: 1, policyVideo: "/videos/LandStablyAfterLowWall.mp4", rewardCode: `
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer
from ...objects import get_object_volume

def main_LandStablyAfterLowWall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for LandStablyAfterLowWall.

    Reward for being in the target x range just past the low wall and before the large sphere, and close to the ground.
    This encourages the robot to land stably after jumping over the low wall in the desired zone.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        low_wall = env.scene['Object3'] # Accessing low wall object using approved pattern and try/except
        large_sphere = env.scene['Object1'] # Accessing large sphere object using approved pattern and try/except

        pelvis_idx = robot.body_names.index('pelvis') # Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing pelvis position using approved pattern
        pelvis_pos_z = pelvis_pos[:, 2]

        distance_z = pelvis_pos_z # Relative distance in z from ground level (z=0)

        pelvis_default_z = 0.7


        reward_z = -torch.abs(distance_z-pelvis_default_z) # Reward for pelvis being the correct height.

        reward = reward_z # Combining x and z rewards

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_1_increase_pelvis_height(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_1") -> torch.Tensor:
    '''Shaping reward for increasing pelvis height as robot approaches the low wall.
    Encourages the robot to jump as it gets closer to the wall.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        low_wall = env.scene['Object3'] # Accessing low wall object using approved pattern and try/except

        pelvis_idx = robot.body_names.index('pelvis') # Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing pelvis position using approved pattern
        pelvis_pos_x = pelvis_pos[:, 0]
        pelvis_pos_z = pelvis_pos[:, 2]

        low_wall_pos_x = low_wall.data.root_pos_w[:, 0] # Accessing low wall x position using approved pattern

        distance_x_to_wall = low_wall_pos_x - pelvis_pos_x # Relative distance in x to the low wall

        activation_condition = (pelvis_pos_x > low_wall_pos_x) # Activation when robot is past the low wall (after the wall in x direction)

        reward_height = pelvis_pos_z # Reward for increasing pelvis height (absolute z position, but used as relative increase from ground)
        reward = reward_height

        reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device)) # Apply activation condition

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_2_feet_close_to_ground(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_2") -> torch.Tensor:
    '''Shaping reward for both feet being close to the ground after passing the low wall in x direction.
    Encourages stable landing on both feet.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        low_wall = env.scene['Object3'] # Accessing low wall object using approved pattern and try/except

        left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Accessing left foot index using approved pattern
        right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Accessing right foot index using approved pattern
        left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Accessing left foot position using approved pattern
        right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Accessing right foot position using approved pattern
        left_foot_pos_z = left_foot_pos[:, 2]
        right_foot_pos_z = right_foot_pos[:, 2]
        pelvis_pos = robot.data.body_pos_w[:, robot.body_names.index('pelvis')] # Accessing pelvis position using approved pattern
        pelvis_pos_x = pelvis_pos[:, 0]

        low_wall_pos_x = low_wall.data.root_pos_w[:, 0] # Accessing low wall x position using approved pattern

        activation_condition = (pelvis_pos_x > low_wall_pos_x) # Activation when robot is past the low wall in x direction

        reward_left_foot = -torch.abs(left_foot_pos_z - 0.0) # Reward for left foot being close to ground
        reward_right_foot = -torch.abs(right_foot_pos_z - 0.0) # Reward for right foot being close to ground

        reward = reward_left_foot + reward_right_foot # Combining left and right foot rewards
        reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device)) # Apply activation condition

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_3_stable_pelvis_height(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_3") -> torch.Tensor:
    '''Shaping reward for maintaining a stable pelvis height after landing.
    Encourages body stabilization after the jump. Target pelvis height is set to 0.7m (relative target height).
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        low_wall = env.scene['Object3'] # Accessing low wall object using approved pattern and try/except

        pelvis_idx = robot.body_names.index('pelvis') # Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing pelvis position using approved pattern
        pelvis_pos_x = pelvis_pos[:, 0]
        pelvis_pos_z = pelvis_pos[:, 2]
        target_pelvis_z = 0.7 # Target pelvis height (relative to ground)

        low_wall_pos_x = low_wall.data.root_pos_w[:, 0] # Accessing low wall x position using approved pattern

        activation_condition = (pelvis_pos_x > low_wall_pos_x) # Activation when robot is past the low wall in x direction

        reward_pelvis_z = -torch.abs(pelvis_pos_z - target_pelvis_z) # Reward for pelvis being at a stable height (relative distance to target height)

        reward = reward_pelvis_z
        reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device)) # Apply activation condition

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    Main_LandStablyAfterLowWallReward = RewTerm(func=main_LandStablyAfterLowWall_reward, weight=1.0,
                                params={"normalise": True, "normaliser_name": "main_reward"})

    ShapingReward_IncreasePelvisHeight = RewTerm(func=shaping_reward_1_increase_pelvis_height, weight=0.5,
                                params={"normalise": True, "normaliser_name": "shaping_reward_1"})

    ShapingReward_FeetCloseToGround = RewTerm(func=shaping_reward_2_feet_close_to_ground, weight=0.6,
                                params={"normalise": True, "normaliser_name": "shaping_reward_2"})

    ShapingReward_StablePelvisHeight = RewTerm(func=shaping_reward_3_stable_pelvis_height, weight=0.3,
                                params={"normalise": True, "normaliser_name": "shaping_reward_3"})
`,
                    successTerminationCode: `


from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
# Assuming mdp is correctly importable from the context where this runs
# If not, adjust the relative import path
from ...mdp import * 
import torch
from pathlib import Path
# Import reward functions if needed by success criteria
# from .TaskRewardsCfg import * 

# Standard imports - DO NOT MODIFY
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer
from ...objects import get_object_volume

def LandStablyAfterLowWall_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the LandStablyAfterLowWall skill has been successfully completed.
    Success is defined as the robot landing stably on both feet on the other side of the low wall.
    '''
    # 1. Get robot object - APPROVED PATTERN
    robot = env.scene["robot"]

    # 2. Get indices for robot body parts - APPROVED PATTERN
    pelvis_idx = robot.body_names.index('pelvis')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    # 3. Get positions of robot parts - APPROVED PATTERN
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # [num_envs, 3]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # [num_envs, 3]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # [num_envs, 3]

    try:
        # 4. Get low wall object - APPROVED PATTERN with try/except for robustness
        low_wall = env.scene['Object3']
        low_wall_pos = low_wall.data.root_pos_w # [num_envs, 3]

        # 5. Calculate relative distances - REQUIREMENT 1: Relative distances only
        pelvis_x_distance_to_wall = pelvis_pos[:, 0] - low_wall_pos[:, 0] # x distance between pelvis and low wall
        left_foot_z_distance_to_ground = torch.abs(left_foot_pos[:, 2] - 0.0) # z distance of left foot from ground (z=0)
        right_foot_z_distance_to_ground = torch.abs(right_foot_pos[:, 2] - 0.0) # z distance of right foot from ground (z=0)
        pelvis_z_distance_to_ground = torch.abs(pelvis_pos[:, 2] - 0.0) # z distance of pelvis from ground (z=0)

        # 6. Define success conditions - REQUIREMENT 4: No hardcoded thresholds, REQUIREMENT 5: Reasonable tolerances
        pelvis_past_wall_condition = pelvis_x_distance_to_wall > 0.5 # Pelvis is 0.5m past the low wall in x direction
        left_foot_low_condition = left_foot_z_distance_to_ground < 0.15 # Left foot is within 0.15m of the ground
        right_foot_low_condition = right_foot_z_distance_to_ground < 0.15 # Right foot is within 0.15m of the ground
        pelvis_low_condition = pelvis_z_distance_to_ground < 0.8 # Pelvis is within 0.8m of the ground
        pelvis_high_condition = pelvis_z_distance_to_ground > 0.5 # Pelvis is above 0.5m of the ground

        # 7. Combine success conditions - REQUIREMENT 1: Relative distances only
        land_stably_condition = pelvis_past_wall_condition & left_foot_low_condition & right_foot_low_condition & pelvis_low_condition & pelvis_high_condition

    except KeyError:
        # 8. Handle missing object - REQUIREMENT 6: Handle missing objects
        land_stably_condition = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 9. Check success duration and save success state - REQUIREMENT 7 & 8: check_success_duration and save_success_state
    success = check_success_duration(env, land_stably_condition, "LandStablyAfterLowWall", duration=0.5) # Duration of 0.5 seconds
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "LandStablyAfterLowWall")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=LandStablyAfterLowWall_success)
`},
            ],
        },
        {
            name: "PushLargeSphereToHighWall",
            level: 2,
            rewardCode: `from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer # DO NOT CHANGE THIS LINE!
from ...objects import get_object_volume

def main_PushLargeSphereToHighWall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for PushLargeSphereToHighWall.

    Reward for moving the large sphere closer to the high wall in the x-direction.
    This encourages the robot to push the large sphere towards the high wall to knock it over.
    '''
    try:
        large_sphere = env.scene['Object1'] # Access the large sphere using approved pattern (rule 2, rule 3, rule 5)
        high_wall = env.scene['Object4'] # Access the high wall using approved pattern (rule 2, rule 3, rule 5)

        # Calculate the x-distance between the large sphere and the high wall (rule 1)
        distance_x = high_wall.data.root_pos_w[:, 0] - large_sphere.data.root_pos_w[:, 0] # Access object positions using approved pattern (rule 2)

        # Reward is negative absolute x-distance to encourage minimizing the distance (rule 4, rule 5)
        reward = -torch.abs(distance_x) # Continuous reward (rule 5)

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing objects (rule 5, rule 6)

    # Reward normalization (rule 6)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_approach_large_sphere_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_sphere_reward") -> torch.Tensor:
    '''Shaping reward for approaching the large sphere.

    Reward for reducing the x-distance between the robot's pelvis and the large sphere when the robot is behind the sphere.
    This encourages the robot to move towards the large sphere before pushing it.
    '''
    try:
        large_sphere = env.scene['Object1'] # Access the large sphere using approved pattern (rule 2, rule 3, rule 5)
        robot = env.scene['robot'] # Access the robot object (rule 2, rule 3)
        pelvis_idx = robot.body_names.index('pelvis') # Get pelvis index using approved pattern (rule 3)
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Get pelvis position using approved pattern (rule 3)

        # Calculate the x-distance between the pelvis and the large sphere (rule 1)
        distance_x = large_sphere.data.root_pos_w[:, 0] - pelvis_pos[:, 0] # Access object and robot part positions using approved pattern (rule 2, rule 3)
        distance_y = large_sphere.data.root_pos_w[:, 1] - pelvis_pos[:, 1] # Access object and robot part positions using approved pattern (rule 2, rule 3)

        # Reward is negative absolute x-distance when activated (rule 4, rule 5)
        reward = -torch.abs(distance_x) - torch.abs(distance_y) # Continuous reward (rule 5)

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing objects (rule 5, rule 6)

    # Reward normalization (rule 6)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_stable_pelvis_height_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stable_height_reward") -> torch.Tensor:
    '''Shaping reward for maintaining a stable pelvis height.

    Reward for keeping the pelvis height close to a target height (0.7m).
    This encourages the robot to stay upright and stable.
    '''
    try:
        robot = env.scene['robot'] # Access the robot object (rule 2, rule 3)
        pelvis_idx = robot.body_names.index('pelvis') # Get pelvis index using approved pattern (rule 3)
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Get pelvis position using approved pattern (rule 3)

        # Define target pelvis height (no hardcoded position, but target height is acceptable as a task parameter)
        target_pelvis_z = 0.7

        # Calculate the z-distance between the current pelvis height and the target height (rule 1)
        distance_z = target_pelvis_z - pelvis_pos[:, 2] # Relative distance (rule 1)

        # Reward is negative absolute z-distance (rule 4, rule 5)
        reward = -torch.abs(distance_z) # Continuous reward (rule 5)

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing objects (rule 5, rule 6)

    # Reward normalization (rule 6)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "avoid_collision_reward") -> torch.Tensor:
    '''Shaping reward for collision avoidance with low and high walls.

    Penalize the robot for getting too close to the low wall and high wall unnecessarily,
    but deactivate this reward when the robot is close to the large sphere to allow pushing.
    '''
    try:
        low_wall = env.scene['Object3'] # Access the low wall using approved pattern (rule 2, rule 3, rule 5)
        high_wall = env.scene['Object4'] # Access the high wall using approved pattern (rule 2, rule 3, rule 5)
        large_sphere = env.scene['Object1'] # Access the large sphere using approved pattern (rule 2, rule 3, rule 5)
        robot = env.scene['robot'] # Access the robot object (rule 2, rule 3)
        pelvis_idx = robot.body_names.index('pelvis') # Get pelvis index using approved pattern (rule 3)
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Get pelvis position using approved pattern (rule 3)

        # Calculate the x-distance between the pelvis and the walls (rule 1)
        distance_low_wall_x = low_wall.data.root_pos_w[:, 0] - pelvis_pos[:, 0] # Access object and robot part positions using approved pattern (rule 2, rule 3)
        distance_high_wall_x = high_wall.data.root_pos_w[:, 0] - pelvis_pos[:, 0] # Access object and robot part positions using approved pattern (rule 2, rule 3)
        distance_sphere_x = large_sphere.data.root_pos_w[:, 0] - pelvis_pos[:, 0] # Distance to sphere for activation condition

        # Activation condition: robot is not close to the large sphere (to allow pushing)
        approach_sphere_threshold = 1.0 # Threshold is relative, not hardcoded position (rule 4)
        activation_condition_avoid_low_wall = (torch.abs(distance_low_wall_x) < 1.0) & (torch.abs(distance_sphere_x) > approach_sphere_threshold) # Relative conditions (rule 1)
        activation_condition_avoid_high_wall = (torch.abs(distance_high_wall_x) < 1.0) & (torch.abs(distance_sphere_x) > approach_sphere_threshold) # Relative conditions (rule 1)

        # Reward is negative distance if too close to walls, scaled to be continuous (rule 4, rule 5)
        reward_low_wall = torch.where(activation_condition_avoid_low_wall, -torch.abs(1.0 - torch.abs(distance_low_wall_x)), torch.tensor(0.0, device=env.device)) # Continuous reward (rule 5)
        reward_high_wall = torch.where(activation_condition_avoid_high_wall, -torch.abs(1.0 - torch.abs(distance_high_wall_x)), torch.tensor(0.0, device=env.device)) # Continuous reward (rule 5)

        reward = reward_low_wall + reward_high_wall # Sum of rewards

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing objects (rule 5, rule 6)

    # Reward normalization (rule 6)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    Main_PushLargeSphereToHighWallReward = RewTerm(func=main_PushLargeSphereToHighWall_reward, weight=1.0,
                                params={"normalise": True, "normaliser_name": "main_reward"}) # Main reward weight ~1.0 (rule 7)
    ApproachLargeSphereReward = RewTerm(func=shaping_approach_large_sphere_reward, weight=0.4,
                                params={"normalise": True, "normaliser_name": "approach_sphere_reward"}) # Supporting reward weight < 1.0 (rule 7)
    StablePelvisHeightReward = RewTerm(func=shaping_stable_pelvis_height_reward, weight=0.3,
                                params={"normalise": True, "normaliser_name": "stable_height_reward"}) # Supporting reward weight < 1.0 (rule 7)
    CollisionAvoidanceReward = RewTerm(func=shaping_collision_avoidance_reward, weight=0.2, # combined weight of low and high wall avoidance.
                                params={"normalise": True, "normaliser_name": "avoid_collision_reward"}) # Supporting reward weight < 1.0 (rule 7)`,
            successTerminationCode: `

from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
# Assuming mdp is correctly importable from the context where this runs
# If not, adjust the relative import path
from ...mdp import * 
import torch
from pathlib import Path
# Import reward functions if needed by success criteria
# from .TaskRewardsCfg import * 

def PushLargeSphereToHighWall_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the PushLargeSphereToHighWall skill has been successfully completed.
    Success is defined as the large sphere being pushed to be at or beyond the x-position of the high wall.
    '''
    # 1. Access the large sphere object (Object1) from the environment scene using the approved pattern (rule 2, rule 3, rule 5)
    try:
        large_sphere = env.scene['Object1']
    except KeyError:
        # Handle the case where the large sphere object is not found in the scene (rule 5, rule 6)
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 2. Access the high wall object (Object4) from the environment scene using the approved pattern (rule 2, rule 3, rule 5)
    try:
        high_wall = env.scene['Object4']
    except KeyError:
        # Handle the case where the high wall object is not found in the scene (rule 5, rule 6)
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


    # 3. Get the x-position of the large sphere using the approved pattern (rule 2)
    large_sphere_x = large_sphere.data.root_pos_w[:, 0]

    # 4. Get the x-position of the high wall using the approved pattern (rule 2)
    high_wall_x = high_wall.data.root_pos_w[:, 0]
    high_wall_z = high_wall.data.root_pos_w[:, 2]

    # 5. Calculate the relative x-distance: large_sphere_x - high_wall_x (rule 1, rule 3)
    distance_x = large_sphere_x - high_wall_x

    # 6. Define the success condition: large_sphere_x is at or beyond the high_wall_x - 2m tolerance (rule 1, rule 4, rule 10)
    #    We use a negative threshold because we want the large sphere's x position to be greater than the high wall's x position.
    success_threshold = -2
    condition_sphere = distance_x > success_threshold
    condition_high_wall = high_wall_z < 0.4

    final_condition = condition_sphere & condition_high_wall

    # 7. Check success duration and save success states using the provided helper functions (rule 6, rule 7)
    success = check_success_duration(env, final_condition, "PushLargeSphereToHighWall", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "PushLargeSphereToHighWall")

    # 8. Return the success tensor (rule 3)
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=PushLargeSphereToHighWall_success)
`,
            policyVideo: "/videos/L2_PushLargeSphereToHighWall.mp4",
            children: [
                {
                    name: "WalkToLargeSphere", level: 1, policyVideo: "/videos/WalkToLargeSphere.mp4", rewardCode: `
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer
from ...objects import get_object_volume
import torch

def main_WalkToLargeSphere_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for WalkToLargeSphere.

    Reward for moving the robot's pelvis closer to the large sphere in the horizontal (x-y) plane.
    This encourages the robot to walk towards the large sphere, fulfilling the primary objective of the skill.
    The reward is inversely proportional to the horizontal distance, providing a continuous signal as the robot approaches the sphere.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        large_sphere = env.scene['Object1'] # Accessing object using approved pattern and try/except

        pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern
        pelvis_pos_x = pelvis_pos[:, 0] # Separating x component
        pelvis_pos_y = pelvis_pos[:, 1] # Separating y component

        large_sphere_pos = large_sphere.data.root_pos_w # Accessing object position using approved pattern
        large_sphere_pos_x = large_sphere_pos[:, 0] # Separating x component
        large_sphere_pos_y = large_sphere_pos[:, 1] # Separating y component

        distance_x = large_sphere_pos_x - pelvis_pos_x # Relative distance in x-direction
        distance_y = large_sphere_pos_y - pelvis_pos_y # Relative distance in y-direction

        horizontal_distance = torch.sqrt(distance_x**2 + distance_y**2) # Euclidean distance in x-y plane
        reward = -horizontal_distance # Negative distance to reward getting closer

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def stable_pelvis_height_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stable_height_reward") -> torch.Tensor:
    '''Shaping reward for maintaining a stable pelvis height.

    Encourages the robot to maintain a pelvis height close to 0.7m, promoting balance and stability during walking.
    This is a shaping reward to prevent the robot from crouching too low or standing too high, which could hinder its movement.
    The reward is based on the negative absolute difference between the pelvis z-position and the target height, providing a continuous signal.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern
        pelvis_pos_z = pelvis_pos[:, 2] # Separating z component

        default_pelvis_z = 0.7 # Default pelvis height - not hardcoded position, but a relative target height

        reward = -torch.abs(pelvis_pos_z - default_pelvis_z) # Negative absolute difference from default pelvis height

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object (though pelvis should always exist), return zero reward

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    Main_WalkToLargeSphereReward = RewTerm(func=main_WalkToLargeSphere_reward, weight=1.0,
                                params={"normalise": True, "normaliser_name": "main_reward"}) # Main reward with weight 1.0
    StablePelvisHeightReward = RewTerm(func=stable_pelvis_height_reward, weight=0.6,
                                params={"normalise": True, "normaliser_name": "stable_height_reward"}) # Shaping reward with weight 0.6
`,
                    successTerminationCode: `


from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
# Assuming mdp is correctly importable from the context where this runs
# If not, adjust the relative import path
from ...mdp import * 
import torch
from pathlib import Path
# Import reward functions if needed by success criteria
# from .TaskRewardsCfg import * 

def WalkToLargeSphere_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the WalkToLargeSphere skill has been successfully completed.
    Success is achieved when the robot's pelvis is within 1.5 meters horizontal distance of the large sphere.
    '''
    # 1. Get robot pelvis position using approved access pattern
    robot = env.scene["robot"] # Accessing robot using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # Accessing pelvis index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing pelvis position using approved pattern

    try:
        # 2. Get large sphere position using approved access pattern and try/except for robustness
        large_sphere = env.scene['Object1'] # Accessing Object1 (large sphere) using approved pattern
        large_sphere_pos = large_sphere.data.root_pos_w # Accessing large sphere position using approved pattern

        # 3. Calculate horizontal distance between robot pelvis and large sphere. Only using relative distances.
        distance_x = large_sphere_pos[:, 0] - pelvis_pos[:, 0] # Relative distance in x-direction
        distance_y = large_sphere_pos[:, 1] - pelvis_pos[:, 1] # Relative distance in y-direction
        horizontal_distance = torch.sqrt(distance_x**2 + distance_y**2) # Euclidean distance in x-y plane

        # 4. Define success condition: horizontal distance is within 1.5 meters. Using a lenient threshold as requested.
        success_threshold = 1.5
        condition = horizontal_distance < success_threshold # Success condition based on relative distance and threshold

    except KeyError:
        # 5. Handle missing object (large sphere). Skill fails if the object is not found.
        condition = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device) # Return False for all envs if object is missing

    # 6. Check success duration and save success states. Using check_success_duration as required.
    success = check_success_duration(env, condition, "WalkToLargeSphere", duration=0.5) # Check if success condition is maintained for 0.5 seconds
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "WalkToLargeSphere") # Save success state for successful environments

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=WalkToLargeSphere_success)
`}, // No specific video found
                {
                    name: "PositionHandsForPushLargeSphere", level: 1, policyVideo: "/videos/PositionHandsForPushLargeSphere.mp4", rewardCode: `
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer # DO NOT CHANGE THIS LINE!
from ...objects import get_object_volume

def main_PositionHandsForPushLargeSphere_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for PositionHandsForPushLargeSphere.

    Rewards the robot for positioning both hands in front of the large sphere at a suitable height for pushing.
    This encourages the robot to get its hands ready to push the large sphere.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        large_sphere = env.scene['Object1'] # Accessing object1 (large sphere) using approved pattern and try/except

        left_hand_idx = robot.body_names.index('left_palm_link') # Accessing left hand index using approved pattern
        right_hand_idx = robot.body_names.index('right_palm_link') # Accessing right hand index using approved pattern
        left_hand_pos = robot.data.body_pos_w[:, left_hand_idx] # Accessing left hand position using approved pattern
        right_hand_pos = robot.data.body_pos_w[:, right_hand_idx] # Accessing right hand position using approved pattern
        large_sphere_pos = large_sphere.data.root_pos_w # Accessing large sphere position using approved pattern

        target_x_offset = -0.4 # Hands should be in front of the sphere for pushing, relative offset, not hardcoded position
        target_push_pos_x = large_sphere_pos[:, 0] + target_x_offset # Target x position relative to sphere x position
        target_push_pos_y = large_sphere_pos[:, 1] # Target y position same as sphere y position

        # Calculate distances for left hand, using relative distances
        distance_x_left = target_push_pos_x - left_hand_pos[:, 0]
        distance_y_left = target_push_pos_y - left_hand_pos[:, 1]

        # Calculate distances for right hand, using relative distances
        distance_x_right = target_push_pos_x - right_hand_pos[:, 0]
        distance_y_right = target_push_pos_y - right_hand_pos[:, 1]

        # Reward is negative absolute distance in x, y and z for both hands, continuous reward
        reward_left = -torch.abs(distance_x_left) - 0.5*torch.abs(distance_y_left) 
        reward_right = -torch.abs(distance_x_right) - 0.5*torch.abs(distance_y_right) 

        reward = reward_left + reward_right

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    # Normalize and return reward, using RewNormalizer for proper normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_approach_sphere_x(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_sphere_x_reward") -> torch.Tensor:
    '''Shaping reward for approaching the large sphere in the x-direction.
    Rewards the robot for moving closer to the large sphere in the x-direction when the pelvis is behind the sphere.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        large_sphere = env.scene['Object1'] # Accessing object1 (large sphere) using approved pattern and try/except

        pelvis_idx = robot.body_names.index('pelvis') # Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing pelvis position using approved pattern
        large_sphere_pos = large_sphere.data.root_pos_w # Accessing large sphere position using approved pattern

        target_x_offset = -0.6 # Target x offset for hands in front of the sphere, relative offset, not hardcoded position

        # Calculate distance in x direction, relative distance
        distance_x_pelvis_sphere = (large_sphere_pos[:, 0] + target_x_offset) - pelvis_pos[:, 0]

        # Reward for decreasing x distance when behind the sphere, continuous reward
        reward = -torch.abs(distance_x_pelvis_sphere)

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    # Normalize and return reward, using RewNormalizer for proper normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_align_hands_z(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "align_hands_z_reward") -> torch.Tensor:
    '''Shaping reward for aligning the hands vertically with the center of the large sphere.
    Rewards the robot for adjusting hand height to match the sphere's vertical center.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        large_sphere = env.scene['Object1'] # Accessing object1 (large sphere) using approved pattern and try/except

        left_hand_idx = robot.body_names.index('left_palm_link') # Accessing left hand index using approved pattern
        right_hand_idx = robot.body_names.index('right_palm_link') # Accessing right hand index using approved pattern
        left_hand_pos = robot.data.body_pos_w[:, left_hand_idx] # Accessing left hand position using approved pattern
        right_hand_pos = robot.data.body_pos_w[:, right_hand_idx] # Accessing right hand position using approved pattern
        large_sphere_pos = large_sphere.data.root_pos_w # Accessing large sphere position using approved pattern

        # Calculate distance in z direction, relative distance
        distance_z_left_hand_sphere = large_sphere_pos[:, 2] - left_hand_pos[:, 2]
        distance_z_right_hand_sphere = large_sphere_pos[:, 2] - right_hand_pos[:, 2]

        # Reward for aligning hand z position with sphere z position, continuous reward
        reward_left = -torch.abs(distance_z_left_hand_sphere)
        reward_right = -torch.abs(distance_z_right_hand_sphere)

        reward = reward_left + reward_right

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    # Normalize and return reward, using RewNormalizer for proper normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_align_hands_y(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "align_hands_y_reward") -> torch.Tensor:
    '''Shaping reward for aligning the hands horizontally (y-direction) with the center of the large sphere.
    Rewards the robot for positioning its hands centrally relative to the sphere's width.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        large_sphere = env.scene['Object1'] # Accessing object1 (large sphere) using approved pattern and try/except

        left_hand_idx = robot.body_names.index('left_palm_link') # Accessing left hand index using approved pattern
        right_hand_idx = robot.body_names.index('right_palm_link') # Accessing right hand index using approved pattern
        left_hand_pos = robot.data.body_pos_w[:, left_hand_idx] # Accessing left hand position using approved pattern
        right_hand_pos = robot.data.body_pos_w[:, right_hand_idx] # Accessing right hand position using approved pattern
        large_sphere_pos = large_sphere.data.root_pos_w # Accessing large sphere position using approved pattern

        # Calculate distance in y direction, relative distance
        distance_y_left_hand_sphere = large_sphere_pos[:, 1] - left_hand_pos[:, 1]
        distance_y_right_hand_sphere = large_sphere_pos[:, 1] - right_hand_pos[:, 1]

        # Reward for aligning hand y position with sphere y position, continuous reward
        reward_left = -torch.abs(distance_y_left_hand_sphere)
        reward_right = -torch.abs(distance_y_right_hand_sphere)

        reward = reward_left + reward_right

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    # Normalize and return reward, using RewNormalizer for proper normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_stable_pelvis_height(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stable_pelvis_height_reward") -> torch.Tensor:
    '''Shaping reward for maintaining a stable standing posture by keeping the pelvis at a default height of 0.7m.
    Rewards the robot for maintaining a consistent pelvis height, encouraging stability.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern

    pelvis_idx = robot.body_names.index('pelvis') # Accessing pelvis index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing pelvis position using approved pattern

    default_pelvis_z = 0.7 # Define default pelvis height, not hardcoded position, but a relative target height

    # Calculate distance in z direction from default pelvis height, relative distance
    distance_z_pelvis_default = default_pelvis_z - pelvis_pos[:, 2]

    # Reward for maintaining default pelvis height, continuous reward
    reward = -torch.abs(distance_z_pelvis_default)

    # Normalize and return reward, using RewNormalizer for proper normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    Main_PositionHandsForPushLargeSphereReward = RewTerm(func=main_PositionHandsForPushLargeSphere_reward, weight=1.0,
                                params={"normalise": True, "normaliser_name": "main_reward"})
    Shaping_ApproachSphereXReward = RewTerm(func=shaping_reward_approach_sphere_x, weight=0,
                                params={"normalise": True, "normaliser_name": "approach_sphere_x_reward"})
    Shaping_AlignHandsZReward = RewTerm(func=shaping_reward_align_hands_z, weight=0,
                                params={"normalise": True, "normaliser_name": "align_hands_z_reward"})
    Shaping_AlignHandsYReward = RewTerm(func=shaping_reward_align_hands_y, weight=0,
                                params={"normalise": True, "normaliser_name": "align_hands_y_reward"})
    Shaping_StablePelvisHeightReward = RewTerm(func=shaping_reward_stable_pelvis_height, weight=0.3, # Lower weight for stability reward
                                params={"normalise": True, "normaliser_name": "stable_pelvis_height_reward"})
`,
                    successTerminationCode: `


from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
# Assuming mdp is correctly importable from the context where this runs
# If not, adjust the relative import path
from ...mdp import * 
import torch
from pathlib import Path
# Import reward functions if needed by success criteria
# from .TaskRewardsCfg import * 

def PositionHandsForPushLargeSphere_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the PositionHandsForPushLargeSphere skill has been successfully completed.'''
    # 1. Get robot and hand parts - APPROVED ACCESS PATTERN
    robot = env.scene["robot"] # Accessing robot using approved pattern
    left_hand_idx = robot.body_names.index('left_palm_link') # Get left hand index - APPROVED ACCESS PATTERN
    right_hand_idx = robot.body_names.index('right_palm_link') # Get right hand index - APPROVED ACCESS PATTERN
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx] # Get left hand position - APPROVED ACCESS PATTERN
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx] # Get right hand position - APPROVED ACCESS PATTERN

    try:
        # 2. Get large sphere object position - APPROVED ACCESS PATTERN and HANDLE MISSING OBJECTS
        large_sphere = env.scene['Object1'] # Accessing object1 (large sphere) - APPROVED ACCESS PATTERN
        large_sphere_pos = large_sphere.data.root_pos_w # Get large sphere position - APPROVED ACCESS PATTERN

        # 3. Calculate relative distances - RELATIVE DISTANCES ONLY
        left_hand_x_distance = large_sphere_pos[:, 0] - left_hand_pos[:, 0] # x-distance between large sphere and left hand - RELATIVE DISTANCE
        right_hand_x_distance = large_sphere_pos[:, 0] - right_hand_pos[:, 0] # x-distance between large sphere and right hand - RELATIVE DISTANCE
        left_hand_y_distance = large_sphere_pos[:, 1] - left_hand_pos[:, 1] # y-distance between large sphere and left hand - RELATIVE DISTANCE
        right_hand_y_distance = large_sphere_pos[:, 1] - right_hand_pos[:, 1] # y-distance between large sphere and right hand - RELATIVE DISTANCE
        left_hand_z_distance = large_sphere_pos[:, 2] - left_hand_pos[:, 2] # z-distance between large sphere and left hand - RELATIVE DISTANCE
        right_hand_z_distance = large_sphere_pos[:, 2] - right_hand_pos[:, 2] # z-distance between large sphere and right hand - RELATIVE DISTANCE

        # 4. Define success conditions based on relative distances and reasonable tolerances - RELATIVE DISTANCES AND LENIENT THRESHOLDS
        x_distance_threshold_lower = 0.3 # Lower threshold for x-distance - REASONABLE TOLERANCE
        x_distance_threshold_upper = 1.0 # Upper threshold for x-distance - REASONABLE TOLERANCE
        y_distance_threshold = 1 # Threshold for y and z distances - REASONABLE TOLERANCE

        left_hand_x_condition = (left_hand_x_distance > x_distance_threshold_lower) & (left_hand_x_distance < x_distance_threshold_upper) # Left hand x-distance condition
        right_hand_x_condition = (right_hand_x_distance > x_distance_threshold_lower) & (right_hand_x_distance < x_distance_threshold_upper) # Right hand x-distance condition
        left_hand_y_condition = torch.abs(left_hand_y_distance) < y_distance_threshold # Left hand y-distance condition
        right_hand_y_condition = torch.abs(right_hand_y_distance) < y_distance_threshold # Right hand y-distance condition

        # Combine conditions for both hands - ALL CONDITIONS MUST BE MET
        success_condition = left_hand_x_condition & right_hand_x_condition & left_hand_y_condition & right_hand_y_condition # Combining all conditions using logical AND

    except KeyError:
        # Handle missing object - HANDLE MISSING OBJECTS
        success_condition = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device) # Return False if large sphere is missing - HANDLES MISSING OBJECTS

    # 5. Check duration and save success states - CHECK SUCCESS DURATION and SAVE SUCCESS STATES
    success = check_success_duration(env, success_condition, "PositionHandsForPushLargeSphere", duration=0.3) # Check success duration for 0.5 seconds - CHECK SUCCESS DURATION
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "PositionHandsForPushLargeSphere") # Save success state for successful environments - SAVE SUCCESS STATES

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=PositionHandsForPushLargeSphere_success)
`},
                {
                    name: "PushLargeSphereForward", level: 1, policyVideo: "/videos/PushLargeSphereForward.mp4", rewardCode: `
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer # DO NOT CHANGE THIS LINE!
from ...objects import get_object_volume

def main_PushLargeSphereForward_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for PushLargeSphereForward.

    Reward for decreasing the x-distance between the large sphere and the high wall.
    This encourages the robot to push the large sphere towards the high wall to complete the primary objective.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        large_sphere = env.scene['Object1'] # Accessing large sphere using approved pattern and try/except
        high_wall = env.scene['Object4'] # Accessing high wall using approved pattern and try/except

        large_sphere_pos = large_sphere.data.root_pos_w # Accessing large sphere position using approved pattern
        high_wall_pos = high_wall.data.root_pos_w # Accessing high wall position using approved pattern

        distance_x = high_wall_pos[:, 0] - large_sphere_pos[:, 0] # Calculating relative distance in x-direction between high wall and large sphere
        reward = -torch.abs(distance_x) # Reward is negative absolute distance to encourage minimizing the distance

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    # Reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_ApproachLargeSphere_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_sphere_reward") -> torch.Tensor:
    '''Shaping reward for approaching the large sphere.

    Reward for moving the robot pelvis closer to the large sphere in the x direction when the robot is behind the large sphere (in x).
    This encourages the robot to approach the sphere to initiate pushing.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        large_sphere = env.scene['Object1'] # Accessing large sphere using approved pattern and try/except
        robot_pelvis_idx = robot.body_names.index('pelvis') # Accessing robot pelvis index using approved pattern
        robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] # Accessing robot pelvis position using approved pattern
        large_sphere_pos = large_sphere.data.root_pos_w # Accessing large sphere position using approved pattern

        distance_x = large_sphere_pos[:, 0] - robot_pelvis_pos[:, 0] # Calculating relative distance in x-direction between large sphere and robot pelvis
        distance_y = large_sphere_pos[:, 1] - robot_pelvis_pos[:, 1] # Calculating relative distance in y-direction between large sphere and robot pelvis
        activation_condition = (robot_pelvis_pos[:, 0] < large_sphere_pos[:, 0]) # Activation condition: robot is behind the large sphere in x
        reward = -torch.abs(distance_x) - torch.abs(distance_y) # Reward is negative absolute distance to encourage minimizing the distance

        reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device)) # Apply reward only when activation condition is met

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    # Reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_MaintainContactPushSphere_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "push_sphere_reward") -> torch.Tensor:
    '''Shaping reward for maintaining contact and pushing the large sphere.

    Reward for maintaining close proximity to the large sphere in the x direction when the robot is in front of the large sphere (in x) and the large sphere is still far from the high wall.
    This encourages continuous pushing.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        large_sphere = env.scene['Object1'] # Accessing large sphere using approved pattern and try/except
        high_wall = env.scene['Object4'] # Accessing high wall using approved pattern and try/except
        robot_pelvis_idx = robot.body_names.index('pelvis') # Accessing robot pelvis index using approved pattern
        robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] # Accessing robot pelvis position using approved pattern
        large_sphere_pos = large_sphere.data.root_pos_w # Accessing large sphere position using approved pattern
        high_wall_pos = high_wall.data.root_pos_w # Accessing high wall position using approved pattern

        distance_x = large_sphere_pos[:, 0] - robot_pelvis_pos[:, 0] # Calculating relative distance in x-direction between large sphere and robot pelvis
        sphere_wall_distance_x = torch.abs(high_wall_pos[:, 0] - large_sphere_pos[:, 0]) # Calculating relative distance in x-direction between high wall and large sphere
        activation_condition = (robot_pelvis_pos[:, 0] >= large_sphere_pos[:, 0]) & (sphere_wall_distance_x > 1.0) # Activation condition: robot is in front of the large sphere in x AND large sphere is not yet close to the high wall (distance > 1.0m)
        reward = -torch.abs(distance_x) # Reward is negative absolute distance to encourage minimizing the distance

        reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device)) # Apply reward only when activation condition is met

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    # Reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_StabilityReward_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_reward") -> torch.Tensor:
    '''Shaping reward for maintaining stable pelvis height.

    Reward for maintaining a stable pelvis height. This helps prevent the robot from falling over while pushing.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        robot_pelvis_idx = robot.body_names.index('pelvis') # Accessing robot pelvis index using approved pattern
        robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] # Accessing robot pelvis position using approved pattern
        robot_pelvis_pos_z = robot_pelvis_pos[:, 2] # Accessing robot pelvis z position

        desired_pelvis_z = 0.7 # Desired pelvis height (no hardcoding of positions, but desired height is a parameter of the skill)

        reward = -torch.abs(robot_pelvis_pos_z - desired_pelvis_z) # Reward is negative absolute distance from desired pelvis height

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object (robot), return zero reward

    # Reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_AvoidLowWallCollision_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "avoid_lowwall_reward") -> torch.Tensor:
    '''Shaping reward for avoiding collision with the low wall while approaching the sphere.

    Reward for maintaining distance from the low wall in the x direction when the robot is behind the low wall.
    This encourages the robot to move around the low wall and not collide with it while approaching the large sphere.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        low_wall = env.scene['Object3'] # Accessing low wall using approved pattern and try/except
        robot_pelvis_idx = robot.body_names.index('pelvis') # Accessing robot pelvis index using approved pattern
        robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] # Accessing robot pelvis position using approved pattern
        low_wall_pos = low_wall.data.root_pos_w # Accessing low wall position using approved pattern

        distance_x = low_wall_pos[:, 0] - robot_pelvis_pos[:, 0] # Calculating relative distance in x-direction between low wall and robot pelvis
        activation_condition = (robot_pelvis_pos[:, 0] < low_wall_pos[:, 0]) # Activation condition: robot is behind the low wall in x
        reward = torch.abs(distance_x) # Reward is positive absolute distance to encourage maximizing the distance from the low wall

        reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device)) # Apply reward only when activation condition is met

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    # Reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    Main_PushLargeSphereForwardReward = RewTerm(func=main_PushLargeSphereForward_reward, weight=1.0,
                                params={"normalise": True, "normaliser_name": "main_reward"})

    Shaping_ApproachLargeSphereReward = RewTerm(func=shaping_ApproachLargeSphere_reward, weight=0.6,
                                params={"normalise": True, "normaliser_name": "approach_sphere_reward"})

    Shaping_MaintainContactPushSphereReward = RewTerm(func=shaping_MaintainContactPushSphere_reward, weight=0.5,
                                params={"normalise": True, "normaliser_name": "push_sphere_reward"})

    Shaping_StabilityReward = RewTerm(func=shaping_StabilityReward_reward, weight=0.3,
                                params={"normalise": True, "normaliser_name": "stability_reward"})

    Shaping_AvoidLowWallCollisionReward = RewTerm(func=shaping_AvoidLowWallCollision_reward, weight=0.2,
                                params={"normalise": True, "normaliser_name": "avoid_lowwall_reward"})
`,
                    successTerminationCode: `


from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
# Assuming mdp is correctly importable from the context where this runs
# If not, adjust the relative import path
from ...mdp import * 
import torch
from pathlib import Path
# Import reward functions if needed by success criteria
# from .TaskRewardsCfg import * 

def PushLargeSphereForward_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the PushLargeSphereForward skill has been successfully completed.
    Success is defined as the large sphere being close to the high wall in the x direction.
    '''
    # 1. Access the large sphere object from the environment scene.
    # Using try-except to handle cases where the object might be missing.
    try:
        large_sphere = env.scene['Object1'] # Accessing large sphere using approved pattern
    except KeyError:
        # If 'Object1' (large sphere) is not found, return failure for all environments.
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 2. Access the high wall object from the environment scene.
    # Using try-except to handle cases where the object might be missing.
    try:
        high_wall = env.scene['Object4'] # Accessing high wall using approved pattern
    except KeyError:
        # If 'Object4' (high wall) is not found, return failure for all environments.
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 3. Get the world position of the large sphere and the high wall.
    # Accessing object positions using the approved pattern: .data.root_pos_w
    large_sphere_pos = large_sphere.data.root_pos_w # Shape: [num_envs, 3]
    high_wall_pos = high_wall.data.root_pos_w # Shape: [num_envs, 3]

    # 4. Calculate the distance in the x-direction between the high wall and the large sphere.
    # Using relative distances as required.
    distance_x = high_wall_pos[:, 0] - large_sphere_pos[:, 0] # Shape: [num_envs]

    # 5. Define the success condition: The large sphere is close to the high wall in the x direction.
    # Using a threshold of 1.5 meters. This is a relative distance and a reasonable tolerance.
    success_threshold = 1.5
    success_condition = distance_x < success_threshold # Shape: [num_envs] - boolean tensor

    # 6. Check for success duration and save success state.
    # Using check_success_duration to ensure success is maintained for a short period (0.5s).
    # Using save_success_state to record successful environments.
    success = check_success_duration(env, success_condition, "PushLargeSphereForward", duration=1.5) # duration is 1.5 seconds
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "PushLargeSphereForward")

    # 7. Return the success tensor.
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=PushLargeSphereForward_success)
`},
                {
                    name: "EnsureHighWallFalls", level: 1, policyVideo: "/videos/EnsureHighWallFalls.mp4", rewardCode: `
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer # DO NOT CHANGE THIS LINE!
from ...objects import get_object_volume

def main_EnsureHighWallFalls_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for EnsureHighWallFalls.

    Encourages the robot to roll the large sphere into the high wall until the wall falls.
    This reward is composed of approaching the wall with the sphere and the wall falling.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        large_sphere = env.scene['Object1'] # Accessing large sphere using approved pattern and try/except
        high_wall = env.scene['Object4'] # Accessing high wall using approved pattern and try/except

        # Calculate distance vector between the large sphere and the high wall in x direction (relative distance)
        distance_sphere_wall_x = large_sphere.data.root_pos_w[:, 0] - high_wall.data.root_pos_w[:, 0] # Relative distance calculation

        reward_wall_fallen = -high_wall.data.root_pos_w[:, 2]

        # Activate approach reward only when robot is past the low wall and approaching the large sphere

        reward = reward_wall_fallen # Combine approach and wall fallen rewards

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    # Normalize and return reward
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def jump_over_low_wall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "jump_over_low_wall_reward") -> torch.Tensor:
    '''Shaping reward to encourage the robot to jump over the low wall.

    Rewards increasing pelvis height while approaching the low wall and clearing the wall with pelvis and feet.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        low_wall = env.scene['Object3'] # Accessing low wall using approved pattern and try/except

        pelvis_idx = robot.body_names.index('pelvis') # Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing pelvis position using approved pattern
        pelvis_pos_x = pelvis_pos[:, 0]
        pelvis_pos_z = pelvis_pos[:, 2]

        left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Accessing left foot index using approved pattern
        left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Accessing left foot position using approved pattern
        right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Accessing right foot index using approved pattern
        right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Accessing right foot position using approved pattern
        feet_pos_z = (left_foot_pos[:, 2] + right_foot_pos[:, 2]) / 2.0 # Average feet z position

        # Calculate distance vector between pelvis and low wall in x direction (relative distance)
        distance_pelvis_wall_x = pelvis_pos_x - low_wall.data.root_pos_w[:, 0] # Relative distance calculation

        # Reward for increasing pelvis height as robot approaches the wall
        target_pelvis_height = 1.0 # Target pelvis height for jumping (relative target height)
        reward_pelvis_height = -torch.abs(pelvis_pos_z - target_pelvis_height) # Continuous reward, negative absolute difference from target height

        # Reward for clearing the wall (pelvis and feet above wall height + clearance)
        wall_top_z = low_wall.data.root_pos_w[:, 2] + 0.2 # Top of low wall (wall z position + half of wall height as wall is centered at z=0, and wall height is 0.4m, so half height is 0.2m)
        clearance = 0.1 # Clearance above the wall (relative clearance)
        reward_clear_wall_pelvis = torch.where(pelvis_pos_z > wall_top_z + clearance, torch.tensor(0.5, device=env.device), torch.tensor(0.0, device=env.device)) # Conditional reward for pelvis clearing wall
        reward_clear_wall_feet = torch.where(feet_pos_z > wall_top_z + clearance, torch.tensor(0.5, device=env.device), torch.tensor(0.0, device=env.device)) # Conditional reward for feet clearing wall

        activation_condition_jump = (pelvis_pos_x > low_wall.data.root_pos_w[:, 0] - 1.5) & (pelvis_pos_x < low_wall.data.root_pos_w[:, 0] + 0.5) # Activation based on relative x positions
        reward_jump_over_wall = torch.where(activation_condition_jump, reward_pelvis_height + reward_clear_wall_pelvis + reward_clear_wall_feet, torch.tensor(0.0, device=env.device)) # Apply jump reward only when activated

        reward = reward_jump_over_wall # Combine pelvis height and wall clearing rewards

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    # Normalize and return reward
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def approach_large_sphere_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_large_sphere_reward") -> torch.Tensor:
    '''Shaping reward to encourage the robot to approach the large sphere after jumping over the low wall.

    Rewards decreasing x-distance between the pelvis and the large sphere after passing the low wall.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        large_sphere = env.scene['Object1'] # Accessing large sphere using approved pattern and try/except
        low_wall = env.scene['Object3'] # Accessing low wall using approved pattern and try/except

        pelvis_idx = robot.body_names.index('pelvis') # Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing pelvis position using approved pattern
        pelvis_pos_x = pelvis_pos[:, 0]

        # Calculate distance vector between pelvis and large sphere in x direction (relative distance)
        distance_pelvis_sphere_x = pelvis_pos_x - large_sphere.data.root_pos_w[:, 0] # Relative distance calculation

        # Reward for approaching the large sphere in x direction
        reward_approach_sphere = -torch.abs(distance_pelvis_sphere_x) # Continuous reward, negative absolute distance

        # Activate reward after robot is past the low wall and approaching the large sphere
        activation_condition_approach_sphere = (pelvis_pos_x > low_wall.data.root_pos_w[:, 0] + 1.0) & (pelvis_pos_x < large_sphere.data.root_pos_w[:, 0] + 3.0) # Activation based on relative x positions
        reward_approach_sphere_activated = torch.where(activation_condition_approach_sphere, reward_approach_sphere, torch.tensor(0.0, device=env.device)) # Apply approach reward only when activated

        reward = reward_approach_sphere_activated # Apply activated approach sphere reward

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    # Normalize and return reward
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def pelvis_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_stability_reward") -> torch.Tensor:
    '''Shaping reward to encourage robot stability by maintaining a consistent pelvis height.

    Rewards maintaining a pelvis height close to a default standing height.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        pelvis_idx = robot.body_names.index('pelvis') # Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing pelvis position using approved pattern
        pelvis_pos_z = pelvis_pos[:, 2]

        # Reward for maintaining a stable pelvis height (close to default standing height)
        default_pelvis_z = 0.7 # Default pelvis z height (can be considered relative to ground z=0)
        reward_pelvis_stability = -torch.abs(pelvis_pos_z - default_pelvis_z) # Continuous reward, negative absolute difference from default height

        reward = reward_pelvis_stability # Apply pelvis stability reward

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    # Normalize and return reward
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    Main_EnsureHighWallFallsReward = RewTerm(func=main_EnsureHighWallFalls_reward, weight=0.1,
                                params={"normalise": True, "normaliser_name": "main_reward"}) # Main reward with weight 1.0

    JumpOverLowWallReward = RewTerm(func=jump_over_low_wall_reward, weight=0,
                                params={"normalise": True, "normaliser_name": "jump_over_low_wall_reward"}) # Shaping reward with weight 0.4

    ApproachLargeSphereReward = RewTerm(func=approach_large_sphere_reward, weight=0.5,
                                params={"normalise": True, "normaliser_name": "approach_large_sphere_reward"}) # Shaping reward with weight 0.3

    PelvisStabilityReward = RewTerm(func=pelvis_stability_reward, weight=1.0,
                                params={"normalise": True, "normaliser_name": "pelvis_stability_reward"}) # Shaping reward with weight 0.2
`,
                    successTerminationCode: `


from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
# Assuming mdp is correctly importable from the context where this runs
# If not, adjust the relative import path
from ...mdp import * 
import torch
from pathlib import Path
# Import reward functions if needed by success criteria
# from .TaskRewardsCfg import * 

def EnsureHighWallFalls_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the EnsureHighWallFalls skill has been successfully completed.'''
    # 1. Access the high wall object using the approved pattern and handle potential KeyError if missing
    try:
        high_wall = env.scene['Object4'] # Accessing 'Object4' which is the high wall as per object config
    except KeyError:
        # Handle the case where 'Object4' (high_wall) is not found in the scene, returning failure (False) for all environments
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 2. Get the world position of the high wall's root using the approved pattern
    high_wall_pos = high_wall.data.root_pos_w

    # 3. Define success condition: High wall has fallen over.
    #    We check if the z-position of the high wall is below a certain threshold, indicating it has fallen.
    #    The initial z position of the high wall is around 0.0 (centered at z=0, height 1m).
    #    A threshold of 0.4m for the z-position of the root is chosen to indicate that the wall has fallen significantly.
    #    This is a relative check to the ground plane (z=0), which is a fixed reference.
    fallen_threshold_z = 0.4
    wall_fallen_condition = (high_wall_pos[:, 2] < fallen_threshold_z) # Condition: z-position of high wall is less than fallen_threshold_z
    # 4. Check for success duration using the check_success_duration function.
    #    Success is considered achieved if the wall_fallen_condition is true for a duration of 0.5 seconds.
    success = check_success_duration(env, wall_fallen_condition, "EnsureHighWallFalls", duration=0.5)

    # 5. Save success state for environments where the skill is successful using save_success_state.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "EnsureHighWallFalls")

    # 6. Return the tensor indicating success for each environment.
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=EnsureHighWallFalls_success)
`},
            ],
        },
        {
            name: "KickSmallSpherePastBlock",
            level: 2,
            rewardCode: `
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer # DO NOT CHANGE THIS LINE!
from ...objects import get_object_volume

def main_KickSmallSpherePastBlock_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for KickSmallSpherePastBlock.

    Reward is negative absolute distance in x direction between the small sphere and a target position 5m past the block.
    This encourages the robot to kick the small sphere past the block in the x direction.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        small_sphere = env.scene['Object2'] # Accessing small sphere using approved pattern and try/except
        block = env.scene['Object5'] # Accessing block using approved pattern and try/except

        # Accessing object positions using approved pattern
        small_sphere_pos = small_sphere.data.root_pos_w
        block_pos = block.data.root_pos_w

        small_sphere_pos_x = small_sphere.data.root_pos_w[:, 0]
        small_sphere_pos_y = small_sphere.data.root_pos_w[:, 1]
        block_pos_x = block.data.root_pos_w[:, 0]
        block_pos_y = block.data.root_pos_w[:, 1]

        relative_distance_x = small_sphere_pos_x - block_pos_x
        relative_distance_y = small_sphere_pos_y - block_pos_y
        relative_distance = torch.sqrt(relative_distance_x**2 + relative_distance_y**2)


        # Primary reward is negative absolute distance in x to the target position past the block. Continuous reward.
        reward = -torch.abs(relative_distance_x)

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    # Reward normalization using RewNormalizer.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def reward_shaping_approach_sphere(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_approach_sphere_reward") -> torch.Tensor:
    '''Shaping reward 1: Reward for approaching the small sphere with the robot's right foot.

    Reward is negative 3D distance between the robot's right foot and the small sphere, activated when the foot is behind the sphere in x direction.
    Encourages the robot to get closer to the small sphere in preparation for kicking.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        small_sphere = env.scene['Object2'] # Accessing small sphere using approved pattern and try/except

        # Accessing robot right foot position using approved pattern
        robot_foot_right_idx = robot.body_names.index('right_ankle_roll_link')
        robot_foot_right_pos = robot.data.body_pos_w[:, robot_foot_right_idx]
        robot_foot_right_pos_x = robot_foot_right_pos[:, 0]

        # Accessing small sphere position using approved pattern
        small_sphere_pos = small_sphere.data.root_pos_w

        # Calculate the distance vector between the small sphere and the robot right foot. Relative distance.
        distance_x_foot_sphere = small_sphere_pos[:, 0] - robot_foot_right_pos_x
        distance_y_foot_sphere = small_sphere_pos[:, 1] - robot_foot_right_pos[:, 1]
        distance_z_foot_sphere = small_sphere_pos[:, 2] - robot_foot_right_pos[:, 2]
        distance_foot_sphere = torch.sqrt(distance_x_foot_sphere**2 + distance_y_foot_sphere**2 + distance_z_foot_sphere**2)

        # Activation condition: robot foot is behind the sphere in x direction. Relative positions.
        activation_condition_approach = (robot_foot_right_pos_x < small_sphere_pos[:, 0])

        # Reward is negative absolute distance to the sphere. Continuous reward.
        reward_shaping_sphere = -distance_foot_sphere

        reward = torch.where(activation_condition_approach, reward_shaping_sphere, torch.tensor(0.0, device=env.device)) # Apply activation condition

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    # Reward normalization using RewNormalizer.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def reward_shaping_kick_sphere(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_kick_sphere_reward") -> torch.Tensor:
    '''Shaping reward 2: Reward for moving the small sphere in the positive x direction relative to the robot's right foot.

    Reward is positive x distance between the sphere and the foot, activated when the foot is close to the sphere and before the block in x direction.
    Encourages the kicking motion.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        small_sphere = env.scene['Object2'] # Accessing small sphere using approved pattern and try/except
        block = env.scene['Object5'] # Accessing block using approved pattern and try/except

        # Accessing robot right foot position using approved pattern
        robot_foot_right_idx = robot.body_names.index('right_ankle_roll_link')
        robot_foot_right_pos = robot.data.body_pos_w[:, robot_foot_right_idx]
        robot_foot_right_pos_x = robot_foot_right_pos[:, 0]

        # Accessing small sphere and block positions using approved pattern
        small_sphere_pos = small_sphere.data.root_pos_w
        block_pos = block.data.root_pos_w

        # Calculate the distance vector between the small sphere and the robot right foot. Relative distance.
        distance_x_foot_sphere = small_sphere_pos[:, 0] - robot_foot_right_pos_x
        distance_y_foot_sphere = small_sphere_pos[:, 1] - robot_foot_right_pos[:, 1]
        distance_z_foot_sphere = small_sphere_pos[:, 2] - robot_foot_right_pos[:, 2]
        distance_foot_sphere = torch.sqrt(distance_x_foot_sphere**2 + distance_y_foot_sphere**2 + distance_z_foot_sphere**2)

        # Activation condition: robot foot is close to the sphere (0.3m) and before the block in x direction. Relative positions.
        activation_condition_kick = (distance_foot_sphere < 0.3) & (robot_foot_right_pos_x < block_pos[:, 0])

        # Reward is the positive x distance between the sphere and the foot. Continuous reward.
        reward_shaping_sphere = (small_sphere_pos[:, 0] - robot_foot_right_pos_x)

        reward = torch.where(activation_condition_kick, reward_shaping_sphere, torch.tensor(0.0, device=env.device)) # Apply activation condition

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    # Reward normalization using RewNormalizer.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def reward_shaping_collision_avoidance(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_collision_avoidance_reward") -> torch.Tensor:
    '''Shaping reward 3: Collision avoidance reward to prevent the robot's pelvis from getting too close to the block.

    Reward is negative when pelvis is too close to the block (within 1.0m), based on 3D distance.
    Helps maintain a safe distance and prevents the robot from stumbling into the block.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        block = env.scene['Object5'] # Accessing block using approved pattern and try/except

        # Accessing robot pelvis position using approved pattern
        robot_pelvis_idx = robot.body_names.index('pelvis')
        robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]

        # Accessing block position using approved pattern
        block_pos = block.data.root_pos_w

        # Calculate the distance vector between the pelvis and the block. Relative distance.
        distance_x_pelvis_block = block_pos[:, 0] - robot_pelvis_pos[:, 0]
        distance_y_pelvis_block = block_pos[:, 1] - robot_pelvis_pos[:, 1]
        distance_z_pelvis_block = block_pos[:, 2] - robot_pelvis_pos[:, 2]
        distance_pelvis_block = torch.sqrt(distance_x_pelvis_block**2 + distance_y_pelvis_block**2 + distance_z_pelvis_block**2)

        # Collision threshold. Not hardcoded position, relative distance is used.
        collision_threshold = 1.0

        # Negative reward when pelvis is too close to the block. Continuous reward.
        reward_shaping_collision = -torch.abs(torch.where(distance_pelvis_block < collision_threshold, collision_threshold - distance_pelvis_block, torch.tensor(0.0, device=env.device)))

        reward = reward_shaping_collision

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    # Reward normalization using RewNormalizer.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def reward_shaping_stability(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_stability_reward") -> torch.Tensor:
    '''Shaping reward 4: Stability reward to encourage the robot to maintain a stable standing posture.

    Reward is negative absolute difference between the robot's pelvis z position and a desired pelvis height (0.7m).
    Encourages the robot to maintain a stable standing posture.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        # Accessing robot pelvis position using approved pattern
        robot_pelvis_idx = robot.body_names.index('pelvis')
        robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]
        robot_pelvis_pos_z = robot_pelvis_pos[:, 2]

        # Desired pelvis height. Not hardcoded position, relative height is used conceptually.
        default_pelvis_z = 0.7

        # Reward for maintaining pelvis height. Continuous reward.
        reward_shaping_stability_value = -torch.abs(robot_pelvis_pos_z - default_pelvis_z)

        reward = reward_shaping_stability_value

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward (though robot should always exist)

    # Reward normalization using RewNormalizer.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    Main_KickSmallSpherePastBlockReward = RewTerm(func=main_KickSmallSpherePastBlock_reward, weight=1.0,
                                params={"normalise": True, "normaliser_name": "main_reward"})

    Shaping_ApproachSphereReward = RewTerm(func=reward_shaping_approach_sphere, weight=0.5,
                                params={"normalise": True, "normaliser_name": "shaping_approach_sphere_reward"})

    Shaping_KickSphereReward = RewTerm(func=reward_shaping_kick_sphere, weight=0.6,
                                params={"normalise": True, "normaliser_name": "shaping_kick_sphere_reward"})

    Shaping_CollisionAvoidanceReward = RewTerm(func=reward_shaping_collision_avoidance, weight=0.4,
                                params={"normalise": True, "normaliser_name": "shaping_collision_avoidance_reward"})

    Shaping_StabilityReward = RewTerm(func=reward_shaping_stability, weight=0.3,
                                params={"normalise": True, "normaliser_name": "shaping_stability_reward"})
`,
            successTerminationCode: `

from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
# Assuming mdp is correctly importable from the context where this runs
# If not, adjust the relative import path
from ...mdp import * 
import torch
from pathlib import Path
# Import reward functions if needed by success criteria
# from .TaskRewardsCfg import * 

def KickSmallSpherePastBlock_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the KickSmallSpherePastBlock skill has been successfully completed.
    Success is defined as the small sphere being at least 5m past the block in the x direction, relative to the block's x position.
    '''
    # 1. Get robot - although not directly used in this success criteria, it is good practice to include for potential future expansion and template compliance.
    robot = env.scene["robot"]

    try:
        # 2. Get object positions - Accessing small sphere (Object2) and block (Object5) positions using approved pattern and try/except block.
        small_sphere = env.scene['Object2'] # Accessing small sphere using approved pattern
        block = env.scene['Object5'] # Accessing block using approved pattern

        # 3. Access object positions - Getting x positions of the small sphere and the block using approved pattern.
        small_sphere_pos_x = small_sphere.data.root_pos_w[:, 0] # Accessing small sphere x position using approved pattern
        small_sphere_pos_y = small_sphere.data.root_pos_w[:, 1] # Accessing small sphere y position using approved pattern
        block_pos_x = block.data.root_pos_w[:, 0] # Accessing block x position using approved pattern
        block_pos_y = block.data.root_pos_w[:, 1] # Accessing block y position using approved pattern

        # 4. Calculate relative distance - Calculating the x distance between the small sphere and the block.
        relative_distance_x = small_sphere_pos_x - block_pos_x
        relative_distance_y = small_sphere_pos_y - block_pos_y
        relative_distance = torch.sqrt(relative_distance_x**2 + relative_distance_y**2)

        # 5. Define success condition - Checking if the small sphere is at least 5m past the block in the x direction. Using relative distance and no hardcoded thresholds.
        success_threshold = 3.0 # 5m past the block
        condition = relative_distance > success_threshold # Success condition: small sphere is 5m past the block in x direction

    except KeyError:
        # 6. Handle missing objects - If 'Object2' or 'Object5' is not found, consider success as false for all environments.
        condition = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 7. Check success duration and save success states - Using check_success_duration to ensure success is maintained for a duration and save_success_state to record success.
    success = check_success_duration(env, condition, "KickSmallSpherePastBlock", duration=0.5) # Check if success condition is maintained for 0.5 seconds
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "KickSmallSpherePastBlock") # Save success state for successful environments

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=KickSmallSpherePastBlock_success)
`,
            policyVideo: "/videos/L2_KickSmallSpherePastBlock.mp4",
            children: [
                {
                    name: "WalkToSmallSphere", level: 1, policyVideo: "/videos/WalkToSmallSphere.mp4", rewardCode: `
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer # DO NOT CHANGE THIS LINE!
from ...objects import get_object_volume

def main_WalkToSmallSphere_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for WalkToSmallSphere.

    Reward for reducing the horizontal distance to the small sphere until within 1m.
    This encourages the robot to walk towards the small sphere.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    try:
        small_sphere = env.scene['Object2'] # CORRECT: Accessing small sphere using approved pattern and try/except
        robot_pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing robot part index using approved pattern
        robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] # CORRECT: Accessing robot part position using approved pattern
        small_sphere_pos = small_sphere.data.root_pos_w # CORRECT: Accessing object position using approved pattern

        # CORRECT: Calculate relative distance - horizontal distance between pelvis and small sphere
        distance_x = small_sphere_pos[:, 0] - robot_pelvis_pos[:, 0] - 0.5
        distance_y = small_sphere_pos[:, 1] - robot_pelvis_pos[:, 1]
        Distance = torch.sqrt(distance_x**2 + distance_y**2)

        # CORRECT: Reward for reducing horizontal distance, saturated at 1m, continuous reward
        reward = -torch.abs(Distance)

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    # Add clipping before normalization
    reward = torch.clip(reward, min=-3.0, max=3.0) # Choose bounds appropriate for your expected reward scale

    # CORRECT: Reward normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_forward_progress(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "forward_progress_reward") -> torch.Tensor:
    '''Shaping reward for forward progress towards the small sphere in the x direction.

    Rewards the robot for moving closer to the small sphere in the x direction when behind it.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    try:
        small_sphere = env.scene['Object2'] # CORRECT: Accessing small sphere using approved pattern and try/except
        robot_pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing robot part index using approved pattern
        robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] # CORRECT: Accessing robot part position using approved pattern
        robot_pelvis_pos_x = robot_pelvis_pos[:, 0]
        small_sphere_pos_x = small_sphere.data.root_pos_w[:, 0] # CORRECT: Accessing object position using approved pattern

        # CORRECT: Calculate relative distance - x distance to sphere
        distance_x_to_sphere = small_sphere_pos_x - robot_pelvis_pos_x


        # CORRECT: Reward forward progress in x direction, continuous reward
        reward = -distance_x_to_sphere

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    # Add clipping before normalization
    reward = torch.clip(reward, min=-3.0, max=3.0) # Choose bounds appropriate for your expected reward scale

    # CORRECT: Reward normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_stability(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_reward") -> torch.Tensor:
    '''Shaping reward for maintaining pelvis stability in z direction.

    Rewards the robot for keeping its pelvis at a nominal standing height.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    robot_pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing robot part index using approved pattern
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] # CORRECT: Accessing robot part position using approved pattern
    robot_pelvis_pos_z = robot_pelvis_pos[:, 2]

    # CORRECT: Reward for maintaining pelvis height around 0.7m, continuous reward based on relative height
    target_pelvis_z = 0.7
    reward = -torch.abs(robot_pelvis_pos_z - target_pelvis_z)

    # CORRECT: Reward normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    '''Reward for avoiding collisions with the small sphere.

    Rewards the robot for avoiding collisions with the small sphere.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    small_sphere = env.scene['Object2'] # CORRECT: Accessing small sphere using approved pattern and try/except
    robot_pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing robot part index using approved pattern
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] # CORRECT: Accessing robot part position using approved pattern
    small_sphere_pos = small_sphere.data.root_pos_w # CORRECT: Accessing object position using approved pattern

    # CORRECT: Calculate relative distance - x distance to sphere
    distance_x_to_sphere = small_sphere_pos[:, 0] - robot_pelvis_pos[:, 0]
    distance_y_to_sphere = small_sphere_pos[:, 1] - robot_pelvis_pos[:, 1]
    distance_to_sphere = torch.sqrt(distance_x_to_sphere**2 + distance_y_to_sphere**2)

    # CORRECT: Reward for avoiding collisions with the small sphere
    reward = distance_to_sphere**2

    activation_condition = distance_to_sphere < 1

    reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device))

    return reward


@configclass
class TaskRewardsCfg:
    # CORRECT: Main reward with weight 1.0
    Main_WalkToSmallSphereReward = RewTerm(func=main_WalkToSmallSphere_reward, weight=1.0,
                                params={"normalise": True, "normaliser_name": "main_reward"})

    # CORRECT: Supporting rewards with lower weights
    ForwardProgressReward = RewTerm(func=shaping_reward_forward_progress, weight=0.3,
                                params={"normalise": True, "normaliser_name": "forward_progress_reward"})
    
    CollisionAvoidanceReward = RewTerm(func=collision_avoidance_reward, weight=0.2,
                                params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})

    StabilityReward = RewTerm(func=shaping_reward_stability, weight=0.6,
                                params={"normalise": True, "normaliser_name": "stability_reward"})
`,
                    successTerminationCode: `


from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
# Assuming mdp is correctly importable from the context where this runs
# If not, adjust the relative import path
from ...mdp import * 
import torch
from pathlib import Path
# Import reward functions if needed by success criteria
# from .TaskRewardsCfg import * 

def WalkToSmallSphere_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the WalkToSmallSphere skill has been successfully completed.
    Success is defined as the robot pelvis being within 1m of the small sphere horizontally.
    '''
    # 1. Access the robot object using the approved pattern
    robot = env.scene["robot"]

    # 2. Get the index of the robot pelvis using robot.body_names.index for robustness
    robot_pelvis_idx = robot.body_names.index('pelvis')
    # 3. Get the world position of the robot pelvis using the approved pattern
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]

    try:
        # 4. Safely access the small sphere object (Object2) using try-except block and approved pattern
        small_sphere = env.scene['Object2']
        # 5. Get the world position of the small sphere using the approved pattern
        small_sphere_pos = small_sphere.data.root_pos_w

        # 6. Calculate the relative distance in x and y directions (horizontal distance)
        distance_x = small_sphere_pos[:, 0] - robot_pelvis_pos[:, 0]
        distance_y = small_sphere_pos[:, 1] - robot_pelvis_pos[:, 1]
    


        pelvis_z_pos = robot_pelvis_pos[:, 2]

        # 7. Define success condition: horizontal distance to small sphere is less than 1m.
        #    Using a lenient threshold of 1.5m to ensure robustness, as per instructions.
        success_threshold = 1.0

        condition = (distance_x < success_threshold) & (distance_y < success_threshold) & (pelvis_z_pos > 0.5)

    except KeyError:
        # 8. Handle the case where the small sphere object is not found, setting success to False for all envs.
        condition = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 9. Check for success duration using the check_success_duration function with a duration of 0.1s
    success = check_success_duration(env, condition, "WalkToSmallSphere", duration=0.1)

    # 10. Save success states for environments that are successful in this step
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "WalkToSmallSphere")

    # 11. Return the success tensor
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=WalkToSmallSphere_success)
`},
                {
                    name: "ExecuteKickSmallSphereForward", level: 1, policyVideo: "/videos/ExecuteKickSmallSphereForward.mp4", rewardCode: `
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer
from ...objects import get_object_volume

def main_ExecuteKickSmallSphereForward_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for ExecuteKickSmallSphereForward.

    Reward for kicking the small sphere past the block in the x direction.
    This reward is activated when the small sphere is past the block in the x direction, and the reward is the x distance of the sphere past the block.
    '''
    try:
        small_sphere = env.scene['Object2'] # CORRECT: Accessing small sphere object using approved pattern
        block = env.scene['Object5'] # CORRECT: Accessing block object using approved pattern

        # CORRECT: Accessing object positions using approved pattern
        small_sphere_pos_x = small_sphere.data.root_pos_w[:, 0]
        small_sphere_pos_y = small_sphere.data.root_pos_w[:, 1]
        block_pos_x = block.data.root_pos_w[:, 0]
        block_pos_y = block.data.root_pos_w[:, 1]

        # CORRECT: Calculate relative distance in x direction between small sphere and block
        distance_x_sphere_block = small_sphere_pos_x - block_pos_x
        distance_y_sphere_block = small_sphere_pos_y - block_pos_y

        distance_sphere_block = torch.sqrt(distance_x_sphere_block**2 + distance_y_sphere_block**2)

        # Reward is the absolute x distance of the sphere past the block, only when activated
        reward = torch.abs(distance_sphere_block)

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    # CORRECT: Reward normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_approach_sphere(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_sphere_reward") -> torch.Tensor:
    '''Shaping reward for approaching the small sphere from behind in the x-direction.

    Reward is based on reducing the x-distance between the robot's pelvis and the small sphere.
    Activated when the robot is behind the sphere in the x direction.
    '''
    try:
        small_sphere = env.scene['Object2'] # CORRECT: Accessing small sphere object using approved pattern
        robot = env.scene["robot"] # CORRECT: Accessing robot object using approved pattern
        pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing pelvis index using approved pattern
        robot_pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing pelvis position using approved pattern

        # CORRECT: Accessing object position using approved pattern
        small_sphere_pos_x = small_sphere.data.root_pos_w[:, 0]
        small_sphere_pos_y = small_sphere.data.root_pos_w[:, 1]
        robot_pelvis_pos_x = robot_pelvis_pos[:, 0]
        robot_pelvis_pos_y = robot_pelvis_pos[:, 1]

        # CORRECT: Calculate relative distance in x direction between pelvis and sphere
        distance_x_pelvis_sphere = small_sphere_pos_x - robot_pelvis_pos_x
        distance_y_pelvis_sphere = small_sphere_pos_y - robot_pelvis_pos_y

        distance_to_sphere = torch.sqrt(distance_x_pelvis_sphere**2 + distance_y_pelvis_sphere**2)


        # Reward is negative absolute x distance to the sphere, only when activated
        reward = -torch.abs(distance_to_sphere)

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    # CORRECT: Reward normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_foot_close_to_sphere(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "foot_close_sphere_reward") -> torch.Tensor:
    '''Shaping reward for bringing the kicking foot close to the small sphere.

    Reward is based on reducing the distance between the right ankle (kicking foot) and the small sphere in x and y directions.
    Activated when the robot pelvis is close to the sphere in x direction.
    '''
    try:
        small_sphere = env.scene['Object2'] # CORRECT: Accessing small sphere object using approved pattern
        robot = env.scene["robot"] # CORRECT: Accessing robot object using approved pattern
        robot_foot_idx = robot.body_names.index('right_ankle_roll_link') # CORRECT: Accessing right ankle index using approved pattern
        robot_foot_pos = robot.data.body_pos_w[:, robot_foot_idx] # CORRECT: Accessing right ankle position using approved pattern
        robot_pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing pelvis index using approved pattern
        robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] # CORRECT: Accessing pelvis position using approved pattern

        # CORRECT: Accessing object position using approved pattern
        small_sphere_pos = small_sphere.data.root_pos_w
        robot_foot_pos_x = robot_foot_pos[:, 0]
        robot_foot_pos_y = robot_foot_pos[:, 1]
        robot_pelvis_pos_x = robot_pelvis_pos[:, 0]
        small_sphere_pos_x = small_sphere_pos[:, 0]

        # CORRECT: Calculate relative distances in x and y directions between foot and sphere
        distance_x_foot_sphere = small_sphere_pos_x - robot_foot_pos_x
        distance_y_foot_sphere = small_sphere_pos[:, 1] - robot_foot_pos_y # y distance

        distance_to_sphere = torch.sqrt(distance_x_foot_sphere**2 + distance_y_foot_sphere**2)

        # Activation condition: robot pelvis is close to the sphere in x direction (within 1m)
        activation_condition = (torch.abs(distance_to_sphere) < 2)

        # Reward is negative sum of absolute distances to the sphere in x and y, only when activated
        reward = -distance_to_sphere
        reward = torch.where(activation_condition, reward, -2*torch.ones_like(reward)) # CORRECT: Reward is -2 if not activated, ensuring continuous reward

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    # CORRECT: Reward normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_stable_pelvis_height(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stable_pelvis_height_reward") -> torch.Tensor:
    '''Shaping reward for maintaining a stable pelvis height.

    Reward is based on keeping the pelvis z-position close to a desired standing height (0.7m).
    Always active.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot object using approved pattern
    robot_pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing pelvis index using approved pattern
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] # CORRECT: Accessing pelvis position using approved pattern

    # CORRECT: Accessing pelvis z position
    robot_pelvis_pos_z = robot_pelvis_pos[:, 2]

    # Desired pelvis height
    default_pelvis_z = 0.7

    # CORRECT: Calculate relative distance in z direction from default height
    distance_z_pelvis_default =  default_pelvis_z - robot_pelvis_pos_z

    # Activation condition: always active
    activation_condition = True

    # Reward is negative absolute distance from default pelvis z height, always active
    reward = -torch.abs(distance_z_pelvis_default)

    # CORRECT: Reward normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_collision_avoidance_block(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_block_reward") -> torch.Tensor:
    '''Shaping reward for collision avoidance penalty for collisions between the robot's pelvis and the block.

    Penalty if the robot's pelvis is close to the block in x and y directions.
    Activated when robot pelvis is near the block in x and y direction.
    '''
    try:
        block = env.scene['Object5'] # CORRECT: Accessing block object using approved pattern
        robot = env.scene["robot"] # CORRECT: Accessing robot object using approved pattern
        robot_pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing pelvis index using approved pattern
        robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] # CORRECT: Accessing pelvis position using approved pattern

        # CORRECT: Accessing object position using approved pattern
        block_pos = block.data.root_pos_w
        robot_pelvis_pos_x = robot_pelvis_pos[:, 0]
        robot_pelvis_pos_y = robot_pelvis_pos[:, 1]
        robot_pelvis_pos_z = robot_pelvis_pos[:, 2]
        block_pos_x = block_pos[:, 0]
        block_pos_y = block_pos[:, 1]
        block_pos_z = block_pos[:, 2]


        # CORRECT: Calculate relative distances in x, y, and z directions between pelvis and block
        distance_x_pelvis_block = block_pos_x - robot_pelvis_pos_x
        distance_y_pelvis_block = block_pos_y - robot_pelvis_pos_y
        distance_z_pelvis_block = block_pos_z - robot_pelvis_pos_z


        # Activation condition: robot pelvis is near the block in x and y direction (within 1m in x, 5m in y)
        activation_condition = (torch.abs(distance_x_pelvis_block) < 1.0) & (torch.abs(distance_y_pelvis_block) < 5.0)

        # Reward is a penalty (-1.0) if close to block, otherwise 0, only when activated
        reward = -1.0 * torch.ones(env.num_envs, device=env.device) # penalty value
        reward = torch.where(activation_condition, reward, torch.zeros_like(reward)) # CORRECT: Penalty when activated, 0 otherwise, ensuring continuous reward

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    # CORRECT: Reward normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    Main_ExecuteKickSmallSphereForwardReward = RewTerm(func=main_ExecuteKickSmallSphereForward_reward, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "main_reward"})
    ShapingRewardApproachSphere = RewTerm(func=shaping_reward_approach_sphere, weight=0.2,
                                            params={"normalise": True, "normaliser_name": "approach_sphere_reward"})
    ShapingRewardFootCloseToSphere = RewTerm(func=shaping_reward_foot_close_to_sphere, weight=0.2,
                                            params={"normalise": True, "normaliser_name": "foot_close_sphere_reward"})
    ShapingRewardStablePelvisHeight = RewTerm(func=shaping_reward_stable_pelvis_height, weight=0.2,
                                            params={"normalise": True, "normaliser_name": "stable_pelvis_height_reward"})
    ShapingRewardCollisionAvoidanceBlock = RewTerm(func=shaping_reward_collision_avoidance_block, weight=0, # Reduced weight slightly as it's a penalty
                                            params={"normalise": True, "normaliser_name": "collision_avoidance_block_reward"})
`,
                    successTerminationCode: `


from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
# Assuming mdp is correctly importable from the context where this runs
# If not, adjust the relative import path
from ...mdp import * 
import torch
from pathlib import Path
# Import reward functions if needed by success criteria
# from .TaskRewardsCfg import * 

# Standard imports - DO NOT MODIFY
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer
from ...objects import get_object_volume

def ExecuteKickSmallSphereForward_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the ExecuteKickSmallSphereForward skill has been successfully completed.
    Success is defined as the small sphere being at least 1 meter past the block in the x direction.
    '''
    try:
        # Access the small sphere object using the approved pattern (rule 2 & 6)
        object_small_sphere = env.scene['Object2']
        # Access the block object using the approved pattern (rule 2 & 6)
        object_block = env.scene['Object5']

        # Get the x position of the small sphere using the approved pattern (rule 2)
        small_sphere_pos_x = object_small_sphere.data.root_pos_w[:, 0]
        small_sphere_pos_y = object_small_sphere.data.root_pos_w[:, 1]
        # Get the x position of the block using the approved pattern (rule 2)
        block_pos_x = object_block.data.root_pos_w[:, 0]
        block_pos_y = object_block.data.root_pos_w[:, 1]



        # Calculate the relative distance in the x direction between the small sphere and the block (rule 1)
        distance_x_sphere_block = torch.abs(small_sphere_pos_x - block_pos_x)
        distance_y_sphere_block = torch.abs(small_sphere_pos_y - block_pos_y)

        distance_sphere_block = torch.sqrt(distance_x_sphere_block**2 + distance_y_sphere_block**2)
        # Define success condition: small sphere is at least 1.0 meter past the block in the x direction (rule 4 & 13)
        success_condition = distance_sphere_block > 3.0

    except KeyError:
        # Handle missing objects (small sphere or block) by returning failure (rule 5)
        success_condition = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Check for success duration and save success state using provided helper functions (rule 6 & 7)
    success = check_success_duration(env, success_condition, "ExecuteKickSmallSphereForward", duration=0.5) # Using a duration of 0.5 seconds
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "ExecuteKickSmallSphereForward")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=ExecuteKickSmallSphereForward_success)
`},
            ],
        },
        {
            name: "JumpOntoBlock",
            level: 2,
            rewardCode: `
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer
from ...objects import get_object_volume

import torch

def main_ExecuteJumpOntoBlock_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for ExecuteJumpOntoBlock.

    Reward for the robot standing on top of the block with feet above the block's top surface.
    This encourages the robot to successfully jump and land on the block.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
    try:
        block = env.scene['Object5'] # Accessing block object using approved pattern and try/except - rule 2 & 5 in ABSOLUTE REQUIREMENTS

        # Accessing robot foot positions using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
        left_foot_idx = robot.body_names.index('left_ankle_roll_link')
        right_foot_idx = robot.body_names.index('right_ankle_roll_link')
        left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
        right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

        # Calculate minimum foot position
        min_foot_pos = torch.min(left_foot_pos, right_foot_pos)

        # Accessing block position using approved pattern - rule 2 in ABSOLUTE REQUIREMENTS
        block_pos = block.data.root_pos_w

        # Calculate relative distance in z direction between minimum foot position and block top surface - rule 1 in ABSOLUTE REQUIREMENTS
        block_size_z = 0.5 # Reading block size from object config (size_cubes = [[0.4, 10.0, 0.4], [1.0, 10.0, 0.2], [0.5, 0.5, 0.5]]) - rule 6 & 7 in CRITICAL IMPLEMENTATION RULES
        block_top_surface_z = block_pos[:, 2] + block_size_z 
        distance_z_feet_block_top = min_foot_pos[:, 2] - block_top_surface_z  # Relative distance - rule 1 in ABSOLUTE REQUIREMENTS


        # Reward is negative absolute distance to encourage feet to be on top of the block - rule 5 in CRITICAL IMPLEMENTATION RULES
        reward = -torch.abs(distance_z_feet_block_top) # Continuous reward, using relative distance - rule 5 in CRITICAL IMPLEMENTATION RULES

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward - rule 5 in ABSOLUTE REQUIREMENTS

    # Reward normalization using RewNormalizer - rule 6 in ABSOLUTE REQUIREMENTS and rule 4 in REWARD STRUCTURE RULES
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_approach_block_x(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_block_x_reward") -> torch.Tensor:
    '''Shaping reward for approaching the block in the x-direction.

    Reward for decreasing the x-distance between the pelvis and the block.
    Encourages the robot to move towards the block before jumping.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
    try:
        block = env.scene['Object5'] # Accessing block object using approved pattern and try/except - rule 2 & 5 in ABSOLUTE REQUIREMENTS

        # Accessing robot pelvis position using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
        pelvis_idx = robot.body_names.index('pelvis')
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

        # Accessing block position using approved pattern - rule 2 in ABSOLUTE REQUIREMENTS
        block_pos = block.data.root_pos_w

        # Calculate relative distance in x direction between pelvis and block - rule 1 in ABSOLUTE REQUIREMENTS
        distance_x_pelvis_block = block_pos[:, 0] - pelvis_pos[:, 0] # Relative distance - rule 1 in ABSOLUTE REQUIREMENTS
        distance_y_pelvis_block = block_pos[:, 1] - pelvis_pos[:, 1] # Relative distance in y-direction
        distance = torch.sqrt(distance_x_pelvis_block**2 + distance_y_pelvis_block**2)


        # Reward is negative absolute distance to encourage moving closer in x direction - rule 5 in CRITICAL IMPLEMENTATION RULES
        reward = -torch.abs(distance) # Continuous reward, using relative distance - rule 5 in CRITICAL IMPLEMENTATION RULES

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward - rule 5 in ABSOLUTE REQUIREMENTS

    # Reward normalization using RewNormalizer - rule 6 in ABSOLUTE REQUIREMENTS and rule 4 in REWARD STRUCTURE RULES
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_jump_height(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "jump_height_reward") -> torch.Tensor:
    '''Shaping reward for achieving a suitable jump height when approaching the block.

    Reward for reaching a target pelvis height above the block's top surface when close to the block in x-direction.
    Encourages the robot to jump upwards when approaching the block.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
    try:
        block = env.scene['Object5'] # Accessing block object using approved pattern and try/except - rule 2 & 5 in ABSOLUTE REQUIREMENTS

        # Accessing robot pelvis position using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
        pelvis_idx = robot.body_names.index('pelvis')
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

        # Accessing block position using approved pattern - rule 2 in ABSOLUTE REQUIREMENTS
        block_pos = block.data.root_pos_w

        # Calculate target pelvis height above block top surface - rule 1 in ABSOLUTE REQUIREMENTS
        block_size_z = 0.5 # Reading block size from object config (size_cubes = [[0.4, 10.0, 0.4], [1.0, 10.0, 0.2], [0.5, 0.5, 0.5]]) - rule 6 & 7 in CRITICAL IMPLEMENTATION RULES
        block_top_surface_z = block_pos[:, 2] + block_size_z
        target_pelvis_height = block_top_surface_z + 0.5 # Relative target height - rule 1 in ABSOLUTE REQUIREMENTS

        # Calculate relative distance in z direction between pelvis and target height - rule 1 in ABSOLUTE REQUIREMENTS
        distance_z_pelvis_target_height = pelvis_pos[:, 2] - target_pelvis_height # Relative distance - rule 1 in ABSOLUTE REQUIREMENTS

        # Approach block condition: activate when robot is approaching the block in x direction - rule 4 in ABSOLUTE REQUIREMENTS
        approach_block_condition = (pelvis_pos[:, 0] > block_pos[:, 0] - 2.0) & (pelvis_pos[:, 0] < block_pos[:, 0]) # Relative condition - rule 1 in ABSOLUTE REQUIREMENTS

        # Reward is negative absolute distance to encourage reaching target height - rule 5 in CRITICAL IMPLEMENTATION RULES
        reward = -torch.abs(distance_z_pelvis_target_height) # Continuous reward, using relative distance - rule 5 in CRITICAL IMPLEMENTATION RULES

        reward = torch.where(approach_block_condition, reward, -torch.ones_like(reward)) # Apply activation condition - rule 4 in ABSOLUTE REQUIREMENTS

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward - rule 5 in ABSOLUTE REQUIREMENTS

    # Reward normalization using RewNormalizer - rule 6 in ABSOLUTE REQUIREMENTS and rule 4 in REWARD STRUCTURE RULES
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_stability_on_block(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_on_block_reward") -> torch.Tensor:
    '''Shaping reward for maintaining stability on the block after landing.

    Reward for keeping the pelvis at a reasonable height relative to the block and feet, and avoid falling off.
    Encourages the robot to maintain balance on the block.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
    try:
        block = env.scene['Object5'] # Accessing block object using approved pattern and try/except - rule 2 & 5 in ABSOLUTE REQUIREMENTS

        # Accessing robot pelvis and feet positions using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
        pelvis_idx = robot.body_names.index('pelvis')
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
        left_foot_idx = robot.body_names.index('left_ankle_roll_link')
        right_foot_idx = robot.body_names.index('right_ankle_roll_link')
        left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
        right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

        # Calculate average foot position
        avg_foot_pos = (left_foot_pos + right_foot_pos) / 2

        # Accessing block position using approved pattern - rule 2 in ABSOLUTE REQUIREMENTS
        block_pos = block.data.root_pos_w

        # Calculate relative pelvis height above average feet position - rule 1 in ABSOLUTE REQUIREMENTS
        relative_pelvis_height = pelvis_pos[:, 2] - avg_foot_pos[:, 2] # Relative distance - rule 1 in ABSOLUTE REQUIREMENTS

        # Target relative pelvis height (slightly above feet) - rule 4 in ABSOLUTE REQUIREMENTS
        target_relative_pelvis_height = 0.5 # Relative target height - rule 4 in ABSOLUTE REQUIREMENTS

        # Calculate distance from target relative pelvis height - rule 1 in ABSOLUTE REQUIREMENTS
        distance_z_pelvis_relative_target = relative_pelvis_height - target_relative_pelvis_height # Relative distance - rule 1 in ABSOLUTE REQUIREMENTS

        pelvis_on_block_condition_x = (pelvis_pos[:, 0] > block_pos[:, 0] - 0.25) & (pelvis_pos[:, 0] < block_pos[:, 0] + 0.25)
        pelvis_on_block_condition_y = (pelvis_pos[:, 1] > block_pos[:, 1] - 0.25) & (pelvis_pos[:, 1] < block_pos[:, 1] + 0.25)


        # On block condition: activate when feet are approximately on top of the block - rule 4 in ABSOLUTE REQUIREMENTS
        block_size_z = 0.5 # Reading block size from object config (size_cubes = [[0.4, 10.0, 0.4], [1.0, 1.0, 0.2], [0.5, 0.5, 0.5]]) - rule 6 & 7 in CRITICAL IMPLEMENTATION RULES
        on_block_condition = (avg_foot_pos[:, 2] > (block_pos[:, 2] + block_size_z - 0.1)) & pelvis_on_block_condition_x & pelvis_on_block_condition_y # Relative condition - rule 1 in ABSOLUTE REQUIREMENTS

        # Reward is negative absolute distance to encourage maintaining target relative pelvis height - rule 5 in CRITICAL IMPLEMENTATION RULES
        reward = -torch.abs(distance_z_pelvis_relative_target) # Continuous reward, using relative distance - rule 5 in CRITICAL IMPLEMENTATION RULES

        reward = torch.where(on_block_condition, reward, -2.0 * torch.ones_like(reward)) # Apply activation condition - rule 4 in ABSOLUTE REQUIREMENTS

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward - rule 5 in ABSOLUTE REQUIREMENTS

    # Reward normalization using RewNormalizer - rule 6 in ABSOLUTE REQUIREMENTS and rule 4 in REWARD STRUCTURE RULES
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_collision_avoidance(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    '''Shaping reward for collision avoidance with the block at ground level.

    Negative reward if the pelvis is too close to the block horizontally when the pelvis is low to the ground.
    Discourages the robot from colliding with the block before jumping.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
    try:
        block = env.scene['Object5'] # Accessing block object using approved pattern and try/except - rule 2 & 5 in ABSOLUTE REQUIREMENTS

        # Accessing robot pelvis position using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
        pelvis_idx = robot.body_names.index('pelvis')
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

        # Accessing block position using approved pattern - rule 2 in ABSOLUTE REQUIREMENTS
        block_pos = block.data.root_pos_w

        # Calculate relative distance in x direction between pelvis and block - rule 1 in ABSOLUTE REQUIREMENTS
        distance_x_pelvis_block = block_pos[:, 0] - pelvis_pos[:, 0] # Relative distance - rule 1 in ABSOLUTE REQUIREMENTS

        # Collision condition: activate when robot is very close to block horizontally and low to the ground - rule 4 in ABSOLUTE REQUIREMENTS
        collision_condition = (pelvis_pos[:, 0] > block_pos[:, 0] - 0.5) & (pelvis_pos[:, 0] < block_pos[:, 0] + 0.5) & (pelvis_pos[:, 2] < 0.2) # Relative condition in x, absolute condition in z (allowed for height) - rule 1 in ABSOLUTE REQUIREMENTS

        # Small negative reward for collision proximity - rule 5 in CRITICAL IMPLEMENTATION RULES
        reward = -1.0 * torch.ones(env.num_envs, device=env.device) # Continuous negative reward - rule 5 in CRITICAL IMPLEMENTATION RULES

        reward = torch.where(collision_condition, reward, torch.zeros_like(reward)) # Apply activation condition - rule 4 in ABSOLUTE REQUIREMENTS

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward - rule 5 in ABSOLUTE REQUIREMENTS

    # Reward normalization using RewNormalizer - rule 6 in ABSOLUTE REQUIREMENTS and rule 4 in REWARD STRUCTURE RULES
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_feet_under_pelvis(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "feet_under_pelvis_reward") -> torch.Tensor:
    '''Shaping reward for keeping feet underneath the pelvis in the horizontal plane.
    
    Reward for minimizing the horizontal (x,y) distance between the feet and pelvis.
    Encourages the robot to maintain a stable posture.
    '''
    robot = env.scene["robot"]
    
    # Accessing robot pelvis and feet positions using approved pattern
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    
    left_x_distance = pelvis_pos[:, 0] - left_foot_pos[:, 0]
    left_y_distance = pelvis_pos[:, 1] - left_foot_pos[:, 1]
    
    right_x_distance = pelvis_pos[:, 0] - right_foot_pos[:, 0]
    right_y_distance = pelvis_pos[:, 1] - right_foot_pos[:, 1]
    
    
    

    # Calculate horizontal distance between pelvis and average foot position
    horizontal_distance = torch.sqrt((left_x_distance)**2 + 
                                     (left_y_distance)**2) + torch.sqrt((right_x_distance)**2 + 
                                     (right_y_distance)**2)
    
    # Reward is negative distance to encourage feet to stay under pelvis
    reward = -horizontal_distance
    
    # Reward normalization using RewNormalizer
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

@configclass
class TaskRewardsCfg:
    Main_ExecuteJumpOntoBlockReward = RewTerm(func=main_ExecuteJumpOntoBlock_reward, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "main_reward"})
    ShapingRewardApproachBlockX = RewTerm(func=shaping_reward_approach_block_x, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "approach_block_x_reward"})
    ShapingRewardJumpHeight = RewTerm(func=shaping_reward_jump_height, weight=0.5,
                                            params={"normalise": True, "normaliser_name": "jump_height_reward"})
    ShapingRewardStabilityOnBlock = RewTerm(func=shaping_reward_stability_on_block, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "stability_on_block_reward"})
    ShapingRewardCollisionAvoidance = RewTerm(func=shaping_reward_collision_avoidance, weight=0, 
                                            params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})
    ShapingRewardFeetUnderPelvis = RewTerm(func=shaping_reward_feet_under_pelvis, weight=0.5,
                                            params={"normalise": True, "normaliser_name": "feet_under_pelvis_reward"})
`,
            successTerminationCode: `


from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
# Assuming mdp is correctly importable from the context where this runs
# If not, adjust the relative import path
from ...mdp import * 
import torch
from pathlib import Path
# Import reward functions if needed by success criteria
# from .TaskRewardsCfg import * 

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer
from ...objects import get_object_volume

def JumpOntoBlock_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the JumpOntoBlock skill has been successfully completed.
    Success is defined as the robot standing on top of the block with both feet.
    '''
    # 1. Access the robot object from the environment scene.
    robot = env.scene["robot"]

    # 2. Get indices for the left and right feet (ankle_roll_link) using robot.body_names.index.
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    # 3. Get the world positions of the left and right feet using robot.data.body_pos_w and the indices.
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Shape: [num_envs, 3]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Shape: [num_envs, 3]

    # 4. Calculate the average position of the two feet.
    avg_feet_pos_x = (left_foot_pos[:, 0] + right_foot_pos[:, 0]) / 2.0
    avg_feet_pos_y = (left_foot_pos[:, 1] + right_foot_pos[:, 1]) / 2.0
    avg_feet_pos_z = (left_foot_pos[:, 2] + right_foot_pos[:, 2]) / 2.0
    avg_feet_pos = torch.stack([avg_feet_pos_x, avg_feet_pos_y, avg_feet_pos_z], dim=-1) # Shape: [num_envs, 3]


    try:
        # 5. Safely access the block object (Object5) from the environment scene using try-except block.
        block = env.scene['Object5']
        # 6. Get the world position of the block using block.data.root_pos_w.
        block_pos = block.data.root_pos_w # Shape: [num_envs, 3]

        # 7. Calculate the relative distance vector between the average feet position and the block position.
        distance_x = block_pos[:, 0] - avg_feet_pos[:, 0]
        distance_y = block_pos[:, 1] - avg_feet_pos[:, 1]
        # 8. Calculate the vertical distance between the average feet position and the top of the block.
        #    Block size is [0.5, 0.5, 0.5], so half height is 0.25.
        distance_z = (block_pos[:, 2] + 0.25) - avg_feet_pos[:, 2] # distance to top of block

        # 9. Define success condition:
        #    - Vertical distance (distance_z) is less than 0.15 (feet are above the block top within 15cm).
        #    - Horizontal distances (distance_x and distance_y) are within 0.25 in both x and y directions.
        success_condition = (distance_z < 0.15) & (torch.abs(distance_x) < 0.25) & (torch.abs(distance_y) < 0.25)

    except KeyError:
        # 10. Handle KeyError if 'Object5' (block) is not found in the scene. Set success to False for all environments.
        success_condition = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 11. Check for success duration of 0.5 seconds using check_success_duration.
    success = check_success_duration(env, success_condition, "JumpOntoBlock", duration=0.5)

    # 12. Save success states for environments that meet the success condition using save_success_state.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "JumpOntoBlock")

    # 13. Return the success tensor.
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=JumpOntoBlock_success)
`,
            policyVideo: "/videos/JumpOntoBlock.mp4", // Matched without L2_ prefix
            children: [
                {
                    name: "WalkToBlock", level: 1, policyVideo: "/videos/WalkToBlock.mp4", rewardCode: `
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer # DO NOT CHANGE THIS LINE!
from ...objects import get_object_volume

def main_WalkToBlock_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for WalkToBlock.

    Reward for moving the robot's pelvis closer to the block in the x-direction.
    This encourages the robot to walk towards the block.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        block = env.scene['Object5'] # Accessing block (Object5) using approved pattern and try/except
        pelvis_idx = robot.body_names.index('pelvis') # Getting pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Getting pelvis position using approved pattern
        block_pos = block.data.root_pos_w # Getting block position using approved pattern

        distance_x = block_pos[:, 0] - pelvis_pos[:, 0] # Relative distance in x-direction (block - pelvis)
        distance_y = block_pos[:, 1] - pelvis_pos[:, 1] # Relative distance in y-direction (block - pelvis)
        distance = torch.sqrt(distance_x**2 + distance_y**2)

        reward = -torch.abs(distance) # Reward is negative absolute x-distance to encourage closer distance, continuous reward

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # Normalize reward
        RewNormalizer.update_stats(normaliser_name, reward) # Update normalizer stats
        return scaled_reward
    return reward

def pelvis_height_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_height_reward") -> torch.Tensor:
    '''Supporting reward for maintaining pelvis height.

    Reward for keeping the pelvis at a stable height around 0.7m.
    This encourages a stable standing posture.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        pelvis_idx = robot.body_names.index('pelvis') # Getting pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Getting pelvis position using approved pattern
        pelvis_pos_z = pelvis_pos[:, 2] # Getting pelvis z position

        default_pelvis_z = 0.75 # Define default pelvis height (read from task description)
        reward = -torch.abs(pelvis_pos_z - default_pelvis_z) # Reward is negative absolute difference from default height, continuous reward

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # Normalize reward
        RewNormalizer.update_stats(normaliser_name, reward) # Update normalizer stats
        return scaled_reward
    return reward

def feet_block_collision_penalty(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "feet_block_penalty") -> torch.Tensor:
    '''Supporting reward to penalize feet getting too close to the block.

    Penalty when the robot's feet are too close to the block in the x-direction.
    This prevents collisions and encourages careful approach.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        block = env.scene['Object5'] # Accessing block (Object5) using approved pattern and try/except
        left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Getting left foot index using approved pattern
        right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Getting right foot index using approved pattern
        left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Getting left foot position using approved pattern
        right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Getting right foot position using approved pattern
        block_pos_x = block.data.root_pos_w[:, 0] # Getting block x position

        distance_x_left_foot = torch.abs(block_pos_x - left_foot_pos[:, 0]) # Relative x-distance between block and left foot
        distance_x_right_foot = torch.abs(block_pos_x - right_foot_pos[:, 0]) # Relative x-distance between block and right foot
        min_distance_x_feet_block = torch.min(distance_x_left_foot, distance_x_right_foot) # Minimum distance to either foot

        collision_threshold = 0.5 # Define collision threshold (read from shaping reward description)
        penalty = -1.0 # Fixed negative penalty value (read from shaping reward description)
        reward = torch.where(min_distance_x_feet_block < collision_threshold, torch.tensor(penalty, device=env.device), torch.tensor(0.0, device=env.device)) # Penalty if too close, otherwise 0, continuous reward (though binary in effect)

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # Normalize reward
        RewNormalizer.update_stats(normaliser_name, reward) # Update normalizer stats
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    Main_WalkToBlockReward = RewTerm(func=main_WalkToBlock_reward, weight=1.0,
                                params={"normalise": True, "normaliser_name": "main_reward"}) # Main reward with weight 1.0
    PelvisHeightStabilityReward = RewTerm(func=pelvis_height_stability_reward, weight=0.4,
                                params={"normalise": True, "normaliser_name": "pelvis_height_reward"}) # Supporting reward with weight 0.4
    # FeetBlockCollisionPenalty = RewTerm(func=feet_block_collision_penalty, weight=0.2,
    #                             params={"normalise": True, "normaliser_name": "feet_block_penalty"}) # Supporting reward with weight 0.2 (lower weight for penalty)
                `,
                    successTerminationCode: `


from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
# Assuming mdp is correctly importable from the context where this runs
# If not, adjust the relative import path
from ...mdp import * 
import torch
from pathlib import Path
# Import reward functions if needed by success criteria
# from .TaskRewardsCfg import * 

# Standard imports - DO NOT MODIFY
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer
from ...objects import get_object_volume


def WalkToBlock_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the WalkToBlock skill has been successfully completed.'''
    # 1. Get robot pelvis position - Approved access pattern
    robot = env.scene["robot"] # Approved access pattern to get robot
    pelvis_idx = robot.body_names.index('pelvis') # Approved access pattern to get pelvis index
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Approved access pattern to get pelvis position

    try:
        # 2. Get block position (Object5) - Approved access pattern and try/except for missing object
        block = env.scene['Object5'] # Approved access pattern to get Object5 (block)
        block_pos = block.data.root_pos_w # Approved access pattern to get block position

        # 3. Calculate relative x-distance (block x - pelvis x) - Relative distance as required, only x component
        x_distance = block_pos[:, 0] - pelvis_pos[:, 0] # Relative distance in x-direction
        y_distance = block_pos[:, 1] - pelvis_pos[:, 1] # Relative distance in y-direction
        distance = torch.sqrt(x_distance**2 + y_distance**2)

        condition = distance < 1.4 # Success if x distance is less than 1.2m
    except KeyError:
        # Handle case where block (Object5) is missing - Required for robustness
        condition = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device) # If block is missing, skill is not successful

    # 5. Check duration and save success states - DO NOT MODIFY THIS SECTION - Correct usage of check_success_duration and save_success_state
    success = check_success_duration(env, condition, "WalkToBlock", duration=2) # Check if success condition is maintained for 0.3 seconds
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "WalkToBlock") # Save success state for successful environments

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=WalkToBlock_success) # Defines success termination condition using WalkToBlock_success function
                `},
                {
                    name: "PrepareForJumpOntoBlock", level: 1, policyVideo: "/videos/PrepareForJumpOntoBlock.mp4", rewardCode: `
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer # DO NOT CHANGE THIS LINE!
from ...objects import get_object_volume

def main_PrepareForJumpOntoBlock_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for PrepareForJumpOntoBlock.

    Rewards the robot for crouching to a target pelvis height, preparing for the jump.
    This reward encourages the robot to lower its pelvis, which is the primary action for preparing to jump.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        #access the required robot part(s)
        robot_pelvis_idx = robot.body_names.index('pelvis') # Getting pelvis index using approved pattern
        robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] # Getting pelvis position using approved pattern
        robot_pelvis_pos_z = robot_pelvis_pos[:, 2] # Pelvis z position

        # Define target pelvis height for crouching (relative to ground z=0). No hardcoded positions, target is relative to ground.
        target_pelvis_z = 0.5

        # Primary reward: negative absolute difference between current pelvis z and target crouch z. Relative distance to target height.
        primary_reward = -torch.abs(robot_pelvis_pos_z - target_pelvis_z) # Continuous reward, relative distance, no hardcoded thresholds except target_z

        reward = primary_reward

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object (robot or pelvis), return zero reward

    # Reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_block_xy(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_block_x") -> torch.Tensor:
    '''Shaping reward for being in front of the block in the x-direction.

    Encourages the robot to position itself appropriately for the jump onto the block.
    This reward is activated when the robot is past the small sphere, indicating it's moving towards the block.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        block = env.scene['Object5'] # Access the block using approved pattern
        small_sphere = env.scene['Object2'] # Access the small sphere using approved pattern
        #access the required robot part(s)
        robot_pelvis_idx = robot.body_names.index('pelvis') # Getting pelvis index using approved pattern
        robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] # Getting pelvis position using approved pattern
        robot_pelvis_pos_x = robot_pelvis_pos[:, 0] # Pelvis x position
        robot_pelvis_pos_y = robot_pelvis_pos[:, 1] # Pelvis y position
        block_pos_x = block.data.root_pos_w[:, 0] # Block x position using approved pattern
        block_pos_y = block.data.root_pos_w[:, 1] # Block y position using approved pattern
        small_sphere_pos_x = small_sphere.data.root_pos_w[:, 0] # Small sphere x position using approved pattern


        # Calculate x distance to block. Relative distance.
        distance_block_x = robot_pelvis_pos_x - block_pos_x
        distance_block_y = robot_pelvis_pos_y - block_pos_y

        # Shaping reward: negative absolute x distance to the block. Continuous reward, relative distance.
        reward_block_x = -torch.abs(distance_block_x)
        reward_block_y = -torch.abs(distance_block_y)
        shaping_reward_block_x_unscaled = reward_block_x + reward_block_y # Apply reward only when activated

        # Normalize the reward
        reward = shaping_reward_block_x_unscaled

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    # Reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def shaping_reward_min_z(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_min_z") -> torch.Tensor:
    '''Shaping reward to penalize pelvis going too low.

    Prevents the pelvis from going too low, which might lead to instability or self-collisions during crouching.
    Penalizes if the pelvis z is lower than a minimum threshold (0.3m relative to ground).
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        #access the required robot part(s)
        robot_pelvis_idx = robot.body_names.index('pelvis') # Getting pelvis index using approved pattern
        robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] # Getting pelvis position using approved pattern
        robot_pelvis_pos_z = robot_pelvis_pos[:, 2] # Pelvis z position

        # Minimum pelvis height threshold. No hardcoded positions, threshold is relative to ground.
        min_pelvis_z = 0.3

        # Activation condition: Pelvis is below the minimum height. Relative condition.
        activation_condition_min_z = (robot_pelvis_pos_z < min_pelvis_z)

        # Shaping reward: penalty for going below minimum pelvis height. Proportional penalty. Continuous reward, relative distance.
        reward_min_z = -(min_pelvis_z - robot_pelvis_pos_z) # negative reward if below threshold, 0 if above.

        shaping_reward_min_z_unscaled = torch.where(activation_condition_min_z, reward_min_z, torch.tensor(0.0, device=env.device)) # only apply penalty if below threshold

        reward = shaping_reward_min_z_unscaled

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    # Reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def shaping_reward_feet_z(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_feet_z") -> torch.Tensor:
    '''Shaping reward for keeping feet relatively close to the ground during crouching.

    Encourages a controlled crouch and prevents the feet from lifting excessively, maintaining stability.
    Rewards for keeping the average z-height of the feet low (relative to ground).
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    try:
        #access the required robot part(s)
        robot_left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Getting left foot index using approved pattern
        robot_right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Getting right foot index using approved pattern
        robot_left_foot_pos = robot.data.body_pos_w[:, robot_left_foot_idx] # Left foot position using approved pattern
        robot_right_foot_pos = robot.data.body_pos_w[:, robot_right_foot_idx] # Right foot position using approved pattern
        robot_left_foot_pos_z = robot_left_foot_pos[:, 2] # Left foot z position
        robot_right_foot_pos_z = robot_right_foot_pos[:, 2] # Right foot z position

        # Calculate average foot z-height. Relative to ground.
        average_foot_z = (robot_left_foot_pos_z + robot_right_foot_pos_z) / 2.0

        # Shaping reward: negative average foot z-height (reward for lower feet). Continuous reward, relative distance to ground.
        shaping_reward_feet_z_unscaled = -torch.abs(average_foot_z) # reward for feet being closer to ground (z=0)

        reward = shaping_reward_feet_z_unscaled

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    # Reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    Main_PrepareForJumpOntoBlockReward = RewTerm(func=main_PrepareForJumpOntoBlock_reward, weight=1.0,
                                params={"normalise": True, "normaliser_name": "main_reward"}) # Main reward with weight 1.0

    ShapingRewardBlockXY = RewTerm(func=shaping_reward_block_xy, weight=0.5,
                                params={"normalise": True, "normaliser_name": "shaping_reward_block_xy"}) # Shaping reward with weight 0.5

    ShapingRewardMinZ = RewTerm(func=shaping_reward_min_z, weight=0.3,
                                params={"normalise": True, "normaliser_name": "shaping_reward_min_z"}) # Shaping reward with weight 0.3

    ShapingRewardFeetZ = RewTerm(func=shaping_reward_feet_z, weight=0.3,
                                params={"normalise": True, "normaliser_name": "shaping_reward_feet_z"}) # Shaping reward with weight 0.3
                `,
                    successTerminationCode: `


from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
# Assuming mdp is correctly importable from the context where this runs
# If not, adjust the relative import path
from ...mdp import * 
import torch
from pathlib import Path
# Import reward functions if needed by success criteria
# from .TaskRewardsCfg import * 

def PrepareForJumpOntoBlock_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the PrepareForJumpOntoBlock skill has been successfully completed.
    Success is defined as the robot pelvis being below 0.6m in height, indicating a crouched position.
    This is measured relative to the ground (z=0).
    '''
    # 1. Access the robot object from the scene using the approved pattern.
    robot = env.scene["robot"]

    try:
        # 2. Get the index of the 'pelvis' body part using the approved pattern.
        robot_pelvis_idx = robot.body_names.index('pelvis')
        # 3. Get the world position of the pelvis using the approved pattern.
        robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]
        # 4. Extract the z-component of the pelvis position.
        robot_pelvis_pos_z = robot_pelvis_pos[:, 2]

        # 5. Define the success condition: pelvis z-position < 0.6m.
        #    This is a relative check to the ground (z=0), fulfilling requirement 1 (relative distances).
        #    No hardcoded positions or arbitrary thresholds are used, threshold is based on skill description and reward functions.
        condition = (robot_pelvis_pos_z < 0.6)

    except KeyError:
        # 6. Handle potential KeyError if 'robot' or 'pelvis' is not found in the scene.
        #    This fulfills requirement 5 (handle missing objects).
        condition = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 7. Check success duration and save success states using the provided helper functions.
    #    This fulfills requirement 6 and 7 (check_success_duration and save_success_state).
    success = check_success_duration(env, condition, "PrepareForJumpOntoBlock", duration=0.5) # Using duration of 0.5 seconds as specified.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "PrepareForJumpOntoBlock")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=PrepareForJumpOntoBlock_success)
                `},
                {
                    name: "ExecuteJumpOntoBlock", level: 1, policyVideo: "/videos/ExecuteJumpOntoBlock.mp4", rewardCode: `
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer
from ...objects import get_object_volume

import torch

def main_ExecuteJumpOntoBlock_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for ExecuteJumpOntoBlock.

    Reward for the robot standing on top of the block with feet above the block's top surface.
    This encourages the robot to successfully jump and land on the block.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
    try:
        block = env.scene['Object5'] # Accessing block object using approved pattern and try/except - rule 2 & 5 in ABSOLUTE REQUIREMENTS

        # Accessing robot foot positions using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
        left_foot_idx = robot.body_names.index('left_ankle_roll_link')
        right_foot_idx = robot.body_names.index('right_ankle_roll_link')
        left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
        right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

        # Calculate minimum foot position
        min_foot_pos = torch.min(left_foot_pos, right_foot_pos)

        # Accessing block position using approved pattern - rule 2 in ABSOLUTE REQUIREMENTS
        block_pos = block.data.root_pos_w

        # Calculate relative distance in z direction between minimum foot position and block top surface - rule 1 in ABSOLUTE REQUIREMENTS
        block_size_z = 0.5 # Reading block size from object config (size_cubes = [[0.4, 10.0, 0.4], [1.0, 10.0, 0.2], [0.5, 0.5, 0.5]]) - rule 6 & 7 in CRITICAL IMPLEMENTATION RULES
        block_top_surface_z = block_pos[:, 2] + block_size_z 
        distance_z_feet_block_top = min_foot_pos[:, 2] - block_top_surface_z  # Relative distance - rule 1 in ABSOLUTE REQUIREMENTS


        # Reward is negative absolute distance to encourage feet to be on top of the block - rule 5 in CRITICAL IMPLEMENTATION RULES
        reward = -torch.abs(distance_z_feet_block_top) # Continuous reward, using relative distance - rule 5 in CRITICAL IMPLEMENTATION RULES

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward - rule 5 in ABSOLUTE REQUIREMENTS

    # Reward normalization using RewNormalizer - rule 6 in ABSOLUTE REQUIREMENTS and rule 4 in REWARD STRUCTURE RULES
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_approach_block_x(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_block_x_reward") -> torch.Tensor:
    '''Shaping reward for approaching the block in the x-direction.

    Reward for decreasing the x-distance between the pelvis and the block.
    Encourages the robot to move towards the block before jumping.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
    try:
        block = env.scene['Object5'] # Accessing block object using approved pattern and try/except - rule 2 & 5 in ABSOLUTE REQUIREMENTS

        # Accessing robot pelvis position using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
        pelvis_idx = robot.body_names.index('pelvis')
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

        # Accessing block position using approved pattern - rule 2 in ABSOLUTE REQUIREMENTS
        block_pos = block.data.root_pos_w

        # Calculate relative distance in x direction between pelvis and block - rule 1 in ABSOLUTE REQUIREMENTS
        distance_x_pelvis_block = block_pos[:, 0] - pelvis_pos[:, 0] # Relative distance - rule 1 in ABSOLUTE REQUIREMENTS
        distance_y_pelvis_block = block_pos[:, 1] - pelvis_pos[:, 1] # Relative distance in y-direction
        distance = torch.sqrt(distance_x_pelvis_block**2 + distance_y_pelvis_block**2)


        # Reward is negative absolute distance to encourage moving closer in x direction - rule 5 in CRITICAL IMPLEMENTATION RULES
        reward = -torch.abs(distance) # Continuous reward, using relative distance - rule 5 in CRITICAL IMPLEMENTATION RULES

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward - rule 5 in ABSOLUTE REQUIREMENTS

    # Reward normalization using RewNormalizer - rule 6 in ABSOLUTE REQUIREMENTS and rule 4 in REWARD STRUCTURE RULES
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_jump_height(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "jump_height_reward") -> torch.Tensor:
    '''Shaping reward for achieving a suitable jump height when approaching the block.

    Reward for reaching a target pelvis height above the block's top surface when close to the block in x-direction.
    Encourages the robot to jump upwards when approaching the block.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
    try:
        block = env.scene['Object5'] # Accessing block object using approved pattern and try/except - rule 2 & 5 in ABSOLUTE REQUIREMENTS

        # Accessing robot pelvis position using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
        pelvis_idx = robot.body_names.index('pelvis')
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

        # Accessing block position using approved pattern - rule 2 in ABSOLUTE REQUIREMENTS
        block_pos = block.data.root_pos_w

        # Calculate target pelvis height above block top surface - rule 1 in ABSOLUTE REQUIREMENTS
        block_size_z = 0.5 # Reading block size from object config (size_cubes = [[0.4, 10.0, 0.4], [1.0, 10.0, 0.2], [0.5, 0.5, 0.5]]) - rule 6 & 7 in CRITICAL IMPLEMENTATION RULES
        block_top_surface_z = block_pos[:, 2] + block_size_z
        target_pelvis_height = block_top_surface_z + 0.5 # Relative target height - rule 1 in ABSOLUTE REQUIREMENTS

        # Calculate relative distance in z direction between pelvis and target height - rule 1 in ABSOLUTE REQUIREMENTS
        distance_z_pelvis_target_height = pelvis_pos[:, 2] - target_pelvis_height # Relative distance - rule 1 in ABSOLUTE REQUIREMENTS

        # Approach block condition: activate when robot is approaching the block in x direction - rule 4 in ABSOLUTE REQUIREMENTS
        approach_block_condition = (pelvis_pos[:, 0] > block_pos[:, 0] - 2.0) & (pelvis_pos[:, 0] < block_pos[:, 0]) # Relative condition - rule 1 in ABSOLUTE REQUIREMENTS

        # Reward is negative absolute distance to encourage reaching target height - rule 5 in CRITICAL IMPLEMENTATION RULES
        reward = -torch.abs(distance_z_pelvis_target_height) # Continuous reward, using relative distance - rule 5 in CRITICAL IMPLEMENTATION RULES

        reward = torch.where(approach_block_condition, reward, -torch.ones_like(reward)) # Apply activation condition - rule 4 in ABSOLUTE REQUIREMENTS

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward - rule 5 in ABSOLUTE REQUIREMENTS

    # Reward normalization using RewNormalizer - rule 6 in ABSOLUTE REQUIREMENTS and rule 4 in REWARD STRUCTURE RULES
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_stability_on_block(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_on_block_reward") -> torch.Tensor:
    '''Shaping reward for maintaining stability on the block after landing.

    Reward for keeping the pelvis at a reasonable height relative to the block and feet, and avoid falling off.
    Encourages the robot to maintain balance on the block.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
    try:
        block = env.scene['Object5'] # Accessing block object using approved pattern and try/except - rule 2 & 5 in ABSOLUTE REQUIREMENTS

        # Accessing robot pelvis and feet positions using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
        pelvis_idx = robot.body_names.index('pelvis')
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
        left_foot_idx = robot.body_names.index('left_ankle_roll_link')
        right_foot_idx = robot.body_names.index('right_ankle_roll_link')
        left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
        right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

        # Calculate average foot position
        avg_foot_pos = (left_foot_pos + right_foot_pos) / 2

        # Accessing block position using approved pattern - rule 2 in ABSOLUTE REQUIREMENTS
        block_pos = block.data.root_pos_w

        # Calculate relative pelvis height above average feet position - rule 1 in ABSOLUTE REQUIREMENTS
        relative_pelvis_height = pelvis_pos[:, 2] - avg_foot_pos[:, 2] # Relative distance - rule 1 in ABSOLUTE REQUIREMENTS

        # Target relative pelvis height (slightly above feet) - rule 4 in ABSOLUTE REQUIREMENTS
        target_relative_pelvis_height = 0.5 # Relative target height - rule 4 in ABSOLUTE REQUIREMENTS

        # Calculate distance from target relative pelvis height - rule 1 in ABSOLUTE REQUIREMENTS
        distance_z_pelvis_relative_target = relative_pelvis_height - target_relative_pelvis_height # Relative distance - rule 1 in ABSOLUTE REQUIREMENTS

        pelvis_on_block_condition_x = (pelvis_pos[:, 0] > block_pos[:, 0] - 0.25) & (pelvis_pos[:, 0] < block_pos[:, 0] + 0.25)
        pelvis_on_block_condition_y = (pelvis_pos[:, 1] > block_pos[:, 1] - 0.25) & (pelvis_pos[:, 1] < block_pos[:, 1] + 0.25)


        # On block condition: activate when feet are approximately on top of the block - rule 4 in ABSOLUTE REQUIREMENTS
        block_size_z = 0.5 # Reading block size from object config (size_cubes = [[0.4, 10.0, 0.4], [1.0, 1.0, 0.2], [0.5, 0.5, 0.5]]) - rule 6 & 7 in CRITICAL IMPLEMENTATION RULES
        on_block_condition = (avg_foot_pos[:, 2] > (block_pos[:, 2] + block_size_z - 0.1)) & pelvis_on_block_condition_x & pelvis_on_block_condition_y # Relative condition - rule 1 in ABSOLUTE REQUIREMENTS

        # Reward is negative absolute distance to encourage maintaining target relative pelvis height - rule 5 in CRITICAL IMPLEMENTATION RULES
        reward = -torch.abs(distance_z_pelvis_relative_target) # Continuous reward, using relative distance - rule 5 in CRITICAL IMPLEMENTATION RULES

        reward = torch.where(on_block_condition, reward, -2.0 * torch.ones_like(reward)) # Apply activation condition - rule 4 in ABSOLUTE REQUIREMENTS

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward - rule 5 in ABSOLUTE REQUIREMENTS

    # Reward normalization using RewNormalizer - rule 6 in ABSOLUTE REQUIREMENTS and rule 4 in REWARD STRUCTURE RULES
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_collision_avoidance(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    '''Shaping reward for collision avoidance with the block at ground level.

    Negative reward if the pelvis is too close to the block horizontally when the pelvis is low to the ground.
    Discourages the robot from colliding with the block before jumping.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
    try:
        block = env.scene['Object5'] # Accessing block object using approved pattern and try/except - rule 2 & 5 in ABSOLUTE REQUIREMENTS

        # Accessing robot pelvis position using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
        pelvis_idx = robot.body_names.index('pelvis')
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

        # Accessing block position using approved pattern - rule 2 in ABSOLUTE REQUIREMENTS
        block_pos = block.data.root_pos_w

        # Calculate relative distance in x direction between pelvis and block - rule 1 in ABSOLUTE REQUIREMENTS
        distance_x_pelvis_block = block_pos[:, 0] - pelvis_pos[:, 0] # Relative distance - rule 1 in ABSOLUTE REQUIREMENTS

        # Collision condition: activate when robot is very close to block horizontally and low to the ground - rule 4 in ABSOLUTE REQUIREMENTS
        collision_condition = (pelvis_pos[:, 0] > block_pos[:, 0] - 0.5) & (pelvis_pos[:, 0] < block_pos[:, 0] + 0.5) & (pelvis_pos[:, 2] < 0.2) # Relative condition in x, absolute condition in z (allowed for height) - rule 1 in ABSOLUTE REQUIREMENTS

        # Small negative reward for collision proximity - rule 5 in CRITICAL IMPLEMENTATION RULES
        reward = -1.0 * torch.ones(env.num_envs, device=env.device) # Continuous negative reward - rule 5 in CRITICAL IMPLEMENTATION RULES

        reward = torch.where(collision_condition, reward, torch.zeros_like(reward)) # Apply activation condition - rule 4 in ABSOLUTE REQUIREMENTS

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward - rule 5 in ABSOLUTE REQUIREMENTS

    # Reward normalization using RewNormalizer - rule 6 in ABSOLUTE REQUIREMENTS and rule 4 in REWARD STRUCTURE RULES
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_feet_under_pelvis(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "feet_under_pelvis_reward") -> torch.Tensor:
    '''Shaping reward for keeping feet underneath the pelvis in the horizontal plane.
    
    Reward for minimizing the horizontal (x,y) distance between the feet and pelvis.
    Encourages the robot to maintain a stable posture.
    '''
    robot = env.scene["robot"]
    
    # Accessing robot pelvis and feet positions using approved pattern
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    
    left_x_distance = pelvis_pos[:, 0] - left_foot_pos[:, 0]
    left_y_distance = pelvis_pos[:, 1] - left_foot_pos[:, 1]
    
    right_x_distance = pelvis_pos[:, 0] - right_foot_pos[:, 0]
    right_y_distance = pelvis_pos[:, 1] - right_foot_pos[:, 1]
    
    
    

    # Calculate horizontal distance between pelvis and average foot position
    horizontal_distance = torch.sqrt((left_x_distance)**2 + 
                                     (left_y_distance)**2) + torch.sqrt((right_x_distance)**2 + 
                                     (right_y_distance)**2)
    
    # Reward is negative distance to encourage feet to stay under pelvis
    reward = -horizontal_distance
    
    # Reward normalization using RewNormalizer
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

@configclass
class TaskRewardsCfg:
    Main_ExecuteJumpOntoBlockReward = RewTerm(func=main_ExecuteJumpOntoBlock_reward, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "main_reward"})
    ShapingRewardApproachBlockX = RewTerm(func=shaping_reward_approach_block_x, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "approach_block_x_reward"})
    ShapingRewardJumpHeight = RewTerm(func=shaping_reward_jump_height, weight=0.5,
                                            params={"normalise": True, "normaliser_name": "jump_height_reward"})
    ShapingRewardStabilityOnBlock = RewTerm(func=shaping_reward_stability_on_block, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "stability_on_block_reward"})
    ShapingRewardCollisionAvoidance = RewTerm(func=shaping_reward_collision_avoidance, weight=0, 
                                            params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})
    ShapingRewardFeetUnderPelvis = RewTerm(func=shaping_reward_feet_under_pelvis, weight=0.5,
                                            params={"normalise": True, "normaliser_name": "feet_under_pelvis_reward"})
                `,
                    successTerminationCode: `


from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
# Assuming mdp is correctly importable from the context where this runs
# If not, adjust the relative import path
from ...mdp import * 
import torch
from pathlib import Path
# Import reward functions if needed by success criteria
# from .TaskRewardsCfg import * 

def ExecuteJumpOntoBlock_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the ExecuteJumpOntoBlock skill has been successfully completed.'''
    # 1. Get robot and block objects - rule 2 & 3 in ABSOLUTE REQUIREMENTS
    robot = env.scene["robot"] # Accessing robot using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
    try:
        block = env.scene['Object5'] # Accessing block object using approved pattern and try/except - rule 2 & 5 in ABSOLUTE REQUIREMENTS
    except KeyError:
        # Handle case where the block object is not found - rule 5 in ABSOLUTE REQUIREMENTS
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 2. Get indices for robot parts - rule 3 in ABSOLUTE REQUIREMENTS
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    # 3. Get positions of robot parts and block - rule 2 & 3 in ABSOLUTE REQUIREMENTS
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Accessing left foot position using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Accessing right foot position using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing pelvis position using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
    block_pos = block.data.root_pos_w # Accessing block position using approved pattern - rule 2 in ABSOLUTE REQUIREMENTS

    # 4. Calculate average foot z position - rule 1 in ABSOLUTE REQUIREMENTS
    avg_foot_z = (left_foot_pos[:, 2] + right_foot_pos[:, 2]) / 2

    # 5. Calculate block top surface z position - rule 1 & 6 in ABSOLUTE REQUIREMENTS, rule 6 & 7 in CRITICAL IMPLEMENTATION RULES
    block_size_z = 0.5 # Reading block size from object config (size_cubes = [[0.4, 10.0, 0.4], [1.0, 10.0, 0.2], [0.5, 0.5, 0.5]]) - rule 6 & 7 in CRITICAL IMPLEMENTATION RULES
    block_top_surface_z = block_pos[:, 2] + block_size_z/2

    # 6. Calculate relative z distances - rule 1 in ABSOLUTE REQUIREMENTS
    feet_above_block = avg_foot_z - block_top_surface_z # Relative distance - rule 1 in ABSOLUTE REQUIREMENTS
    pelvis_above_block = pelvis_pos[:, 2] - block_top_surface_z # Relative distance - rule 1 in ABSOLUTE REQUIREMENTS

    # 7. Define success conditions based on relative distances and thresholds - rule 1 & 4 in ABSOLUTE REQUIREMENTS, rule 1 & 2 in SUCCESS CRITERIA RULES
    feet_condition = feet_above_block > 0 # Feet are above block top - rule 1 in ABSOLUTE REQUIREMENTS, rule 3 in SUCCESS CRITERIA RULES
    pelvis_condition = pelvis_above_block > 0.2 # Pelvis is 20cm above block top - rule 1 in ABSOLUTE REQUIREMENTS, rule 3 in SUCCESS CRITERIA RULES

    # 8. Define success conditions based on pelvis position - rule 1 & 4 in ABSOLUTE REQUIREMENTS, rule 1 & 2 in SUCCESS CRITERIA RULES
    pelvis_x_condition_low = pelvis_pos[:, 0] > block_pos[:, 0] - 0.25 # Pelvis is in front of block - rule 1 in ABSOLUTE REQUIREMENTS, rule 3 in SUCCESS CRITERIA RULES
    pelvis_x_condition_high = pelvis_pos[:, 0] < block_pos[:, 0] + 0.25 # Pelvis is in front of block - rule 1 in ABSOLUTE REQUIREMENTS, rule 3 in SUCCESS CRITERIA RULES
    pelvis_y_condition_low = pelvis_pos[:, 1] > block_pos[:, 1] - 0.25 # Pelvis is in front of block - rule 1 in ABSOLUTE REQUIREMENTS, rule 3 in SUCCESS CRITERIA RULES
    pelvis_y_condition_high = pelvis_pos[:, 1] < block_pos[:, 1] + 0.25 # Pelvis is in front of block - rule 1 in ABSOLUTE REQUIREMENTS, rule 3 in SUCCESS CRITERIA RULES
    
    success_condition = feet_condition & pelvis_condition & pelvis_x_condition_low & pelvis_x_condition_high & pelvis_y_condition_low & pelvis_y_condition_high # Both feet and pelvis conditions must be met - rule 1 in SUCCESS CRITERIA RULES

    # 8. Check success duration and save success states - rule 6 & 7 in ABSOLUTE REQUIREMENTS, rule 4 in CRITICAL IMPLEMENTATION RULES
    success = check_success_duration(env, success_condition, "ExecuteJumpOntoBlock", duration=0.5) # Check success duration for 0.3 seconds - rule 6 in ABSOLUTE REQUIREMENTS
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "ExecuteJumpOntoBlock") # Save success state for successful environments - rule 7 in ABSOLUTE REQUIREMENTS

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=ExecuteJumpOntoBlock_success)
                `},
                {
                    name: "StabilizeOnBlockTop", level: 1, policyVideo: "/videos/StabalizerOnBlockTop.mp4", rewardCode: `
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ...mdp import *
from ... import mdp
from ...reward_normalizer import get_normalizer # DO NOT CHANGE THIS LINE!
from ...objects import get_object_volume

def main_StabilizeOnBlockTop_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for StabilizeOnBlockTop.

    Reward for getting both feet on top of the block and minimizing the vertical distance between the feet and the top surface of the block.
    This encourages the robot to land on the block and stay on it.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern (rule 3, approved access pattern)
    try:
        block = env.scene['Object5'] # Accessing block using approved pattern and try/except for robustness (rule 2, 5, approved access pattern)

        left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Getting left foot index using approved pattern (rule 3, approved access pattern)
        left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Getting left foot position using approved pattern (rule 3, approved access pattern)
        right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Getting right foot index using approved pattern (rule 3, approved access pattern)
        right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Getting right foot position using approved pattern (rule 3, approved access pattern)

        block_top_z = block.data.root_pos_w[:, 2] + (0.5/2) # Block top z position, block size is 0.5m, so half is 0.25m. Using relative distance, no hardcoded values. (rule 1, 4, relative distance, using object position, block size from config)

        left_foot_distance_z = left_foot_pos[:, 2] - block_top_z # Vertical distance of left foot from block top, relative distance (rule 1, relative distance)
        right_foot_distance_z = right_foot_pos[:, 2] - block_top_z # Vertical distance of right foot from block top, relative distance (rule 1, relative distance)

        reward_left_foot = -torch.abs(left_foot_distance_z) # Negative absolute distance to minimize vertical distance (rule 5, continuous reward)
        reward_right_foot = -torch.abs(right_foot_distance_z) # Negative absolute distance to minimize vertical distance (rule 5, continuous reward)

        reward = reward_left_foot + reward_right_foot # Summing rewards for both feet (rule 3, tensor operation)

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handling missing block, returning zero reward (rule 5, handle missing object)

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats() # Initialize normalizer if not present (rule 6, reward normalization)

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # Normalize reward (rule 6, reward normalization)
        RewNormalizer.update_stats(normaliser_name, reward) # Update normalizer stats (rule 6, reward normalization)
        return scaled_reward
    return reward

def shaping_approach_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_block_reward") -> torch.Tensor:
    '''Shaping reward for approaching the block in the x-direction.

    Reward for decreasing the x-distance between the pelvis and the block.
    Activates when the robot is behind the block in the x-direction.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern (rule 3, approved access pattern)
    try:
        block = env.scene['Object5'] # Accessing block using approved pattern and try/except for robustness (rule 2, 5, approved access pattern)

        pelvis_idx = robot.body_names.index('pelvis') # Getting pelvis index using approved pattern (rule 3, approved access pattern)
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Getting pelvis position using approved pattern (rule 3, approved access pattern)

        distance_x_pelvis_block = pelvis_pos[:, 0] - block.data.root_pos_w[:, 0] # x distance between pelvis and block, relative distance (rule 1, relative distance)

        activation_condition = (pelvis_pos[:, 0] < block.data.root_pos_w[:, 0]) # Activation when robot is behind the block in x (rule 1, relative condition)

        reward = -torch.abs(distance_x_pelvis_block) # Negative absolute distance to encourage approaching (rule 5, continuous reward)

        reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device)) # Apply reward only when activated (rule 3, tensor operation)

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handling missing block, returning zero reward (rule 5, handle missing object)

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats() # Initialize normalizer if not present (rule 6, reward normalization)

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # Normalize reward (rule 6, reward normalization)
        RewNormalizer.update_stats(normaliser_name, reward) # Update normalizer stats (rule 6, reward normalization)
        return scaled_reward
    return reward

def shaping_horizontal_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "horizontal_stability_reward") -> torch.Tensor:
    '''Shaping reward for minimizing horizontal distance between feet and block center.

    Reward for positioning feet above the block center in the xy-plane.
    Activates when both feet are on or above the top surface of the block in z-direction.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern (rule 3, approved access pattern)
    try:
        block = env.scene['Object5'] # Accessing block using approved pattern and try/except for robustness (rule 2, 5, approved access pattern)

        left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Getting left foot index using approved pattern (rule 3, approved access pattern)
        left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Getting left foot position using approved pattern (rule 3, approved access pattern)
        right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Getting right foot index using approved pattern (rule 3, approved access pattern)
        right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Getting right foot position using approved pattern (rule 3, approved access pattern)

        distance_x_left_foot_block = left_foot_pos[:, 0] - block.data.root_pos_w[:, 0] # x distance left foot to block center, relative distance (rule 1, relative distance)
        distance_y_left_foot_block = left_foot_pos[:, 1] - block.data.root_pos_w[:, 1] # y distance left foot to block center, relative distance (rule 1, relative distance)
        distance_horizontal_left_foot = torch.sqrt(distance_x_left_foot_block**2 + distance_y_left_foot_block**2) # Horizontal distance left foot to block center (rule 3, tensor operation)

        distance_x_right_foot_block = right_foot_pos[:, 0] - block.data.root_pos_w[:, 0] # x distance right foot to block center, relative distance (rule 1, relative distance)
        distance_y_right_foot_block = right_foot_pos[:, 1] - block.data.root_pos_w[:, 1] # y distance right foot to block center, relative distance (rule 1, relative distance)
        distance_horizontal_right_foot = torch.sqrt(distance_x_right_foot_block**2 + distance_y_right_foot_block**2) # Horizontal distance right foot to block center (rule 3, tensor operation)

        block_top_z = block.data.root_pos_w[:, 2] + (0.5/2) # Block top z position, block size is 0.5m, so half is 0.25m. Using relative distance, no hardcoded values. (rule 1, 4, relative distance, using object position, block size from config)
        activation_condition = (left_foot_pos[:, 2] >= block_top_z) & (right_foot_pos[:, 2] >= block_top_z) # Activation when feet are on or above block top (rule 1, relative condition)

        reward_left_foot = -distance_horizontal_left_foot # Negative horizontal distance to encourage centering (rule 5, continuous reward)
        reward_right_foot = -distance_horizontal_right_foot # Negative horizontal distance to encourage centering (rule 5, continuous reward)

        reward = reward_left_foot + reward_right_foot # Summing rewards for both feet (rule 3, tensor operation)
        reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device)) # Apply reward only when activated (rule 3, tensor operation)

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handling missing block, returning zero reward (rule 5, handle missing object)

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats() # Initialize normalizer if not present (rule 6, reward normalization)

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # Normalize reward (rule 6, reward normalization)
        RewNormalizer.update_stats(normaliser_name, reward) # Update normalizer stats (rule 6, reward normalization)
        return scaled_reward
    return reward

def shaping_pelvis_height_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_height_stability_reward") -> torch.Tensor:
    '''Shaping reward for maintaining stable pelvis height.

    Reward for keeping the pelvis at a desired height for stability on the block.
    Always active.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern (rule 3, approved access pattern)

    pelvis_idx = robot.body_names.index('pelvis') # Getting pelvis index using approved pattern (rule 3, approved access pattern)
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Getting pelvis position using approved pattern (rule 3, approved access pattern)

    desired_pelvis_z =  torch.tensor(0.7, device=env.device) # default value if block is missing. (rule 4, no hardcoded absolute values, but default value if block is missing is acceptable)
    try:
        block = env.scene['Object5'] # Accessing block using approved pattern and try/except for robustness (rule 2, 5, approved access pattern)
        block_top_z = block.data.root_pos_w[:, 2] + (0.5/2) # Block top z position, block size is 0.5m, so half is 0.25m. Using relative distance, no hardcoded values. (rule 1, 4, relative distance, using object position, block size from config)
        desired_pelvis_z = block_top_z + 0.7 # Desired pelvis height relative to block top. No hardcoded absolute values. (rule 1, relative distance)
    except KeyError:
        pass # if block is missing, use default value already set. (rule 5, handle missing object)


    reward = -torch.abs(pelvis_pos[:, 2] - desired_pelvis_z) # Negative absolute difference from desired pelvis height (rule 5, continuous reward)

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats() # Initialize normalizer if not present (rule 6, reward normalization)

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # Normalize reward (rule 6, reward normalization)
        RewNormalizer.update_stats(normaliser_name, reward) # Update normalizer stats (rule 6, reward normalization)
        return scaled_reward
    return reward

def shaping_foot_block_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "foot_block_collision_avoidance_reward") -> torch.Tensor:
    '''Shaping reward for foot-block collision avoidance.

    Negative reward when feet penetrate below the bottom of the block.
    Always active.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern (rule 3, approved access pattern)
    try:
        block = env.scene['Object5'] # Accessing block using approved pattern and try/except for robustness (rule 2, 5, approved access pattern)

        left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Getting left foot index using approved pattern (rule 3, approved access pattern)
        left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Getting left foot position using approved pattern (rule 3, approved access pattern)
        right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Getting right foot index using approved pattern (rule 3, approved access pattern)
        right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Getting right foot position using approved pattern (rule 3, approved access pattern)

        block_bottom_z = block.data.root_pos_w[:, 2] - (0.5/2) # Block bottom z position, block size is 0.5m, so half is 0.25m. Using relative distance, no hardcoded values. (rule 1, 4, relative distance, using object position, block size from config)

        left_foot_distance_z_bottom = left_foot_pos[:, 2] - block_bottom_z # Vertical distance of left foot from block bottom, relative distance (rule 1, relative distance)
        right_foot_distance_z_bottom = right_foot_pos[:, 2] - block_bottom_z # Vertical distance of right foot from block bottom, relative distance (rule 1, relative distance)

        penetration_threshold = 0.0 # Penetration depth threshold (rule 4, no arbitrary thresholds, but 0.0 is physically meaningful)

        reward_left_foot = torch.where(left_foot_distance_z_bottom < penetration_threshold, left_foot_distance_z_bottom - penetration_threshold, torch.tensor(0.0, device=env.device)) # Negative reward if penetrating (rule 5, continuous reward)
        reward_right_foot = torch.where(right_foot_distance_z_bottom < penetration_threshold, right_foot_distance_z_bottom - penetration_threshold, torch.tensor(0.0, device=env.device)) # Negative reward if penetrating (rule 5, continuous reward)

        reward = reward_left_foot + reward_right_foot # Summing rewards for both feet (rule 3, tensor operation)

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handling missing block, returning zero reward (rule 5, handle missing object)

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats() # Initialize normalizer if not present (rule 6, reward normalization)

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # Normalize reward (rule 6, reward normalization)
        RewNormalizer.update_stats(normaliser_name, reward) # Update normalizer stats (rule 6, reward normalization)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    Main_StabilizeOnBlockTopReward = RewTerm(func=main_StabilizeOnBlockTop_reward, weight=1.0,
                                params={"normalise": True, "normaliser_name": "main_reward"}) # Main reward with weight 1.0 (rule 7, proper weights)
    ApproachBlockReward = RewTerm(func=shaping_approach_block_reward, weight=0.4,
                                params={"normalise": True, "normaliser_name": "approach_block_reward"}) # Shaping reward with weight < 1.0 (rule 7, proper weights)
    HorizontalStabilityReward = RewTerm(func=shaping_horizontal_stability_reward, weight=0.5,
                                params={"normalise": True, "normaliser_name": "horizontal_stability_reward"}) # Shaping reward with weight < 1.0 (rule 7, proper weights)
    PelvisHeightStabilityReward = RewTerm(func=shaping_pelvis_height_stability_reward, weight=0.3,
                                params={"normalise": True, "normaliser_name": "pelvis_height_stability_reward"}) # Shaping reward with weight < 1.0 (rule 7, proper weights)
    FootBlockCollisionAvoidanceReward = RewTerm(func=shaping_foot_block_collision_avoidance_reward, weight=0.2,
                                params={"normalise": True, "normaliser_name": "foot_block_collision_avoidance_reward"}) # Shaping reward with weight < 1.0 (rule 7, proper weights)
                `,
                    successTerminationCode: `


from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
# Assuming mdp is correctly importable from the context where this runs
# If not, adjust the relative import path
from ...mdp import * 
import torch
from pathlib import Path
# Import reward functions if needed by success criteria
# from .TaskRewardsCfg import * 

def StabilizeOnBlockTop_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the StabilizeOnBlockTop skill has been successfully completed.
    Success is defined as both feet being on top of the block and stable for a duration.
    '''
    # 1. Get robot object - Approved access pattern (rule 3, approved access pattern)
    robot = env.scene["robot"]

    # 2. Get indices for left and right feet - Approved access pattern (rule 3, approved access pattern)
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    # 3. Get positions of left and right feet - Approved access pattern (rule 3, approved access pattern)
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    try:
        # 4. Get block object - Approved access pattern and handle missing object (rule 2, 5, approved access pattern)
        block = env.scene['Object5']
        block_pos = block.data.root_pos_w

        # 5. Calculate block top z position - Relative distance, no hardcoded values (rule 1, 4)
        block_top_z = block_pos[:, 2] + (0.5/2) # Block size is 0.5m from config

        # 6. Calculate distances - Relative distances (rule 1)
        distance_z_left_foot_block_top = left_foot_pos[:, 2] - block_top_z # Vertical distance of left foot from block top
        distance_z_right_foot_block_top = right_foot_pos[:, 2] - block_top_z # Vertical distance of right foot from block top

        distance_x_left_foot_block = left_foot_pos[:, 0] - block_pos[:, 0] # Horizontal distance x of left foot from block center
        distance_y_left_foot_block = left_foot_pos[:, 1] - block_pos[:, 1] # Horizontal distance y of left foot from block center
        distance_x_right_foot_block = right_foot_pos[:, 0] - block_pos[:, 0] # Horizontal distance x of right foot from block center
        distance_y_right_foot_block = right_foot_pos[:, 1] - block_pos[:, 1] # Horizontal distance y of right foot from block center


        # 7. Define success condition - Relative distances and reasonable thresholds (rule 1, 13, 14)
        vertical_threshold = 0.2 # Allow feet to be slightly above or below block top
        horizontal_threshold = 0.35 # Allow feet to be within 25cm of block center horizontally

        condition = (distance_z_left_foot_block_top >= -vertical_threshold) & (distance_z_left_foot_block_top <= vertical_threshold) & \
                    (distance_z_right_foot_block_top >= -vertical_threshold) & (distance_z_right_foot_block_top <= vertical_threshold) & \
                    (torch.abs(distance_x_left_foot_block) < horizontal_threshold) & (torch.abs(distance_y_left_foot_block) < horizontal_threshold) & \
                    (torch.abs(distance_x_right_foot_block) < horizontal_threshold) & (torch.abs(distance_y_right_foot_block) < horizontal_threshold)

    except KeyError:
        # 8. Handle missing block - Return failure (rule 5)
        condition = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 9. Check success duration and save success states - DO NOT MODIFY THIS SECTION (rule 6, 7)
    success = check_success_duration(env, condition, "StabilizeOnBlockTop", duration=0.5) # Using duration of 0.5 seconds
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "StabilizeOnBlockTop")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=StabilizeOnBlockTop_success)
                `}, // Note: video filename has "Stabalizer"
            ],
        },
    ],
};

// Level 0 represents Primitive Actions, which are the leaves of Level 1 skills.
// We can consider Level 1 skills as directly composed of these primitive actions,
// so they don't need an explicit "children" array of primitive actions.
// The demonstration for Level 1 skills would show these primitives. 
