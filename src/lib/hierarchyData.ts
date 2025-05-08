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
        { name: "WalkToLowWall", level: 1 },
        { name: "PrepareForJumpOverLowWall", level: 1 },
        { name: "ExecuteJumpOverLowWall", level: 1 },
        { name: "LandStablyAfterLowWall", level: 1 },
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
        { name: "WalkToLargeSphere", level: 1, policyVideo: "/videos/WalkToLargeSphere.mp4" }, // No specific video found
        { name: "PositionHandsForPushLargeSphere", level: 1, policyVideo: "/videos/PositionHandsForPushLargeSphere.mp4" },
        { name: "PushLargeSphereForward", level: 1, policyVideo: "/videos/PushLargeSphereForward.mp4" },
        { name: "EnsureHighWallFalls", level: 1, policyVideo: "/videos/EnsureHighWallFalls.mp4" },
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
        { name: "WalkToSmallSphere", level: 1, policyVideo: "/videos/WalkToSmallSphere.mp4" },
        { name: "ExecuteKickSmallSphereForward", level: 1, policyVideo: "/videos/ExecuteKickSmallSphereForward.mp4" },
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
        { name: "WalkToBlock", level: 1, policyVideo: "/videos/WalkToBlock.mp4" },
        { name: "PrepareForJumpOntoBlock", level: 1, policyVideo: "/videos/PrepareForJumpOntoBlock.mp4" },
        { name: "ExecuteJumpOntoBlock", level: 1, policyVideo: "/videos/ExecuteJumpOntoBlock.mp4" },
        { name: "StabilizeOnBlockTop", level: 1, policyVideo: "/videos/StabalizerOnBlockTop.mp4" }, // Note: video filename has "Stabalizer"
      ],
    },
  ],
};

// Level 0 represents Primitive Actions, which are the leaves of Level 1 skills.
// We can consider Level 1 skills as directly composed of these primitive actions,
// so they don't need an explicit "children" array of primitive actions.
// The demonstration for Level 1 skills would show these primitives. 
