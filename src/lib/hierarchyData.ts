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
  rewardCode: "// Placeholder: Obstacle Course Reward Code",
  successTerminationCode: "// Placeholder: Obstacle Course Success Termination Code",
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
from ...reward_normalizer import get_normalizer # DO NOT CHANGE THIS LINE!
from ...objects import get_object_volume

def main_CelebrateOnBlock_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for CelebrateOnBlock.

    This reward combines all shaping rewards to encourage the robot to perform the entire CelebrateOnBlock skill.
    It is a weighted sum of horizontal proximity to the block, stability on the block, raising hands, and collision avoidance.
    '''
    # Initialize reward to zero
    reward = torch.zeros(env.num_envs, device=env.device)

    # Add shaping reward 1: Horizontal Proximity to Block
    reward += shaping_reward_horizontal_proximity_block(env, normalise=normalise, normaliser_name="horizontal_proximity_reward")

    # Add shaping reward 2: Stability on Block
    reward += shaping_reward_stability_on_block(env, normalise=normalise, normaliser_name="stability_reward")

    # Add shaping reward 3: Raising Hands
    reward += shaping_reward_raising_hands(env, normalise=normalise, normaliser_name="raising_hands_reward")

    # Add shaping reward 4: Collision Avoidance
    reward += shaping_reward_collision_avoidance(env, normalise=normalise, normaliser_name="collision_avoidance_reward")

    # Normalize and return
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_horizontal_proximity_block(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "horizontal_proximity_reward") -> torch.Tensor:
    '''Shaping reward for horizontal proximity to the block.
    Encourages the robot to move towards the block in the x-y plane before jumping.
    '''
    try:
        block = env.scene['Object5'] # Access the block using approved pattern and handle potential KeyError
        robot = env.scene["robot"] # Access the robot using approved pattern
        pelvis_idx = robot.body_names.index('pelvis') # Getting the index of the pelvis using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Getting the position of the pelvis using approved pattern

        block_pos = block.data.root_pos_w # Access the block position using approved pattern

        # Calculate the distance vector between the pelvis and the block in x and y directions (relative distance)
        distance_x = pelvis_pos[:, 0] - block_pos[:, 0] # Relative distance in x
        distance_y = pelvis_pos[:, 1] - block_pos[:, 1] # Relative distance in y

        # Reward for being close to the block in x and y directions. Continuous reward based on negative absolute distance.
        reward_x = -torch.abs(distance_x)
        reward_y = -torch.abs(distance_y)

        # Activation condition: activate only when far from block horizontally (e.g., more than 1.0m in x or y direction from the block's center).
        activation_condition = (torch.abs(distance_x) > 1.0) | (torch.abs(distance_y) > 1.0)

        reward = reward_x + reward_y
        reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device)) # Apply activation condition

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    # Normalize and return
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_stability_on_block(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_reward") -> torch.Tensor:
    '''Shaping reward for stability on the block.
    Penalize excessive vertical movement of the pelvis when the feet are on the block.
    '''
    try:
        block = env.scene['Object5'] # Access the block using approved pattern and handle potential KeyError
        robot = env.scene["robot"] # Access the robot using approved pattern
        pelvis_idx = robot.body_names.index('pelvis') # Getting the index of the pelvis using approved pattern
        pelvis_vel_z = robot.data.body_vel_w[:, pelvis_idx][:, 2] # Getting the vertical velocity of the pelvis using approved pattern
        left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Getting the index of the left foot using approved pattern
        right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Getting the index of the right foot using approved pattern
        left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Getting the position of the left foot using approved pattern
        right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Getting the position of the right foot using approved pattern

        block_pos_z = block.data.root_pos_w[:, 2] # z position of the block (bottom) using approved pattern
        block_size_z = 0.5 # Hardcoded block z size from object config (rule 6 & 7 in critical implementation rules)
        block_top_z = block_pos_z + block_size_z # z position of the top of the block

        feet_on_block_condition = (left_foot_pos[:, 2] > block_top_z - 0.1) & (right_foot_pos[:, 2] > block_top_z - 0.1) # feet are roughly on the block (relative height)

        # Reward for low vertical pelvis velocity when feet are on block. Continuous reward based on negative absolute velocity.
        reward_stability = -torch.abs(pelvis_vel_z)

        reward = torch.where(feet_on_block_condition, reward_stability, torch.tensor(0.0, device=env.device)) # Activate only when feet are on block

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    # Normalize and return
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_raising_hands(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "raising_hands_reward") -> torch.Tensor:
    '''Shaping reward for raising hands above the pelvis during celebration jumps.
    Encourages the celebratory action after the robot is stable on the block.
    '''
    try:
        block = env.scene['Object5'] # Access the block using approved pattern and handle potential KeyError
        robot = env.scene["robot"] # Access the robot using approved pattern
        left_hand_idx = robot.body_names.index('left_palm_link') # Getting the index of the left hand using approved pattern
        right_hand_idx = robot.body_names.index('right_palm_link') # Getting the index of the right hand using approved pattern
        left_hand_pos = robot.data.body_pos_w[:, left_hand_idx] # Getting the position of the left hand using approved pattern
        right_hand_pos = robot.data.body_pos_w[:, right_hand_idx] # Getting the position of the right hand using approved pattern
        pelvis_idx = robot.body_names.index('pelvis') # Getting the index of the pelvis using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Getting the position of the pelvis using approved pattern
        left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Getting the index of the left foot using approved pattern
        right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Getting the index of the right foot using approved pattern
        left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Getting the position of the left foot using approved pattern
        right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Getting the position of the right foot using approved pattern

        block_pos_z = block.data.root_pos_w[:, 2] # z position of the block (bottom) using approved pattern
        block_size_z = 0.5 # Hardcoded block z size from object config (rule 6 & 7 in critical implementation rules)
        block_top_z = block_pos_z + block_size_z # z position of the top of the block

        feet_on_block_condition = (left_foot_pos[:, 2] > block_top_z - 0.1) & (right_foot_pos[:, 2] > block_top_z - 0.1) # feet are roughly on the block (relative height)

        # Calculate the distance vector between the hands and the pelvis in z direction (relative distance)
        distance_left_hand_pelvis_z = left_hand_pos[:, 2] - pelvis_pos[:, 2] # Relative distance in z
        distance_right_hand_pelvis_z = right_hand_pos[:, 2] - pelvis_pos[:, 2] # Relative distance in z

        # Reward for hands being above pelvis in z direction when feet are on block. Continuous reward using relu to only reward positive distance.
        reward_hands_up = -(torch.relu(-distance_left_hand_pelvis_z) + torch.relu(-distance_right_hand_pelvis_z)) # only positive when hands are above pelvis

        reward = torch.where(feet_on_block_condition, reward_hands_up, torch.tensor(0.0, device=env.device)) # Activate only when feet are on block

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    # Normalize and return
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_collision_avoidance(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    '''Shaping reward for collision avoidance.
    Penalize close proximity between hands and feet, and hands and block during celebration jumps.
    '''
    try:
        block = env.scene['Object5'] # Access the block using approved pattern and handle potential KeyError
        robot = env.scene["robot"] # Access the robot using approved pattern
        left_hand_idx = robot.body_names.index('left_palm_link') # Getting the index of the left hand using approved pattern
        right_hand_idx = robot.body_names.index('right_palm_link') # Getting the index of the right hand using approved pattern
        left_hand_pos = robot.data.body_pos_w[:, left_hand_idx] # Getting the position of the left hand using approved pattern
        right_hand_pos = robot.data.body_pos_w[:, right_hand_idx] # Getting the position of the right hand using approved pattern
        left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Getting the index of the left foot using approved pattern
        right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Getting the index of the right foot using approved pattern
        left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Getting the position of the left foot using approved pattern
        right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Getting the position of the right foot using approved pattern

        block_pos = block.data.root_pos_w # Access the block position using approved pattern

        # Calculate distances between hands and feet (relative distances)
        distance_left_hand_left_foot = torch.sqrt(torch.sum((left_hand_pos - left_foot_pos)**2, dim=-1, keepdim=True)) # Relative distance
        distance_right_hand_right_foot = torch.sqrt(torch.sum((right_hand_pos - right_foot_pos)**2, dim=-1, keepdim=True)) # Relative distance
        distance_left_hand_right_foot = torch.sqrt(torch.sum((left_hand_pos - right_foot_pos)**2, dim=-1, keepdim=True)) # Relative distance
        distance_right_hand_left_foot = torch.sqrt(torch.sum((right_hand_pos - left_foot_pos)**2, dim=-1, keepdim=True)) # Relative distance

        # Calculate distances between hands and block (relative distances)
        distance_left_hand_block = torch.sqrt(torch.sum((left_hand_pos - block_pos)**2, dim=-1, keepdim=True)) # Relative distance
        distance_right_hand_block = torch.sqrt(torch.sum((right_hand_pos - block_pos)**2, dim=-1, keepdim=True)) # Relative distance

        collision_threshold = 0.3 # Collision threshold (relative distance)
        # Collision avoidance reward, penalize if hands are too close to feet or block. Continuous reward using relu to penalize distances below threshold.
        reward_collision_hands_feet = -(torch.relu(collision_threshold - distance_left_hand_left_foot) +
                                         torch.relu(collision_threshold - distance_right_hand_right_foot) +
                                         torch.relu(collision_threshold - distance_left_hand_right_foot) +
                                         torch.relu(collision_threshold - distance_right_hand_left_foot))
        reward_collision_hands_block = -(torch.relu(collision_threshold - distance_left_hand_block) +
                                          torch.relu(collision_threshold - distance_right_hand_block))

        reward = reward_collision_hands_feet + reward_collision_hands_block

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

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
    Main_CelebrateOnBlockReward = RewTerm(func=main_CelebrateOnBlock_reward, weight=1.0,
                                params={"normalise": True, "normaliser_name": "main_reward"})

    ShapingRewardHorizontalProximityBlock = RewTerm(func=shaping_reward_horizontal_proximity_block, weight=0.4,
                                params={"normalise": True, "normaliser_name": "horizontal_proximity_reward"})

    ShapingRewardStabilityOnBlock = RewTerm(func=shaping_reward_stability_on_block, weight=0.3,
                                params={"normalise": True, "normaliser_name": "stability_reward"})

    ShapingRewardRaisingHands = RewTerm(func=shaping_reward_raising_hands, weight=0.3,
                                params={"normalise": True, "normaliser_name": "raising_hands_reward"})

    ShapingRewardCollisionAvoidance = RewTerm(func=shaping_reward_collision_avoidance, weight=0.1, # Reduced weight as it's a penalty and should be less dominant
                                params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})
`,
      successTerminationCode: "// Placeholder: JumpOverLowWall Success Termination Code",
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
      rewardCode: "// Placeholder: PushLargeSphereToHighWall Reward Code",
      successTerminationCode: "// Placeholder: PushLargeSphereToHighWall Success Termination Code",
      policyVideo: "/videos/L2_PushLargeSphereToHighWall.mp4",
      children: [
        { name: "WalkToLargeSphere", level: 1, policyVideo: "" }, // No specific video found
        { name: "PositionHandsForPushLargeSphere", level: 1, policyVideo: "/videos/PositionHandsForPushLargeSphere.mp4" },
        { name: "PushLargeSphereForward", level: 1, policyVideo: "/videos/PushLargeSphereForward.mp4" },
        { name: "EnsureHighWallFalls", level: 1, policyVideo: "/videos/EnsureHighWallFalls.mp4" },
      ],
    },
    {
      name: "KickSmallSpherePastBlock",
      level: 2,
      rewardCode: "// Placeholder: KickSmallSpherePastBlock Reward Code",
      successTerminationCode: "// Placeholder: KickSmallSpherePastBlock Success Termination Code",
      policyVideo: "/videos/L2_KickSmallSpherePastBlock.mp4",
      children: [
        { name: "WalkToSmallSphere", level: 1, policyVideo: "/videos/WalkToSmallSphere.mp4" },
        { name: "ExecuteKickSmallSphereForward", level: 1, policyVideo: "/videos/ExecuteKickSmallSphereForward.mp4" },
      ],
    },
    {
      name: "JumpOntoBlock",
      level: 2,
      rewardCode: "// Placeholder: JumpOntoBlock Reward Code",
      successTerminationCode: "// Placeholder: JumpOntoBlock Success Termination Code",
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
