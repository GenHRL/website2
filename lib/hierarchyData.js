export const hierarchyData = {
  id: 'obstacle-course',
  title: 'Obstacle Course',
  description: 'The robot should jump over a low wall, push a large sphere into a high wall to knock it down and pass over it. The robot should then walk to a small sphere and kick it past a block. Finally the robot should walk to the block and jump onto it.',
  paperTitle: "GenHRL: Generative Hierarchical Reinforcement Learning",
  paperAbstract: "Defining effective multi-level skill hierarchies and their corresponding learning objectives is a core challenge in robotics and reinforcement learning. Large Language Models (LLMs) offer powerful new capabilities for tackling this challenge through automated generation and reasoning. This paper introduces GenHRL, an LLM-driven framework that automates the pipeline from high-level natural language task descriptions to learned hierarchical skills. GenHRL autonomously generates: (1) task-specific simulation environments, (2) multi-level skill decompositions, and (3) executable code defining intrinsic reward and termination functions for each skill. This automation avoids the need for manual reward engineering, predefined skill sets, offline datasets, and enables end-to-end hierarchical policy learning via standard reinforcement learning algorithms. Empirical evaluations on complex robotic humanoid simulation tasks demonstrate that GenHRL significantly enhances learning efficiency and final performance compared to non-hierarchical baselines.",
  videoPlaceholder: '/videos/obstacle_course_overview.mp4',
  rewardFunctionCode: '# Placeholder for Obstacle Course Overall Reward Function\n# This could describe the reward for completing the entire sequence.\nprint("Reward for entire Obstacle Course")',
  successFunctionCode: '# Placeholder for Obstacle Course Overall Success Condition\n# This could describe the criteria for successfully completing all sub-tasks.\nprint("Success for entire Obstacle Course")',
  children: [
    {
      id: 'level2-jump',
      title: 'JumpOverLowWall',
      color: 'red',
      parentId: 'obstacle-course',
      videoPlaceholder: '/videos/level2_jump_over_low_wall.mp4',
      rewardFunctionCode: '# Placeholder for JumpOverLowWall Reward Function (Level 2)\nprint("Reward for JumpOverLowWall")',
      successFunctionCode: '# Placeholder for JumpOverLowWall Success Function (Level 2)\nprint("Success for JumpOverLowWall")',
      children: [
        { 
          id: 'level1-jump-walk', title: 'WalkToLowWall', color: 'orange',
          parentId: 'level2-jump',
          videoPlaceholder: '/videos/level1_walk_to_low_wall.mp4',
          rewardFunctionCode: '# Placeholder for WalkToLowWall Reward Function (Level 1)\nprint("Reward for WalkToLowWall")',
          successFunctionCode: '# Placeholder for WalkToLowWall Success Function (Level 1)\nprint("Success for WalkToLowWall")'
        },
        { 
          id: 'level1-jump-prepare', title: 'PrepareForJumpOverLowWall', color: 'orange',
          parentId: 'level2-jump',
          videoPlaceholder: '/videos/level1_prepare_for_jump.mp4',
          rewardFunctionCode: '# Placeholder for PrepareForJumpOverLowWall Reward Function (Level 1)\nprint("Reward for PrepareForJumpOverLowWall")',
          successFunctionCode: '# Placeholder for PrepareForJumpOverLowWall Success Function (Level 1)\nprint("Success for PrepareForJumpOverLowWall")'
        },
        { 
          id: 'level1-jump-execute', title: 'ExecuteJumpOverLowWall', color: 'red-dark',
          parentId: 'level2-jump',
          videoPlaceholder: '/videos/level1_execute_jump.mp4',
          rewardFunctionCode: '# Placeholder for ExecuteJumpOverLowWall Reward Function (Level 1)\nprint("Reward for ExecuteJumpOverLowWall")',
          successFunctionCode: '# Placeholder for ExecuteJumpOverLowWall Success Function (Level 1)\nprint("Success for ExecuteJumpOverLowWall")'
        },
        { 
          id: 'level1-jump-land', title: 'LandStablyAfterLowWall', color: 'pink',
          parentId: 'level2-jump',
          videoPlaceholder: '/videos/level1_land_stably.mp4',
          rewardFunctionCode: '# Placeholder for LandStablyAfterLowWall Reward Function (Level 1)\nprint("Reward for LandStablyAfterLowWall")',
          successFunctionCode: '# Placeholder for LandStablyAfterLowWall Success Function (Level 1)\nprint("Success for LandStablyAfterLowWall")'
        },
      ],
    },
    {
      id: 'level2-push',
      title: 'PushLargeSphereToHighWall',
      color: 'purple',
      parentId: 'obstacle-course',
      videoPlaceholder: '/videos/level2_push_large_sphere.mp4',
      rewardFunctionCode: '# Placeholder for PushLargeSphereToHighWall Reward Function (Level 2)\nprint("Reward for PushLargeSphereToHighWall")',
      successFunctionCode: '# Placeholder for PushLargeSphereToHighWall Success Function (Level 2)\nprint("Success for PushLargeSphereToHighWall")',
      children: [
        { id: 'level1-push-walk', title: 'WalkToLargeSphere', color: 'magenta', parentId: 'level2-push', videoPlaceholder: '/videos/level1_walk_to_large_sphere.mp4', rewardFunctionCode: '# Placeholder for WalkToLargeSphere Reward Function (Level 1)\nprint("Reward for WalkToLargeSphere")', successFunctionCode: '# Placeholder for WalkToLargeSphere Success Function (Level 1)\nprint("Success for WalkToLargeSphere")' },
        { id: 'level1-push-position', title: 'PositionHandsForPushLargeSphere', color: 'magenta', parentId: 'level2-push', videoPlaceholder: '/videos/level1_position_hands_push.mp4', rewardFunctionCode: '# Placeholder for PositionHandsForPushLargeSphere Reward Function (Level 1)\nprint("Reward for PositionHandsForPushLargeSphere")', successFunctionCode: '# Placeholder for PositionHandsForPushLargeSphere Success Function (Level 1)\nprint("Success for PositionHandsForPushLargeSphere")' },
        { id: 'level1-push-execute', title: 'PushLargeSphereForward', color: 'purple-dark', parentId: 'level2-push', videoPlaceholder: '/videos/level1_push_large_sphere_forward.mp4', rewardFunctionCode: '# Placeholder for PushLargeSphereForward Reward Function (Level 1)\nprint("Reward for PushLargeSphereForward")', successFunctionCode: '# Placeholder for PushLargeSphereForward Success Function (Level 1)\nprint("Success for PushLargeSphereForward")' },
        { id: 'level1-push-ensure', title: 'EnsureHighWallFalls', color: 'blue-violet', parentId: 'level2-push', videoPlaceholder: '/videos/level1_ensure_high_wall_falls.mp4', rewardFunctionCode: '# Placeholder for EnsureHighWallFalls Reward Function (Level 1)\nprint("Reward for EnsureHighWallFalls")', successFunctionCode: '# Placeholder for EnsureHighWallFalls Success Function (Level 1)\nprint("Success for EnsureHighWallFalls")' },
      ],
    },
    {
      id: 'level2-kick',
      title: 'KickSmallSpherePastBlock',
      color: 'blue',
      parentId: 'obstacle-course',
      videoPlaceholder: '/videos/level2_kick_small_sphere.mp4',
      rewardFunctionCode: '# Placeholder for KickSmallSpherePastBlock Reward Function (Level 2)\nprint("Reward for KickSmallSpherePastBlock")',
      successFunctionCode: '# Placeholder for KickSmallSpherePastBlock Success Function (Level 2)\nprint("Success for KickSmallSpherePastBlock")',
      children: [
        { id: 'level1-kick-walk', title: 'WalkToSmallSphere', color: 'blue-light', parentId: 'level2-kick', videoPlaceholder: '/videos/level1_walk_to_small_sphere.mp4', rewardFunctionCode: '# Placeholder for WalkToSmallSphere Reward Function (Level 1)\nprint("Reward for WalkToSmallSphere")', successFunctionCode: '# Placeholder for WalkToSmallSphere Success Function (Level 1)\nprint("Success for WalkToSmallSphere")' },
        { id: 'level1-kick-execute', title: 'ExecuteKickSmallSphereForward', color: 'blue-dark', parentId: 'level2-kick', videoPlaceholder: '/videos/level1_execute_kick_small_sphere.mp4', rewardFunctionCode: '# Placeholder for ExecuteKickSmallSphereForward Reward Function (Level 1)\nprint("Reward for ExecuteKickSmallSphereForward")', successFunctionCode: '# Placeholder for ExecuteKickSmallSphereForward Success Function (Level 1)\nprint("Success for ExecuteKickSmallSphereForward")' },
      ],
    },
    {
      id: 'level2-jump-block',
      title: 'JumpOntoBlock',
      color: 'green',
      parentId: 'obstacle-course',
      videoPlaceholder: '/videos/level2_jump_onto_block.mp4',
      rewardFunctionCode: '# Placeholder for JumpOntoBlock Reward Function (Level 2)\nprint("Reward for JumpOntoBlock")',
      successFunctionCode: '# Placeholder for JumpOntoBlock Success Function (Level 2)\nprint("Success for JumpOntoBlock")',
      children: [
        { id: 'level1-jump-block-walk', title: 'WalkToBlock', color: 'green-medium', parentId: 'level2-jump-block', videoPlaceholder: '/videos/level1_walk_to_block.mp4', rewardFunctionCode: '# Placeholder for WalkToBlock Reward Function (Level 1)\nprint("Reward for WalkToBlock")', successFunctionCode: '# Placeholder for WalkToBlock Success Function (Level 1)\nprint("Success for WalkToBlock")' },
        { id: 'level1-jump-block-prepare', title: 'PrepareForJumpOntoBlock', color: 'green-medium', parentId: 'level2-jump-block', videoPlaceholder: '/videos/level1_prepare_jump_block.mp4', rewardFunctionCode: '# Placeholder for PrepareForJumpOntoBlock Reward Function (Level 1)\nprint("Reward for PrepareForJumpOntoBlock")', successFunctionCode: '# Placeholder for PrepareForJumpOntoBlock Success Function (Level 1)\nprint("Success for PrepareForJumpOntoBlock")' },
        { id: 'level1-jump-block-execute', title: 'ExecuteJumpOntoBlock', color: 'green-dark', parentId: 'level2-jump-block', videoPlaceholder: '/videos/level1_execute_jump_block.mp4', rewardFunctionCode: '# Placeholder for ExecuteJumpOntoBlock Reward Function (Level 1)\nprint("Reward for ExecuteJumpOntoBlock")', successFunctionCode: '# Placeholder for ExecuteJumpOntoBlock Success Function (Level 1)\nprint("Success for ExecuteJumpOntoBlock")' },
        { id: 'level1-jump-block-stabilize', title: 'StabilizeOnBlockTop', color: 'green-dark', parentId: 'level2-jump-block', videoPlaceholder: '/videos/level1_stabilize_block_top.mp4', rewardFunctionCode: '# Placeholder for StabilizeOnBlockTop Reward Function (Level 1)\nprint("Reward for StabilizeOnBlockTop")', successFunctionCode: '# Placeholder for StabilizeOnBlockTop Success Function (Level 1)\nprint("Success for StabilizeOnBlockTop")' },
      ],
    },
  ],
  primitiveActions: {
    id: 'level0',
    title: 'Primitive actions',
  }
}; 