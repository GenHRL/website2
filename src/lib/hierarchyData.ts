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
  policyVideo: "", // Placeholder: No video for top level, or specify a path
  children: [
    {
      name: "JumpOverLowWall",
      level: 2,
      rewardCode: "// Placeholder: JumpOverLowWall Reward Code",
      successTerminationCode: "// Placeholder: JumpOverLowWall Success Termination Code",
      policyVideo: "", // e.g., "/videos/JumpOverLowWall.mp4"
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
      policyVideo: "", // e.g., "/videos/PushLargeSphereToHighWall.mp4"
      children: [
        { name: "WalkToLargeSphere", level: 1 },
        { name: "PositionHandsForPushLargeSphere", level: 1 },
        { name: "PushLargeSphereForward", level: 1 },
        { name: "EnsureHighWallFalls", level: 1 },
      ],
    },
    {
      name: "KickSmallSpherePastBlock",
      level: 2,
      rewardCode: "// Placeholder: KickSmallSpherePastBlock Reward Code",
      successTerminationCode: "// Placeholder: KickSmallSpherePastBlock Success Termination Code",
      policyVideo: "", // e.g., "/videos/KickSmallSpherePastBlock.mp4"
      children: [
        { name: "WalkToSmallSphere", level: 1 },
        { name: "ExecuteKickSmallSphereForward", level: 1 },
      ],
    },
    {
      name: "JumpOntoBlock",
      level: 2,
      rewardCode: "// Placeholder: JumpOntoBlock Reward Code",
      successTerminationCode: "// Placeholder: JumpOntoBlock Success Termination Code",
      policyVideo: "", // e.g., "/videos/JumpOntoBlock.mp4"
      children: [
        { name: "WalkToBlock", level: 1 },
        { name: "PrepareForJumpOntoBlock", level: 1 },
        { name: "ExecuteJumpOntoBlock", level: 1 },
        { name: "StabilizeOnBlockTop", level: 1 },
      ],
    },
  ],
};

// Level 0 represents Primitive Actions, which are the leaves of Level 1 skills.
// We can consider Level 1 skills as directly composed of these primitive actions,
// so they don't need an explicit "children" array of primitive actions.
// The demonstration for Level 1 skills would show these primitives. 