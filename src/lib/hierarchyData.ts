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
      rewardCode: "// Placeholder: JumpOverLowWall Reward Code",
      successTerminationCode: "// Placeholder: JumpOverLowWall Success Termination Code",
      policyVideo: "/videos/L2_JumpOverLowWall.mp4",
      children: [
        { name: "WalkToLowWall", level: 1, policyVideo: "/videos/WalkToWall.mp4" },
        { name: "PrepareForJumpOverLowWall", level: 1, policyVideo: "/videos/PrepareToJump.mp4" },
        { name: "ExecuteJumpOverLowWall", level: 1, policyVideo: "/videos/ExecuteJumpOverLowWall.mp4" },
        { name: "LandStablyAfterLowWall", level: 1, policyVideo: "/videos/LandStablyAfterLowWall.mp4" },
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