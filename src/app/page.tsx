import SkillTree from '@/components/SkillTree';

export default function HomePage() {
  const openReviewUrl = "https://openreview.net/forum?id=vPwAh0eL0D&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3Drobot-learning.org%2FCoRL%2F2025%2FConference%2FAuthors%23your-submissions)"; // Replace # with your actual OpenReview URL

  return (
    <main className="container mx-auto p-4">
      <header className="my-6 text-center">
        <h1 className="text-4xl font-bold">GenHRL: Generative Hierarchical Reinforcement Learning</h1>
        <p className="text-lg text-gray-700 mt-2">Authors Anonymous</p>
        
        <div className="mt-6 max-w-3xl mx-auto text-left">
          <h2 className="text-2xl font-semibold mb-2 text-center">Abstract</h2>
          <p className="text-gray-700 leading-relaxed">
            Defining effective multi-level skill hierarchies and their corresponding learning objectives is a core challenge in robotics and reinforcement learning. Large Language Models (LLMs) offer powerful new capabilities for tackling this challenge through automated generation and reasoning. This paper introduces GenHRL, an LLM-driven framework that automates the pipeline from high-level natural language task descriptions to learned hierarchical skills. GenHRL autonomously generates: (1) task-specific simulation environments, (2) multi-level skill decompositions, and (3) executable code defining intrinsic reward and termination functions for each skill. This automation avoids the need for manual reward engineering, predefined skill sets, offline datasets, and enables end-to-end hierarchical policy learning via standard reinforcement learning algorithms. Empirical evaluations on complex robotic humanoid simulation tasks demonstrate that GenHRL significantly enhances learning efficiency and final performance compared to non-hierarchical baselines.
          </p>
          <div className="mt-4 text-center">
            <a
              href={openReviewUrl}
              target="https://openreview.net/forum?id=vPwAh0eL0D&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3Drobot-learning.org%2FCoRL%2F2025%2FConference%2FAuthors%23your-submissions)"
              rel="noopener noreferrer"
              className="inline-block bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition-colors"
            >
              View on OpenReview
            </a>
          </div>
        </div>

        <div className="my-8 max-w-4xl mx-auto">
          <h2 className="text-2xl font-semibold mb-4 text-center">Method Overview</h2>
          <div className="w-full">
            <img 
              src="/method_drawing.svg"  // Updated to new SVG file
              alt="Method Diagram"
              className="h-auto w-full max-w-2xl mx-auto block" // Styling
            />
          </div>
          <div className="mt-4 text-left">
            <h3 className="text-xl font-semibold mb-2">Method Description:</h3>
            <p className="text-gray-700 leading-relaxed">
            Our GenHRL system uses Large Language Models (LLMs) to automatically build complex robot skills from simple language instructions. As shown above, the process starts (Stage 1) when a user provides a task description. The LLM interprets this, designs a suitable simulation environment, and breaks the task down into a multi-level hierarchy of described skills. Next (Stage 2), the LLM translates these skill descriptions into executable code that defines the goals (rewards) and completion rules (terminations) for learning each skill, which can be optionally checked by a human. Finally (Stage 3), GenHRL uses this generated environment and code to automatically train policies for the entire hierarchy using reinforcement learning, resulting in an agent capable of performing the complex task.
            </p>
          </div>
        </div>

        <div className="text-center">
          <p className="text-xl text-gray-600 mt-8"> 
            Interactive Skill Hierarchy
          </p>
        </div>
      </header>
      <SkillTree />
      <footer className="mt-8 text-center text-gray-500">
        <p>Level 0 skills are Primitive Actions and are implicitly part of Level 1 skills.</p>
      </footer>
    </main>
  );
}
