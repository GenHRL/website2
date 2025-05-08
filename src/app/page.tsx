import SkillTree from '@/components/SkillTree';
import Image from 'next/image';

// This will be replaced at build time by the actual value of NEXT_PUBLIC_BASE_PATH
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || ""; 

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
            <Image 
              src={`${basePath}/method_drawing.svg`}  // Construct the path with basePath
              alt="Method Diagram"
              width={800} // Placeholder width, adjust as needed
              height={600} // Placeholder height, adjust as needed
              className="h-auto w-full max-w-2xl mx-auto block" // Styling
            />
          </div>
          <div className="mt-4 text-left">
            <h3 className="text-xl font-semibold mb-2">Method Description:</h3>
            <p className="text-gray-700 leading-relaxed">
              Our GenHRL system employs Large Language Models (LLMs) to automatically construct complex robot skills from simple language instructions. The process, illustrated above, unfolds in three main stages:
            </p>
            <ul className="list-disc list-outside text-gray-700 leading-relaxed ml-5 mt-2">
              <li className="mb-2">
                <strong>Stage 1: Task Interpretation and Decomposition.</strong> The user provides a task description. The LLM then interprets this, designs a suitable simulation environment, and breaks the task down into a multi-level hierarchy of described skills.
              </li>
              <li className="mb-2">
                <strong>Stage 2: Code Generation.</strong> The LLM translates these skill descriptions into executable code that defines the goals (rewards) and completion rules (terminations) for learning each skill. This generated code can be optionally checked by a human.
              </li>
              <li className="mb-2">
                <strong>Stage 3: Hierarchical Reinforcement Learning.</strong> GenHRL utilizes this generated environment and code to automatically train policies for the entire skill hierarchy using reinforcement learning, ultimately resulting in an agent capable of performing the complex task.
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-8 max-w-4xl mx-auto text-left">
          <h3 className="text-xl font-semibold mb-2">Demonstration Context:</h3>
          <p className="text-gray-700 leading-relaxed mb-2">
            Below, we showcase the GenHRL system in action. For this demonstration, GenHRL was provided with the following high-level task description:
          </p>
          <div className="bg-gray-100 border border-gray-300 p-3 rounded-md my-3 shadow-sm">
            <em className="text-gray-700 leading-relaxed">
              &quot;The robot should jump over a low wall, push a large sphere into a high wall to knock it down and pass over it. The robot should then walk to a small sphere and kick it past a block. Finally the robot should walk to the block and jump onto it.&quot;
            </em>
          </div>
          <p className="text-gray-700 leading-relaxed mb-2">
            From this instruction, the system first generates the corresponding simulation environment, including the layout of all necessary objects. It then automatically designs the multi-level skill hierarchy that you can explore interactively below.
          </p>
          <p className="text-gray-700 leading-relaxed mb-2">
            Furthermore, GenHRL generates the underlying executable code for each skill, including the reward functions that guide learning and the success termination conditions that define task completion. Each higher-level skill in the hierarchy is composed of and utilizes the skills immediately below it to achieve its objective.
          </p>
          <p className="text-gray-700 leading-relaxed">
            To explore the generated hierarchy, simply click on any skill box. This will reveal tabs for the <strong>Video Demo</strong>, which shows a successfully trained policy for that skill, as well as the <strong>Reward Code</strong> and <strong>Success Code</strong> generated by the LLM.
          </p>
        </div>

        <div className="text-center">
          <p className="text-xl text-gray-600 mt-8"> 
            ----------------------------------------------------------
          </p>
        </div>
      </header>

      {/* Instructional text for SkillTree in an info box */}
      <div className="my-8 p-4 bg-blue-50 border border-blue-200 rounded-lg max-w-3xl mx-auto shadow">
        <p className="text-md text-blue-800 leading-relaxed text-center">
          <strong>How to Use the Interactive Demo:</strong> Explore the skill hierarchy below, generated by our system. Click on any skill to view its details, including a video demonstration, the reward code, and the success termination code. The hierarchy is displayed with all levels expanded by default for a complete overview.
        </p>
      </div>

      <SkillTree />

      {/* Results Section */}
      <section className="my-12 text-left">
        <h2 className="text-3xl font-bold mb-8 text-center">Results</h2>

        {/* L1 Results Graph */}
        <div className="mb-10">
          <h3 className="text-2xl font-semibold mb-3 text-center">Level 2 Skill Training Using Only Level 1 Skills</h3>
          <div className="w-full flex justify-center">
            <Image 
              src={`${basePath}/L1_results.svg`}
              alt="Results of training Level 2 skills using Level 1 sub-skills"
              width={800}
              height={500}
              className="h-auto w-full max-w-5xl mx-auto block border border-gray-300 rounded shadow-md"
            />
          </div>
          <p className="mt-2 text-sm text-gray-600 leading-relaxed text-center">
            Figure 1: Success rates of Level 2 skills when policies are learned over their constituent Level 1 skills. This demonstrates the effectiveness of the first layer of hierarchy in acquiring complex behaviors.
          </p>
        </div>

        {/* Obstacle Course Task Success Graph */}
        <div className="mb-10">
          <h3 className="text-2xl font-semibold mb-3 text-center">Task Training Using Only Level 2 Skills</h3>
          <div className="w-full flex justify-center">
            <Image 
              src={`${basePath}/Obstacle_Course_Task_Success_combined.svg`}
              alt="Task success rates for the full Obstacle Course using Level 2 skills"
              width={800}
              height={500}
              className="h-auto w-full max-w-5xl mx-auto block border border-gray-300 rounded shadow-md"
            />
          </div>
          <p className="mt-2 text-sm text-gray-600 leading-relaxed text-center">
            Figure 2: Comparison of task-level policy learning using the composed Level 2 skills versus attempting to learn the entire task with a flat PPO agent. Direct PPO struggles to learn meaningful behaviors, whereas breaking the task into a hierarchy allows for successful learning.
          </p>
        </div>

        {/* Zero-shot Results Table */}
        <div className="mb-10">
          <h3 className="text-2xl font-semibold mb-3 text-center">Zero-Shot Composition of Level 2 Skills</h3>
          <p className="text-gray-700 leading-relaxed mb-4 text-center">
            The following table shows the success rates when applying the pre-trained Level 2 skills sequentially on the full obstacle course, following an order determined by the LLM, without any task-level policy learning or reward function. This demonstrates the zero-shot compositional capabilities of the generated skills.
          </p>
          
          <div className="overflow-x-auto shadow-md rounded-lg">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-100">
                <tr>
                  <th colSpan={4} className="px-6 py-3 text-center text-lg font-semibold text-gray-700 tracking-wider border-b border-gray-300">
                    Zero-shot completion rates
                  </th>
                </tr>
                <tr>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Low Wall</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Large Sphere</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Small Sphere</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Block</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-800">94% (5.5)</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-800">62% (8.4)</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-800">40% (7.1)</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-800">30% (7.1)</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-sm text-gray-600 leading-relaxed text-center">
            Table 1: Success percentages of zero-shot composition of Level 2 skills in the obstacle environment. Percentages are the means of five seeds. Brackets show the standard deviations. The percentages include all previous tasks, such that the 30% task success of finishing on the block means that 30% of runs completed all obstacles.
          </p>
        </div>
      </section>

      <footer className="mt-8 text-center text-gray-500">
        <p>Level 0 skills are Primitive Actions and are implicitly part of Level 1 skills.</p>
      </footer>
    </main>
  );
}
