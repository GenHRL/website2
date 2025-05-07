'use client';

import { useState } from 'react';
import { Skill } from '@/lib/hierarchyData';

// This will be replaced at build time by the actual value of NEXT_PUBLIC_BASE_PATH
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || ""; 

interface SkillNodeProps {
  skill: Skill;
  skillPath: string; // e.g., "Obstacle Course" or "Obstacle Course/JumpOverLowWall"
  openContentPath: string | null; // Path of the node whose content is currently open (from SkillTree state)
  expandedChildrenPaths: Set<string>; // Set of paths whose children are expanded (from SkillTree state)
  onToggleContent: (path: string) => void; // Callback to toggle content visibility (accordion)
  onToggleChildren: (path: string) => void; // Callback to toggle children expansion
}

const SkillNode: React.FC<SkillNodeProps> = ({ 
  skill, 
  skillPath,
  openContentPath,
  expandedChildrenPaths,
  onToggleContent,
  onToggleChildren
}) => {
  const hasChildren = skill.children && skill.children.length > 0;
  
  // Determine if this node's content should be open
  const isContentOpen = openContentPath === skillPath;
  // Determine if this node's children should be expanded (with defensive check)
  const isChildrenExpanded = expandedChildrenPaths?.has(skillPath) ?? false;

  // State for the active tab within this node's content
  const [activeTab, setActiveTab] = useState<'video' | 'reward' | 'success'>('video');

  const levelColors: { [key: number]: string } = {
    3: 'bg-gray-200 hover:bg-gray-300',
    2: 'bg-red-200 hover:bg-red-300',
    1: 'bg-orange-200 hover:bg-orange-300',
  };
  const nodeColor = levelColors[skill.level] || 'bg-blue-200 hover:bg-blue-300';

  const handleHeaderClick = () => {
    // Clicking the header always toggles the content visibility for this node
    onToggleContent(skillPath);
    // If it has children, it also toggles their expansion
    if (hasChildren) {
      onToggleChildren(skillPath);
    }
    // Reset to video tab when collapsing/re-opening content? Optional.
    // if (!isContentOpen) setActiveTab('video'); 
  };

  // Helper for tab button styling
  const getTabClassName = (tabName: 'video' | 'reward' | 'success') => {
    return `px-3 py-1 text-sm rounded-t-md cursor-pointer transition-colors ${ 
      activeTab === tabName 
        ? 'bg-gray-200 font-semibold' 
        : 'bg-gray-100 hover:bg-gray-200' 
    }`;
  };

  return (
    <div className="ml-4 my-2">
      <div
        className={`p-2 rounded cursor-pointer ${nodeColor} border border-gray-400`}
        onClick={handleHeaderClick}
      >
        <div className="flex justify-between items-center">
          <span>
            {hasChildren && (isChildrenExpanded ? '▼' : '►')}{' '}
            {skill.name} (Level {skill.level})
          </span>
          {/* Details button removed */}
        </div>
      </div>

      {/* Content Area */}
      {isContentOpen && (
        <div className="mt-1 border border-gray-300 rounded bg-gray-50 shadow-sm">
          {/* Tabs */}
          <div className="flex border-b border-gray-300 bg-gray-100 rounded-t-md">
            <button className={getTabClassName('video')} onClick={() => setActiveTab('video')}>
              Video Demo
            </button>
            <button className={getTabClassName('reward')} onClick={() => setActiveTab('reward')}>
              Reward Code
            </button>
            <button className={getTabClassName('success')} onClick={() => setActiveTab('success')}>
              Success Code
            </button>
          </div>

          {/* Tab Content */}
          <div className="p-2">
            {activeTab === 'video' && (
              <div>
                {/* <h5 className="font-medium mb-1">Policy Video Demonstration:</h5> Removed title, implied by tab */}
                {skill.policyVideo ? (
                  <video controls width="100%" key={skillPath} className="max-w-full">
                    <source src={`${basePath}${skill.policyVideo}`} type="video/mp4" />
                    Your browser does not support the video tag.
                  </video>
                ) : (
                  <p className="text-xs italic text-center p-4">Placeholder for policy video.</p>
                )}
              </div>
            )}

            {activeTab === 'reward' && (
              <div>
                {/* <h5 className="font-medium mb-1">Reward Code:</h5> */}
                <pre className="bg-gray-100 p-2 rounded text-xs overflow-auto h-60 border border-gray-200">
                  {skill.rewardCode || '// Placeholder for reward code'}
                </pre>
              </div>
            )}

            {activeTab === 'success' && (
              <div>
                {/* <h5 className="font-medium mb-1">Success Termination Code:</h5> */}
                <pre className="bg-gray-100 p-2 rounded text-xs overflow-auto h-60 border border-gray-200">
                  {skill.successTerminationCode || '// Placeholder for success termination code'}
                </pre>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Children are rendered if isChildrenExpanded is true */}
      {hasChildren && isChildrenExpanded && (
        <div className="ml-4 border-l-2 border-gray-300 pl-2">
          {skill.children?.map((child) => {
            const childPath = `${skillPath}/${child.name}`;
            return (
              // Recursively render child nodes, passing down the state and handlers
              <SkillNode 
                key={child.name} 
                skill={child} 
                skillPath={childPath}
                openContentPath={openContentPath} // Pass down the single open content path
                expandedChildrenPaths={expandedChildrenPaths} // Pass down the set of expanded paths
                onToggleContent={onToggleContent} // Pass down the handler
                onToggleChildren={onToggleChildren} // Pass down the handler
              />
            );
          })}
        </div>
      )}
    </div>
  );
};

export default SkillNode; 