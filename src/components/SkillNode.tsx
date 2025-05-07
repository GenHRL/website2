'use client';

import { useState, useEffect } from 'react';
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
  const [playTabFlash, setPlayTabFlash] = useState(false); // State for flash animation

  // Effect to trigger tab flash animation
  useEffect(() => {
    if (isContentOpen) {
      setPlayTabFlash(true);
      const timer = setTimeout(() => {
        setPlayTabFlash(false);
      }, 3000); // Increased duration for sequential flash (e.g., 1s delay + 1.5s animation)
      return () => clearTimeout(timer); // Cleanup timer on unmount or if isContentOpen changes
    }
  }, [isContentOpen]);

  const levelColors: { [key: number]: string } = {
    3: 'bg-blue-700 hover:bg-blue-800 text-white',      // Darkest blue for Level 3
    2: 'bg-blue-500 hover:bg-blue-600 text-white',      // Medium blue for Level 2
    1: 'bg-blue-300 hover:bg-blue-400 text-gray-800', // Lighter blue for Level 1
  };
  const nodeColor = levelColors[skill.level] || 'bg-blue-200 hover:bg-blue-300 text-gray-700'; // Fallback with explicit text color

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
    let baseClasses = `px-3 py-1 text-base rounded-t-md cursor-pointer transition-colors`; // Changed text-sm to text-base
    if (activeTab === tabName) {
      baseClasses += ' bg-gray-200 font-semibold';
    } else {
      baseClasses += ' bg-gray-100 hover:bg-gray-200';
    }
    if (playTabFlash) {
      baseClasses += ' sunset-flash-tab'; // Use new animation class
    }
    return baseClasses;
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
            <button 
              className={getTabClassName('video')} 
              onClick={() => setActiveTab('video')}
              style={playTabFlash ? { animationDelay: '0s' } : {}}
            >
              Video Demo
            </button>
            <button 
              className={getTabClassName('reward')} 
              onClick={() => setActiveTab('reward')}
              style={playTabFlash ? { animationDelay: '0.4s' } : {}} // Stagger delay
            >
              Reward Code
            </button>
            <button 
              className={getTabClassName('success')} 
              onClick={() => setActiveTab('success')}
              style={playTabFlash ? { animationDelay: '0.8s' } : {}} // Stagger delay
            >
              Success Code
            </button>
          </div>

          {/* Tab Content */}
          <div className="p-2">
            {activeTab === 'video' && (
              <div>
                {/* <h5 className="font-medium mb-1">Policy Video Demonstration:</h5> Removed title, implied by tab */}
                {skill.policyVideo ? (
                  <video controls width="100%" key={skillPath} className="w-4/5 mx-auto rounded">
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