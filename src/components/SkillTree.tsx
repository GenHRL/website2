'use client';

import { useState } from 'react';
import { hierarchy, Skill } from '@/lib/hierarchyData';
import SkillNode from './SkillNode';

// Helper function to get all paths of nodes that have children
const getAllExpandablePaths = (skill: Skill, currentPath: string, paths: Set<string>) => {
  if (skill.children && skill.children.length > 0) {
    paths.add(currentPath);
    skill.children.forEach(child => {
      getAllExpandablePaths(child, `${currentPath}/${child.name}`, paths);
    });
  }
};

const SkillTree: React.FC = () => {
  const [openContentPath, setOpenContentPath] = useState<string | null>(hierarchy.name); 
  
  // Initialize expanded paths to show all levels by default
  const initialExpandedPaths = new Set<string>();
  getAllExpandablePaths(hierarchy, hierarchy.name, initialExpandedPaths);
  const [expandedChildrenPaths, setExpandedChildrenPaths] = useState<Set<string>>(
    initialExpandedPaths
  );

  const handleToggleContent = (path: string) => {
    setOpenContentPath(prevPath => (prevPath === path ? null : path)); 
  };

  const handleToggleChildren = (path: string) => {
    setExpandedChildrenPaths(prevPaths => {
      const newPaths = new Set(prevPaths);
      if (newPaths.has(path)) {
        newPaths.delete(path);
      } else {
        newPaths.add(path);
      }
      return newPaths;
    });
  };

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Interactive Skill Hierarchy</h1>
      <SkillNode
          skill={hierarchy} 
          skillPath={hierarchy.name} 
          openContentPath={openContentPath}
          expandedChildrenPaths={expandedChildrenPaths}
          onToggleContent={handleToggleContent}
          onToggleChildren={handleToggleChildren}
        />
    </div>
  );
};

export default SkillTree; 