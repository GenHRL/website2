'use client';

import { useState } from 'react';
import { hierarchy } from '@/lib/hierarchyData';
import SkillNode from './SkillNode';

const SkillTree: React.FC = () => {
  const [openContentPath, setOpenContentPath] = useState<string | null>(hierarchy.name); 
  const [expandedChildrenPaths, setExpandedChildrenPaths] = useState<Set<string>>(
    new Set([hierarchy.name]) // Expand top-level children by default
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
      <h1 className="text-2xl font-bold mb-4">{hierarchy.name} Details</h1>
      <SkillNode
          skill={hierarchy} // The root skill
          skillPath={hierarchy.name} // The path for the root skill
          openContentPath={openContentPath}
          expandedChildrenPaths={expandedChildrenPaths}
          onToggleContent={handleToggleContent}
          onToggleChildren={handleToggleChildren}
        />
    </div>
  );
};

export default SkillTree; 