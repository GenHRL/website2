import Head from 'next/head';
import Image from 'next/image';
import { useState, useEffect } from 'react';
import { hierarchyData } from '../lib/hierarchyData';
import styles from '../styles/Home.module.css';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/cjs/styles/prism';

const ItemDetails = ({ item, activeTab, setActiveTab, getAssetPath }) => {
  if (!item || !item.rewardFunctionCode) return null;

  return (
    <div className={styles.detailsContainer}>
      <h3>{item.title} - Details</h3>
      <div className={styles.tabs}>
        <button
          className={`${styles.tabButton} ${activeTab === 'reward' ? styles.active : ''}`}
          onClick={() => setActiveTab('reward')}
        >
          Reward Function
        </button>
        <button
          className={`${styles.tabButton} ${activeTab === 'success' ? styles.active : ''}`}
          onClick={() => setActiveTab('success')}
        >
          Success Function
        </button>
        <button
          className={`${styles.tabButton} ${activeTab === 'video' ? styles.active : ''}`}
          onClick={() => setActiveTab('video')}
        >
          Policy Video
        </button>
      </div>
      {activeTab === 'reward' && (
        <div>
          <h4>Reward Function (Python)</h4>
          <SyntaxHighlighter language="python" style={vscDarkPlus} className={styles.codeBlock}>
            {item.rewardFunctionCode}
          </SyntaxHighlighter>
        </div>
      )}
      {activeTab === 'success' && (
        <div>
          <h4>Success Function (Python)</h4>
          <SyntaxHighlighter language="python" style={vscDarkPlus} className={styles.codeBlock}>
            {item.successFunctionCode}
          </SyntaxHighlighter>
        </div>
      )}
      {activeTab === 'video' && (
        <div>
          <h4>Policy Video</h4>
          {item.videoPlaceholder ? (
            <video controls width="100%" className={styles.videoPlayer} src={getAssetPath(item.videoPlaceholder)} key={item.videoPlaceholder + activeTab + item.id}>
              Your browser does not support the video tag. <track kind="captions" />
            </video>
          ) : <p>Video not available.</p>}
        </div>
      )}
    </div>
  );
};

export default function HomePage() {
  const [selectedItem, setSelectedItem] = useState(null);
  const [openLevel2Id, setOpenLevel2Id] = useState(null);
  const [activeTab, setActiveTab] = useState('reward');

  useEffect(() => {
    if (hierarchyData && hierarchyData.rewardFunctionCode) {
      setSelectedItem(hierarchyData);
    }
  }, []);

  const handleItemClick = (item) => {
    if (selectedItem && selectedItem.id === item.id) {
      if (item.id !== hierarchyData.id) {
         setSelectedItem(null); 
      }
    } else {
      if (item.rewardFunctionCode) {
        setSelectedItem(item);
        setActiveTab('reward');
      } else {
        setSelectedItem(null);
      }
    }
  };

  const toggleLevel2Children = (level2ItemId) => {
    const newOpenLevel2Id = openLevel2Id === level2ItemId ? null : level2ItemId;
    setOpenLevel2Id(newOpenLevel2Id);
    if (newOpenLevel2Id === null && selectedItem) {
      if (selectedItem.parentId === level2ItemId) {
        setSelectedItem(null);
      }
    }
  };
  
  const getAssetPath = (path) => path;

  const methodExplanationText = `
    **Please replace this with your detailed method explanation.**
    GenHRL operates by first...
    The key components are:
    1.  **Component A:** Does foo.
    2.  **Component B:** Does bar.
    This approach allows for...
  `;
  const methodImagePath = "/images/method_diagram.png";

  return (
    <div className={styles.container}>
      <Head>
        <title>{hierarchyData.paperTitle || 'GenHRL Project'}</title>
        <meta name="description" content={hierarchyData.paperAbstract || 'Interactive HRL project page'} />
        <link rel="icon" href={getAssetPath('/favicon.ico')} />
      </Head>

      <header className={styles.header}>
        <h1 className={styles.paperTitle}>{hierarchyData.paperTitle}</h1>
        <p className={styles.authors}>Anonymous Authors (Update as needed)</p>
        <div className={styles.abstractContainer}>
          <h2>Abstract</h2>
          <p className={styles.abstractText}>{hierarchyData.paperAbstract}</p>
        </div>
      </header>

      <section className={styles.methodSection}>
        <h2>Our Method: GenHRL</h2>
        <div className={styles.methodLayout}>
          <div className={styles.methodImageContainer}>
            <img 
              src={getAssetPath(methodImagePath)}
              alt="GenHRL Method Diagram" 
              style={{ width: '100%', height: 'auto', objectFit: 'contain' }} 
            />
            <p className={styles.caption}>Fig. 1: Overview of the GenHRL framework. (Placeholder caption)</p>
          </div>
          <div className={styles.methodTextContainer}>
            {methodExplanationText.split('\n').map((line, index) => (
              line.startsWith('**') && line.endsWith('**') ? <strong key={index}>{line.substring(2, line.length - 2)}</strong> :
              line.startsWith('1. ') || line.startsWith('2. ') || line.startsWith('3. ') ? <p key={index} style={{marginLeft: '20px'}}>{line}</p> :
              <p key={index}>{line}</p>
            ))}
          </div>
        </div>
      </section>
      
      <hr className={styles.divider} />

      <main className={styles.mainHierarchy}>
        <div className={styles.taskTitleContainer}>
             <h2 className={styles.taskTitle} onClick={() => handleItemClick(hierarchyData)} style={{cursor: 'pointer'}}>
                {hierarchyData.title} (Click to see details/video)
             </h2>
             <p className={styles.taskDescription}>{hierarchyData.description}</p>
        </div>

        <div className={styles.level2ContainerTree}>
          {hierarchyData.children.map((level2Item) => (
            <div key={level2Item.id} className={styles.level2Node}>
              <div 
                className={`${styles.itemCard} ${styles[level2Item.color.toLowerCase()]} ${selectedItem && selectedItem.id === level2Item.id && selectedItem.rewardFunctionCode ? styles.selected : ''} ${openLevel2Id === level2Item.id ? styles.open : ''}`}
                onClick={() => {
                  toggleLevel2Children(level2Item.id);
                  handleItemClick(level2Item);
                }}
              >
                {level2Item.title}
                 <span className={`${styles.arrow} ${openLevel2Id === level2Item.id ? styles.arrowDown : styles.arrowRight}`}></span>
              </div>
              
              {openLevel2Id === level2Item.id && level2Item.children && (
                <div className={styles.level1ContainerTree}>
                  {level2Item.children.map((level1Item) => (
                    <div key={level1Item.id} className={styles.level1Node}>
                       <div
                         className={`${styles.itemCard} ${styles[level1Item.color.toLowerCase()]} ${selectedItem && selectedItem.id === level1Item.id ? styles.selected : ''}`}
                         onClick={() => handleItemClick(level1Item)}
                       >
                        {level1Item.title}
                       </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
        
        <ItemDetails item={selectedItem} activeTab={activeTab} setActiveTab={setActiveTab} getAssetPath={getAssetPath} />

        <div className={styles.primitiveActionsBar}>
            {hierarchyData.primitiveActions.title}
        </div>
      </main>

      <footer className={styles.footer}>
        <p>.</p>
      </footer>
    </div>
  );
} 