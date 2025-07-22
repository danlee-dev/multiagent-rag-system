import React from 'react';
import './SourcesPanel.css';

const SourcesPanel = ({ sources, isVisible, onToggle }) => {
  console.log('SourcesPanel received sources:', sources);

  if (!sources) {
    return (
      <div className={`sources-panel ${isVisible ? 'sources-panel-open' : 'sources-panel-closed'}`}>
        <div className="sources-panel-header">
          <h3>실시간 수집한 출처 확인하기</h3>
          <button className="sources-close-btn" onClick={onToggle}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18"/>
              <line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        </div>
        <div className="sources-panel-content">
          <div style={{padding: '20px', color: '#ff6b6b'}}>
            디버그: sources가 null입니다
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`sources-panel ${isVisible ? 'sources-panel-open' : 'sources-panel-closed'}`}>
        <div className="sources-panel-header">
          <h3>실시간 수집한 출처 확인하기</h3>
          <button className="sources-close-btn" onClick={onToggle}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18"/>
              <line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        </div>

        <div className="sources-panel-content">
          <div className="sources-summary">
            <p>총 <span className="sources-count">{sources.total_count}</span>개의 출처를 찾았습니다</p>
          </div>

          <div className="sources-list">
            {sources.sources && sources.sources.map((source, index) => (
              <div key={index} className="source-item">
                <div className="source-header">
                  <div className="source-index">{index + 1}</div>
                  <div className="source-title">{source.title || '제목 없음'}</div>
                </div>

                {source.url && (
                  <div className="source-url">
                    <a href={source.url} target="_blank" rel="noopener noreferrer">
                      {source.url}
                    </a>
                  </div>
                )}

                {source.snippet && (
                  <div className="source-snippet">
                    {source.snippet}
                  </div>
                )}

                {source.relevance_score && (
                  <div className="source-relevance">
                    관련도: {Math.round(source.relevance_score * 100)}%
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
  );
};

export default SourcesPanel;
