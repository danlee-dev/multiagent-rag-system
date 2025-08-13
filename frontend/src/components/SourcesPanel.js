import React from 'react';
import './SourcesPanel.css';

const SourcesPanel = ({ sources, isVisible, onToggle, dataDict = {} }) => {
  console.log('SourcesPanel received:', {
    sources: !!sources,
    sourcesCount: sources?.sources?.length || sources?.length || 0,
    dataDict: !!dataDict,
    dataDictSize: Object.keys(dataDict).length
  });
  
  console.log('SourcesPanel detailed sources:', sources);
  console.log('SourcesPanel detailed dataDict:', dataDict);
  
  // 디버깅: sources 구조 상세 분석
  if (sources) {
    console.log('🔍 Sources 구조 분석:');
    console.log('  - sources 타입:', typeof sources);
    console.log('  - Array.isArray(sources):', Array.isArray(sources));
    console.log('  - sources.sources 존재:', !!sources.sources);
    console.log('  - sources.sources 타입:', typeof sources.sources);
    if (sources.sources) {
      console.log('  - sources.sources 길이:', sources.sources.length);
      console.log('  - sources.sources[0] 샘플:', sources.sources[0]);
      console.log('  - sources.sources[0].id:', sources.sources[0]?.id);
    }
  }

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

  // sources가 dataDict 형태로 직접 전달될 수 있음
  let sourcesList = [];
  let totalCount = 0;

  // sources가 dataDict 형태인지 확인 (숫자 키를 가진 객체)
  if (sources && typeof sources === 'object' && !Array.isArray(sources) && !sources.sources) {
    console.log('✅ sources가 dataDict 형태로 전달됨, 키 개수:', Object.keys(sources).length);
    
    // 키를 숫자로 정렬
    const sortedIndexes = Object.keys(sources).map(k => parseInt(k)).sort((a, b) => a - b);
    
    sourcesList = sortedIndexes.map(actualIndex => {
      const data = sources[actualIndex];
      return {
        id: actualIndex,
        title: data.title || '제목 없음',
        content: data.content || '',
        snippet: data.content ? (data.content.length > 200 ? data.content.substring(0, 200) + '...' : data.content) : '',
        url: data.url || data.source_url || '',
        source_type: data.source || 'unknown',
        score: data.score || 0,
        document_type: data.document_type || 'unknown'
      };
    });
    
    totalCount = sourcesList.length;
    console.log(`✅ dataDict 형태 sources로 출처 목록 생성: ${totalCount}개`);
  }
  // 기존 sources 형태
  else if (sources && sources.sources && Array.isArray(sources.sources)) {
    console.log('✅ 기존 sources 형태 사용');
    sourcesList = sources.sources.map((source) => ({
      ...source,
      id: source.id
    }));
    totalCount = sources.total_count || sources.sources.length;
  }
  // dataDict fallback
  else if (Object.keys(dataDict).length > 0) {
    console.log('✅ dataDict fallback 사용');
    const sortedIndexes = Object.keys(dataDict).map(k => parseInt(k)).sort((a, b) => a - b);
    
    sourcesList = sortedIndexes.map(actualIndex => {
      const data = dataDict[actualIndex];
      return {
        id: actualIndex,
        title: data.title || '제목 없음',
        content: data.content || '',
        snippet: data.content ? (data.content.length > 200 ? data.content.substring(0, 200) + '...' : data.content) : '',
        url: data.url || data.source_url || '',
        source_type: data.source || 'unknown',
        score: data.score || 0,
        document_type: data.document_type || 'unknown'
      };
    });
    
    totalCount = sourcesList.length;
  } else {
    console.log('❌ 표시할 출처 데이터가 없음');
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
            <p>총 <span className="sources-count">{totalCount}</span>개의 출처를 찾았습니다</p>
            {Object.keys(dataDict).length > 0 && (
              <p style={{fontSize: '12px', color: '#888', marginTop: '4px'}}>
                실제 인덱스 기반으로 재구성됨
              </p>
            )}
          </div>

          <div className="sources-list">
            {sourcesList.map((source, index) => {
              // >> 핵심 수정: 실제 인덱스 번호 표시 (백엔드에서 보낸 실제 인덱스 사용)
              const displayIndex = source.id;
              
              // 디버깅: displayIndex 계산 과정
              console.log(`🔍 소스 ${index}: source.id=${source.id}, displayIndex=${displayIndex}`);
              
              // 에러 체크: id가 없으면 경고
              if (source.id === undefined || source.id === null) {
                console.error(`❌ 소스 ${index}에 id가 없습니다:`, source);
              }

              return (
                <div key={source.id || index} className="source-item">
                  <div className="source-header">
                    <div className="source-index" title={`실제 인덱스: ${displayIndex}`}>
                      {displayIndex}
                    </div>
                    <div className="source-title">{source.title || '제목 없음'}</div>
                    {source.source_type && (
                      <div className="source-type-badge">
                        {source.source_type === 'web_search' ? '웹' :
                         source.source_type === 'vector_db' ? '문서' :
                         source.source_type === '_search' ? 'Vector DB' :
                         source.source_type === 'rdb_search' ? 'DB' :
                         source.source_type === 'graph_db' ? '그래프' :
                         source.source_type}
                      </div>
                    )}
                  </div>

                  {source.url && (
                    <div className="source-url">
                      <a href={source.url} target="_blank" rel="noopener noreferrer">
                        {source.url.length > 60 ? source.url.substring(0, 60) + '...' : source.url}
                      </a>
                    </div>
                  )}

                  {(source.snippet || source.content) && (
                    <div className="source-snippet">
                      {source.snippet ||
                       (source.content && source.content.length > 200 ?
                        source.content.substring(0, 200) + '...' :
                        source.content)}
                    </div>
                  )}

                  <div className="source-meta">
                    {source.score !== undefined && (
                      <div className="source-relevance">
                        관련도: {Math.round((source.score || 0) * 100)}%
                      </div>
                    )}
                    {source.document_type && (
                      <div className="source-doc-type">
                        타입: {source.document_type}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
  );
};

export default SourcesPanel;
