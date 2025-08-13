import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import './SourceRenderer.css';

const SourceRenderer = ({ content, sources = [], isStreaming = false, dataDict = {} }) => {
  const [hoveredSource, setHoveredSource] = useState(null);
  const [hoveredIndex, setHoveredIndex] = useState(null);

  console.log("SourceRenderer 렌더링 시작:");
  console.log("- content 길이:", content?.length || 0);
  console.log("- sources 개수:", sources?.length || 0);
  console.log("- isStreaming:", isStreaming);
  console.log("- dataDict 크기:", Object.keys(dataDict || {}).length);
  console.log("- dataDict 키들:", Object.keys(dataDict || {}));
  console.log("- dataDict 전체 내용:", dataDict);

  // [SOURCE:번호] 또는 [SOURCE:번호, 번호, ...] 형식을 파싱해서 출처 버튼으로 변환
  const parseContentWithSources = (text) => {
    if (!text) {
      return [{ type: 'text', content: '' }];
    }

    // 개선된 정규식: [SOURCE:1] 또는 [SOURCE:1, 2, 3] 형식 모두 지원
    const sourcePattern = /\[SOURCE:([\d\s,]+)\]/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = sourcePattern.exec(text)) !== null) {
      // 매칭 이전의 텍스트 추가
      if (match.index > lastIndex) {
        parts.push({
          type: 'text',
          content: text.slice(lastIndex, match.index)
        });
      }

      // 여러 번호를 파싱 (쉼표로 구분된 번호들)
      const numbersString = match[1]; // "1" 또는 "1, 2, 3" 등
      const sourceNumbers = numbersString
        .split(',')
        .map(num => parseInt(num.trim()))
        .filter(num => !isNaN(num)); // 유효한 숫자만 필터링

      // 각 번호에 대해 개별 버튼 생성
      sourceNumbers.forEach((sourceNumber, index) => {
        // >> 핵심 수정: 전체 데이터 딕셔너리에서 직접 매핑
        let sourceData = null;

        console.log(`🔍 SOURCE:${sourceNumber} 매핑 시작`);
        console.log(`   - dataDict 타입:`, typeof dataDict);
        console.log(`   - dataDict 키들:`, Object.keys(dataDict || {}));
        console.log(`   - 찾는 키: ${sourceNumber} (타입: ${typeof sourceNumber})`);

        // 전체 데이터 딕셔너리에서 직접 조회 (숫자와 문자열 키 모두 시도)
        if (dataDict) {
          // 숫자 키로 먼저 시도
          if (dataDict[sourceNumber]) {
            sourceData = dataDict[sourceNumber];
            console.log(`✅ 전체 데이터 딕셔너리에서 직접 매핑 성공 (숫자 키): SOURCE:${sourceNumber}`);
          } 
          // 문자열 키로 시도
          else if (dataDict[String(sourceNumber)]) {
            sourceData = dataDict[String(sourceNumber)];
            console.log(`✅ 전체 데이터 딕셔너리에서 직접 매핑 성공 (문자열 키): SOURCE:${sourceNumber}`);
          }
          
          if (sourceData) {
            console.log(`   - 출처 제목:`, sourceData.title?.substring(0, 50) + '...');
            console.log(`   - 출처 내용:`, sourceData.content?.substring(0, 100) + '...');
          } else {
            console.log(`❌ 전체 데이터 딕셔너리에서 매핑 실패: SOURCE:${sourceNumber}`);
            console.log(`   - 사용 가능한 키들: ${Object.keys(dataDict || {})}`);
            
            // fallback: sources 배열에서 시도
            if (sources && sources.length > 0) {
              console.log(`🔄 sources 배열 fallback 시도`);
              const arrayIndex = sourceNumber - 1;
              if (arrayIndex >= 0 && arrayIndex < sources.length && sources[arrayIndex]) {
                sourceData = sources[arrayIndex];
                console.log(`✅ sources 배열 fallback 성공: SOURCE:${sourceNumber} → 배열[${arrayIndex}]`);
              }
            }
          }
        } else {
          console.log(`❌ dataDict가 null/undefined`);
        }

        // 버튼 생성 (데이터 있든 없든)
        parts.push({
          type: 'source',
          sourceNumber: sourceNumber, // 실제 인덱스 번호 사용
          sourceData: sourceData, // null일 수도 있음
          isLoading: !sourceData && isStreaming // 로딩 상태
        });

        console.log(`출처 ${sourceNumber} 버튼 생성됨 ${sourceData ? '(데이터 있음)' : '(로딩 상태)'}`);

        // 여러 번호가 있을 때 버튼 사이에 약간의 간격 추가
        if (index < sourceNumbers.length - 1) {
          parts.push({
            type: 'text',
            content: ' '
          });
        }
      });

      lastIndex = match.index + match[0].length;
    }

    // 마지막 남은 텍스트 추가
    if (lastIndex < text.length) {
      parts.push({
        type: 'text',
        content: text.slice(lastIndex)
      });
    }

    return parts;
  };

  const handleSourceContainerEnter = (event, sourceData, sourceNumber, buttonIndex) => {
    if (!sourceData) return; // 데이터가 없으면 툴팁 표시 안함

    setHoveredSource({
      number: sourceNumber,
      data: sourceData
    });
    setHoveredIndex(buttonIndex);
  };

  const handleSourceContainerLeave = () => {
    setHoveredSource(null);
    setHoveredIndex(null);
  };

  const handleTooltipEnter = () => {
    // 현재 hoveredSource 상태 유지
  };

  const handleTooltipLeave = () => {
    setHoveredSource(null);
    setHoveredIndex(null);
  };

  const parts = parseContentWithSources(content);

  return (
    <div className="source-renderer">
      {parts.map((part, index) => {
        if (part.type === 'text') {
          return (
            <ReactMarkdown
              key={index}
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeRaw]}
              components={{
                p: ({ children }) => <span>{children}</span>,
                div: ({ children }) => <span>{children}</span>,
                table: ({ node, ...props }) => (
                  <div className="table-container">
                    <table {...props} />
                  </div>
                ),
                img: () => null,
              }}
            >
              {part.content}
            </ReactMarkdown>
          );
        } else if (part.type === 'source') {
          return (
            <div
              key={index}
              className="source-container"
              onMouseEnter={(e) => handleSourceContainerEnter(e, part.sourceData, part.sourceNumber, index)}
              onMouseLeave={handleSourceContainerLeave}
              style={{ position: 'relative', display: 'inline-block' }}
            >
              <button
                className={`source-button ${part.isLoading ? 'loading' : ''} ${!part.sourceData ? 'no-data' : ''}`}
                onClick={() => {
                  if (part.sourceData && (part.sourceData.url || part.sourceData.source_url)) {
                    window.open(part.sourceData.url || part.sourceData.source_url, '_blank');
                  }
                }}
                disabled={!part.sourceData}
                title={part.isLoading ? '출처 로딩 중...' : (part.sourceData ? part.sourceData.title : '출처 정보 없음')}
              >
                {part.sourceNumber}
              </button>

              {/* 출처 툴팁 - 현재 호버된 특정 버튼에만 표시 */}
              {hoveredSource && hoveredIndex === index && part.sourceData && (
                <div
                  style={{
                    position: 'absolute',
                    bottom: '100%',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    width: '300px',
                    maxHeight: '180px',
                    overflow: 'hidden',
                    zIndex: 1000,
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #444',
                    borderRadius: '8px',
                    padding: '10px',
                    fontSize: '12px',
                    lineHeight: '1.4',
                    color: '#fff',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
                    marginBottom: '2px'
                  }}
                  onMouseEnter={handleTooltipEnter}
                  onMouseLeave={handleTooltipLeave}
                >
                  {/* 제목 */}
                  <div style={{
                    fontWeight: 'bold',
                    marginBottom: '6px',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    fontSize: '13px'
                  }}>
                    {hoveredSource.data.title}
                  </div>

                  {/* URL */}
                  {(hoveredSource.data.url || hoveredSource.data.source_url) && (
                    <div style={{
                      fontSize: '10px',
                      color: '#888',
                      marginBottom: '8px',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap'
                    }}>
                      {(hoveredSource.data.url || hoveredSource.data.source_url).replace(/^https?:\/\//, '')}
                    </div>
                  )}

                  {/* 내용 */}
                  <div style={{
                    fontSize: '11px',
                    maxHeight: '80px',
                    overflow: 'hidden',
                    marginBottom: '8px',
                    lineHeight: '1.4',
                    color: '#ccc'
                  }}>
                    {hoveredSource.data.content && hoveredSource.data.content.length > 150
                      ? hoveredSource.data.content.substring(0, 150) + '...'
                      : hoveredSource.data.content
                    }
                  </div>

                  {/* 하단 정보 */}
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    fontSize: '10px',
                    marginTop: '6px'
                  }}>
                    <span style={{ color: '#666' }}>
                      {hoveredSource.data.source_type === 'web_search' ? '웹 검색' :
                       hoveredSource.data.source_type === 'vector_db' ? '문서' :
                       hoveredSource.data.source_type === 'rdb_search' ? '데이터베이스' :
                       hoveredSource.data.source_type === 'graph_db' ? '그래프 DB' :
                       hoveredSource.data.source || '출처'}
                    </span>
                    {(hoveredSource.data.url || hoveredSource.data.source_url) && (
                      <a
                        href={hoveredSource.data.url || hoveredSource.data.source_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{
                          color: '#4a9eff',
                          textDecoration: 'none',
                          fontSize: '10px'
                        }}
                        onClick={(e) => e.stopPropagation()}
                      >
                        링크 열기 →
                      </a>
                    )}
                  </div>
                </div>
              )}
            </div>
          );
        }
        return null;
      })}
    </div>
  );
};

export default SourceRenderer;
