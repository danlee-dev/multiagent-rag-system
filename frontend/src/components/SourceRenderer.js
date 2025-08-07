import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import './SourceRenderer.css';

const SourceRenderer = ({ content, sources = [], isStreaming = false, sectionMappings = [] }) => {
  const [hoveredSource, setHoveredSource] = useState(null);
  const [hoveredIndex, setHoveredIndex] = useState(null); // 현재 호버된 버튼의 인덱스

  // [SOURCE:번호] 또는 [SOURCE:번호, 번호, ...] 형식을 파싱해서 출처 버튼으로 변환
  const parseContentWithSources = (text) => {
    if (!text) {
      return [{ type: 'text', content: '' }];
    }

    console.log("SourceRenderer 렌더링 시작:");
  console.log("- content 길이:", content?.length || 0);
  console.log("- sources 개수:", sources?.length || 0);
  console.log("- isStreaming:", isStreaming);
  console.log("- sectionMappings:", sectionMappings);
  console.log("- sectionMappings 타입:", typeof sectionMappings);
  console.log("- sectionMappings 배열 여부:", Array.isArray(sectionMappings));
  console.log("- sectionMappings 길이:", sectionMappings?.length || 0);

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
        // ✅ 개선: 매핑 정보를 사용해서 출처 찾기
        let sourceData = null;
        let actualIndex = sourceNumber;

        console.log(`🔍 SOURCE:${sourceNumber} 매핑 시작`);
        console.log(`   - sectionMappings:`, sectionMappings);
        console.log(`   - sources 길이:`, sources?.length);

        if (sources && sources.length > 0) {
          // 🔥 핵심 수정: 섹션 매핑 사용 (1-based SOURCE를 0-based로 변환)
          if (sectionMappings && Array.isArray(sectionMappings) && sectionMappings.length > sourceNumber - 1) {
            const globalIndex = sectionMappings[sourceNumber - 1]; // SOURCE:1 -> 매핑[0], SOURCE:2 -> 매핑[1]

            console.log(`   - SOURCE:${sourceNumber} → 매핑 인덱스: ${globalIndex}`);

            // 매핑된 인덱스가 sources 범위 내에 있는지 확인
            if (typeof globalIndex === 'number' && globalIndex >= 0 && globalIndex < sources.length && sources[globalIndex]) {
              sourceData = sources[globalIndex];
              actualIndex = sourceNumber; // 버튼 표시는 그대로 1-based
              console.log(`✅ 매핑 성공: SOURCE:${sourceNumber} → 전체 인덱스 ${globalIndex}`);
              console.log(`   - 출처 제목:`, sourceData.title?.substring(0, 50) + '...');
              console.log(`   - 출처 내용:`, sourceData.content?.substring(0, 100) + '...');
            } else {
              console.log(`❌ 매핑 실패: 섹션 SOURCE:${sourceNumber} → 전체 인덱스 ${globalIndex} (범위 벗어남 또는 데이터 없음)`);
              console.log(`   - sources.length: ${sources.length}, globalIndex: ${globalIndex}, 유효성: ${typeof globalIndex === 'number' && globalIndex >= 0 && globalIndex < sources.length}`);
            }
          }

          // 2. 매핑이 없거나 실패한 경우 기본 로직 사용 (1-based를 0-based로)
          if (!sourceData) {
            const fallbackIndex = sourceNumber - 1;
            if (fallbackIndex >= 0 && fallbackIndex < sources.length && sources[fallbackIndex]) {
              sourceData = sources[fallbackIndex];
              actualIndex = sourceNumber; // 버튼 표시는 그대로 1-based
              console.log(`🔄 기본 로직 사용: SOURCE:${sourceNumber} → 인덱스 ${fallbackIndex}`);
            } else {
              console.log(`❌ 기본 로직도 실패: SOURCE:${sourceNumber} → 인덱스 ${fallbackIndex} (범위 벗어남 또는 데이터 없음)`);
              console.log(`   - sources.length: ${sources.length}, fallbackIndex: ${fallbackIndex}`);

              // 🔧 범위를 벗어나는 경우, 사용 가능한 마지막 인덱스로 대체
              if (sources.length > 0) {
                const safeIndex = Math.min(fallbackIndex, sources.length - 1);
                if (sources[safeIndex]) {
                  sourceData = sources[safeIndex];
                  actualIndex = sourceNumber; // 버튼 표시는 원래 번호 유지
                  console.log(`🔧 안전 인덱스 사용: SOURCE:${sourceNumber} → 인덱스 ${safeIndex} (최대 ${sources.length - 1})`);
                }
              }
            }
          }
        }

        // 🔥 핵심 수정: sourceData가 없어도 항상 버튼으로 렌더링
        parts.push({
          type: 'source',
          sourceNumber: actualIndex,
          sourceData: sourceData, // null일 수도 있음
          isLoading: !sourceData && isStreaming // 로딩 상태 추가
        });

        if (sourceData) {
          console.log(`출처 ${sourceNumber} 버튼 생성됨 (데이터 있음)`);
        } else {
          console.log(`출처 ${sourceNumber} 버튼 생성됨 (로딩 상태)`);
        }

        // 여러 번호가 있을 때 버튼 사이에 약간의 간격 추가 (마지막 번호가 아닌 경우)
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
    setHoveredIndex(buttonIndex); // 현재 호버된 버튼의 인덱스 저장
  };

  const handleSourceContainerLeave = () => {
    setHoveredSource(null);
    setHoveredIndex(null);
  };

  // 툴팁에 마우스가 올라갔을 때 툴팁 유지
  const handleTooltipEnter = () => {
    // 현재 hoveredSource 상태 유지 (툴팁이 사라지지 않도록)
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
                // 헤더 크기 보존 - ReactMarkdown 기본 헤더 태그 사용
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
                disabled={!part.sourceData} // 데이터가 없으면 클릭 비활성화
                title={part.isLoading ? '출처 로딩 중...' : (part.sourceData ? part.sourceData.title : '출처 정보 없음')}
              >
                {part.sourceNumber}
              </button>

              {/* 출처 툴팁 - 현재 호버된 특정 버튼에만 표시 */}
              {hoveredSource && hoveredIndex === index && part.sourceData && (
                <div
                  style={{
                    position: 'absolute',
                    bottom: '100%', // 버튼 바로 위에 붙임
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
                    marginBottom: '2px' // 버튼과 약간의 간격
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
                       hoveredSource.data.source_type === 'vector_db' ? '문서' : '출처'}
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
