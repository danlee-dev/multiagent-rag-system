import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import './SourceRenderer.css';

const SourceRenderer = ({ content, sources = [], isStreaming = false, sectionMappings = [] }) => {
  const [hoveredSource, setHoveredSource] = useState(null);
  const [hoverPosition, setHoverPosition] = useState({ x: 0, y: 0 });

  // [SOURCE:번호] 또는 [SOURCE:번호, 번호, ...] 형식을 파싱해서 출처 버튼으로 변환
  const parseContentWithSources = (text) => {
    if (!text) {
      return [{ type: 'text', content: '' }];
    }

    console.log("SourceRenderer - 받은 텍스트:", text);
    console.log("SourceRenderer - 받은 출처:", sources);
    console.log("SourceRenderer - 스트리밍 상태:", isStreaming);
    console.log("SourceRenderer - 섹션 매핑:", sectionMappings);

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

        if (sources && sources.length > 0) {
          // 1. 매핑 정보가 있으면 매핑된 인덱스 사용
          if (sectionMappings && sectionMappings.length > sourceNumber) {
            const globalIndex = sectionMappings[sourceNumber];
            if (sources[globalIndex]) {
              sourceData = sources[globalIndex];
              actualIndex = sourceNumber;
              console.log(`매핑 사용: 섹션 인덱스 ${sourceNumber} → 전체 인덱스 ${globalIndex}`);
            }
          }
          
          // 2. 매핑이 없거나 실패한 경우 기존 로직 사용
          if (!sourceData) {
            // 먼저 직접 인덱스로 시도 (0-based)
            if (sources[sourceNumber]) {
              sourceData = sources[sourceNumber];
              actualIndex = sourceNumber;
            }
            // 1-based 인덱스로 시도
            else if (sourceNumber > 0 && sources[sourceNumber - 1]) {
              sourceData = sources[sourceNumber - 1];
              actualIndex = sourceNumber;
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

  const handleSourceContainerEnter = (event, sourceData, sourceNumber) => {
    // 로딩 중이거나 데이터가 없으면 툴팁 표시 안함
    if (!sourceData) return;

    const rect = event.target.getBoundingClientRect();
    const tooltipHeight = 200; // 예상 tooltip 높이
    const tooltipWidth = 320; // 예상 tooltip 너비
    const viewportHeight = window.innerHeight;
    const viewportWidth = window.innerWidth;

    // tooltip이 위쪽으로 나갈 공간이 충분한지 확인
    const spaceAbove = rect.top;
    const spaceBelow = viewportHeight - rect.bottom;

    // 좌우 위치 조정
    let x = rect.left + rect.width / 2;
    const spaceLeft = x - tooltipWidth / 2;
    const spaceRight = x + tooltipWidth / 2;

    if (spaceLeft < 10) {
      x = tooltipWidth / 2 + 10; // 왼쪽 여백 확보
    } else if (spaceRight > viewportWidth - 10) {
      x = viewportWidth - tooltipWidth / 2 - 10; // 오른쪽 여백 확보
    }

    let position = {
      x: x,
      y: rect.top - 10
    };

    // 위쪽 공간이 부족하면 아래쪽에 표시
    if (spaceAbove < tooltipHeight && spaceBelow > spaceAbove) {
      position.y = rect.bottom + 10;
      position.showBelow = true;
    } else {
      position.showBelow = false;
    }

    setHoverPosition(position);
    setHoveredSource({ data: sourceData, number: sourceNumber });
  };

  const handleSourceContainerLeave = () => {
    setHoveredSource(null);
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
              onMouseEnter={(e) => handleSourceContainerEnter(e, part.sourceData, part.sourceNumber)}
              onMouseLeave={handleSourceContainerLeave}
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

              {/* 출처 툴팁 - 데이터가 있을 때만 표시 */}
              {hoveredSource && hoveredSource.number === part.sourceNumber && part.sourceData && (
                <div
                  className={`source-tooltip ${hoverPosition.showBelow ? 'show-below' : 'show-above'}`}
                  style={{
                    position: 'fixed',
                    left: hoverPosition.x,
                    top: hoverPosition.y,
                    transform: hoverPosition.showBelow
                      ? 'translateX(-50%) translateY(0%)'
                      : 'translateX(-50%) translateY(-100%)',
                    zIndex: 1000
                  }}
                >
                  <div className="source-tooltip-header">
                    <h4>{hoveredSource.data.title}</h4>
                    {(hoveredSource.data.url || hoveredSource.data.source_url) && (
                      <span className="source-url">
                        {(hoveredSource.data.url || hoveredSource.data.source_url).replace(/^https?:\/\//, '')}
                      </span>
                    )}
                  </div>
                  <div className="source-tooltip-content">
                    {hoveredSource.data.content}
                  </div>
                  <div className="source-tooltip-footer">
                    <span className="source-type">
                      {hoveredSource.data.source_type === 'web_search' ? '웹 검색' :
                       hoveredSource.data.source_type === 'vector_db' ? '문서' : '출처'}
                    </span>
                    {(hoveredSource.data.url || hoveredSource.data.source_url) && (
                      <a
                        href={hoveredSource.data.url || hoveredSource.data.source_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="source-link"
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
