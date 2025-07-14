"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import { ChartComponent } from "../components/ChartComponent";
import "./globals.css";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [query, setQuery] = useState("");
  const [reportMarkdown, setReportMarkdown] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [conversationId, setConversationId] = useState("");
  const [statusMessage, setStatusMessage] = useState("");
  const [chartsData, setChartsData] = useState([]);

  // 스크롤 관리
  const [userScrolledUp, setUserScrolledUp] = useState(false);
  const userScrolledUpRef = useRef(userScrolledUp);

  // 차트 중복 방지를 위한 ID 추적
  const processedChartIds = useRef(new Set());

  useEffect(() => {
    userScrolledUpRef.current = userScrolledUp;
  }, [userScrolledUp]);

  const scrollToBottom = useCallback(() => {
    if (!userScrolledUpRef.current) {
      setTimeout(() => {
        window.scrollTo({
          top: document.documentElement.scrollHeight,
          behavior: "smooth",
        });
      }, 100);
    }
  }, []);

  const handleScroll = useCallback(() => {
    if (!isStreaming) return;
    const scrollable = document.documentElement;
    const isAtBottom =
      scrollable.scrollHeight - scrollable.scrollTop <=
      scrollable.clientHeight + 15;
    setUserScrolledUp(!isAtBottom);
  }, [isStreaming]);

  useEffect(() => {
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, [handleScroll]);

  useEffect(() => {
    if (isStreaming) {
      scrollToBottom();
    }
  }, [reportMarkdown, chartsData.length, isStreaming, scrollToBottom]);

  // 차트 고유 ID 생성 함수 (기존 방식 유지)
  const generateChartId = (chartData) => {
    let sampleData = "";
    if (chartData.data) {
      if (Array.isArray(chartData.data)) {
        sampleData = JSON.stringify(chartData.data.slice(0, 2));
      } else {
        sampleData = JSON.stringify(chartData.data);
      }
    }

    const chartKey = JSON.stringify({
      type: chartData.type || "",
      title: chartData.title || "",
      data_sample: sampleData,
    });

    console.log("프론트엔드 차트 ID 생성:", chartKey.substring(0, 100) + "...");
    return chartKey;
  };

  // 스트리밍 처리 (일반 요청 분기 제거)
  const handleSubmit = async () => {
    if (!query.trim() || isStreaming) return;

    console.log("스트리밍 요청 시작");

    setIsStreaming(true);
    setReportMarkdown("");
    setChartsData([]);
    processedChartIds.current.clear();
    setStatusMessage("요청을 보내는 중...");
    setUserScrolledUp(false);

    try {
      const res = await fetch(`${API_BASE_URL}/api/query/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
        body: JSON.stringify({
          query: query.trim(),
          conversation_id: conversationId || undefined,
        }),
      });

      console.log("Response status:", res.status);

      if (!res.body) throw new Error("Response body is null");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      console.log("스트리밍 읽기 시작");

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          console.log("스트리밍 완료");
          break;
        }

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        const events = buffer.split("\n\n");
        buffer = events.pop() || "";

        for (const eventText of events) {
          if (!eventText.trim()) continue;

          if (eventText.startsWith("data: ")) {
            try {
              const data = JSON.parse(eventText.slice(6));
              console.log("파싱된 데이터 타입:", data.type);

              if (data.conversation_id) {
                setConversationId(data.conversation_id);
              }

              switch (data.type) {
                case "status":
                  console.log("상태 업데이트:", data.content);
                  setStatusMessage(data.content);
                  break;

                case "text_chunk":
                  console.log("텍스트 청크 수신:", data.content.length, "자");
                  setReportMarkdown((prev) => prev + data.content);
                  break;

                case "chart":
                  console.log("차트 데이터 수신");
                  console.log(
                    "받은 차트 데이터:",
                    JSON.stringify(data.chart_data).substring(0, 200) + "..."
                  );

                  const chartId = generateChartId(data.chart_data);

                  // 중복 차트 체크
                  if (processedChartIds.current.has(chartId)) {
                    console.log(
                      "중복 차트 무시:",
                      chartId.substring(0, 100) + "..."
                    );
                    break;
                  }

                  // 새로운 차트 추가
                  processedChartIds.current.add(chartId);
                  console.log(
                    "새 차트 추가, 총 차트 수:",
                    processedChartIds.current.size
                  );

                  // 차트 데이터 추가
                  setChartsData((prevCharts) => {
                    const newCharts = [...prevCharts, data.chart_data];
                    console.log(
                      "차트 배열 업데이트, 총 개수:",
                      newCharts.length
                    );
                    return newCharts;
                  });

                  // 차트 플레이스홀더 추가
                  setReportMarkdown((prev) => {
                    const chartIndex = processedChartIds.current.size - 1;
                    const newMarkdown =
                      prev + `\n\n[CHART-PLACEHOLDER-${chartIndex}]\n\n`;
                    console.log(
                      "마크다운 업데이트, 플레이스홀더 추가:",
                      `[CHART-PLACEHOLDER-${chartIndex}]`
                    );
                    return newMarkdown;
                  });
                  break;

                case "complete":
                  console.log("완료 이벤트 수신");
                  setStatusMessage(
                    `분석이 완료되었습니다. (차트 ${processedChartIds.current.size}개 생성)`
                  );
                  setIsStreaming(false);
                  setTimeout(() => scrollToBottom(), 500);
                  return;

                case "error":
                  console.log("오류 이벤트 수신:", data.content);
                  setStatusMessage(`스트리밍 오류: ${data.content}`);
                  setIsStreaming(false);
                  return;

                default:
                  console.log("알 수 없는 이벤트 타입:", data.type);
              }
            } catch (parseError) {
              console.error(
                "JSON 파싱 오류:",
                parseError,
                "원본:",
                eventText.substring(0, 100)
              );
            }
          }
        }
      }
    } catch (error) {
      console.error("API 오류:", error);
      setStatusMessage(
        `오류 발생: ${
          error instanceof Error ? error.message : "알 수 없는 오류"
        }`
      );
    } finally {
      setIsStreaming(false);
    }
  };

  // 차트 렌더링 (기존 방식 유지)
  const renderReportWithCharts = () => {
    if (!reportMarkdown) return null;

    const parts = reportMarkdown.split(/(\[CHART-PLACEHOLDER-\d+\])/g);

    return parts.map((part, index) => {
      const match = part.match(/\[CHART-PLACEHOLDER-(\d+)\]/);
      if (match) {
        const chartIndex = parseInt(match[1], 10);
        const chartConfig = chartsData[chartIndex];

        if (chartConfig) {
          const chartKey = `chart-${chartIndex}-${chartConfig.type}-${
            chartConfig.title || "untitled"
          }-${index}`;

          return (
            <div key={chartKey} className="chart-wrapper">
              <div className="chart-fade-in">
                <ChartComponent
                  chartConfig={JSON.parse(JSON.stringify(chartConfig))}
                />
              </div>
            </div>
          );
        } else {
          return (
            <div
              key={`chart-skeleton-${chartIndex}-${index}`}
              className="chart-skeleton-pulse"
            >
              <div className="skeleton-chart">
                <div className="skeleton-title"></div>
                <div className="skeleton-body"></div>
              </div>
              <p className="chart-loading-text">
                차트 #{chartIndex}를 실시간 생성하는 중...
              </p>
            </div>
          );
        }
      }

      return (
        <ReactMarkdown
          key={`md-${index}-${part.length}`}
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeRaw]}
        >
          {part}
        </ReactMarkdown>
      );
    });
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Crowdworks</h1>
        <p>Multi-Agent RAG System</p>
        {conversationId && (
          <div className="conversation-info">
            <small>대화 ID: {conversationId}</small>
          </div>
        )}
      </header>

      <main className="app-main">
        <form
          className="query-form"
          onSubmit={(e) => {
            e.preventDefault();
            handleSubmit();
          }}
        >
          <div className="input-group">
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="질문을 입력하세요... (예: 퀴노아의 영양성분과 마케팅 전략을 분석해줘)"
              rows={4}
              disabled={isStreaming}
            />
          </div>
          <div className="button-group">
            <button
              type="submit"
              disabled={isStreaming || !query.trim()}
              className="btn btn-secondary"
            >
              {isStreaming ? "분석 중..." : "분석 시작"}
            </button>
          </div>
        </form>

        {(isStreaming || reportMarkdown) && (
          <div className="response-section">
            <h2>분석 결과</h2>
            {isStreaming && !reportMarkdown.trim() && (
              <div className="loading">
                <div className="spinner"></div>
                <p>{statusMessage}</p>
              </div>
            )}
            <div className="markdown-content">{renderReportWithCharts()}</div>
            {isStreaming && (
              <div className="streaming-status">
                <div className="status-indicator">
                  <div className="pulse-dot"></div>
                  <span>{statusMessage}</span>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>RAG System v1.0.0</p>
      </footer>
    </div>
  );
}
