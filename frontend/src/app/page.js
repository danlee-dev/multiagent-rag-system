"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import { ChartComponent } from "../components/ChartComponent";
import SourcesPanel from "../components/SourcesPanel";
import "./globals.css";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [query, setQuery] = useState("");
  const [currentConversation, setCurrentConversation] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [conversationId, setConversationId] = useState("");
  const [statusMessage, setStatusMessage] = useState("");
  const [currentStreamingMessage, setCurrentStreamingMessage] = useState("");
  const [currentStreamingCharts, setCurrentStreamingCharts] = useState([]);

  // 사이드바 관련 상태
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [conversations, setConversations] = useState([]);

  // 출처 패널 관련 상태 추가
  const [sourcesData, setSourcesData] = useState(null);
  const [sourcesPanelVisible, setSourcesPanelVisible] = useState(false);

  // Claude 스타일 실시간 검색 결과 상태
  const [currentSearchResults, setCurrentSearchResults] = useState([]);
  const [searchResultsVisible, setSearchResultsVisible] = useState({});
  const [conversationSearchResults, setConversationSearchResults] = useState({}); // 대화별 검색 결과

  // 스크롤 관리
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // 차트 중복 방지를 위한 ID 추적
  const processedChartIds = useRef(new Set());

  // 메시지 끝으로 스크롤
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [currentConversation, currentStreamingMessage, currentStreamingCharts]);

  // 로컬 스토리지에서 대화 히스토리 로드
  useEffect(() => {
    const savedConversations = localStorage.getItem("chatConversations");
    if (savedConversations) {
      try {
        setConversations(JSON.parse(savedConversations));
      } catch (error) {
        console.error("대화 히스토리 로드 오류:", error);
        setConversations([]);
      }
    }

    // 검색 결과도 로컬 스토리지에서 로드
    const savedSearchResults = localStorage.getItem("currentSearchResults");
    if (savedSearchResults) {
      try {
        const searchResults = JSON.parse(savedSearchResults);
        setCurrentSearchResults(searchResults);
      } catch (error) {
        console.error("검색 결과 로드 오류:", error);
      }
    }

    const savedSearchVisible = localStorage.getItem("searchResultsVisible");
    if (savedSearchVisible) {
      try {
        setSearchResultsVisible(JSON.parse(savedSearchVisible));
      } catch (error) {
        console.error("검색 결과 표시 상태 로드 오류:", error);
      }
    }
  }, []);  // 대화 히스토리 저장
  const saveConversations = useCallback((newConversations) => {
    localStorage.setItem("chatConversations", JSON.stringify(newConversations));
    setConversations(newConversations);
  }, []);

  // 검색 결과 저장
  const saveSearchResults = useCallback((searchResults, visibleState) => {
    localStorage.setItem("currentSearchResults", JSON.stringify(searchResults));
    localStorage.setItem("searchResultsVisible", JSON.stringify(visibleState));
  }, []);

  // 새 채팅 시작
  const startNewChat = () => {
    setCurrentConversation([]);
    setConversationId("");
    setCurrentStreamingMessage("");
    setCurrentStreamingCharts([]);
    processedChartIds.current.clear();
    setQuery("");
    setSourcesData(null);
    setSourcesPanelVisible(false);
    // 새 채팅 시작할 때만 검색 결과 초기화
    setCurrentSearchResults([]);
    setSearchResultsVisible({});
    setConversationSearchResults({});
    // 로컬 스토리지에서도 제거
    localStorage.removeItem("currentSearchResults");
    localStorage.removeItem("searchResultsVisible");
    localStorage.removeItem("conversationSearchResults");
  };

  // 기존 대화 로드
  const loadConversation = (conv) => {
    setCurrentConversation(conv.messages || []);
    setConversationId(conv.id);
    setCurrentStreamingMessage("");
    setCurrentStreamingCharts([]);
    processedChartIds.current.clear();
    setQuery("");
    setSourcesData(null);
    setSourcesPanelVisible(false);
    // 기존 대화 로드할 때만 검색 결과 초기화
    setCurrentSearchResults([]);
    setSearchResultsVisible({});
    // 로컬 스토리지에서도 제거
    localStorage.removeItem("currentSearchResults");
    localStorage.removeItem("searchResultsVisible");
  };

  // 출처 패널 토글
  const toggleSourcesPanel = () => {
    setSourcesPanelVisible(!sourcesPanelVisible);
  };

  // 차트 고유 ID 생성 함수
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

    return chartKey;
  };

  // 메시지 전송
  const handleSubmit = async () => {
    console.log("handleSubmit 호출됨, query:", query, "isStreaming:", isStreaming);

    if (!query.trim() || isStreaming) {
      console.log("조건 불만족으로 반환:", {
        queryTrimmed: query.trim(),
        isStreaming: isStreaming
      });
      return;
    }

    console.log("API 요청 시작...");

    const userMessage = {
      id: Date.now(),
      type: "user",
      content: query.trim(),
      timestamp: new Date().toISOString(),
    };

    setCurrentConversation((prev) => [...prev, userMessage]);
    const currentQuery = query.trim();
    setQuery("");
    setIsStreaming(true);
    setCurrentStreamingMessage("");
    setCurrentStreamingCharts([]);
    processedChartIds.current.clear();
    setStatusMessage("생각하는 중...");
    setSourcesData(null); // 새 요청 시 출처 데이터 초기화
    // 검색 결과는 유지 - 새 쿼리에서도 이전 검색 박스 보존

    // 빈 어시스턴트 메시지 추가 (스트리밍용)
    const assistantMessage = {
      id: Date.now() + 1,
      type: "assistant",
      content: "",
      charts: [],
      timestamp: new Date().toISOString(),
      isStreaming: true,
      sources: null, // 출처 정보 초기화
    };

    setCurrentConversation((prev) => [...prev, assistantMessage]);

    try {
      console.log("API_BASE_URL 값:", API_BASE_URL);
      console.log("API 요청 URL:", `${API_BASE_URL}/query/stream`);
      console.log("요청 데이터:", {
        query: currentQuery,
        session_id: conversationId || undefined,
      });

      console.log("fetch 요청 시작!");

      const res = await fetch(`${API_BASE_URL}/query/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
        body: JSON.stringify({
          query: currentQuery,
          session_id: conversationId || undefined,
        }),
      }).catch(error => {
        console.error("fetch 요청 자체가 실패:", error);
        throw new Error(`Network request failed: ${error.message}`);
      });

      console.log("fetch 응답 받음:", res.status, res.statusText);
      console.log("응답 헤더:", res.headers);

      if (!res.ok) {
        console.error("HTTP 오류 응답:", res.status, res.statusText);
        const errorText = await res.text();
        console.error("오류 내용:", errorText);
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      if (!res.body) {
        console.error("Response body is null");
        throw new Error("Response body is null");
      }

      console.log("스트리밍 시작...");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finalContent = "";
      let finalCharts = [];
      let currentStep = 0;
      let totalSteps = 0;
      let finalSources = null; // 최종 출처 정보 저장

      while (true) {
        console.log("스트리밍 청크 읽기 시도...");
        const { done, value } = await reader.read();
        console.log("청크 읽기 결과:", { done, valueLength: value?.length });

        if (done) {
          console.log("스트리밍 완료");
          break;
        }

        const chunk = decoder.decode(value, { stream: true });
        console.log("디코딩된 청크:", chunk.substring(0, 100) + (chunk.length > 100 ? '...' : ''));
        buffer += chunk;

        const events = buffer.split("\n\n");
        buffer = events.pop() || "";

        for (const eventText of events) {
          if (!eventText.trim()) continue;

          if (eventText.startsWith("data: ")) {
            try {
              const data = JSON.parse(eventText.slice(6));
              console.log(">> 받은 스트리밍 데이터:", data.type, data);

              if (data.session_id && !conversationId) {
                setConversationId(data.session_id);
              }

              switch (data.type) {
                case "status":
                  setStatusMessage(data.message);
                  break;

                case "chart":
                  finalCharts.push(data.chart_data);
                  break;

                case "plan":
                  totalSteps = data.total_steps || data.plan?.steps?.length || 0;
                  setStatusMessage(`실행 계획: ${totalSteps}개 단계`);
                  console.log("실행 계획:", data.plan);
                  break;

                case "step_start":
                  currentStep = data.step;

                  // Claude 스타일 도구 사용 상태 표시
                  let statusText = `단계 ${data.step}/${totalSteps}: ${data.description}`;
                  if (data.tool && data.query) {
                    if (data.status === "searching") {
                      statusText = `${data.tool}로 검색 중: "${data.query}"`;
                    } else if (data.status === "processing") {
                      statusText = `${data.tool}로 처리 중`;
                    }
                  }

                  setStatusMessage(statusText);
                  break;

                case "search_results":
                  // Claude 스타일 실시간 검색 결과 표시
                  console.log("검색 결과 받음:", data); // 디버깅 로그
                  const searchResultData = {
                    step: data.step,
                    tool_name: data.tool_name || "unknown",
                    query: data.query || "",
                    results: data.results,
                    timestamp: new Date().toISOString(),
                    conversationId: conversationId || data.session_id || Date.now().toString()
                  };
                  console.log("처리된 검색 데이터:", searchResultData); // 디버깅 로그

                  // 현재 대화의 검색 결과에 추가
                  const currentConvId = conversationId || data.session_id || Date.now().toString();

                  setConversationSearchResults(prev => {
                    const newResults = {
                      ...prev,
                      [currentConvId]: [...(prev[currentConvId] || []), searchResultData]
                    };
                    // 로컬 스토리지에 저장
                    localStorage.setItem("conversationSearchResults", JSON.stringify(newResults));
                    return newResults;
                  });

                  setCurrentSearchResults(prev => {
                    const newResults = [...prev, searchResultData];
                    // 로컬 스토리지에 저장
                    localStorage.setItem("currentSearchResults", JSON.stringify(newResults));
                    return newResults;
                  });

                  // 자동으로 검색 결과 펼치기 (최신 검색만)
                  setSearchResultsVisible(prev => {
                    const newVisible = {
                      ...prev,
                      [`${data.step}-latest`]: true
                    };
                    // 로컬 스토리지에 저장
                    localStorage.setItem("searchResultsVisible", JSON.stringify(newVisible));
                    return newVisible;
                  });

                  // 임시 출처 데이터 업데이트 (실시간 표시용)
                  const tempSources = {
                    total_count: data.results.length,
                    sources: data.results.map((result, index) => ({
                      id: `temp_${data.step}_${index}`,
                      title: result.title,
                      content: result.content_preview,
                      url: result.url,
                      source_type: result.source,
                      relevance_score: result.relevance_score,
                      document_type: result.document_type
                    }))
                  };

                  setSourcesData(tempSources);
                  break;

                case "section_start":
                  finalContent += `\n\n## ${data.title}\n\n`;
                  setCurrentConversation((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMessage.id
                        ? { ...msg, content: finalContent }
                        : msg
                    )
                  );
                  break;

                case "content":
                  // SimpleAnswerer의 스트리밍 답변 처리
                  finalContent += data.chunk;
                  setCurrentConversation((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMessage.id
                        ? { ...msg, content: finalContent }
                        : msg
                    )
                  );
                  break;

                case "content_chunk":
                  // 최종 보고서 및 기타 컨텐츠 표시 (context integration 제외)
                  finalContent += data.chunk;
                  setCurrentConversation((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMessage.id
                        ? { ...msg, content: finalContent }
                        : msg
                    )
                  );
                  break;

                case "section_end":
                  finalContent += "\n\n";
                  setCurrentConversation((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMessage.id
                        ? { ...msg, content: finalContent }
                        : msg
                    )
                  );
                  break;

                case "step_complete":
                  setStatusMessage(
                    `단계 ${data.step} 완료 (${data.step}/${totalSteps})`
                  );
                  break;

                case "chart":
                  const chartId = generateChartId(data.chart_data);
                  if (!processedChartIds.current.has(chartId)) {
                    processedChartIds.current.add(chartId);
                    finalCharts.push(data.chart_data);
                    const chartIndex = processedChartIds.current.size - 1;
                    finalContent += `\n\n[CHART-PLACEHOLDER-${chartIndex}]\n\n`;

                    setCurrentConversation((prev) =>
                      prev.map((msg) =>
                        msg.id === assistantMessage.id
                          ? {
                              ...msg,
                              content: finalContent,
                              charts: [...finalCharts],
                            }
                          : msg
                      )
                    );
                  }
                  break;

                case "complete":
                  // 완료시 최종 출처 정보 저장
                  if (data.sources) {
                    finalSources = data.sources;
                  }
                  setStatusMessage("완료");
                  break;

                case "final_complete":
                  setStatusMessage("");
                  // 스트리밍 완료 - 최종 메시지 업데이트 (출처 정보 보존)
                  setCurrentConversation((prev) => {
                    const currentConvId = conversationId || data.session_id || Date.now().toString();
                    const messageSearchResults = conversationSearchResults[currentConvId] || currentSearchResults;

                    const newConversation = prev.map((msg) =>
                      msg.id === assistantMessage.id
                        ? {
                            ...msg,
                            content: finalContent,
                            charts: finalCharts,
                            isStreaming: false,
                            sources: finalSources, // 최종 출처 정보 저장
                            searchResults: messageSearchResults, // 해당 메시지의 검색 결과 저장
                          }
                        : msg
                    );

                    // 대화 히스토리 업데이트
                    const conversationData = {
                      id: conversationId || Date.now().toString(),
                      title:
                        currentQuery.slice(0, 30) +
                        (currentQuery.length > 30 ? "..." : ""),
                      messages: newConversation,
                      lastUpdated: new Date().toISOString(),
                    };

                    const updatedConversations = conversations.filter(
                      (c) => c.id !== conversationData.id
                    );
                    updatedConversations.unshift(conversationData);
                    saveConversations(updatedConversations.slice(0, 50));

                    return newConversation;
                  });

                  // 현재 스트리밍용 검색 결과 초기화 (다음 질문을 위해)
                  setCurrentSearchResults([]);

                  setIsStreaming(false);
                  console.log("스트리밍 완료 - 검색 결과 및 출처 정보 유지");
                  return;

                case "error":
                  setStatusMessage(`오류: ${data.message}`);
                  setIsStreaming(false);
                  return;

                case "result":
                  // 기타 결과 데이터는 로그로만 출력 (context integration 결과 제외)
                  console.log("처리 결과:", data.data);
                  break;

                default:
                  console.log("알 수 없는 이벤트 타입:", data.type, data);
                  break;
              }
            } catch (parseError) {
              console.error("JSON 파싱 오류:", parseError);
            }
          }
        }
      }
    } catch (error) {
      console.error("=== API 오류 상세 정보 ===");
      console.error("오류 타입:", error.name);
      console.error("오류 메시지:", error.message);
      console.error("오류 스택:", error.stack);
      console.error("========================");
      setStatusMessage(`오류: ${error.message}`);
      setIsStreaming(false);
    }
  };

  // Enter 키 처리
  const handleKeyPress = (e) => {
    console.log("키 눌림:", e.key, "Shift:", e.shiftKey);

    if (e.key === "Enter" && !e.shiftKey) {
      console.log("Enter 키 감지, handleSubmit 호출");
      e.preventDefault();
      handleSubmit();
    }
  };

  // 대화 삭제
  const deleteConversation = (convId) => {
    const updatedConversations = conversations.filter((c) => c.id !== convId);
    saveConversations(updatedConversations);

    if (conversationId === convId) {
      startNewChat();
    }
  };

  // 메시지 렌더링 (차트 포함)
  const renderMessageContent = (message) => {
    const content = message.content || "";
    const charts = message.charts || [];
    const parts = content.split(/(\[CHART-PLACEHOLDER-\d+\])/g);

    return parts.map((part, index) => {
      const match = part.match(/\[CHART-PLACEHOLDER-(\d+)\]/);
      if (match) {
        const chartIndex = parseInt(match[1], 10);
        const chartConfig = charts[chartIndex];
        if (chartConfig) {
          const chartKey = `chart-${chartIndex}-${chartConfig.type}-${
            chartConfig.title || "untitled"
          }-${index}`;
          return (
            <div key={chartKey} className="message-chart">
              <ChartComponent chartConfig={chartConfig} />
            </div>
          );
        } else {
          return (
            <div
              key={`chart-loading-${chartIndex}-${index}`}
              className="chart-loading"
            >
              <div className="chart-skeleton">
                <div className="skeleton-title"></div>
                <div className="skeleton-body"></div>
              </div>
              <span className="chart-loading-text">차트 생성 중...</span>
            </div>
          );
        }
      }
      return (
        <ReactMarkdown
          key={`md-${index}`}
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeRaw]}
          components={{
            table: ({ node, ...props }) => (
              <div className="table-container">
                <table {...props} />
              </div>
            ),
            img: () => null,
          }}
        >
          {part}
        </ReactMarkdown>
      );
    });
  };

  // textarea 자동 높이 조절
  const adjustTextareaHeight = useCallback(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 120) + "px";
    }
  }, []);

  useEffect(() => {
    adjustTextareaHeight();
  }, [query, adjustTextareaHeight]);

  return (
    <div className="chat-app">
      {/* 사이드바 */}
      <div
        className={`sidebar ${sidebarOpen ? "sidebar-open" : "sidebar-closed"}`}
      >
        <div className="sidebar-header">
          <button
            className="sidebar-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            title={sidebarOpen ? "사이드바 닫기" : "사이드바 열기"}
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
              <line x1="9" y1="9" x2="15" y2="15" />
              <line x1="15" y1="9" x2="9" y2="15" />
            </svg>
          </button>
          {sidebarOpen && (
            <button className="new-chat-btn" onClick={startNewChat}>
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <line x1="12" y1="5" x2="12" y2="19" />
                <line x1="5" y1="12" x2="19" y2="12" />
              </svg>
              새 채팅
            </button>
          )}
        </div>

        {sidebarOpen && (
          <div className="conversations-list">
            {conversations.length === 0 ? (
              <div className="no-conversations">
                <p>대화 기록이 없습니다</p>
              </div>
            ) : (
              conversations.map((conv) => (
                <div
                  key={conv.id}
                  className={`conversation-item ${
                    conversationId === conv.id ? "active" : ""
                  }`}
                  onClick={() => loadConversation(conv)}
                >
                  <div className="conversation-content">
                    <div className="conversation-title">{conv.title}</div>
                    <div className="conversation-date">
                      {new Date(conv.lastUpdated).toLocaleDateString("ko-KR")}
                    </div>
                  </div>
                  <button
                    className="delete-conversation"
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteConversation(conv.id);
                    }}
                    title="대화 삭제"
                  >
                    <svg
                      width="12"
                      height="12"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                    >
                      <line x1="18" y1="6" x2="6" y2="18" />
                      <line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                  </button>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      {/* 메인 채팅 영역 */}
      <div className={`chat-main ${sourcesPanelVisible ? "chat-main-with-sources" : ""}`}>
        {/* 메시지 영역 */}
        <div className="messages-container">
          {currentConversation.length === 0 ? (
            <div className="welcome-screen">
              <div className="welcome-content">
                <h1>안녕하세요!</h1>
                <p>무엇을 도와드릴까요?</p>
              </div>
            </div>
          ) : (
            <>
              {currentConversation.map((message, messageIndex) => (
                <div key={message.id}>
                  <div className={`message-wrapper ${message.type}`}>
                    {message.type === "assistant" && (
                      <div className="assistant-avatar">
                        <svg
                          width="24"
                          height="24"
                          viewBox="0 0 24 24"
                          fill="none"
                        >
                          <circle cx="12" cy="12" r="10" fill="#10a37f" />
                          <path
                            d="M8 12h8M12 8v8"
                            stroke="white"
                            strokeWidth="2"
                            strokeLinecap="round"
                          />
                        </svg>
                      </div>
                    )}
                    <div className="message-content">
                      {/* 완료된 어시스턴트 메시지 위에 해당 검색 결과 먼저 표시 */}
                      {message.type === "assistant" && !message.isStreaming && message.searchResults && (
                        <div className="claude-search-results">
                          {message.searchResults.map((searchData, index) => (
                            <div key={`search-${searchData.step}-${index}`} className="search-result-section">
                              <div
                                className="search-result-header"
                                onClick={() => setSearchResultsVisible(prev => {
                                  const newVisible = {
                                    ...prev,
                                    [`${message.id}-${searchData.step}-${index}`]: !prev[`${message.id}-${searchData.step}-${index}`]
                                  };
                                  localStorage.setItem("searchResultsVisible", JSON.stringify(newVisible));
                                  return newVisible;
                                })}
                              >
                                <div className="search-info">
                                  <span className="search-tool">{searchData.tool_name}</span>
                                  {searchData.query && (
                                    <span className="search-query">
                                      "{searchData.query.length > 50 ? searchData.query.substring(0, 50) + '...' : searchData.query}"
                                    </span>
                                  )}
                                  <span className="result-count">{searchData.results.length}개 결과</span>
                                </div>
                                <div className="toggle-icon">
                                  {searchResultsVisible[`${message.id}-${searchData.step}-${index}`] ? '▼' : '▶'}
                                </div>
                              </div>

                              {searchResultsVisible[`${message.id}-${searchData.step}-${index}`] && (
                                <div className="search-result-list">
                                  {searchData.results.map((result, resultIndex) => (
                                    <div key={resultIndex} className="search-result-item">
                                      <div className="result-header">
                                        <span className="result-title">{result.title}</span>
                                        <span className="result-source">{result.source}</span>
                                      </div>
                                      <div className="result-preview">{result.content_preview}</div>
                                      {result.url && (
                                        <div className="result-url">
                                          <a href={result.url} target="_blank" rel="noopener noreferrer">
                                            {result.url}
                                          </a>
                                        </div>
                                      )}
                                      <div className="result-meta">
                                        <span>관련성: {(result.relevance_score * 100).toFixed(0)}%</span>
                                        <span>타입: {result.document_type}</span>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}

                      {/* 그 다음에 메시지 콘텐츠(보고서) 표시 */}
                      {renderMessageContent(message)}

                      {/* 출처 보기 버튼 - 완료된 메시지에만 표시 */}
                      {message.type === "assistant" && !message.isStreaming && (
                        <div className="message-actions">
                          {message.sources && message.sources.total_count > 0 && (
                            <button
                              className="sources-simple-btn"
                              onClick={() => {
                                setSourcesData(message.sources);
                                if (!sourcesPanelVisible) {
                                  toggleSourcesPanel();
                                }
                              }}
                            >
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                                <polyline points="14,2 14,8 20,8" />
                              </svg>
                              {message.sources.total_count}개 출처
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}

              {/* 스트리밍 중일 때만 상태 표시 */}
              {isStreaming && (
                <div className="streaming-status">
                  <div className="status-content">
                    <div className="pulse-dot"></div>
                    <span>{statusMessage}</span>
                  </div>
                </div>
              )}

              {/* 현재 스트리밍 중인 검색 결과만 임시 표시 */}
              {isStreaming && currentSearchResults.length > 0 && (
                <div className="claude-search-results">
                  {currentSearchResults.map((searchData, index) => (
                    <div key={`search-${searchData.step}-${index}`} className="search-result-section">
                      <div
                        className="search-result-header"
                        onClick={() => setSearchResultsVisible(prev => {
                          const newVisible = {
                            ...prev,
                            [`current-${searchData.step}-${index}`]: !prev[`current-${searchData.step}-${index}`]
                          };
                          localStorage.setItem("searchResultsVisible", JSON.stringify(newVisible));
                          return newVisible;
                        })}
                      >
                        <div className="search-info">
                          <span className="search-tool">{searchData.tool_name}</span>
                          {searchData.query && (
                            <span className="search-query">
                              "{searchData.query.length > 50 ? searchData.query.substring(0, 50) + '...' : searchData.query}"
                            </span>
                          )}
                          <span className="result-count">{searchData.results.length}개 결과</span>
                        </div>
                        <div className="toggle-icon">
                          {searchResultsVisible[`current-${searchData.step}-${index}`] ? '▼' : '▶'}
                        </div>
                      </div>

                      {searchResultsVisible[`current-${searchData.step}-${index}`] && (
                        <div className="search-result-list">
                          {searchData.results.map((result, resultIndex) => (
                            <div key={resultIndex} className="search-result-item">
                              <div className="result-header">
                                <span className="result-title">{result.title}</span>
                                <span className="result-source">{result.source}</span>
                              </div>
                              <div className="result-preview">{result.content_preview}</div>
                              {result.url && (
                                <div className="result-url">
                                  <a href={result.url} target="_blank" rel="noopener noreferrer">
                                    {result.url}
                                  </a>
                                </div>
                              )}
                              <div className="result-meta">
                                <span>관련성: {(result.relevance_score * 100).toFixed(0)}%</span>
                                <span>타입: {result.document_type}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* 입력 영역 */}
        <div className="input-area">
          <div className="input-container">
            <textarea
              ref={textareaRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="메시지 보내기..."
              disabled={isStreaming}
              className="message-input"
              rows={1}
            />
            <button
              onClick={handleSubmit}
              disabled={isStreaming || !query.trim()}
              className="send-button"
              title="메시지 전송"
            >
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22,2 15,22 11,13 2,9 22,2" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* 출처 패널 */}
      <SourcesPanel
        sources={sourcesData}
        isVisible={sourcesPanelVisible}
        onToggle={toggleSourcesPanel}
      />
    </div>
  );
}
