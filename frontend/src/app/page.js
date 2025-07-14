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
  const [currentConversation, setCurrentConversation] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [conversationId, setConversationId] = useState("");
  const [statusMessage, setStatusMessage] = useState("");
  const [currentStreamingMessage, setCurrentStreamingMessage] = useState("");
  const [currentStreamingCharts, setCurrentStreamingCharts] = useState([]);

  // 사이드바 관련 상태
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [conversations, setConversations] = useState([]);

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
      setConversations(JSON.parse(savedConversations));
    }
  }, []);

  // 대화 히스토리 저장
  const saveConversations = useCallback((newConversations) => {
    localStorage.setItem("chatConversations", JSON.stringify(newConversations));
    setConversations(newConversations);
  }, []);

  // 새 채팅 시작
  const startNewChat = () => {
    setCurrentConversation([]);
    setConversationId("");
    setCurrentStreamingMessage("");
    setCurrentStreamingCharts([]);
    processedChartIds.current.clear();
    setQuery("");
  };

  // 기존 대화 로드
  const loadConversation = (conv) => {
    setCurrentConversation(conv.messages || []);
    setConversationId(conv.id);
    setCurrentStreamingMessage("");
    setCurrentStreamingCharts([]);
    processedChartIds.current.clear();
    setQuery("");
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
    if (!query.trim() || isStreaming) return;

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

    // 빈 어시스턴트 메시지 추가 (스트리밍용)
    const assistantMessage = {
      id: Date.now() + 1,
      type: "assistant",
      content: "",
      charts: [],
      timestamp: new Date().toISOString(),
      isStreaming: true,
    };

    setCurrentConversation((prev) => [...prev, assistantMessage]);

    try {
      const res = await fetch(`${API_BASE_URL}/api/query/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
        body: JSON.stringify({
          query: currentQuery,
          conversation_id: conversationId || undefined,
        }),
      });

      if (!res.body) throw new Error("Response body is null");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finalContent = "";
      let finalCharts = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        const events = buffer.split("\n\n");
        buffer = events.pop() || "";

        for (const eventText of events) {
          if (!eventText.trim()) continue;

          if (eventText.startsWith("data: ")) {
            try {
              const data = JSON.parse(eventText.slice(6));

              if (data.conversation_id && !conversationId) {
                setConversationId(data.conversation_id);
              }

              switch (data.type) {
                case "status":
                  setStatusMessage(data.content);
                  break;

                case "text_chunk":
                  finalContent += data.content;
                  setCurrentConversation((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMessage.id
                        ? { ...msg, content: finalContent }
                        : msg
                    )
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
                  // 스트리밍 완료 - 최종 메시지 업데이트
                  setCurrentConversation((prev) => {
                    const newConversation = prev.map((msg) =>
                      msg.id === assistantMessage.id
                        ? {
                            ...msg,
                            content: finalContent,
                            charts: finalCharts,
                            isStreaming: false,
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

                  setIsStreaming(false);
                  setStatusMessage("");
                  return;

                case "error":
                  setStatusMessage(`오류: ${data.content}`);
                  setIsStreaming(false);
                  return;
              }
            } catch (parseError) {
              console.error("JSON 파싱 오류:", parseError);
            }
          }
        }
      }
    } catch (error) {
      console.error("API 오류:", error);
      setStatusMessage(`오류: ${error.message}`);
      setIsStreaming(false);
    }
  };

  // Enter 키 처리
  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
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
          components={{ img: () => null }} // disable markdown <img> tag
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
      <div className="chat-main">
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
              {currentConversation.map((message) => (
                <div
                  key={message.id}
                  className={`message-wrapper ${message.type}`}
                >
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
                    {renderMessageContent(message)}
                  </div>
                </div>
              ))}

              {isStreaming && (
                <div className="streaming-status">
                  <div className="status-content">
                    <div className="pulse-dot"></div>
                    <span>{statusMessage}</span>
                  </div>
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
    </div>
  );
}
