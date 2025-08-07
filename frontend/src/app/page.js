"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import { ChartComponent } from "../components/ChartComponent";
import SourcesPanel from "../components/SourcesPanel";
import SourceRenderer from "../components/SourceRenderer";
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

  // 🔍 디버깅: currentSearchResults 변경사항 추적
  const setCurrentSearchResultsDebug = (newResults) => {
    console.log("🔍 currentSearchResults 변경:", {
      이전: currentSearchResults.length,
      새로운: Array.isArray(newResults) ? newResults.length : "함수",
      스택: new Error().stack?.split('\n')[2]?.trim()
    });
    setCurrentSearchResults(newResults);
  };
  const [searchResultsVisible, setSearchResultsVisible] = useState({});
  const [conversationSearchResults, setConversationSearchResults] = useState({}); // 대화별 검색 결과

  // 섹션별 매핑 정보 상태 추가
  const [sectionMappings, setSectionMappings] = useState({});

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
        const parsedConversations = JSON.parse(savedConversations);
        setConversations(parsedConversations);

        // ✅ 현재 대화가 있으면 검색 결과도 복원
        if (currentConversation.length > 0) {
          const restoredSearchResults = [];
          currentConversation.forEach(message => {
            if (message.searchResults && Array.isArray(message.searchResults)) {
              message.searchResults.forEach(result => {
                restoredSearchResults.push({
                  ...result,
                  messageId: String(message.id)
                });
              });
            }
          });
          if (restoredSearchResults.length > 0) {
            setCurrentSearchResultsDebug(restoredSearchResults);
            console.log(`페이지 로드 시 검색 결과 복원: ${restoredSearchResults.length}개`);
          }
        }
      } catch (error) {
        console.error("대화 히스토리 로드 오류:", error);
        setConversations([]);
      }
    }

    // 🚫 검색 결과 로컬 스토리지 복원 제거 - 세션별 관리로 변경
    // 검색 결과는 각 세션마다 독립적으로 관리되므로 전역 복원 불필요
    console.log("검색 결과 전역 복원 건너뜀 - 세션별 관리");

    // ✅ 페이지 로드 시 localStorage에서 검색 결과 복원
    const savedCurrentSearchResults = localStorage.getItem("currentSearchResults");
    if (savedCurrentSearchResults) {
      try {
        const parsedSearchResults = JSON.parse(savedCurrentSearchResults);
        if (Array.isArray(parsedSearchResults) && parsedSearchResults.length > 0) {
          setCurrentSearchResultsDebug(parsedSearchResults);
          console.log(`페이지 로드 시 localStorage에서 검색 결과 복원: ${parsedSearchResults.length}개`);
        }
      } catch (error) {
        console.error("검색 결과 복원 오류:", error);
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

    // conversationSearchResults 로드
    const savedConversationSearchResults = localStorage.getItem("conversationSearchResults");
    if (savedConversationSearchResults) {
      try {
        setConversationSearchResults(JSON.parse(savedConversationSearchResults));
        console.log("대화별 검색 결과 복원 완료");
      } catch (error) {
        console.error("대화별 검색 결과 로드 오류:", error);
      }
    }

    // 스트리밍 중에 저장된 대화 복원
    const savedStreamingConversation = localStorage.getItem("currentStreamingConversation");
    if (savedStreamingConversation) {
      try {
        const { messages, isStreaming } = JSON.parse(savedStreamingConversation);
        if (isStreaming && Array.isArray(messages) && messages.length > 0) {
          setCurrentConversation(messages);
          setIsStreaming(true);
          console.log("스트리밍 중이던 대화 복원됨:", messages.length, "개 메시지");
        }
      } catch (error) {
        console.error("스트리밍 대화 복원 오류:", error);
        localStorage.removeItem("currentStreamingConversation");
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
    setCurrentSearchResultsDebug([]);
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

    // ✅ 검색 결과 복원: 메시지에 저장된 searchResults를 currentSearchResults로 복원
    const restoredSearchResults = [];
    if (conv.messages) {
      conv.messages.forEach(message => {
        if (message.searchResults && Array.isArray(message.searchResults)) {
          // 각 검색 결과에 messageId 추가
          message.searchResults.forEach(result => {
            restoredSearchResults.push({
              ...result,
              messageId: String(message.id)
            });
          });
        }
      });
    }

    setCurrentSearchResultsDebug(restoredSearchResults);
    console.log(`대화 ${conv.id} 로드 완료 - 검색 결과 ${restoredSearchResults.length}개 복원`);

    setSearchResultsVisible({});
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

    // 🆕 Claude 스타일: 새로운 질문 시작할 때 현재 검색 결과 초기화 (세션별 관리)
    setCurrentSearchResultsDebug([]);
    console.log("🔄 새 질문 시작: 검색 결과 초기화 (세션별 관리)");

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

    // 🔥 핵심 수정: 스트리밍 시작 즉시 현재 대화 상태를 localStorage에 저장
    const tempConversationWithNewMessages = [...currentConversation, userMessage, assistantMessage];
    const tempConversationData = {
      id: conversationId || Date.now().toString(),
      title: currentQuery.slice(0, 30) + (currentQuery.length > 30 ? "..." : ""),
      messages: tempConversationWithNewMessages,
      lastUpdated: new Date().toISOString(),
      isStreaming: true, // 🆕 스트리밍 상태 표시
    };

    // 임시 대화 데이터를 localStorage에 저장 (스트리밍 중 복원용)
    localStorage.setItem("currentStreamingConversation", JSON.stringify(tempConversationData));
    console.log("🔄 스트리밍 중 대화 상태 저장:", tempConversationData.id);

    try {
      console.log("API_BASE_URL 값:", API_BASE_URL);
      console.log("API 요청 URL:", `${API_BASE_URL}/query/stream`);
      console.log("요청 데이터:", {
        query: currentQuery,
        session_id: conversationId || undefined,
        message_id: assistantMessage.id, // 🆕 메시지 ID 추가
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
          message_id: String(assistantMessage.id), // 🆕 메시지 ID를 문자열로 변환
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
                  // 차트가 생성되는 즉시 UI 업데이트
                  setCurrentConversation((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMessage.id
                        ? {
                            ...msg,
                            charts: [...finalCharts],
                          }
                        : msg
                    )
                  );
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

                  // 🆕 중간 검색 여부 확인
                  const isIntermediateSearch = data.is_intermediate_search || false;
                  const sectionContext = data.section_context || null;

                  if (isIntermediateSearch && sectionContext) {
                    console.log("중간 검색 감지:", sectionContext);
                  }

                  const searchResultData = {
                    step: data.step,
                    tool_name: data.tool_name || "unknown",
                    query: data.query || "",
                    results: data.results,
                    timestamp: new Date().toISOString(),
                    conversationId: conversationId || data.session_id || Date.now().toString(),
                    messageId: data.message_id || assistantMessage.id, // 🆕 메시지 ID 추가
                    // 🆕 중간 검색 정보 추가
                    isIntermediateSearch: isIntermediateSearch,
                    sectionContext: sectionContext
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

                  // 🆕 모든 검색 결과를 currentSearchResults에 추가 (메시지별 필터링은 렌더링 시 처리)
                  setCurrentSearchResultsDebug(prev => {
                    const newResults = [...prev, searchResultData];

                    // 로컬 스토리지에는 모든 검색 결과 저장
                    localStorage.setItem("currentSearchResults", JSON.stringify(newResults));
                    console.log(`🔍 검색 결과 추가 (총 ${newResults.length}개):`, searchResultData);

                    // ✅ 즉시 메시지에 검색 결과 저장 (새로고침 대비)
                    setCurrentConversation(prevMessages => {
                      return prevMessages.map(msg => {
                        if (msg.id === assistantMessage.id && msg.type === "assistant") {
                          // 현재 메시지의 모든 검색 결과 수집 (디버깅 강화)
                          const messageSearchResults = newResults.filter(result => {
                            // 🔥 숫자와 문자열 모두 처리하도록 수정
                            const resultMsgId = String(result.messageId);
                            const assistantMsgId = String(assistantMessage.id);
                            const match = resultMsgId === assistantMsgId;
                            console.log(`🔍 검색 결과 매칭 확인:`, {
                              resultMessageId: result.messageId,
                              resultMsgIdString: resultMsgId,
                              assistantMessageId: assistantMessage.id,
                              assistantMsgIdString: assistantMsgId,
                              match: match
                            });
                            return match;
                          });
                          console.log(`🔍 메시지 ${assistantMessage.id}에 검색 결과 저장: ${messageSearchResults.length}개`);
                          console.log(`🔍 전체 검색 결과:`, newResults.map(r => ({ messageId: r.messageId, query: r.query })));
                          return {
                            ...msg,
                            searchResults: messageSearchResults
                          };
                        }
                        return msg;
                      });
                    });

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
                      score: result.score,
                      document_type: result.document_type
                    }))
                  };

                  setSourcesData(tempSources);
                  break;

                case "section_header":
                  // 섹션 헤더를 별도 배열로 관리하여 올바른 마크다운 형식 유지
                  setCurrentConversation((prev) => {
                    const updated = [...prev];
                    if (updated.length > 0 && updated[updated.length - 1].id === assistantMessage.id) {
                      const lastMessage = updated[updated.length - 1];
                      // 섹션 헤더를 별도 배열로 관리
                      if (!lastMessage.sectionHeaders) {
                        lastMessage.sectionHeaders = [];
                      }
                      lastMessage.sectionHeaders.push({
                        id: `header-${Date.now()}-${Math.random()}`,
                        title: data.title,
                        timestamp: Date.now()
                      });
                      // 현재 content에 임시로 헤더 추가 (렌더링에서 별도 처리할 예정)
                      return updated.map((msg) =>
                        msg.id === assistantMessage.id
                          ? { ...msg, content: finalContent }
                          : msg
                      );
                    }
                    return updated;
                  });
                  break;

                case "sources":
                  // 실시간 출처 정보 업데이트 (즉시 버튼으로 전환)
                  console.log("실시간 출처 데이터 받음:", data);
                  setSourcesData(data);

                  // 현재 스트리밍 중인 메시지에도 출처 정보 추가
                  setCurrentConversation((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMessage.id
                        ? { ...msg, sources: data }
                        : msg
                    )
                  );
                  break;

                case "section_mapping":
                  // 섹션별 매핑 정보 저장
                  console.log("섹션 매핑 정보 받음:", data);
                  console.log("섹션 제목:", data.section_title);
                  console.log("매핑 데이터:", data.section_to_global_mapping);
                  console.log("데이터 타입:", typeof data.section_to_global_mapping);
                  console.log("배열 여부:", Array.isArray(data.section_to_global_mapping));

                  const mappingKey = `${conversationId || data.session_id || Date.now()}-${data.section_title}`;
                  console.log("생성된 매핑 키:", mappingKey);

                  setSectionMappings(prev => {
                    const newMappings = {
                      ...prev,
                      [mappingKey]: data.section_to_global_mapping
                    };
                    console.log("업데이트된 전체 매핑:", newMappings);
                    return newMappings;
                  });
                  break;

                case "section_start":
                  // 섹션 시작 시 적절한 마크다운 헤더와 간격 추가
                  const sectionHeader = `\n\n## ${data.title}\n\n`;
                  finalContent += sectionHeader;
                  setCurrentConversation((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMessage.id
                        ? {
                            ...msg,
                            content: (msg.content || "") + sectionHeader,
                            isStreaming: true
                          }
                        : msg
                    )
                  );
                  break;

                case "content":
                  // 실시간 스트리밍: 청크를 즉시 기존 내용에 추가
                  finalContent += data.chunk;
                  setCurrentConversation((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMessage.id
                        ? {
                            ...msg,
                            content: finalContent,
                            isStreaming: true
                          }
                        : msg
                    )
                  );
                  break;


                case "section_end":
                  const sectionEnd = "\n\n";
                  finalContent += sectionEnd;
                  setCurrentConversation((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMessage.id
                        ? {
                            ...msg,
                            content: (msg.content || "") + sectionEnd,
                            isStreaming: true
                          }
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
                    const chartPlaceholder = `\n\n[CHART-PLACEHOLDER-${chartIndex}]\n\n`;
                    finalContent += chartPlaceholder;

                    setCurrentConversation((prev) =>
                      prev.map((msg) =>
                        msg.id === assistantMessage.id
                          ? {
                              ...msg,
                              content: (msg.content || "") + chartPlaceholder,
                              charts: [...finalCharts],
                              isStreaming: true
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
                  // 스트리밍 완료 - 최종 메시지 업데이트 (검색 결과는 이미 실시간으로 저장됨)
                  setCurrentConversation((prev) => {
                    const newConversation = prev.map((msg) =>
                      msg.id === assistantMessage.id
                        ? {
                            ...msg,
                            // 기존 searchResults 보존하면서 다른 속성들만 업데이트
                            charts: finalCharts,
                            isStreaming: false,
                            sources: finalSources, // 최종 출처 정보 저장
                          }
                        : msg
                    );

                    // 디버깅: final_complete 후 searchResults 확인
                    const updatedMessage = newConversation.find(m => m.id === assistantMessage.id);
                    console.log("final_complete - 최종 메시지 상태:", {
                      messageId: assistantMessage.id,
                      hasSearchResults: !!updatedMessage?.searchResults,
                      searchResultsLength: updatedMessage?.searchResults?.length || 0,
                      isStreaming: updatedMessage?.isStreaming
                    });

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

                    // 스트리밍 완료 후 임시 저장된 대화 정리
                    localStorage.removeItem('currentStreamingConversation');

                    return newConversation;
                  });

                  console.log("검색 결과 유지됨 - Claude 스타일 지속성 적용");

                  setIsStreaming(false);
                  console.log("스트리밍 완료 - 검색 결과 및 출처 정보 유지");
                  return;

                case "error":
                  setStatusMessage(`오류: ${data.message}`);
                  setIsStreaming(false);
                  // 에러 발생 시 임시 저장된 대화 정리
                  localStorage.removeItem('currentStreamingConversation');
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
      // API 오류 발생 시 임시 저장된 대화 정리
      localStorage.removeItem('currentStreamingConversation');
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
  // renderMessageContent 함수 수정
  const renderMessageContent = (message) => {
    const content = message.content || "";
    const charts = message.charts || [];
    const sectionHeaders = message.sectionHeaders || [];

    // ✅ 수정: 실시간 출처 데이터 우선 사용
    let sources = [];
    if (message.sources) {
      // 완료된 메시지의 출처 사용
      if (Array.isArray(message.sources)) {
        sources = message.sources;
      } else if (message.sources.sources && Array.isArray(message.sources.sources)) {
        sources = message.sources.sources;
      }
    } else if (message.isStreaming && sourcesData && sourcesData.sources) {
      // 🔥 스트리밍 중에는 전역 sourcesData 사용
      sources = sourcesData.sources;
    }

    console.log("렌더링할 출처 정보:", sources); // 디버깅용

    // 섹션 헤더들을 먼저 렌더링
    const headerElements = sectionHeaders.map((header) => (
      <div key={header.id} className="section-header">
        <h2 className="section-title">{header.title}</h2>
      </div>
    ));

    const parts = content.split(/(\[CHART-PLACEHOLDER-\d+\])/g);

    const contentElements = parts.map((part, index) => {
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

      // ✅ 수정: [SOURCE:번호] 형식이 있는 경우 SourceRenderer 사용
      if (part.includes('[SOURCE:')) {
        // 🔥 핵심 수정: 현재 텍스트 부분에서 섹션 제목 추출하여 정확한 매핑 찾기
        let currentSectionMapping = [];

        // 1. 현재 part에서 가장 가까운 섹션 헤더 찾기 (## 제목)
        const content = message.content || "";
        const partIndex = content.indexOf(part);
        let nearestSectionTitle = "";

        if (partIndex !== -1) {
          // 현재 part 이전의 텍스트에서 가장 마지막 ## 헤더 찾기
          const beforePart = content.substring(0, partIndex);
          const sectionHeaders = beforePart.match(/## (.+)/g);
          if (sectionHeaders && sectionHeaders.length > 0) {
            const lastHeader = sectionHeaders[sectionHeaders.length - 1];
            nearestSectionTitle = lastHeader.replace(/^## /, '').trim();
          }
        }

        // 2. 섹션 제목으로 매핑 정보 찾기
        if (nearestSectionTitle) {
          for (const [mappingKey, mapping] of Object.entries(sectionMappings)) {
            // mappingKey 형태: "conversationId-섹션제목"
            if (mappingKey.includes(nearestSectionTitle)) {
              currentSectionMapping = mapping;
              console.log(`정확한 섹션 매핑 찾음: ${nearestSectionTitle} -> ${JSON.stringify(mapping)}`);
              break;
            }
          }
        }

        // 3. 매핑을 찾지 못한 경우 기본값 사용
        if (currentSectionMapping.length === 0) {
          // 가장 최근 매핑 사용 (fallback)
          const mappingValues = Object.values(sectionMappings);
          if (mappingValues.length > 0) {
            currentSectionMapping = mappingValues[mappingValues.length - 1];
            console.log(`기본 매핑 사용: ${JSON.stringify(currentSectionMapping)}`);
          }
        }

        return (
          <SourceRenderer
            key={`source-${index}`}
            content={part}
            sources={sources} // 실시간 sources 전달
            isStreaming={message.isStreaming} // 스트리밍 상태도 전달
            sectionMappings={currentSectionMapping} // 🔥 정확한 섹션 매핑 전달
          />
        );
      }

      // 일반 마크다운 렌더링 (헤더 크기 보존)
      const cleanPart = part; // 헤더 제거하지 않고 그대로 유지
      if (cleanPart.trim()) {
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
            {cleanPart}
          </ReactMarkdown>
        );
      }
      return null;
    }).filter(Boolean); // null 요소 제거

    // 헤더와 컨텐츠를 함께 반환
    return (
      <div className="message-content-wrapper">
        {headerElements}
        {contentElements}
      </div>
    );
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
                      {message.type === "assistant" && !message.isStreaming && message.searchResults && message.searchResults.length > 0 && (
                        <div className="claude-search-results">
                          {console.log("렌더링 중인 완료된 메시지 검색 결과:", message.searchResults)} {/* 디버깅 로그 */}
                          {message.searchResults.map((searchData, index) => (
                            <div key={`search-${searchData.step}-${index}`} className={`search-result-section ${searchData.isIntermediateSearch ? 'intermediate-search' : ''}`}>
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
                                  {/* 🆕 중간 검색 표시 */}
                                  {searchData.isIntermediateSearch && searchData.sectionContext && (
                                    <span className="intermediate-badge">
                                      📊 {searchData.sectionContext.section_title}
                                    </span>
                                  )}
                                  {searchData.query && (
                                    <span className="search-query">
                                      "{searchResultsVisible[`${message.id}-${searchData.step}-${index}`]
                                        ? searchData.query
                                        : (searchData.query.length > 50 ? searchData.query.substring(0, 50) + '...' : searchData.query)}"
                                    </span>
                                  )}
                                  <span className="result-count">{searchData.results.length}개 결과</span>
                                  {/* 🆕 중간 검색 이유 표시 */}
                                  {searchData.isIntermediateSearch && searchData.sectionContext && (
                                    <span className="search-reason">
                                      {searchData.sectionContext.search_reason}
                                    </span>
                                  )}
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
                                        <span>관련성: {((result.score || result.relevance_score || 0) * 100).toFixed(0)}%</span>
                                        <span>타입: {result.document_type || result.type || 'unknown'}</span>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}

                      {/* 🆕 현재 스트리밍 중인 메시지에서만 실시간 검색 결과 표시 */}
                      {message.type === "assistant" && message.isStreaming && isStreaming && currentSearchResults.length > 0 && (
                        <div className="claude-search-results">
                          {currentSearchResults.map((searchData, index) => (
                            <div key={`search-${searchData.step}-${index}`} className={`search-result-section ${searchData.isIntermediateSearch ? 'intermediate-search' : ''}`}>
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
                                  {/* 🆕 중간 검색 표시 */}
                                  {searchData.isIntermediateSearch && searchData.sectionContext && (
                                    <span className="intermediate-badge">
                                      📊 {searchData.sectionContext.section_title}
                                    </span>
                                  )}
                                  {searchData.query && (
                                    <span className="search-query">
                                      "{searchResultsVisible[`${message.id}-${searchData.step}-${index}`]
                                        ? searchData.query
                                        : (searchData.query.length > 50 ? searchData.query.substring(0, 50) + '...' : searchData.query)}"
                                    </span>
                                  )}
                                  <span className="result-count">{searchData.results.length}개 결과</span>
                                  {/* 🆕 중간 검색 이유 표시 */}
                                  {searchData.isIntermediateSearch && searchData.sectionContext && (
                                    <span className="search-reason">
                                      {searchData.sectionContext.search_reason}
                                    </span>
                                  )}
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
                                        <span>관련성: {((result.score || result.relevance_score || 0) * 100).toFixed(0)}%</span>
                                        <span>타입: {result.document_type || result.type || 'unknown'}</span>
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
