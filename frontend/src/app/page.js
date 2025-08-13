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
  const [searchResultsVisible, setSearchResultsVisible] = useState({});
  const [conversationSearchResults, setConversationSearchResults] = useState({});

  // 섹션별 매핑 정보 상태 추가
  const [sectionMappings, setSectionMappings] = useState({});

  // >> 핵심 추가: 데이터 딕셔너리 상태들
  const [fullDataDict, setFullDataDict] = useState({}); // 전체 데이터 딕셔너리
  const [sectionDataDicts, setSectionDataDicts] = useState({}); // 섹션별 데이터 딕셔너리

  // 실시간 생각 스트리밍 상태들
  const [statusMessages, setStatusMessages] = useState([]); // 상태 메시지들 배열
  const [statusToggleOpen, setStatusToggleOpen] = useState(false); // 토글 열림/닫힘
  const [streamingStartTime, setStreamingStartTime] = useState(null); // 시작 시간
  const [elapsedTime, setElapsedTime] = useState(0); // 경과 시간 (초)
  const [isStreamingCompleted, setIsStreamingCompleted] = useState(false); // 완료 여부

  // 자동 스크롤 제어 상태
  const [autoScrollEnabled, setAutoScrollEnabled] = useState(true); // 자동 스크롤 활성화 여부

  // 🔍 디버깅: currentSearchResults 변경사항 추적
  const setCurrentSearchResultsDebug = (newResults) => {
    console.log("🔍 currentSearchResults 변경:", {
      이전: currentSearchResults.length,
      새로운: Array.isArray(newResults) ? newResults.length : "함수",
      스택: new Error().stack?.split('\n')[2]?.trim()
    });
    setCurrentSearchResults(newResults);
  };

  // 스크롤 관리
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const textareaRef = useRef(null);

  // 차트 중복 방지를 위한 ID 추적
  const processedChartIds = useRef(new Set());

  // 메시지 끝으로 스크롤 (자동 스크롤이 활성화된 경우에만)
  const scrollToBottom = useCallback(() => {
    if (autoScrollEnabled) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [autoScrollEnabled]);

  // 스크롤 위치가 하단에 있는지 확인하는 함수
  const isScrolledToBottom = useCallback(() => {
    if (!messagesContainerRef.current) return true;
    
    const container = messagesContainerRef.current;
    const threshold = 100; // 하단에서 100px 이내면 하단으로 간주
    
    return container.scrollHeight - container.scrollTop - container.clientHeight < threshold;
  }, []);

  // 스크롤 이벤트 핸들러
  const handleScroll = useCallback(() => {
    if (!messagesContainerRef.current) return;
    
    const scrolledToBottom = isScrolledToBottom();
    
    // 사용자가 하단으로 스크롤하면 자동 스크롤 재개
    if (scrolledToBottom && !autoScrollEnabled) {
      setAutoScrollEnabled(true);
    }
    // 사용자가 하단에서 벗어나면 자동 스크롤 일시정지
    else if (!scrolledToBottom && autoScrollEnabled) {
      setAutoScrollEnabled(false);
    }
  }, [autoScrollEnabled, isScrolledToBottom]);

  // 스크롤 이벤트 리스너 등록
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    container.addEventListener('scroll', handleScroll, { passive: true });
    
    return () => {
      container.removeEventListener('scroll', handleScroll);
    };
  }, [handleScroll]);

  // 메시지나 스트리밍 내용이 변경될 때 자동 스크롤
  useEffect(() => {
    scrollToBottom();
  }, [currentConversation, currentStreamingMessage, currentStreamingCharts, scrollToBottom]);

  // 실시간 경과 시간 업데이트
  useEffect(() => {
    let interval = null;
    
    if (isStreaming && streamingStartTime && !isStreamingCompleted) {
      interval = setInterval(() => {
        const now = Date.now();
        const elapsed = Math.floor((now - streamingStartTime) / 1000);
        setElapsedTime(elapsed);
      }, 1000);
    }
    
    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [isStreaming, streamingStartTime, isStreamingCompleted]);

  // 로컬 스토리지에서 대화 히스토리 로드
  useEffect(() => {
    const savedConversations = localStorage.getItem("chatConversations");
    if (savedConversations) {
      try {
        const parsedConversations = JSON.parse(savedConversations);
        setConversations(parsedConversations);

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

    const savedConversationSearchResults = localStorage.getItem("conversationSearchResults");
    if (savedConversationSearchResults) {
      try {
        setConversationSearchResults(JSON.parse(savedConversationSearchResults));
        console.log("대화별 검색 결과 복원 완료");
      } catch (error) {
        console.error("대화별 검색 결과 로드 오류:", error);
      }
    }

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
  }, []);

  // 대화 히스토리 저장
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
    setCurrentSearchResultsDebug([]);
    setSearchResultsVisible({});
    setConversationSearchResults({});

    // >> 데이터 딕셔너리 초기화
    setFullDataDict({});
    setSectionDataDicts({});

    // 실시간 생각 스트리밍 상태 초기화
    setStatusMessages([]);
    setStatusToggleOpen(false);
    setStreamingStartTime(null);
    setElapsedTime(0);
    setIsStreamingCompleted(false);
    setStatusMessage("");

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

    // >> 데이터 딕셔너리 복원
    if (conv.messages && conv.messages.length > 0) {
      const lastAssistantMessage = conv.messages
        .reverse()
        .find(msg => msg.type === "assistant" && !msg.isStreaming);

      if (lastAssistantMessage) {
        if (lastAssistantMessage.fullDataDict) {
          setFullDataDict(lastAssistantMessage.fullDataDict);
          console.log("전체 데이터 딕셔너리 복원:", Object.keys(lastAssistantMessage.fullDataDict).length, "개");
        }
        if (lastAssistantMessage.sectionDataDicts) {
          setSectionDataDicts(lastAssistantMessage.sectionDataDicts);
          console.log("섹션 데이터 딕셔너리 복원:", Object.keys(lastAssistantMessage.sectionDataDicts).length, "개");
        }
      }
      conv.messages.reverse(); // 순서 복원
    }

    const restoredSearchResults = [];
    if (conv.messages) {
      conv.messages.forEach(message => {
        if (message.searchResults && Array.isArray(message.searchResults)) {
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

  // 경과 시간 포맷 함수
  const formatElapsedTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
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
    setSourcesData(null);

    // 실시간 생각 스트리밍 상태 초기화 및 시작
    const currentStreamingStartTime = Date.now();
    
    setStatusMessages([{
      id: Date.now(),
      message: "생각하는 중...",
      timestamp: Date.now()
    }]);
    setStatusToggleOpen(false);
    setStreamingStartTime(currentStreamingStartTime);
    setElapsedTime(0);
    setIsStreamingCompleted(false);

    setCurrentSearchResultsDebug([]);
    console.log("🔄 새 질문 시작: 검색 결과 초기화 (세션별 관리)");
    const assistantMessage = {
      id: Date.now() + 1,
      type: "assistant",
      content: "",
      charts: [],
      timestamp: new Date().toISOString(),
      isStreaming: true,
      streamingStartTime: currentStreamingStartTime,  // 메시지에 스트리밍 시작 시간 저장
      sources: null,
    };

    setCurrentConversation((prev) => [...prev, assistantMessage]);

    const tempConversationWithNewMessages = [...currentConversation, userMessage, assistantMessage];
    const tempConversationData = {
      id: conversationId || Date.now().toString(),
      title: currentQuery.slice(0, 30) + (currentQuery.length > 30 ? "..." : ""),
      messages: tempConversationWithNewMessages,
      lastUpdated: new Date().toISOString(),
      isStreaming: true,
    };

    localStorage.setItem("currentStreamingConversation", JSON.stringify(tempConversationData));
    console.log("🔄 스트리밍 중 대화 상태 저장:", tempConversationData.id);

    try {
      console.log("API_BASE_URL 값:", API_BASE_URL);
      console.log("API 요청 URL:", `${API_BASE_URL}/query/stream`);
      console.log("요청 데이터:", {
        query: currentQuery,
        session_id: conversationId || undefined,
        message_id: assistantMessage.id,
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
          message_id: String(assistantMessage.id),
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
                  // 안전한 접근: data.data.message 또는 data.message
                  const statusMessage = data.data?.message || data.message || "처리 중...";
                  setStatusMessage(statusMessage);
                  
                  // 실시간 생각 스트리밍에 상태 메시지 추가
                  setStatusMessages(prev => {
                    const newMessage = {
                      id: Date.now() + Math.random(),
                      message: statusMessage,
                      timestamp: Date.now()
                    };
                    console.log("🔄 새 상태 메시지 추가:", statusMessage, "현재 배열 길이:", prev.length + 1);
                    return [...prev, newMessage];
                  });
                  break;

                // >> 새로운 이벤트 타입: 전체 데이터 딕셔너리
                case "full_data_dict":
                  console.log("🎯 full_data_dict 이벤트 받음, data 구조:", data);
                  // main.py에서 data를 펼쳐서 보내므로 data.data_dict로 직접 접근
                  const dataDict = data.data_dict;  // 두 가지 경우 모두 처리
                  if (dataDict) {
                    console.log("✅ 전체 데이터 딕셔너리 받음:", Object.keys(dataDict).length, "개");
                    console.log("📊 데이터 딕셔너리 키들:", Object.keys(dataDict));

                    setFullDataDict(dataDict);

                    // dataDict를 sourcesData에도 직접 설정
                    setSourcesData(dataDict);
                    console.log("✅ setSourcesData에도 dataDict 설정 완료");

                    // 즉시 현재 메시지에도 저장
                    setCurrentConversation((prev) =>
                      prev.map((msg) =>
                        msg.id === assistantMessage.id
                          ? { ...msg, fullDataDict: dataDict }
                          : msg
                      )
                    );
                    console.log("✅ 현재 메시지에도 fullDataDict 저장 완료");
                  } else {
                    console.error("❌ 데이터 딕셔너리를 찾을 수 없음, data 구조:", JSON.stringify(data, null, 2));
                  }
                  break;

                case "chart":
                  if (data.chart_data) {
                    finalCharts.push(data.chart_data);
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
                  }
                  break;

                case "plan":
                  totalSteps = data.total_steps || data.data?.plan?.steps?.length || data.plan?.steps?.length || 0;
                  setStatusMessage(`실행 계획: ${totalSteps}개 단계`);
                  console.log("실행 계획:", data.data?.plan || data.plan);
                  break;

                case "step_start":
                  currentStep = data.step || data.data?.step;

                  let statusText = `단계 ${currentStep}/${totalSteps}: ${data.description || data.data?.description || "처리 중"}`;
                  if ((data.tool || data.data?.tool) && (data.query || data.data?.query)) {
                    const tool = data.tool || data.data?.tool;
                    const query = data.query || data.data?.query;
                    const status = data.status || data.data?.status;
                    
                    if (status === "searching") {
                      statusText = `${tool}로 검색 중: "${query}"`;
                    } else if (data.status === "processing") {
                      statusText = `${data.tool}로 처리 중`;
                    }
                  }

                  setStatusMessage(statusText);
                  break;

                case "search_results":
                  console.log("검색 결과 받음:", data);

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
                    messageId: data.message_id || assistantMessage.id,
                    isIntermediateSearch: isIntermediateSearch,
                    sectionContext: sectionContext
                  };
                  console.log("처리된 검색 데이터:", searchResultData);

                  const currentConvId = conversationId || data.session_id || Date.now().toString();

                  setConversationSearchResults(prev => {
                    const newResults = {
                      ...prev,
                      [currentConvId]: [...(prev[currentConvId] || []), searchResultData]
                    };
                    localStorage.setItem("conversationSearchResults", JSON.stringify(newResults));
                    return newResults;
                  });

                  setCurrentSearchResultsDebug(prev => {
                    const newResults = [...prev, searchResultData];

                    localStorage.setItem("currentSearchResults", JSON.stringify(newResults));
                    console.log(`🔍 검색 결과 추가 (총 ${newResults.length}개):`, searchResultData);

                    setCurrentConversation(prevMessages => {
                      return prevMessages.map(msg => {
                        if (msg.id === assistantMessage.id && msg.type === "assistant") {
                          const messageSearchResults = newResults.filter(result => {
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

                  setSearchResultsVisible(prev => {
                    const newVisible = {
                      ...prev,
                      [`${data.step}-latest`]: true
                    };
                    localStorage.setItem("searchResultsVisible", JSON.stringify(newVisible));
                    return newVisible;
                  });

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

                case "section_mapping":
                  console.log("섹션 매핑 정보 받음:", data);
                  console.log("섹션 제목:", data.section_title);
                  console.log("섹션 데이터 딕셔너리:", data.section_data_dict);
                  console.log("사용된 인덱스:", data.section_indexes);

                  const sectionKey = `${conversationId || data.session_id || Date.now()}-${data.section_title}`;
                  console.log("생성된 섹션 키:", sectionKey);

                  setSectionDataDicts(prev => {
                    const newSectionDicts = {
                      ...prev,
                      [sectionKey]: {
                        dataDict: data.section_data_dict,
                        indexes: data.section_indexes,
                        title: data.section_title
                      }
                    };
                    console.log("업데이트된 섹션 딕셔너리:", newSectionDicts);
                    return newSectionDicts;
                  });
                  break;

                case "section_header":
                  setCurrentConversation((prev) => {
                    const updated = [...prev];
                    if (updated.length > 0 && updated[updated.length - 1].id === assistantMessage.id) {
                      const lastMessage = updated[updated.length - 1];
                      if (!lastMessage.sectionHeaders) {
                        lastMessage.sectionHeaders = [];
                      }
                      lastMessage.sectionHeaders.push({
                        id: `header-${Date.now()}-${Math.random()}`,
                        title: data.title,
                        timestamp: Date.now()
                      });
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
                  // sources 이벤트는 더 이상 사용하지 않음 (full_data_dict만 사용)
                  console.log("⚠️ sources 이벤트 수신 (무시됨):", data);
                  break;

                case "section_start":
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
                  setStatusMessage("완료");
                  console.log("complete 이벤트 수신, 현재 fullDataDict 상태:", {
                    hasFullDataDict: !!fullDataDict,
                    fullDataDictSize: Object.keys(fullDataDict || {}).length
                  });
                  break;

                case "final_complete":
                  setStatusMessage("");
                  
                  // 최종 경과 시간 계산 (먼저 계산)
                  const finalElapsedTime = streamingStartTime ? Math.floor((Date.now() - streamingStartTime) / 1000) : 0;
                  
                  console.log("🔥 finalElapsedTime 계산 확인:", {
                    streamingStartTime,
                    currentTime: Date.now(),
                    timeDiff: streamingStartTime ? (Date.now() - streamingStartTime) : 0,
                    finalElapsedTime,
                    hasStreamingStartTime: !!streamingStartTime
                  });
                  
                  // 실시간 경과 시간 상태도 최종 시간으로 업데이트
                  setElapsedTime(finalElapsedTime);
                  
                  // 실시간 생각 스트리밍 완료 처리
                  setIsStreamingCompleted(true);
                  
                  // 최종 상태 메시지 생성 - 함수형 업데이트로 최신 상태 사용
                  setStatusMessages(prevMessages => {
                    const finalMessages = [...prevMessages, {
                      id: Date.now() + Math.random(),
                      message: "보고서 생성 완료",
                      timestamp: Date.now(),
                      isCompleted: true
                    }];
                    
                    console.log("🔥 setStatusMessages 내부 값 확인:", {
                      finalElapsedTime,
                      streamingStartTime,
                      hasStreamingStartTime: !!streamingStartTime
                    });
                    
                    return finalMessages;
                  });
                  
                  // 메시지에 저장할 때도 최신 상태 사용
                  setCurrentConversation((prevConversation) => {
                    console.log("🚨 final_complete - 메시지 업데이트 시작:", {
                      targetMessageId: assistantMessage.id,
                      conversationLength: prevConversation.length,
                      assistantMessages: prevConversation.filter(m => m.type === "assistant").map(m => ({id: m.id, isStreaming: m.isStreaming}))
                    });
                    
                    const newConversation = prevConversation.map((msg) => {
                      if (msg.id === assistantMessage.id) {
                        console.log('🔥 final_complete - 메시지 업데이트:', {
                          messageId: msg.id,
                          finalElapsedTime: finalElapsedTime,
                          streamingStartTime: streamingStartTime,
                          fullDataDictToSave: !!fullDataDict,
                          fullDataDictSize: Object.keys(fullDataDict || {}).length,
                          fullDataDictKeys: Object.keys(fullDataDict || {}).slice(0, 5),
                          msgCurrentFullDataDict: !!msg.fullDataDict,
                          msgCurrentFullDataDictSize: Object.keys(msg.fullDataDict || {}).length
                        });

                        // 메시지에서 이미 저장된 fullDataDict를 우선 사용
                        const finalFullDataDict = msg.fullDataDict || fullDataDict;

                        // 메시지의 streamingStartTime이 있다면 그것을 사용해서 정확한 경과시간 계산
                        const messageStreamingStartTime = msg.streamingStartTime || streamingStartTime;
                        const accurateElapsedTime = messageStreamingStartTime ? 
                          Math.floor((Date.now() - messageStreamingStartTime) / 1000) : finalElapsedTime;

                        console.log("🚨 정확한 시간 계산 확인:", {
                          msgId: msg.id,
                          msgStreamingStartTime: messageStreamingStartTime,
                          globalStreamingStartTime: streamingStartTime,
                          currentTime: Date.now(),
                          timeDifference: messageStreamingStartTime ? (Date.now() - messageStreamingStartTime) : 0,
                          accurateElapsedTime,
                          finalElapsedTime,
                          willSaveAsTotal: accurateElapsedTime
                        });

                        const updatedMessage = {
                          ...msg,
                          charts: finalCharts,
                          isStreaming: false,
                          // >> 우선순위: 메시지에 저장된 것 > 상태의 것
                          fullDataDict: finalFullDataDict,
                          sectionDataDicts: sectionDataDicts,
                          // 상태 메시지들과 시간 정보 저장 (이전에 이미 생성된 상태 사용)
                          statusMessages: [...statusMessages, {
                            id: Date.now() + Math.random(),
                            message: "보고서 생성 완료",
                            timestamp: Date.now(),
                            isCompleted: true
                          }],
                          streamingStartTime: messageStreamingStartTime,
                          totalElapsedTime: accurateElapsedTime  // 더 정확한 경과 시간 사용
                        };
                        
                        console.log("🔥 메시지 업데이트 - 시간 저장 확인:", {
                          messageId: msg.id,
                          finalElapsedTime,
                          accurateElapsedTime,
                          globalStreamingStartTime: streamingStartTime,
                          messageStreamingStartTime,
                          updatedMessage: {
                            totalElapsedTime: updatedMessage.totalElapsedTime,
                            streamingStartTime: updatedMessage.streamingStartTime,
                            hasStatusMessages: !!updatedMessage.statusMessages
                          }
                        });
                        
                        return updatedMessage;
                      }
                      return msg;
                    });

                    console.log("🚨 final_complete - conversation 업데이트 완료:", {
                      targetMessageId: assistantMessage.id,
                      foundUpdatedMessage: !!newConversation.find(m => m.id === assistantMessage.id),
                      updatedMessageData: newConversation.find(m => m.id === assistantMessage.id)?.totalElapsedTime
                    });
                    
                    // 대화 저장
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

                    localStorage.removeItem('currentStreamingConversation');

                    return newConversation;
                  });

                  console.log("검색 결과 및 데이터 딕셔너리 유지됨");
                  setIsStreaming(false);
                  console.log("스트리밍 완료 - 검색 결과 및 출처 정보 유지");
                  break;

                case "error":
                  setStatusMessage(`오류: ${data.message}`);
                  setIsStreaming(false);
                  localStorage.removeItem('currentStreamingConversation');
                  return;

                case "result":
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

  // >> renderMessageContent 함수 수정 - 실제 인덱스 기반 출처 매핑
  const renderMessageContent = (message) => {
    const content = message.content || "";
    const charts = message.charts || [];
    const sectionHeaders = message.sectionHeaders || [];

    // >> 간단하게: 현재 전역 상태 우선 사용
    const messageFullDataDict = fullDataDict || message.fullDataDict || {};
    const messageSectionDataDicts = sectionDataDicts || message.sectionDataDicts || {};

    console.log("렌더링 중인 메시지의 데이터 딕셔너리:", {
      messageId: message.id,
      hasFullDataDict: !!messageFullDataDict,
      fullDataDictSize: Object.keys(messageFullDataDict).length,
      hasSectionDataDicts: !!messageSectionDataDicts,
      sectionDataDictsSize: Object.keys(messageSectionDataDicts).length
    });

    // 실시간 출처 데이터 우선 사용
    let sources = [];
    if (message.sources) {
      if (Array.isArray(message.sources)) {
        sources = message.sources;
      } else if (message.sources.sources && Array.isArray(message.sources.sources)) {
        sources = message.sources.sources;
      }
    } else if (message.isStreaming && sourcesData && sourcesData.sources) {
      sources = sourcesData.sources;
    }

    console.log("렌더링할 출처 정보:", sources);

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

      // >> 핵심 수정: [SOURCE:번호] 형식 처리 시 항상 전체 데이터 딕셔너리 사용
      if (part.includes('[SOURCE:')) {
        console.log('SOURCE 패턴 감지, 전체 데이터 딕셔너리 사용:', Object.keys(messageFullDataDict).length, '개');

        return (
          <SourceRenderer
            key={`source-${index}`}
            content={part}
            sources={sources}
            isStreaming={message.isStreaming}
            dataDict={messageFullDataDict} // >> 핵심: 항상 전체 데이터 딕셔너리 사용
          />
        );
      }

      // 일반 마크다운 렌더링
      const cleanPart = part;
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
    }).filter(Boolean);

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
        <div className="messages-container" ref={messagesContainerRef}>
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
                      {/* 어시스턴트 메시지에서 실시간 생각 스트리밍 UI를 맨 위에 표시 */}
                      {message.type === "assistant" && (() => {
                        // 현재 스트리밍 중인 메시지이거나, 완료된 메시지 중 상태 정보가 있는 경우 표시
                        const isCurrentStreaming = message.isStreaming && isStreaming;
                        const messageStatusMessages = message.statusMessages || [];
                        const hasStatusHistory = !message.isStreaming && messageStatusMessages.length > 0;
                        
                        console.log("🎯 상단 스트리밍 메시지 렌더링 조건 체크:", {
                          messageId: message.id,
                          isStreaming,
                          messageIsStreaming: message.isStreaming,
                          isCurrentStreaming,
                          hasStatusHistory,
                          messageStatusMessagesLength: messageStatusMessages.length,
                          currentStatusMessagesLength: statusMessages.length,
                          shouldShow: isCurrentStreaming || hasStatusHistory,
                          messageData: {
                            hasStatusMessages: !!message.statusMessages,
                            hasStreamingStartTime: !!message.streamingStartTime,
                            hasTotalElapsedTime: !!message.totalElapsedTime,
                            statusMessages: message.statusMessages
                          }
                        });
                        
                        return isCurrentStreaming || hasStatusHistory;
                      })() && (
                        <div className="thinking-stream">
                          <div 
                            className="thinking-stream-header"
                            onClick={() => setStatusToggleOpen(!statusToggleOpen)}
                          >
                            <div className="thinking-stream-title">
                              <div className="pulse-dot"></div>
                              <span>
                                {(() => {
                                  const isCurrentStreaming = message.isStreaming && isStreaming;
                                  
                                  // 시간 표시 로직 개선
                                  let displayTime;
                                  if (isCurrentStreaming) {
                                    // 현재 스트리밍 중인 메시지: 실시간 경과 시간 사용
                                    displayTime = elapsedTime;
                                  } else if (message.totalElapsedTime) {
                                    // 완료된 메시지이고 저장된 시간이 있는 경우: 저장된 시간 사용
                                    displayTime = message.totalElapsedTime;
                                  } else {
                                    // 저장된 시간이 없는 경우 0으로 설정
                                    displayTime = 0;
                                  }
                                  
                                  // 디버깅 로그 추가
                                  console.log("🚨 UI 시간 표시 최종 확인:", {
                                    messageId: message.id,
                                    step: "UI_RENDER",
                                    isCurrentStreaming,
                                    globalElapsedTime: elapsedTime,
                                    messageTotalElapsedTime: message.totalElapsedTime,
                                    messageStreamingStartTime: message.streamingStartTime,
                                    globalStreamingStartTime: streamingStartTime,
                                    finalDisplayTime: displayTime,
                                    displayTimeSource: isCurrentStreaming ? "global_elapsed" : 
                                                     (message.totalElapsedTime ? "message_total" : "zero"),
                                    willShowTime: displayTime,
                                    formattedTime: formatElapsedTime(displayTime),
                                    messageComplete: !message.isStreaming
                                  });
                                  
                                  if (statusToggleOpen) {
                                    return isCurrentStreaming ? `생각하는 중...` : `생각 과정`;
                                  } else {
                                    return isCurrentStreaming ? 
                                      `생각하는 중... (${formatElapsedTime(displayTime)})` :
                                      `생각 완료 (${formatElapsedTime(displayTime)})`;
                                  }
                                })()}
                              </span>
                            </div>
                            <div className="thinking-stream-toggle">
                              {statusToggleOpen ? '▼' : '▶'}
                            </div>
                          </div>
                          
                          {statusToggleOpen && (
                            <div className="thinking-stream-content">
                              {(() => {
                                // 현재 스트리밍 중이면 실시간 상태 메시지, 아니면 저장된 메시지 사용
                                const displayMessages = message.isStreaming && isStreaming ? statusMessages : (message.statusMessages || []);
                                const displayStartTime = message.isStreaming && isStreaming ? streamingStartTime : message.streamingStartTime;
                                
                                return displayMessages.map((status) => (
                                  <div 
                                    key={status.id} 
                                    className={`thinking-step ${status.isCompleted ? 'completed' : ''}`}
                                  >
                                    <div className="step-indicator">
                                      {status.isCompleted ? '✓' : '●'}
                                    </div>
                                    <div className="step-content">
                                      <span className="step-message">{status.message}</span>
                                      <span className="step-time">
                                        {displayStartTime ? formatElapsedTime(Math.floor((status.timestamp - displayStartTime) / 1000)) : '0:00'}
                                      </span>
                                    </div>
                                  </div>
                                ));
                              })()}
                            </div>
                          )}
                        </div>
                      )}

                      {/* 완료된 어시스턴트 메시지 위에 해당 검색 결과 먼저 표시 */}
                      {message.type === "assistant" && !message.isStreaming && message.searchResults && message.searchResults.length > 0 && (
                        <div className="claude-search-results">
                          {console.log("렌더링 중인 완료된 메시지 검색 결과:", message.searchResults)}
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

                      {/* 현재 스트리밍 중인 메시지에서만 실시간 검색 결과 표시 */}
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
                      {message.type === "assistant" && !message.isStreaming && Object.keys(fullDataDict).length > 0 && (
                        <div className="message-actions">
                          <button
                            className="sources-simple-btn"
                            onClick={() => {
                              console.log('소스 패널 열기, fullDataDict:', Object.keys(fullDataDict).length, '개');
                              setFullDataDict(fullDataDict); // 현재 전역 상태 사용
                              if (!sourcesPanelVisible) {
                                toggleSourcesPanel();
                              }
                            }}
                          >
                            {Object.keys(fullDataDict).length}개 출처
                          </button>
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
              onKeyDown={handleKeyPress}
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
        dataDict={fullDataDict} // >> 전체 데이터 딕셔너리 전달
      />
    </div>
  );
}
