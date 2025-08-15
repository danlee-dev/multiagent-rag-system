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

  // 새로운 메시지별 상태 관리 시스템
  const [messageStates, setMessageStates] = useState({}); // 메시지별 상태 {messageId: {status, startTime, endTime, statusHistory}}
  const [statusToggleOpen, setStatusToggleOpen] = useState(false); // 토글 열림/닫힘

  // 자동 스크롤 제어 상태
  const [autoScrollEnabled, setAutoScrollEnabled] = useState(true); // 자동 스크롤 활성화 여부

  // 팀 선택 관련 상태
  const [selectedTeam, setSelectedTeam] = useState(null); // 선택된 팀 정보
  const [availableTeams, setAvailableTeams] = useState([]); // 사용 가능한 팀 목록
  const [teamSectionExpanded, setTeamSectionExpanded] = useState(false); // 팀 섹션 확장 상태
  const [aiAutoEnabled, setAiAutoEnabled] = useState(true); // AI 자동 선택 활성화 상태
  const [teamDropupOpen, setTeamDropupOpen] = useState(false); // 팀 선택 드롭업 상태
  const [abortController, setAbortController] = useState(null); // 스트리밍 중단용

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

  // 실시간 경과 시간 업데이트 (1초마다)
  useEffect(() => {
    const interval = setInterval(() => {
      setMessageStates(prev => {
        const newStates = { ...prev };
        let hasActiveStreaming = false;
        
        Object.keys(newStates).forEach(messageId => {
          const state = newStates[messageId];
          if (state.isActive && !state.isCompleted) {
            hasActiveStreaming = true;
            const now = Date.now();
            const elapsedSeconds = Math.floor((now - state.startTime) / 1000);
            newStates[messageId] = { ...state, elapsedSeconds };
          }
        });
        
        if (hasActiveStreaming) {
          // 로컬스토리지에 저장
          localStorage.setItem('messageStates', JSON.stringify(newStates));
        }
        
        return newStates;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // 드롭업 외부 클릭 시 닫기
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (!event.target.closest('.team-tag-container')) {
        setTeamDropupOpen(false);
      }
    };

    if (teamDropupOpen) {
      document.addEventListener('click', handleClickOutside);
      return () => document.removeEventListener('click', handleClickOutside);
    }
  }, [teamDropupOpen]);

  // 스트리밍 시작 시 드롭업 닫기
  useEffect(() => {
    if (isStreaming) {
      setTeamDropupOpen(false);
    }
  }, [isStreaming]);

  // 로컬 스토리지에서 데이터 로드
  useEffect(() => {
    // 대화 히스토리 로드
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

    // 메시지 상태 로드
    const savedMessageStates = localStorage.getItem("messageStates");
    if (savedMessageStates) {
      try {
        const parsedStates = JSON.parse(savedMessageStates);
        setMessageStates(parsedStates);
        console.log(`메시지 상태 복원: ${Object.keys(parsedStates).length}개`);
      } catch (error) {
        console.error("메시지 상태 로드 오류:", error);
        setMessageStates({});
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
    // 🔑 핵심: 스트리밍 상태 먼저 해제
    setIsStreaming(false);

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

    // 메시지 상태 초기화
    setMessageStates({});
    setStatusToggleOpen(false);
    setStatusMessage("");
    localStorage.removeItem('messageStates');

    localStorage.removeItem("currentSearchResults");
    localStorage.removeItem("searchResultsVisible");
    localStorage.removeItem("conversationSearchResults");

    // AI 자동 선택을 기본값으로 설정
    setAiAutoEnabled(true);
    if (availableTeams.length > 0) {
      const autoSelectTeam = availableTeams.find(team => team.id === "AI_AUTO");
      if (autoSelectTeam) {
        setSelectedTeam(autoSelectTeam);
        console.log("🤖 새 채팅 시작 - AI 자동 선택이 기본값으로 설정되었습니다");
      }
    }

    console.log("새 채팅이 시작되었습니다. 스트리밍 상태:", false);
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

  // 메시지 상태 관리 헬퍼 함수들
  const initializeMessageState = (messageId) => {
    const startTime = Date.now();
    const initialState = {
      messageId,
      startTime,
      endTime: null,
      elapsedSeconds: 0,
      isActive: true,
      isCompleted: false,
      statusHistory: [{
        id: Date.now(),
        message: "생각하는 중...",
        timestamp: startTime,
        elapsedSeconds: 0
      }]
    };
    
    setMessageStates(prev => {
      const newStates = { ...prev, [messageId]: initialState };
      localStorage.setItem('messageStates', JSON.stringify(newStates));
      return newStates;
    });
    
    return startTime;
  };

  const addMessageStatus = (messageId, statusMessage) => {
    setMessageStates(prev => {
      const currentState = prev[messageId];
      if (!currentState) return prev;
      
      const now = Date.now();
      const elapsedSeconds = Math.floor((now - currentState.startTime) / 1000);
      
      const newStatus = {
        id: Date.now() + Math.random(),
        message: statusMessage,
        timestamp: now,
        elapsedSeconds
      };
      
      const updatedState = {
        ...currentState,
        elapsedSeconds,
        statusHistory: [...currentState.statusHistory, newStatus]
      };
      
      const newStates = { ...prev, [messageId]: updatedState };
      localStorage.setItem('messageStates', JSON.stringify(newStates));
      
      return newStates;
    });
  };

  const completeMessageState = (messageId, wasAborted = false) => {
    setMessageStates(prev => {
      const currentState = prev[messageId];
      if (!currentState) return prev;
      
      const now = Date.now();
      const finalElapsedSeconds = Math.floor((now - currentState.startTime) / 1000);
      
      const completionStatus = {
        id: Date.now() + Math.random(),
        message: wasAborted ? "보고서 생성 중지" : "보고서 생성 완료",
        timestamp: now,
        elapsedSeconds: finalElapsedSeconds,
        isCompleted: !wasAborted,
        wasAborted: wasAborted
      };
      
      const completedState = {
        ...currentState,
        endTime: now,
        elapsedSeconds: finalElapsedSeconds,
        isActive: false,
        isCompleted: true,
        statusHistory: [...currentState.statusHistory, completionStatus]
      };
      
      const newStates = { ...prev, [messageId]: completedState };
      localStorage.setItem('messageStates', JSON.stringify(newStates));
      
      return newStates;
    });
  };

  const getMessageState = (messageId) => {
    return messageStates[messageId] || null;
  };

  // 스트리밍 중단 함수
  const stopGeneration = () => {
    if (abortController) {
      console.log("🛑 사용자가 생성을 중단했습니다");
      abortController.abort();
      setAbortController(null);
      setIsStreaming(false);
      setStatusMessage("생성이 중단되었습니다");
      
      // 현재 스트리밍 중인 메시지 상태 업데이트
      if (currentConversation.length > 0) {
        const lastMessage = currentConversation[currentConversation.length - 1];
        if (lastMessage && lastMessage.type === "assistant" && lastMessage.isStreaming) {
          // 메시지 상태 완료 처리 (타이머 중단) - 중단 상태로 표시
          completeMessageState(lastMessage.id, true);
          
          setCurrentConversation(prev => 
            prev.map(msg => 
              msg.id === lastMessage.id 
                ? { ...msg, isStreaming: false, wasAborted: true }
                : msg
            )
          );
        }
      }
    }
  };


  // 팀 목록 로드 함수
  const loadAvailableTeams = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/teams`);
      const data = await response.json();

      // 백엔드에서 받은 데이터 검증 및 변환
      const rawTeams = data.teams || [];
      console.log("백엔드에서 받은 teams 데이터:", rawTeams);

      // 각 팀 객체가 올바른 구조인지 확인하고 정규화
      const teams = rawTeams.map((team, index) => {
        if (typeof team === 'string') {
          // 문자열인 경우 객체로 변환
          return {
            id: team,
            name: team,
            description: `${team} 전용 응답`
          };
        } else if (team && typeof team === 'object') {
          // 이미 객체인 경우 필수 필드 확인
          return {
            id: team.id || team.name || `team-${index}`,
            name: team.name || team.id || `팀 ${index}`,
            description: team.description || `${team.name || team.id} 전용 응답`
          };
        } else {
          // 잘못된 데이터인 경우 기본값
          return {
            id: `team-${index}`,
            name: `팀 ${index}`,
            description: `팀 ${index} 전용 응답`
          };
        }
      });

      console.log("정규화된 teams 데이터:", teams);

      // AI 자동 선택 옵션 추가
      const teamsWithAuto = [
        {
          id: "AI_AUTO",
          name: "AI 자동 선택",
          description: "AI가 질문을 분석하여 가장 적합한 전문가를 자동으로 선택합니다"
        },
        ...teams
      ];

      setAvailableTeams(teamsWithAuto);
      
      // AI 자동 선택을 기본값으로 설정
      if (!selectedTeam) {
        const autoSelectTeam = teamsWithAuto.find(team => team.id === "AI_AUTO");
        if (autoSelectTeam) {
          setSelectedTeam(autoSelectTeam);
          console.log("🤖 AI 자동 선택이 기본값으로 설정되었습니다");
        }
      }
      
      console.log("사용 가능한 팀 목록:", teamsWithAuto);
    } catch (error) {
      console.error("팀 목록 로드 실패:", error);
      // 기본 팀 목록 설정
      setAvailableTeams([
        {id: "기본", name: "기본", description: "기본 응답"}
      ]);
    }
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

    setCurrentSearchResultsDebug([]);
    console.log("🔄 새 질문 시작: 검색 결과 초기화 (세션별 관리)");
    
    const assistantMessage = {
      id: Date.now() + 1,
      type: "assistant",
      content: "",
      charts: [],
      timestamp: new Date().toISOString(),
      isStreaming: true,
      sources: null,
    };

    // 메시지 상태 초기화
    const messageStartTime = initializeMessageState(assistantMessage.id);
    
    console.log("🕐 메시지 상태 초기화:", {
      messageId: assistantMessage.id,
      startTime: messageStartTime
    });

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
      console.log("🎭 팀 선택 상태:", {
        selectedTeam: selectedTeam,
        selectedTeamId: selectedTeam?.id,
        selectedTeamName: selectedTeam?.name,
      });

      // AbortController 생성
      const controller = new AbortController();
      setAbortController(controller);

      // AI 자동 선택이 활성화된 경우 적절한 팀 추천받기
      let finalTeamId = null;
      
      if (aiAutoEnabled) {
        finalTeamId = "AI_AUTO";
      } else if (selectedTeam && selectedTeam.id !== "AI_AUTO") {
        finalTeamId = selectedTeam.id;
      }

      if (aiAutoEnabled || finalTeamId === "AI_AUTO") {
        console.log("🤖 AI 자동 선택 활성화 - 적절한 팀 추천 요청", {
          reason: !selectedTeam ? "팀 선택 없음" : "AI_AUTO 선택됨",
          selectedTeam: selectedTeam?.name || "없음"
        });
        
        // 사용자가 팀을 선택하지 않은 경우 AI 자동 선택으로 UI 업데이트
        if (!selectedTeam) {
          const autoSelectTeam = availableTeams.find(team => team.id === "AI_AUTO");
          if (autoSelectTeam) {
            setSelectedTeam(autoSelectTeam);
            console.log("🎯 프론트엔드 UI를 AI 자동 선택으로 업데이트");
          }
        }
        
        try {
          const suggestResponse = await fetch(`${API_BASE_URL}/teams/suggest`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              query: currentQuery
            }),
          });

          const suggestData = await suggestResponse.json();
          finalTeamId = suggestData.suggested_team || null;
          console.log("🤖 AI 추천 결과:", finalTeamId);

          // UI에 추천된 팀 표시 (옵션)
          if (finalTeamId && finalTeamId !== "기본") {
            console.log(`🎯 AI가 "${finalTeamId}" 팀을 추천했습니다`);
          }
        } catch (suggestError) {
          console.error("팀 추천 API 오류:", suggestError);
          finalTeamId = null; // 오류 시 팀 선택 없음으로 처리
        }
      }

      console.log("📤 최종 요청 데이터:", {
        query: currentQuery,
        session_id: conversationId || undefined,
        message_id: assistantMessage.id,
        team_id: finalTeamId,
      });

      console.log("fetch 요청 시작!");

      const res = await fetch(`${API_BASE_URL}/query/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
        signal: controller.signal, // AbortController 시그널 추가
        body: JSON.stringify({
          query: currentQuery,
          session_id: conversationId || undefined,
          message_id: String(assistantMessage.id),
          team_id: finalTeamId, // AI 자동 선택 고려한 최종 팀 ID
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

                  // 메시지 상태에 상태 추가
                  addMessageStatus(assistantMessage.id, statusMessage);
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

                  // 메시지 상태 완료 처리
                  completeMessageState(assistantMessage.id);

                  // 메시지에 상태 저장
                  setCurrentConversation((prevConversation) => {
                    const newConversation = prevConversation.map((msg) => {
                      if (msg.id === assistantMessage.id) {
                        const messageState = getMessageState(assistantMessage.id);
                        
                        const updatedMessage = {
                          ...msg,
                          charts: finalCharts,
                          isStreaming: false,
                          fullDataDict: msg.fullDataDict || fullDataDict,
                          sectionDataDicts: sectionDataDicts,
                          // 메시지 상태 저장
                          messageState: messageState
                        };
                        
                        return updatedMessage;
                      }
                      return msg;
                    });

                    // 대화 저장
                    const conversationData = {
                      id: conversationId || Date.now().toString(),
                      title: currentQuery.slice(0, 30) + (currentQuery.length > 30 ? "..." : ""),
                      messages: newConversation,
                      lastUpdated: new Date().toISOString(),
                    };

                    const updatedConversations = conversations.filter((c) => c.id !== conversationData.id);
                    updatedConversations.unshift(conversationData);
                    saveConversations(updatedConversations.slice(0, 50));

                    localStorage.removeItem('currentStreamingConversation');

                    return newConversation;
                  });

                  setIsStreaming(false);
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
      
      // AbortError는 사용자가 의도적으로 중단한 것이므로 별도 처리
      if (error.name === 'AbortError') {
        console.log("🛑 요청이 중단되었습니다");
        setStatusMessage("생성이 중단되었습니다");
      } else {
        setStatusMessage(`오류: ${error.message}`);
      }
      
      setIsStreaming(false);
      setAbortController(null);
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
    // 중단된 메시지 처리
    if (message.wasAborted) {
      return (
        <div className="message-content">
          {message.content && (
            <SourceRenderer 
              content={message.content}
              sources={[]}
              isStreaming={false}
              dataDict={{}}
            />
          )}
          <div className="generation-stopped">
            <div className="stopped-icon"></div>
            <div className="stopped-content">
              <div className="stopped-title">생성이 중단되었습니다</div>
              <div className="stopped-subtitle">사용자 요청에 의해 응답 생성이 중단되었습니다</div>
            </div>
          </div>
        </div>
      );
    }

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

  // 컴포넌트 마운트 시 팀 목록 로드
  useEffect(() => {
    loadAvailableTeams();
  }, []);

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

        {/* 팀 선택 토글 UI */}
        {sidebarOpen && (
          <div className="team-selection-section">
            <button
              className="team-selection-header"
              onClick={() => setTeamSectionExpanded(!teamSectionExpanded)}
            >
              <div className="team-selection-title">
                <span className="team-selection-label">응답 전문 분야</span>
                {selectedTeam && (
                  <span className="selected-team-indicator">
                    {selectedTeam.name}
                  </span>
                )}
              </div>
              <svg
                className={`team-expand-icon ${teamSectionExpanded ? 'expanded' : ''}`}
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <polyline points="6,9 12,15 18,9" />
              </svg>
            </button>

            {teamSectionExpanded && (
              <div className="team-buttons-container">
                {availableTeams.map((team, index) => (
                  <button
                    key={`team-${team.id || index}`}
                    className={`team-toggle-btn ${
                      team.id === 'AI_AUTO' ? (aiAutoEnabled ? 'active' : '') : (selectedTeam?.id === team.id ? 'active' : '')
                    } ${team.id === 'AI_AUTO' ? 'ai-auto-btn' : ''}`}
                    onClick={() => {
                      if (team.id === 'AI_AUTO') {
                        // AI 자동 선택 클릭 시 토글 상태 변경
                        setAiAutoEnabled(!aiAutoEnabled);
                        setSelectedTeam(aiAutoEnabled ? null : team);
                        console.log("🤖 AI 자동 선택 토글:", !aiAutoEnabled);
                      } else {
                        // 일반 팀 선택
                        if (selectedTeam?.id === team.id) {
                          console.log("🎭 팀 선택 해제:", team.name);
                          setSelectedTeam(null);
                          setAiAutoEnabled(true); // 팀 해제 시 AI 자동 선택 활성화
                        } else {
                          console.log("🎭 팀 선택됨:", team.name, "ID:", team.id);
                          setSelectedTeam(team);
                          setAiAutoEnabled(false); // 수동 팀 선택 시 AI 자동 선택 비활성화
                        }
                      }
                    }}
                    title={team.description}
                  >
                    <div className="team-button-content">
                      <span className="team-name">{team.name}</span>
                      {((team.id === 'AI_AUTO' && aiAutoEnabled) || (team.id !== 'AI_AUTO' && selectedTeam?.id === team.id)) && (
                        <svg
                          className="team-check-icon"
                          width="16"
                          height="16"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                        >
                          <polyline points="20,6 9,17 4,12" />
                        </svg>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

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
                      {/* 어시스턴트 메시지에서 상태 표시 */}
                      {message.type === "assistant" && (() => {
                        // 현재 스트리밍 중이거나 완료된 메시지의 상태가 있는 경우 표시
                        const messageState = getMessageState(message.id);
                        const storedMessageState = message.messageState;
                        const hasState = messageState || storedMessageState;
                        
                        return hasState;
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
                                  const messageState = getMessageState(message.id) || message.messageState;
                                  const isCurrentStreaming = message.isStreaming && isStreaming;
                                  
                                  let displayTime = 0;
                                  if (messageState) {
                                    displayTime = messageState.elapsedSeconds || 0;
                                  }

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
                                const messageState = getMessageState(message.id) || message.messageState;
                                const statusHistory = messageState?.statusHistory || [];

                                return statusHistory.map((status) => (
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
                                        {formatElapsedTime(status.elapsedSeconds || 0)}
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
                              console.log('소스 패널 토글, fullDataDict:', Object.keys(fullDataDict).length, '개');
                              setFullDataDict(fullDataDict); // 현재 전역 상태 사용
                              toggleSourcesPanel(); // 항상 토글
                            }}
                          >
                            {sourcesPanelVisible ? 
                              `출처 패널 닫기 (${Object.keys(fullDataDict).length}개)` : 
                              `${Object.keys(fullDataDict).length}개 출처 보기`
                            }
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
            <div className="textarea-container">
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
              {isStreaming ? (
                <button
                  onClick={stopGeneration}
                  className="stop-button"
                  title="생성 중단"
                >
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <rect x="6" y="6" width="12" height="12" rx="2" />
                  </svg>
                </button>
              ) : (
                <button
                  onClick={handleSubmit}
                  disabled={!query.trim()}
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
              )}
            </div>

            {/* AI 자동 선택 토글 및 담당자 태그 - 입력창 아래 */}
            <div className="input-controls">
              <div className="ai-auto-toggle">
                <button
                  className={`ai-auto-btn ${aiAutoEnabled ? 'active' : ''} ${isStreaming ? 'disabled' : ''}`}
                  onClick={() => {
                    if (isStreaming) return; // 스트리밍 중에는 클릭 무시
                    setAiAutoEnabled(!aiAutoEnabled);
                    if (!aiAutoEnabled) {
                      // AI 자동 선택 활성화 시 selectedTeam을 AI_AUTO로 설정
                      const autoSelectTeam = availableTeams.find(team => team.id === "AI_AUTO");
                      if (autoSelectTeam) {
                        setSelectedTeam(autoSelectTeam);
                      }
                    }
                    console.log("🤖 AI 자동 선택 토글:", !aiAutoEnabled);
                  }}
                  disabled={isStreaming}
                  title={isStreaming ? "보고서 생성 중에는 변경할 수 없습니다" : (aiAutoEnabled ? "AI 자동 선택 비활성화" : "AI 자동 선택 활성화")}
                >
                  <span className="ai-auto-text">
                    {aiAutoEnabled ? "AI 자동" : "수동"}
                  </span>
                </button>
              </div>
              
              {/* 담당자 태그 */}
              <div className="team-tag-container">
                <button 
                  className={`team-tag-button ${isStreaming ? 'disabled' : ''}`}
                  onClick={() => {
                    if (isStreaming) return; // 스트리밍 중에는 클릭 무시
                    setTeamDropupOpen(!teamDropupOpen);
                  }}
                  disabled={isStreaming}
                  title={isStreaming ? "보고서 생성 중에는 변경할 수 없습니다" : "담당자 선택"}
                >
                  {aiAutoEnabled ? (
                    <span className="tag tag-auto">AI 자동 선택</span>
                  ) : selectedTeam && selectedTeam.id !== "AI_AUTO" ? (
                    <span className="tag tag-manual">{selectedTeam.name}</span>
                  ) : (
                    <span className="tag tag-none">담당자 미선택</span>
                  )}
                  <svg className="dropup-arrow" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="18,15 12,9 6,15" />
                  </svg>
                </button>

                {/* 드롭업 메뉴 */}
                {teamDropupOpen && !isStreaming && (
                  <div className="team-dropup-menu">
                    {availableTeams.map((team, index) => (
                      <button
                        key={team.id}
                        className={`team-dropup-item ${
                          team.id === 'AI_AUTO' ? (aiAutoEnabled ? 'active' : '') : 
                          (selectedTeam?.id === team.id ? 'active' : '')
                        }`}
                        onClick={() => {
                          if (team.id === 'AI_AUTO') {
                            setAiAutoEnabled(true);
                            setSelectedTeam(team);
                          } else {
                            setAiAutoEnabled(false);
                            setSelectedTeam(team);
                          }
                          setTeamDropupOpen(false);
                          console.log("🎭 드롭업에서 팀 선택:", team.name);
                        }}
                        style={{
                          animationDelay: `${index * 50}ms`
                        }}
                      >
                        <span className="team-dropup-name">{team.name}</span>
                        {team.description && (
                          <span className="team-dropup-desc">{team.description}</span>
                        )}
                        {((team.id === 'AI_AUTO' && aiAutoEnabled) || (team.id !== 'AI_AUTO' && selectedTeam?.id === team.id)) && (
                          <svg className="team-dropup-check" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <polyline points="20,6 9,17 4,12" />
                          </svg>
                        )}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
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
