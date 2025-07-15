import os
import uuid
import json
import re
from dotenv import load_dotenv
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


load_dotenv()

# env_checker 사용
from .env_checker import check_api_keys

check_api_keys()

# 이제 다른 모듈들 import
from .mock_databases import create_mock_databases
from .models import (
    AgentType,
    DatabaseType,
    QueryPlan,
    AgentMessage,
    MessageType,
    SearchResult,
    CriticResult,
    StreamingAgentState,
    ExecutionStrategy,
    ComplexityLevel,
    SimpleAgentMemory,  # 새로 추가
)
from .agents import (
    PlanningAgent,
    RetrieverAgentX,
    RetrieverAgentY,
    CriticAgent1,
    CriticAgent2,
    ContextIntegratorAgent,
    ReportGeneratorAgent,
    SimpleAnswererAgent,
)
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 새로운 계층 메모리 시스템 import
from .hierarchical_memory import HierarchicalMemorySystem, ConversationMemory


# Request/Response 모델들
class QueryRequest(BaseModel):
    query: str
    conversation_id: str = None
    user_id: str = "default_user"  # 사용자 ID 추가


class QueryResponse(BaseModel):
    final_answer: str
    chart_data: list = []
    conversation_id: str
    processing_time: float


class StreamChunk(BaseModel):
    type: str  # "text" | "chart" | "complete"
    content: str = ""
    chart_data: dict = None
    conversation_id: str = ""


# RAGWorkflow 전체 파이프라인
class RAGWorkflow:
    """RAG System LangGraph 워크플로우 - 4단계 복잡도 지원"""

    def __init__(self):
        print("\n>> RAG 워크플로우 초기화 시작")

        # Mock Databases 초기화
        self.graph_db, self.vector_db, self.rdb = create_mock_databases()

        # 계층적 메모리 시스템 초기화 (있으면 사용)
        try:
            self.hierarchical_memory = HierarchicalMemorySystem(
                max_short_term=100, max_long_term=1000
            )
        except:
            self.hierarchical_memory = None
            print("- 계층 메모리 시스템 사용 불가")

        # Agent들 초기화
        self.planning_agent = PlanningAgent()
        self.simple_answerer = SimpleAnswererAgent(self.vector_db)
        self.critic1 = CriticAgent1()
        self.context_integrator = ContextIntegratorAgent()
        self.critic2 = CriticAgent2()
        self.report_generator = ReportGeneratorAgent()

        # RetrieverAgentY 초기화 (ReAct 에이전트 포함)
        self.retriever_y = RetrieverAgentY(self.vector_db, self.rdb)

        self.memory = MemorySaver()

        # 워크플로우 그래프 생성
        self.workflow = self._create_workflow()
        print(">> RAG 워크플로우 초기화 완료")

    def _create_workflow(self):
        """4단계 복잡도를 지원하는 메인 워크플로우 생성"""
        self.graph = StateGraph(StreamingAgentState)

        # 노드 추가 (스트리밍 버전들 사용)
        self.graph.add_node("planning", self.planning_node)
        self.graph.add_node("simple_answer", self.simple_answer_streaming_node)
        self.graph.add_node("basic_search", self.basic_search_node)  # 새로 추가
        self.graph.add_node("parallel_retrieval", self.parallel_retrieval_node)
        self.graph.add_node("critic1", self.critic1_node)
        self.graph.add_node("context_integration", self.context_integration_node)
        self.graph.add_node("critic2", self.critic2_node)
        self.graph.add_node("report_generation", self.streaming_report_generation_node)

        # 엣지 정의 - 4단계 라우팅
        self.graph.set_entry_point("planning")

        # Planning 후 복잡도별 분기
        self.graph.add_conditional_edges(
            "planning",
            self.route_by_4level_complexity,
            {
                "simple": "simple_answer",
                "medium": "basic_search",
                "complex": "parallel_retrieval",
                "super_complex": "parallel_retrieval",  # 일단 같은 노드 사용
            },
        )

        # 각 경로별 종료 및 연결
        self.graph.add_edge("simple_answer", END)
        self.graph.add_edge("basic_search", "context_integration")  # 바로 통합으로
        self.graph.add_edge("parallel_retrieval", "critic1")

        self.graph.add_conditional_edges(
            "critic1",
            self.check_info_sufficient,
            {"sufficient": "context_integration", "insufficient": "planning"},
        )

        self.graph.add_edge("context_integration", "critic2")
        self.graph.add_conditional_edges(
            "critic2",
            self.check_context_sufficient,
            {"sufficient": "report_generation", "insufficient": "planning"},
        )
        self.graph.add_edge("report_generation", END)

        return self.graph.compile(checkpointer=self.memory)

    def route_by_4level_complexity(self, state: StreamingAgentState) -> str:
        print("\n>> route_by_4level_complexity 시작")

        if not state.query_plan:
            print("- query_plan 없음, medium으로 기본 라우팅")
            state.execution_mode = ExecutionStrategy.BASIC_SEARCH
            return "medium"

        execution_strategy = state.query_plan.execution_strategy
        print(f"- query_plan.execution_strategy: {execution_strategy}")

        # execution_mode를 여기서 설정!
        state.execution_mode = execution_strategy
        print(f"- state.execution_mode 설정: {state.execution_mode}")

        # 라우팅 결정 (Enum 비교)
        if execution_strategy == ExecutionStrategy.DIRECT_ANSWER:
            print("- 라우팅: simple")
            return "simple"
        elif execution_strategy == ExecutionStrategy.BASIC_SEARCH:
            print("- 라우팅: medium")
            return "medium"
        elif execution_strategy == ExecutionStrategy.FULL_REACT:
            print("- 라우팅: complex")
            return "complex"
        elif execution_strategy == ExecutionStrategy.MULTI_AGENT:
            print("- 라우팅: super_complex")
            return "super_complex"
        else:
            print(f"- 알 수 없는 전략 ({execution_strategy}), medium으로 기본 라우팅")
            state.execution_mode = ExecutionStrategy.BASIC_SEARCH
            return "medium"

    async def planning_node(self, state: StreamingAgentState) -> StreamingAgentState:
        print("\n>>> PLANNING 단계 시작")

        # 관련 메모리 검색 (메모리 시스템이 있으면)
        if self.hierarchical_memory:
            try:
                user_id = getattr(state, "user_id", "default_user")
                relevant_memories = self.hierarchical_memory.retrieve_relevant_memories(
                    state.original_query, user_id, top_k=3
                )

                # 메모리 컨텍스트를 planning에 활용
                memory_context = self._format_memory_context(relevant_memories)
                if memory_context:
                    print(f"- 관련 메모리 컨텍스트 활용: {len(relevant_memories)}개")
                    state.memory_context = memory_context
            except Exception as e:
                print(f"- 메모리 검색 실패: {e}")

        return await self.planning_agent.plan(state)

    async def simple_answer_streaming_node(
        self, state: StreamingAgentState
    ) -> StreamingAgentState:
        print("\n>>> STREAMING SIMPLE ANSWER 단계 시작")

        full_answer = ""
        async for chunk in self.simple_answerer.answer_streaming(state):
            full_answer += chunk

        state.final_answer = full_answer

        # 단순 답변도 메모리에 저장
        await self._save_conversation_memory(state, full_answer, importance=0.3)

        print(f"\n- 단순 답변 스트리밍 완료 (길이: {len(full_answer)}자)")
        return state

    async def basic_search_node(
        self, state: StreamingAgentState
    ) -> StreamingAgentState:
        """새로 추가: MEDIUM 복잡도용 기본 검색 노드"""
        print("\n>>> BASIC_SEARCH 단계 시작")

        try:
            # RetrieverAgentY가 basic_search 모드로 실행됨
            state = await self.retriever_y.search(state)

            # 기본 검색 완료 표시
            state.search_complete = True

            print(f"- 기본 검색 완료: {len(state.multi_source_results_stream)}개 결과")

        except Exception as e:
            print(f"- 기본 검색 오류: {e}")
            state.search_complete = False

        print(">>> BASIC_SEARCH 단계 완료")
        return state

    async def parallel_retrieval_node(
        self, state: StreamingAgentState
    ) -> StreamingAgentState:
        """수정된 병렬 검색 노드 - 복잡도별 처리"""
        print("\n>>> PARALLEL RETRIEVAL 단계 시작")
        print(f"- 실행 모드: {getattr(state, 'execution_mode', 'unknown')}")

        try:
            # Graph DB 검색 (모든 복잡도에서 실행)
            retriever_x = RetrieverAgentX(self.graph_db)
            x_task = asyncio.create_task(retriever_x.search(state))

            # RetrieverAgentY는 복잡도별로 다르게 동작
            y_task = asyncio.create_task(self.retriever_y.search(state))

            # 병렬 실행
            await asyncio.gather(x_task, y_task)
            state.search_complete = True

            # 복잡도별 추가 처리
            execution_mode = getattr(state, "execution_mode", None)
            if execution_mode and "super" in str(execution_mode).lower():
                print("- SUPER_COMPLEX: 추가 검증 단계 실행")
                # 추가 검증이나 크로스체크 로직 추가 가능

            # 검색된 정보를 지식 메모리에 저장
            await self._save_knowledge_memory(state)

        except Exception as e:
            print(f"- 병렬 검색 오류: {e}")
            state.search_complete = False

        print(">>> PARALLEL RETRIEVAL 단계 완료")
        return state

    async def critic1_node(self, state: StreamingAgentState) -> StreamingAgentState:
        print("\n>>> CRITIC_1 시작")

        # 결과 검증
        has_valid_results = (
            len(state.graph_results_stream) > 0
            or len(state.multi_source_results_stream) > 0
        )

        if not has_valid_results:
            state.info_sufficient = False
            state.critic1_result = CriticResult(
                status="insufficient",
                confidence=0.8,
                reasoning="검색 결과가 부족합니다.",
                suggestion="다시 검색해주세요.",
            )
        else:
            state = await self.critic1.evaluate(state)

        # should_terminate() 대신 직접 체크
        if not state.info_sufficient and state.current_iteration >= state.max_iterations:
            print(f"- 최대 반복 횟수({state.max_iterations}) 도달, 진행 허용")
            state.info_sufficient = True
        elif not state.info_sufficient:
            print(f"- 정보 부족, 반복 시도 ({state.current_iteration + 1}/{state.max_iterations})")
            # reset_for_new_iteration() 대신 수동 리셋
            state.current_iteration += 1
            # 필요시 다른 상태도 리셋
            state.search_complete = False

        return state


    async def context_integration_node(
        self, state: StreamingAgentState
    ) -> StreamingAgentState:
        """수정된 컨텍스트 통합 노드 - 복잡도별 처리"""
        print("\n>>> CONTEXT INTEGRATION 시작")
        execution_mode = getattr(state, "execution_mode", None)
        print(f"- 실행 모드: {execution_mode}")

        # 메모리에서 추가 컨텍스트 가져오기
        if self.hierarchical_memory:
            try:
                additional_context = (
                    self.hierarchical_memory.get_working_memory_context()
                )
                if additional_context:
                    state.additional_context = additional_context
                    print("- 작업 메모리 컨텍스트 통합")
            except Exception as e:
                print(f"- 메모리 컨텍스트 가져오기 실패: {e}")

        # 복잡도별 통합 전략
        if execution_mode and "basic" in str(execution_mode).lower():
            # 간단한 통합
            print("- 기본 검색 결과 간단 통합")
            state = await self._simple_integration(state)
        else:
            # 기존 복잡한 통합 로직
            print("- 복합 검색 결과 고급 통합")
            state = await self.context_integrator.integrate(state)

        print(">>> CONTEXT INTEGRATION 완료")
        return state

    async def critic2_node(self, state: StreamingAgentState) -> StreamingAgentState:
        print("\n>>> CRITIC_2 시작")

        state = await self.critic2.evaluate(state)

        # should_terminate() 대신 직접 체크
        if not state.context_sufficient and state.current_iteration >= state.max_iterations:
            print(f"- 최대 반복 횟수({state.max_iterations}) 도달, 진행 허용")
            state.context_sufficient = True
        elif not state.context_sufficient:
            print(f"- 컨텍스트 부족, 반복 시도 ({state.current_iteration + 1}/{state.max_iterations})")
            # reset_for_new_iteration() 대신 수동 리셋
            state.current_iteration += 1
            state.search_complete = False

        return state

    async def _simple_integration(
        self, state: StreamingAgentState
    ) -> StreamingAgentState:
        """MEDIUM 복잡도용 간단한 컨텍스트 통합"""

        all_results = state.multi_source_results_stream
        if not all_results:
            state.integrated_context = "검색된 정보가 없어 기본 답변을 제공합니다."
            return state

        # 간단한 요약 및 통합
        context_summary = ""
        for result in all_results[:5]:  # 상위 5개만 사용
            context_summary += f"- {result.content[:200]}\n"

        # 간단한 LLM 통합
        try:
            from langchain_openai import ChatOpenAI

            chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

            prompt = f"""
사용자 질문: {state.original_query}

검색 결과:
{context_summary}

위 정보를 바탕으로 사용자 질문에 대한 간결하고 명확한 답변을 작성하세요.
- 핵심 내용만 정리
- 실용적인 정보 위주
- 300자 이내로 간결하게

답변:
"""

            response = await chat.ainvoke(prompt)
            state.integrated_context = response.content

        except Exception as e:
            print(f"- 간단 통합 오류: {e}")
            state.integrated_context = f"검색된 정보: {context_summary[:500]}"

        return state

    async def critic2_node(self, state: StreamingAgentState) -> StreamingAgentState:
        state = await self.critic2.evaluate(state)

        if not state.context_sufficient and state.should_terminate():
            state.context_sufficient = True
        elif not state.context_sufficient:
            state.reset_for_new_iteration()

        return state

    async def streaming_report_generation_node(
        self, state: StreamingAgentState
    ) -> StreamingAgentState:
        """수정된 보고서 생성 노드 - 복잡도별 처리"""
        print("\n>>> STREAMING REPORT GENERATION 시작")
        execution_mode = getattr(state, "execution_mode", None)
        print(f"- 실행 모드: {execution_mode}")

        if execution_mode and "basic" in str(execution_mode).lower():
            # 간단한 답변 생성
            print("- 기본 답변 생성 모드")
            state.final_answer = state.integrated_context
        else:
            # 기존 복잡한 보고서 생성
            print("- 고급 보고서 생성 모드")
            full_answer = ""
            async for chunk in self.report_generator.generate_streaming(state):
                full_answer += chunk
            state.final_answer = full_answer

        # 최종 답변을 메모리에 저장
        complexity_level = getattr(state, "query_plan", {}).get(
            "estimated_complexity", "medium"
        )
        importance = 0.4 if "medium" in str(complexity_level).lower() else 0.8
        await self._save_conversation_memory(state, state.final_answer, importance)

        print(">>> STREAMING REPORT GENERATION 완료")
        return state

    # 조건 함수들
    def check_info_sufficient(self, state: StreamingAgentState) -> str:
        return "sufficient" if state.info_sufficient else "insufficient"

    def check_context_sufficient(self, state: StreamingAgentState) -> str:
        return "sufficient" if state.context_sufficient else "insufficient"

    # 메모리 관련 헬퍼 메서드들
    async def _save_conversation_memory(
        self, state: StreamingAgentState, answer: str, importance: float = 0.5
    ):
        """지능적 대화 메모리 저장"""
        if not self.hierarchical_memory:
            return

        try:
            user_id = getattr(state, "user_id", "default_user")
            context_used = []

            # 사용된 컨텍스트 정보 수집
            if hasattr(state, "graph_results_stream"):
                context_used.extend([str(r) for r in state.graph_results_stream])
            if hasattr(state, "multi_source_results_stream"):
                context_used.extend([str(r) for r in state.multi_source_results_stream])

            # 지능적 메모리 저장 시도, 실패시 기본 방식
            try:
                await self.hierarchical_memory.add_conversation_memory_smart(
                    user_id=user_id,
                    query=state.original_query,
                    response=answer,
                    context_used=context_used,
                )
            except Exception as e:
                print(f"- 지능적 메모리 저장 실패: {e}")
                # 기본 방식으로 폴백
                self.hierarchical_memory.add_conversation_memory(
                    user_id=user_id,
                    query=state.original_query,
                    response=answer,
                    context_used=context_used,
                    importance=importance,
                )
        except Exception as e:
            print(f"- 메모리 저장 실패: {e}")

    async def _save_knowledge_memory(self, state: StreamingAgentState):
        """검색된 정보를 지식 메모리에 저장"""
        if not self.hierarchical_memory:
            return

        try:
            if hasattr(state, "graph_results_stream") and state.graph_results_stream:
                for result in state.graph_results_stream:
                    # 토픽 추출 (간단한 키워드 기반)
                    topic = self._extract_topic_from_query(state.original_query)

                    self.hierarchical_memory.add_knowledge_memory(
                        topic=topic,
                        key_facts=[str(result)],
                        sources=["graph_db"],
                        importance=0.6,
                    )
        except Exception as e:
            print(f"- 지식 메모리 저장 실패: {e}")

    def _extract_topic_from_query(self, query: str) -> str:
        """쿼리에서 주요 토픽 추출"""
        words = query.split()
        food_keywords = ["시세", "가격", "농산물", "수산물", "축산물", "식재료", "품목"]

        for word in words:
            if any(keyword in word for keyword in food_keywords):
                return word

        return words[0] if words else "기타"

    def _format_memory_context(self, memories: list) -> str:
        """메모리를 컨텍스트 문자열로 포맷팅"""
        if not memories:
            return ""

        context_parts = ["관련 이전 정보:"]
        for memory in memories:
            if hasattr(memory, "query") and hasattr(memory, "response"):
                context_parts.append(f"- 이전 질문: {memory.query}")
                context_parts.append(f"  답변 요약: {memory.response[:100]}...")
            else:
                context_parts.append(f"- {str(memory)[:100]}...")

        return "\n".join(context_parts)

    def _update_user_interaction_pattern(self, user_id: str, query: str):
        """사용자 상호작용 패턴 업데이트"""
        if not self.hierarchical_memory:
            return

        try:
            query_length = len(query)
            query_words = len(query.split())

            food_keywords = [
                "시세",
                "가격",
                "농산물",
                "수산물",
                "축산물",
                "식재료",
                "품목",
                "트렌드",
            ]
            mentioned_keywords = [kw for kw in food_keywords if kw in query]

            preferences = {
                "average_query_length": query_length,
                "average_query_words": query_words,
                "preferred_topics": mentioned_keywords,
                "last_interaction": datetime.now().isoformat(),
            }

            expertise_areas = []
            if "시세" in query or "가격" in query:
                expertise_areas.append("시장분석")
            if "트렌드" in query:
                expertise_areas.append("동향분석")
            if any(kw in query for kw in ["농산물", "수산물", "축산물"]):
                expertise_areas.append("품목전문")

            self.hierarchical_memory.update_user_profile(
                user_id=user_id,
                preferences=preferences,
                expertise_areas=expertise_areas,
            )
        except Exception as e:
            print(f"- 사용자 패턴 업데이트 실패: {e}")

    def get_memory_summary(self, user_id: str = None) -> Dict[str, Any]:
        """메모리 요약 정보 반환"""
        if not self.hierarchical_memory:
            return {"status": "메모리 시스템 없음"}

        try:
            stats = self.hierarchical_memory.get_memory_stats()

            if user_id:
                user_profile = self.hierarchical_memory._get_user_profile(user_id)
                if user_profile:
                    stats["user_profile"] = {
                        "expertise_areas": user_profile.expertise_areas,
                        "interaction_count": user_profile.access_count,
                        "last_seen": (
                            user_profile.last_accessed.isoformat()
                            if user_profile.last_accessed
                            else None
                        ),
                    }

            return stats
        except Exception as e:
            return {"error": f"메모리 통계 조회 실패: {e}"}

    def save_memory_checkpoint(self, filepath: str = "memory_checkpoint.json"):
        """메모리 체크포인트 저장"""
        if self.hierarchical_memory:
            try:
                self.hierarchical_memory.save_memory_state(filepath)
                print(f"메모리 체크포인트 저장 완료: {filepath}")
            except Exception as e:
                print(f"메모리 저장 실패: {e}")
        else:
            print("메모리 시스템이 없어서 저장할 수 없습니다.")

    def load_memory_checkpoint(self, filepath: str = "memory_checkpoint.json"):
        """메모리 체크포인트 로드"""
        if self.hierarchical_memory:
            try:
                self.hierarchical_memory.load_memory_state(filepath)
                print(f"메모리 체크포인트 로드 완료: {filepath}")
            except Exception as e:
                print(f"메모리 로드 실패: {e}")
        else:
            print("메모리 시스템이 없어서 로드할 수 없습니다.")

    async def stream_api(
        self, query: str, conversation_id: str = None, user_id: str = "default_user"
    ):
        """API용 멀티 이벤트 스트림 실행 메서드 - execution_mode 수정"""
        if not conversation_id:
            conversation_id = f"api-streaming-{uuid.uuid4()}"

        config = {"configurable": {"thread_id": conversation_id}}

        async def event_generator():
            try:
                print(f"\n>> 실시간 스트리밍 워크플로우 시작 (사용자: {user_id})")
                yield f"data: {json.dumps({'type': 'status', 'content': 'AI가 분석을 시작합니다...'})}\n\n"
                await asyncio.sleep(0)

                # 사용자 프로필 업데이트
                self._update_user_interaction_pattern(user_id, query)

                # execution_mode를 None으로 초기화하여 planning에서 결정하도록 함
                new_input = {
                    "original_query": query,
                    "user_id": user_id,
                    # query_plan과 execution_mode는 Optional이므로 생략
                    "final_answer": "",
                    "info_sufficient": False,
                    "context_sufficient": False,
                    "search_complete": False,
                    "graph_results_stream": [],
                    "multi_source_results_stream": [],
                    "integrated_context": "",
                    "current_iteration": 0,
                    "max_iterations": 3,
                    "memory_context": "",
                    "additional_context": "",
                }

                last_state = None
                is_simple_path = False
                is_medium_path = False

                # 4단계 복잡도에 맞춘 상태 메시지
                status_messages = {
                    "planning": "쿼리 분석 및 계획 수립 중...",
                    "simple_answer": "간단한 답변 생성 중...",
                    "basic_search": "기본 정보 검색 중...",
                    "parallel_retrieval": "관련 정보 수집 중...",
                    "critic1": "정보 충분성 검토 중...",
                    "context_integration": "수집된 정보 통합 중...",
                    "critic2": "최종 품질 검토 중...",
                    "report_generation": "전문 보고서 작성을 시작합니다...",
                }

                async for step in self.workflow.astream(new_input, config=config):
                    node_name = list(step.keys())[0]
                    last_state = step[node_name]

                    print(f"\n>> 노드 처리됨: {node_name}")

                    if node_name in status_messages:
                        yield f"data: {json.dumps({'type': 'status', 'content': status_messages[node_name]})}\n\n"
                        await asyncio.sleep(0)

                    # 경로 판별
                    if node_name == "simple_answer":
                        is_simple_path = True
                        break
                    elif node_name == "basic_search":
                        is_medium_path = True
                        continue
                    elif node_name == "context_integration" and is_medium_path:
                        break
                    elif node_name == "critic2":
                        break

                if not last_state:
                    raise ValueError("워크플로우가 어떤 결과도 반환하지 않았습니다.")

                # 경로별 처리
                if is_simple_path:
                    print("\n>> 단순 경로 처리 (SIMPLE)")
                    simple_answer = last_state.get(
                        "final_answer", "답변을 찾지 못했습니다."
                    )

                    for i in range(0, len(simple_answer), 7):
                        chunk = simple_answer[i : i + 7]
                        yield f"data: {json.dumps({'type': 'text_chunk', 'content': chunk})}\n\n"
                        await asyncio.sleep(0.02)

                    yield f"data: {json.dumps({'type': 'complete', 'final_answer': simple_answer, 'conversation_id': conversation_id})}\n\n"

                elif is_medium_path:
                    print("\n>> 중간 복잡도 경로 처리 (MEDIUM)")
                    medium_answer = last_state.get(
                        "integrated_context", "답변을 찾지 못했습니다."
                    )

                    for i in range(0, len(medium_answer), 10):
                        chunk = medium_answer[i : i + 10]
                        yield f"data: {json.dumps({'type': 'text_chunk', 'content': chunk})}\n\n"
                        await asyncio.sleep(0.03)

                    yield f"data: {json.dumps({'type': 'complete', 'final_answer': medium_answer, 'conversation_id': conversation_id})}\n\n"

                else:
                    print("\n>> 복잡한 보고서 경로 처리 (COMPLEX/SUPER_COMPLEX)")

                    state_obj = StreamingAgentState(
                        original_query=last_state.get("original_query", query),
                        user_id=last_state.get("user_id", user_id),
                        query_plan=last_state.get("query_plan"),
                        final_answer=last_state.get("final_answer", ""),
                        info_sufficient=last_state.get("info_sufficient", False),
                        context_sufficient=last_state.get("context_sufficient", False),
                        search_complete=last_state.get("search_complete", False),
                        graph_results_stream=last_state.get("graph_results_stream", []),
                        multi_source_results_stream=last_state.get(
                            "multi_source_results_stream", []
                        ),
                        integrated_context=last_state.get("integrated_context", ""),
                        current_iteration=last_state.get("current_iteration", 0),
                        max_iterations=last_state.get("max_iterations", 1),
                        memory_context=last_state.get("memory_context", ""),
                        additional_context=last_state.get("additional_context", ""),
                        execution_mode=last_state.get(
                            "execution_mode", "multi_agent"
                        ),  # 기본값 설정
                    )

                    full_report_text = ""
                    text_buffer = ""
                    chart_counter = 0
                    sent_chart_ids = set()

                    def generate_chart_id(chart_data):
                        data_sample = chart_data.get("data", [])
                        if isinstance(data_sample, list) and len(data_sample) > 0:
                            sample_data = str(data_sample[:2])
                        else:
                            sample_data = str(data_sample)

                        chart_key = json.dumps(
                            {
                                "type": chart_data.get("type", ""),
                                "title": chart_data.get("title", ""),
                                "data_sample": sample_data,
                            },
                            sort_keys=True,
                        )
                        return hash(chart_key)

                    def maybe_unfinished_chart(text):
                        chart_prefixes = [
                            "{",
                            "{{",
                            "{{C",
                            "{{CH",
                            "{{CHA",
                            "{{CHAR",
                            "{{CHART",
                            "{{CHART_",
                            "{{CHART_S",
                            "{{CHART_ST",
                            "{{CHART_STA",
                            "{{CHART_STAR",
                            "{{CHART_START",
                            "{{CHART_START}",
                            "{{CHART_START}}",
                        ]
                        return any(
                            text.rstrip().endswith(prefix) for prefix in chart_prefixes
                        )

                    # 리포트 생성기가 있으면 사용, 없으면 기본 답변
                    if hasattr(self, "report_generator"):
                        async for chunk in self.report_generator.generate_streaming(
                            state_obj
                        ):
                            full_report_text += chunk
                            text_buffer += chunk

                            # 완전한 차트 마커 패턴 찾기
                            while (
                                "{{CHART_START}}" in text_buffer
                                and "{{CHART_END}}" in text_buffer
                            ):
                                match = re.search(
                                    r"\{\{CHART_START\}\}(.*?)\{\{CHART_END\}\}",
                                    text_buffer,
                                    re.DOTALL,
                                )
                                if not match:
                                    break

                                start, end = match.span()
                                before = text_buffer[:start]
                                after = text_buffer[end:]
                                chart_json_str = match.group(1).strip()

                                # 마커 앞의 텍스트 먼저 전송
                                if before.strip():
                                    yield f"data: {json.dumps({'type': 'text_chunk', 'content': before})}\n\n"
                                    await asyncio.sleep(0.01)

                                # 차트 JSON 처리 및 중복 체크
                                try:
                                    chart_data = json.loads(chart_json_str)
                                    chart_id = generate_chart_id(chart_data)

                                    # 중복 차트 체크
                                    if chart_id not in sent_chart_ids:
                                        sent_chart_ids.add(chart_id)
                                        yield f"data: {json.dumps({'type': 'chart', 'chart_data': chart_data})}\n\n"
                                        chart_counter += 1
                                        print(
                                            f"- 차트 #{chart_counter} 전송 완료 (ID: {chart_id})"
                                        )
                                    else:
                                        print(f"- 중복 차트 무시 (ID: {chart_id})")

                                    await asyncio.sleep(0.01)
                                except json.JSONDecodeError as e:
                                    print(f"JSON 파싱 실패: {e}")
                                    print(f"문제가 된 JSON: {chart_json_str}")

                                # 처리된 부분 제거
                                text_buffer = after

                            # 차트 마커 없는 일반 텍스트 전송
                            if (
                                not maybe_unfinished_chart(text_buffer)
                                and "{{CHART_START}}" not in text_buffer
                                and "{{CHART_END}}" not in text_buffer
                                and len(text_buffer.strip()) > 10
                            ):
                                yield f"data: {json.dumps({'type': 'text_chunk', 'content': text_buffer})}\n\n"
                                await asyncio.sleep(0.01)
                                text_buffer = ""

                        # 남은 텍스트 전송
                        if text_buffer.strip():
                            yield f"data: {json.dumps({'type': 'text_chunk', 'content': text_buffer})}\n\n"
                    else:
                        # report_generator가 없으면 기본 답변
                        fallback_answer = last_state.get(
                            "integrated_context", "답변을 생성할 수 없습니다."
                        )
                        yield f"data: {json.dumps({'type': 'text_chunk', 'content': fallback_answer})}\n\n"

                    # 완료 이벤트
                    yield f"data: {json.dumps({'type': 'complete', 'conversation_id': conversation_id, 'total_charts': chart_counter})}\n\n"

                # 메모리 통계 출력
                if self.hierarchical_memory:
                    try:
                        memory_stats = self.hierarchical_memory.get_memory_stats()
                        print(f"\n>> 메모리 통계: {memory_stats}")
                    except Exception as e:
                        print(f"- 메모리 통계 조회 실패: {e}")

            except Exception as e:
                import traceback

                print(f"\n>> 스트리밍 오류 발생: {str(e)}")
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )


app = FastAPI(
    title="RAG System API",
    description="Multi-Agent RAG System with Hierarchical Memory",
    version="2.0.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_workflow = RAGWorkflow()

@app.post("/api/query/stream")
async def stream_query(request: QueryRequest):
    response = await rag_workflow.stream_api(
        query=request.query,
        conversation_id=request.conversation_id,
        user_id=request.user_id,
    )
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no"
    return response

@app.get("/api/memory/stats")
async def get_memory_stats(user_id: str = None):
    return rag_workflow.get_memory_summary(user_id=user_id)

@app.post("/api/memory/save")
async def save_memory_checkpoint():
    rag_workflow.save_memory_checkpoint()
    return {"status": "메모리 체크포인트가 저장되었습니다."}

@app.post("/api/memory/load")
async def load_memory_checkpoint():
    rag_workflow.load_memory_checkpoint()
    return {"status": "메모리 체크포인트가 로드되었습니다."}

@app.get("/api/health")
async def health_check():
    memory_stats = rag_workflow.get_memory_summary()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "memory_info": memory_stats,
    }
