# 표준 라이브러리
import asyncio
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Set

# 서드파티 라이브러리
import requests

# LangChain 관련
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 로컬 모듈
from .models import (
    AgentMessage,
    AgentType,
    ComplexityLevel,
    CriticResult,
    DatabaseType,
    ExecutionStrategy,
    MessageType,
    QueryPlan,
    SearchResult,
    StreamingAgentState,
)
from .search_tools import (
    debug_web_search,
    mock_graph_db_search,
    mock_rdb_search,
    mock_vector_search,
)
from .utils import create_agent_message

class PlanningAgent:
    """
    4단계 복잡도 분류를 지원하는 향상된 계획 수립 에이전트
    - SIMPLE: 직접 답변 가능
    - MEDIUM: 기본 검색 + 간단 분석
    - COMPLEX: 풀 ReAct 에이전트 활용
    - SUPER_COMPLEX: 다중 에이전트 협업
    """

    def __init__(self):
        self.chat = ChatOpenAI(model="gpt-4o", temperature=0)
        self.agent_type = AgentType.PLANNING

    async def plan(self, state: StreamingAgentState) -> StreamingAgentState:
        """질문을 4단계로 분석하고 최적의 실행 계획 수립"""
        print(">> PLANNING 단계 시작 (4단계 복잡도 분류)")
        query = state.original_query
        print(f"- 원본 쿼리: {query}")

        # 이전 단계 피드백 수집
        feedback_context = self._collect_feedback(state)
        if feedback_context:
            print(f"- 피드백 반영: {feedback_context}")

        # 4단계 복잡도 분석
        complexity_analysis = await self._analyze_query_complexity_4levels(
            query, feedback_context
        )
        print(f"- 복잡도 분석 결과: {complexity_analysis}")

        # 복잡도별 실행 계획 수립
        execution_plan = await self._create_execution_plan_by_complexity(
            query, complexity_analysis, feedback_context
        )

        # 복잡도와 전략 값을 소문자로 변환
        complexity_mapping = {
            "SIMPLE": "simple",
            "MEDIUM": "medium",
            "COMPLEX": "complex",
            "SUPER_COMPLEX": "super_complex",
            "simple": "simple",
            "medium": "medium",
            "complex": "complex",
            "super_complex": "super_complex",
        }

        strategy_mapping = {
            "direct_answer": "direct_answer",
            "basic_search": "basic_search",
            "full_react": "full_react",
            "multi_agent": "multi_agent",
        }

        # 원본 값들
        raw_complexity = complexity_analysis.get("complexity_level", "MEDIUM")
        raw_strategy = complexity_analysis.get("execution_strategy", "basic_search")

        # 매핑된 값들 (소문자)
        mapped_complexity = complexity_mapping.get(raw_complexity, "medium")
        mapped_strategy = strategy_mapping.get(raw_strategy, "basic_search")

        print(f"- 복잡도 매핑: {raw_complexity} → {mapped_complexity}")
        print(f"- 전략 매핑: {raw_strategy} → {mapped_strategy}")

        state.query_plan = QueryPlan(
            original_query=query,
            sub_queries=[execution_plan],
            estimated_complexity=mapped_complexity,  # 매핑된 소문자 값 사용
            execution_strategy=mapped_strategy,  # 매핑된 소문자 값 사용
            resource_requirements=complexity_analysis.get("resource_requirements", {}),
        )

        state.planning_complete = True
        print(f">> PLANNING 단계 완료 - 복잡도: {mapped_complexity}")
        return state

    def _collect_feedback(self, state: StreamingAgentState) -> Optional[str]:
        """이전 단계의 피드백을 수집"""
        if state.critic2_result and state.critic2_result.status == "insufficient":
            return f"최종 검수 피드백: {state.critic2_result.suggestion}"
        elif state.critic1_result and state.critic1_result.status == "insufficient":
            return f"초기 수집 피드백: {state.critic1_result.suggestion}"
        return None

    async def _analyze_query_complexity_4levels(
        self, query: str, feedback: Optional[str] = None
    ) -> Dict:
        """질문을 4단계 복잡도로 분석"""

        feedback_section = ""
        if feedback:
            feedback_section = f"""
## 이전 시도 피드백
{feedback}

위 피드백을 고려하여 분석해주세요.
"""

        prompt = f"""당신은 세계 최고 수준의 AI 시스템 아키텍트입니다. 사용자의 질문을 4단계 복잡도로 정확히 분류해야 합니다.

{feedback_section}

## 분석 대상 질문
"{query}"

## 4단계 복잡도 분류 기준

### SIMPLE (직접 답변)
- 단일 정보 요청, 기본 정의, 간단한 계산
- 추가 검색이나 분석 불필요
- 1-2개 문장으로 답변 가능
- 예: "아마란스가 뭐야?", "칼로리 알려줘"

### MEDIUM (기본 검색 + 간단 분석)
- 최신 정보나 간단한 비교가 필요
- 1-2개 소스에서 정보 수집 후 종합
- 단순한 분석이나 요약 필요
- 예: "오늘 채소 시세는?", "A와 B 차이점은?"

### COMPLEX (풀 ReAct 에이전트)
- 다단계 추론과 여러 소스 종합 필요
- 전략적 사고와 맥락적 분석 필요
- 복잡한 의사결정 지원
- 예: "마케팅 전략 수립해줘", "시장 분석 보고서"

### SUPER_COMPLEX (다중 에이전트 협업)
- 매우 복잡한 다영역 분석
- 장기적 계획이나 종합적 전략 필요
- 여러 전문가 관점 종합 필요
- 예: "글로벌 진출 전략", "5년 사업 계획"

## 실행 전략 매핑

### SIMPLE → "direct_answer"
- SimpleAnswererAgent만 사용
- 즉시 답변 생성

### MEDIUM → "basic_search"
- 기본 Vector/RDB 검색
- 간단한 LLM 분석
- ReAct 없이 직접 처리

### COMPLEX → "full_react"
- 완전한 ReAct 에이전트 활용
- 다단계 검색 및 추론
- 상세한 분석 및 종합

### SUPER_COMPLEX → "multi_agent"
- 여러 전문 에이전트 협업
- 단계별 검증 및 피드백
- 종합적 보고서 생성

## 출력 형식
다음 JSON 형식으로만 응답하세요:

```json
{{
  "complexity_level": "SIMPLE|MEDIUM|COMPLEX|SUPER_COMPLEX",
  "execution_strategy": "direct_answer|basic_search|full_react|multi_agent",
  "reasoning": "판단 근거를 2-3문장으로 설명",
  "resource_requirements": {{
    "search_needed": true|false,
    "react_needed": true|false,
    "multi_agent_needed": true|false,
    "estimated_time": "fast|medium|slow|very_slow"
  }},
  "expected_output_type": "simple_text|analysis|report|comprehensive_strategy"
}}
```

정확한 JSON 형식을 준수하고, 질문의 진짜 복잡도를 신중히 판단하세요."""

        try:
            response = await self.chat.ainvoke(prompt)
            # JSON 파싱 시도
            import json

            # 코드 블록이 있다면 제거
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)
            return result

        except Exception as e:
            print(f"복잡도 분석 파싱 오류: {e}")
            return {
                "complexity_level": "MEDIUM",
                "execution_strategy": "basic_search",
                "reasoning": "분석 중 오류 발생으로 기본값 적용",
                "resource_requirements": {
                    "search_needed": True,
                    "react_needed": False,
                    "multi_agent_needed": False,
                    "estimated_time": "medium",
                },
                "expected_output_type": "analysis",
            }

    async def _create_execution_plan_by_complexity(
        self, query: str, complexity_analysis: Dict, feedback: Optional[str] = None
    ) -> str:
        """복잡도별 맞춤 실행 계획 생성"""

        complexity_level = complexity_analysis["complexity_level"]
        execution_strategy = complexity_analysis["execution_strategy"]

        if complexity_level == "SIMPLE":
            return f"직접 답변 생성: {query}"

        elif complexity_level == "MEDIUM":
            return f"기본 검색 후 분석: {query} - Vector DB 및 RDB 검색 활용"

        elif complexity_level == "COMPLEX":
            return await self._create_complex_execution_plan(
                query, complexity_analysis, feedback
            )

        elif complexity_level == "SUPER_COMPLEX":
            return await self._create_super_complex_execution_plan(
                query, complexity_analysis, feedback
            )

        else:
            return f"기본 실행 계획: {query}"

    async def _create_complex_execution_plan(
        self, query: str, analysis: Dict, feedback: Optional[str] = None
    ) -> str:
        """COMPLEX 레벨 실행 계획 - 기존 ToT 방식 활용"""

        feedback_section = ""
        if feedback:
            feedback_section = f"""
## 중요: 이전 시도 피드백
{feedback}

위 피드백을 반드시 해결하는 새로운 접근법을 포함해야 합니다.
"""

        prompt = f"""당신은 전략 컨설턴트입니다. 복합적 분석이 필요한 질문에 대한 체계적 실행 계획을 수립하세요.

{feedback_section}

## 클라이언트 요청
"{query}"

## 요청 분석 결과
- 복잡도: {analysis.get('complexity_level', 'COMPLEX')}
- 예상 결과물: {analysis.get('expected_output_type', 'analysis')}
- 판단 근거: {analysis.get('reasoning', '')}

## 실행 계획 수립 지침

다음 단계로 체계적인 실행 계획을 작성하세요:

1. **정보 수집 전략**: 어떤 정보를 어떤 방식으로 수집할지
2. **분석 접근법**: 수집된 정보를 어떻게 분석하고 종합할지
3. **결과물 구성**: 최종 답변을 어떤 형태로 제공할지

## 출력 형식
구체적인 실행 지시문 형태로 작성 (200-300자):

**실행 계획:**
[ReAct 에이전트가 그대로 실행할 수 있을 정도로 구체적이고 명확한 계획]
"""

        try:
            response = await self.chat.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"복합 계획 생성 오류: {e}")
            return f"'{query}' 요청에 대한 체계적 분석과 전략적 솔루션을 제공합니다."

    async def _create_super_complex_execution_plan(
        self, query: str, analysis: Dict, feedback: Optional[str] = None
    ) -> str:
        """SUPER_COMPLEX 레벨 실행 계획 - 다중 에이전트 협업"""

        prompt = f"""당신은 McKinsey 시니어 파트너입니다. 매우 복잡한 전략적 과제에 대한 다단계 실행 계획을 수립하세요.

## 전략적 과제
"{query}"

## 다중 에이전트 협업 계획

다음 관점에서 종합적 실행 계획을 수립하세요:

1. **1단계 - 현황 분석**: 시장/상황 분석 전문가 관점
2. **2단계 - 전략 수립**: 전략 기획 전문가 관점
3. **3단계 - 실행 방안**: 실행 전문가 관점
4. **4단계 - 리스크 관리**: 위험 관리 전문가 관점
5. **5단계 - 종합 검증**: 통합 검증 및 최종 제안

각 단계별로 어떤 정보를 수집하고 어떻게 분석할지 구체적으로 명시하세요.

**다단계 실행 계획 (300-400자):**
"""

        try:
            response = await self.chat.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"초복합 계획 생성 오류: {e}")
            return f"'{query}' 요청에 대한 다단계 협업 분석과 종합적 전략을 제공합니다."



class RetrieverAgent:
    """통합 검색 에이전트 - 복잡도별 차등 + 병렬 처리"""

    def __init__(self, vector_db=None, rdb=None, graph_db=None):
        self.vector_db = vector_db
        self.rdb = rdb
        self.graph_db = graph_db

        # 사용 가능한 도구들
        self.available_tools = [
            debug_web_search,
            mock_vector_search,
            mock_rdb_search,
            mock_graph_db_search,
        ]

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.chat = ChatOpenAI(model="gpt-3.5-turbo")

        # 날짜 정보
        self.current_date = datetime.now()
        self.current_date_str = self.current_date.strftime("%Y년 %m월 %d일")
        self.current_year = self.current_date.year

        # ReAct 에이전트 (복잡한 쿼리용)
        self.react_agent_executor = self._create_react_agent()

        # 병렬 처리용 스레드 풀
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    async def search(self, state: StreamingAgentState) -> StreamingAgentState:
        """복잡도별 차등 + 병렬 검색 실행"""
        print(">> 통합 RETRIEVER 시작")

        if not state.query_plan or not state.query_plan.sub_queries:
            print("- 처리할 쿼리가 없어 RETRIEVER를 종료합니다.")
            return state

        # 복잡도 및 실행 전략 결정
        complexity_level = state.get_complexity_level()
        execution_strategy = self._determine_execution_strategy(state, complexity_level)

        original_query = state.query_plan.sub_queries[0]

        print(f"- 복잡도: {complexity_level}")
        print(f"- 실행 전략: {execution_strategy}")
        print(f"- 원본 쿼리: {original_query}")

        # 실행 전략에 따른 병렬 검색
        if execution_strategy == ExecutionStrategy.DIRECT_ANSWER:
            print("- SIMPLE: 검색 생략")
            return state

        elif execution_strategy == ExecutionStrategy.BASIC_SEARCH:
            print("- BASIC: 기본 병렬 검색")
            return await self._execute_basic_parallel_search(state, original_query)

        elif execution_strategy == ExecutionStrategy.FULL_REACT:
            print("- COMPLEX: 풀 병렬 검색 + ReAct")
            return await self._execute_full_parallel_search(state, original_query)

        elif execution_strategy == ExecutionStrategy.MULTI_AGENT:
            print("- SUPER_COMPLEX: 다단계 병렬 검색")
            return await self._execute_multi_stage_parallel_search(state, original_query)

        else:
            return await self._execute_basic_parallel_search(state, original_query)

    async def _execute_basic_parallel_search(
        self, state: StreamingAgentState, query: str
    ) -> StreamingAgentState:
        """기본 병렬 검색 (BASIC 복잡도)"""
        print("\n>> 기본 병렬 검색 실행")

        # 병렬로 실행할 검색 작업들
        search_tasks = []

        # 1. Vector DB 검색
        if self.vector_db:
            search_tasks.append(self._async_vector_search(query))

        # 2. Graph DB 검색
        if self.graph_db:
            search_tasks.append(self._async_graph_search(query))

        # 3. 간단한 웹 검색
        search_tasks.append(self._async_web_search(query))

        try:
            # 모든 검색을 병렬로 실행
            start_time = time.time()
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            execution_time = time.time() - start_time

            print(f"- 병렬 검색 완료: {execution_time:.2f}초")

            # 결과 처리
            total_results = 0
            for result_group in search_results:
                if isinstance(result_group, Exception):
                    print(f"- 검색 오류: {result_group}")
                    continue

                if isinstance(result_group, list):
                    for result in result_group:
                        state.add_multi_source_result(result)
                        total_results += 1

            # 간단한 LLM 분석 (결과가 있을 때만)
            if total_results > 0:
                analysis_result = await self._simple_llm_analysis(
                    query, state.multi_source_results_stream
                )
                if analysis_result:
                    state.add_multi_source_result(analysis_result)
                    total_results += 1

            state.add_step_result("basic_parallel_search", {
                "execution_time": execution_time,
                "total_results": total_results,
                "search_types": len(search_tasks)
            })

            print(f"- 총 {total_results}개 결과 추가")

        except Exception as e:
            print(f"- 기본 병렬 검색 실패: {e}")
            fallback_result = self._create_fallback_result(query, "basic_parallel_error")
            state.add_multi_source_result(fallback_result)

        return state

    async def _execute_full_parallel_search(
        self, state: StreamingAgentState, query: str
    ) -> StreamingAgentState:
        """풀 병렬 검색 (COMPLEX 복잡도)"""
        print("\n>> 풀 병렬 검색 + ReAct 실행")

        # 1단계: 모든 DB 병렬 검색
        db_search_tasks = [
            self._async_vector_search(query),
            self._async_graph_search(query),
            self._async_rdb_search(query),
            self._async_web_search(query)
        ]

        # 2단계: ReAct 에이전트 (병렬 실행)
        react_task = self._async_react_search(query)

        try:
            start_time = time.time()

            # DB 검색들과 ReAct를 병렬로 실행
            all_tasks = db_search_tasks + [react_task]
            results = await asyncio.gather(*all_tasks, return_exceptions=True)

            execution_time = time.time() - start_time
            print(f"- 풀 병렬 검색 완료: {execution_time:.2f}초")

            # DB 검색 결과 처리
            total_results = 0
            for i, result_group in enumerate(results[:-1]):  # ReAct 제외
                if isinstance(result_group, Exception):
                    print(f"- DB 검색 {i} 오류: {result_group}")
                    continue

                if isinstance(result_group, list):
                    for result in result_group:
                        state.add_multi_source_result(result)
                        total_results += 1

            # ReAct 결과 처리
            react_result = results[-1]
            if not isinstance(react_result, Exception) and react_result:
                state.add_multi_source_result(react_result)
                total_results += 1

            state.add_step_result("full_parallel_search", {
                "execution_time": execution_time,
                "total_results": total_results,
                "db_searches": len(db_search_tasks),
                "react_included": True
            })

            print(f"- 총 {total_results}개 결과 추가")

        except Exception as e:
            print(f"- 풀 병렬 검색 실패: {e}")
            fallback_result = self._create_fallback_result(query, "full_parallel_error")
            state.add_multi_source_result(fallback_result)

        return state

    async def _execute_multi_stage_parallel_search(
        self, state: StreamingAgentState, query: str
    ) -> StreamingAgentState:
        """다단계 병렬 검색 (SUPER_COMPLEX 복잡도)"""
        print("\n>> 다단계 병렬 검색 실행")

        try:
            # 1단계: 초기 정보 수집 (병렬)
            print("- 1단계: 초기 정보 수집")
            await self._execute_full_parallel_search(state, query)

            # 2단계: 키워드 확장 및 심화 검색 (병렬)
            print("- 2단계: 키워드 확장 검색")
            expanded_keywords = await self._generate_expanded_keywords(query)

            expanded_search_tasks = []
            for keyword in expanded_keywords[:3]:  # 상위 3개만
                expanded_search_tasks.extend([
                    self._async_vector_search(keyword),
                    self._async_graph_search(keyword)
                ])

            if expanded_search_tasks:
                expanded_results = await asyncio.gather(
                    *expanded_search_tasks, return_exceptions=True
                )

                for result_group in expanded_results:
                    if isinstance(result_group, list):
                        for result in result_group:
                            state.add_multi_source_result(result)

            # 3단계: 전략적 종합 분석
            print("- 3단계: 전략적 종합")
            synthesis_result = await self._strategic_synthesis(
                query, state.multi_source_results_stream
            )
            if synthesis_result:
                state.add_multi_source_result(synthesis_result)

            # 4단계: 최종 검증
            print("- 4단계: 최종 검증")
            validation_result = await self._final_validation(
                query, state.multi_source_results_stream
            )
            if validation_result:
                state.add_multi_source_result(validation_result)

            state.add_step_result("multi_stage_parallel_search", {
                "stages_completed": 4,
                "expanded_keywords": len(expanded_keywords),
                "total_results": len(state.multi_source_results_stream)
            })

        except Exception as e:
            print(f"- 다단계 병렬 검색 실패: {e}")
            fallback_result = self._create_fallback_result(query, "multi_stage_error")
            state.add_multi_source_result(fallback_result)

        return state

    # ========== 개별 검색 메서드들 (비동기) ==========

    async def _async_vector_search(self, query: str) -> List[SearchResult]:
        """비동기 Vector DB 검색"""
        try:
            print(f"  └ Vector DB 검색: {query[:30]}...")

            # 실제로는 비동기이지만 mock은 동기라서 스레드 풀 사용
            loop = asyncio.get_event_loop()
            vector_results = await loop.run_in_executor(
                self.thread_pool,
                lambda: mock_vector_search.invoke({"query": query})
            )

            results = []
            if isinstance(vector_results, dict) and "results" in vector_results:
                for i, doc in enumerate(vector_results["results"][:3]):
                    result = SearchResult(
                        source="vector_db",
                        content=doc.get("content", ""),
                        relevance_score=doc.get("similarity_score", 0.7),
                        metadata={"search_type": "vector", "rank": i + 1},
                        search_query=query,
                    )
                    results.append(result)

            print(f"    ✓ Vector DB: {len(results)}개 결과")
            return results

        except Exception as e:
            print(f"    ✗ Vector DB 오류: {e}")
            return []

    async def _async_graph_search(self, query: str) -> List[SearchResult]:
        """비동기 Graph DB 검색"""
        try:
            print(f"  └ Graph DB 검색: {query[:30]}...")

            # 키워드 최적화
            keywords = await self._optimize_keywords(query)

            loop = asyncio.get_event_loop()

            results = []
            for keyword in keywords[:2]:  # 상위 2개만
                graph_result = await loop.run_in_executor(
                    self.thread_pool,
                    lambda k=keyword: mock_graph_db_search.invoke({"query": k})
                )

                if isinstance(graph_result, dict) and "nodes" in graph_result:
                    for node in graph_result["nodes"][:2]:  # 각 키워드당 2개씩
                        result = SearchResult(
                            source="graph_db",
                            content=f"{node['properties'].get('name', 'Unknown')}: {str(node['properties'])}",
                            relevance_score=0.8,
                            metadata=node,
                            search_query=keyword,
                        )
                        results.append(result)

            print(f"    ✓ Graph DB: {len(results)}개 결과")
            return results

        except Exception as e:
            print(f"    ✗ Graph DB 오류: {e}")
            return []

    async def _async_rdb_search(self, query: str) -> List[SearchResult]:
        """비동기 RDB 검색"""
        try:
            print(f"  └ RDB 검색: {query[:30]}...")

            loop = asyncio.get_event_loop()
            rdb_results = await loop.run_in_executor(
                self.thread_pool,
                lambda: mock_rdb_search.invoke({"query": query})
            )

            results = []
            if isinstance(rdb_results, dict) and "results" in rdb_results:
                for i, doc in enumerate(rdb_results["results"][:2]):
                    result = SearchResult(
                        source="rdb",
                        content=doc.get("content", ""),
                        relevance_score=0.8,
                        metadata={"search_type": "rdb", "rank": i + 1},
                        search_query=query,
                    )
                    results.append(result)

            print(f"    ✓ RDB: {len(results)}개 결과")
            return results

        except Exception as e:
            print(f"    ✗ RDB 오류: {e}")
            return []

    async def _async_web_search(self, query: str) -> List[SearchResult]:
        """비동기 웹 검색"""
        try:
            print(f"  └ Web 검색: {query[:30]}...")

            loop = asyncio.get_event_loop()
            web_results = await loop.run_in_executor(
                self.thread_pool,
                lambda: debug_web_search.invoke({"query": query})
            )

            results = []
            if isinstance(web_results, str) and len(web_results) > 50:
                result = SearchResult(
                    source="web_search",
                    content=web_results,
                    relevance_score=0.75,
                    metadata={"search_type": "web"},
                    search_query=query,
                )
                results.append(result)

            print(f"    ✓ Web: {len(results)}개 결과")
            return results

        except Exception as e:
            print(f"    ✗ Web 검색 오류: {e}")
            return []

    async def _async_react_search(self, query: str) -> Optional[SearchResult]:
        """비동기 ReAct 검색"""
        try:
            print(f"  └ ReAct 검색: {query[:30]}...")

            if not self.react_agent_executor:
                return None

            enhanced_prompt = self._create_enhanced_query_prompt(query)

            result = await asyncio.wait_for(
                self.react_agent_executor.ainvoke({"input": enhanced_prompt}),
                timeout=120
            )

            output = result.get("output", "")
            if len(output) > 50:
                search_result = SearchResult(
                    source="react_agent",
                    content=output,
                    relevance_score=0.9,
                    metadata={"search_type": "react"},
                    search_query=query,
                )
                print(f"    ✓ ReAct: 분석 완료")
                return search_result

            return None

        except Exception as e:
            print(f"    ✗ ReAct 오류: {e}")
            return None

    # ========== 유틸리티 메서드들 ==========

    def _determine_execution_strategy(self, state: StreamingAgentState, complexity_level: str) -> ExecutionStrategy:
        """실행 전략 결정"""
        execution_strategy = state.execution_mode
        if not execution_strategy and state.query_plan:
            execution_strategy = state.query_plan.execution_strategy

        if not execution_strategy:
            if complexity_level == "super_complex":
                execution_strategy = ExecutionStrategy.MULTI_AGENT
            elif complexity_level == "complex":
                execution_strategy = ExecutionStrategy.FULL_REACT
            elif complexity_level == "medium":
                execution_strategy = ExecutionStrategy.BASIC_SEARCH
            else:
                execution_strategy = ExecutionStrategy.DIRECT_ANSWER

        return execution_strategy

    async def _optimize_keywords(self, query: str) -> List[str]:
        """Graph DB용 키워드 최적화"""
        prompt = f"""
        다음 질문에서 검색에 효과적인 핵심 키워드 2-3개를 추출하세요:

        질문: {query}

        키워드들을 쉼표로 구분해서 답변해주세요:
        """

        response = await self.chat.ainvoke(prompt)
        keywords = [kw.strip() for kw in response.content.split(",")]
        return keywords[:3]

    async def _generate_expanded_keywords(self, query: str) -> List[str]:
        """확장 키워드 생성 (SUPER_COMPLEX용)"""
        prompt = f"""
        다음 질문과 관련된 확장 검색 키워드를 생성하세요:

        원본 질문: {query}

        관련 키워드, 유사어, 상위/하위 개념을 포함하여 5개의 확장 키워드를 제시하세요:
        """

        response = await self.llm.ainvoke(prompt)
        keywords = [kw.strip() for kw in response.content.split(",")]
        return keywords[:5]

    async def _simple_llm_analysis(self, query: str, search_results: List[SearchResult]) -> Optional[SearchResult]:
        """간단한 LLM 분석"""
        try:
            if not search_results:
                return None

            context = ""
            for result in search_results[-5:]:
                context += f"- {result.source}: {result.content[:200]}\n"

            prompt = f"""
검색 결과를 바탕으로 질문에 대한 간단한 분석을 제공하세요.

질문: {query}

검색 결과:
{context}

간단한 분석 (200자 이내):
"""

            response = await self.llm.ainvoke(prompt)

            return SearchResult(
                source="llm_analysis",
                content=response.content,
                relevance_score=0.85,
                metadata={"analysis_type": "simple_llm"},
                search_query=query,
            )

        except Exception as e:
            print(f"- LLM 분석 오류: {e}")
            return None

    async def _strategic_synthesis(
        self, query: str, search_results: List[SearchResult]
    ) -> SearchResult:
        """전략적 종합 분석 (SUPER_COMPLEX용)"""
        try:
            # 모든 검색 결과 종합
            context = ""
            for result in search_results:
                context += f"출처({result.source}): {result.content[:300]}\n\n"

            prompt = f"""
당신은 전략 컨설턴트입니다. 수집된 모든 정보를 종합하여 전략적 인사이트를 제공하세요.

사용자 요청: {query}

수집된 정보:
{context}

전략적 종합 분석:
1. 핵심 발견사항 (3가지)
2. 전략적 시사점 (2가지)
3. 실행 가능한 제안 (2가지)

500자 이내로 체계적으로 정리해주세요.
"""

            response = await self.llm.ainvoke(prompt)

            return SearchResult(
                source="strategic_synthesis",
                content=response.content,
                relevance_score=0.95,
                metadata={
                    "analysis_type": "strategic_synthesis",
                    "total_sources": len(search_results),
                    "synthesis_date": self.current_date_str,
                },
                search_query=query,
            )

        except Exception as e:
            print(f"- 전략적 종합 오류: {e}")
            return None

    async def _final_validation(
        self, query: str, search_results: List[SearchResult]
    ) -> SearchResult:
        """최종 검증 분석 (SUPER_COMPLEX용)"""
        try:
            # 최근 분석 결과들만 검증
            recent_results = (
                search_results[-3:] if len(search_results) >= 3 else search_results
            )

            context = ""
            for result in recent_results:
                context += f"{result.source}: {result.content[:200]}\n"

            prompt = f"""
최종 검증자로서 분석 결과의 일관성과 완성도를 평가하고 보완하세요.

원본 요청: {query}

분석 결과들:
{context}

최종 검증 및 보완:
- 분석의 일관성 검토
- 누락된 중요 관점 식별
- 최종 결론 및 권고사항

300자 이내로 간결하게 제시하세요.
"""

            response = await self.llm.ainvoke(prompt)

            return SearchResult(
                source="final_validation",
                content=response.content,
                relevance_score=0.9,
                metadata={
                    "analysis_type": "final_validation",
                    "validated_sources": len(recent_results),
                    "validation_date": self.current_date_str,
                },
                search_query=query,
            )

        except Exception as e:
            print(f"- 최종 검증 오류: {e}")
            return None

    def _create_enhanced_query_prompt(self, query: str) -> str:
        """ReAct용 향상된 프롬프트"""
        return f"""
    Research this topic thoroughly: {query}

    Current Date: {self.current_date_str}

    Please use the available tools to research this topic.

    IMPORTANT: Use this exact format for tools:
    Action: tool_name
    Action Input: search_query

    DO NOT use parentheses or function call syntax.

    Begin your research now.
    """

    def _create_react_agent(self):
        """ReAct 에이전트 생성"""
        try:
            # 기본 ReAct 프롬프트 가져오기
            base_prompt = hub.pull("hwchase17/react")

            # 매우 명확한 시스템 지시사항
            system_instruction = f"""
    You are an expert research assistant for agricultural and food industry analysis.
    Current Date: {self.current_date_str}

    CRITICAL: TOOL USAGE FORMAT
    When using tools, you MUST follow this EXACT format:

    Action: tool_name
    Action Input: your_query_here

    NEVER use function call syntax like tool_name("query") - this will cause errors.

    CORRECT EXAMPLES:
    Action: debug_web_search
    Action Input: MZ세대 소비 패턴 2025

    Action: mock_vector_search
    Action Input: 농산물 가격 동향 분석

    Action: mock_rdb_search
    Action Input: 사과 영양성분 데이터

    Action: mock_graph_db_search
    Action Input: 농업 연구기관 관계 분석

    AVAILABLE TOOLS:
    1. debug_web_search - For latest web information, current events, breaking news
    2. mock_vector_search - For document content analysis, research papers, news articles
    3. mock_rdb_search - For structured data, statistics, numerical information
    4. mock_graph_db_search - For entity relationships, knowledge graph analysis

    RESEARCH STRATEGY:
    1. Start with the most relevant tool for the query type
    2. Use multiple tools if comprehensive analysis is needed
    3. Always analyze results before providing final answer
    4. Synthesize information from all sources

    Remember: Use the EXACT Action/Action Input format shown above.
    """

            # 프롬프트 템플릿 생성
            react_prompt = PromptTemplate(
                template=system_instruction + "\n\n" + base_prompt.template,
                input_variables=base_prompt.input_variables
            )

            # ReAct 에이전트 생성
            react_agent_runnable = create_react_agent(
                self.llm, self.available_tools, react_prompt
            )

            # 에이전트 실행기 - 더 관대한 설정
            return AgentExecutor(
                agent=react_agent_runnable,
                tools=self.available_tools,
                verbose=True,
                handle_parsing_errors=True,  # 파싱 에러 자동 처리
                max_iterations=6,  # 반복 횟수 증가
                max_execution_time=200,  # 실행 시간 충분히 확보
                early_stopping_method="generate",
                return_intermediate_steps=True,
            )

        except Exception as e:
            print(f"ReAct 에이전트 초기화 실패: {e}")
            return None


    def _create_fallback_result(self, query: str, error_type: str) -> SearchResult:
        """폴백 결과 생성"""
        return SearchResult(
            source=f"fallback_{error_type}",
            content=f"검색 중 오류가 발생했습니다. 질문: {query}",
            relevance_score=0.3,
            metadata={"error_type": error_type},
            search_query=query,
        )


# CriticAgent1: 정보량 충분성 평가
class CriticAgent1:
    def __init__(self):
        self.chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.agent_type = AgentType.CRITIC_1

    async def evaluate(self, state: StreamingAgentState) -> StreamingAgentState:
        print(">> CRITIC_1 시작")
        graph_results = state.graph_results_stream
        multi_results = state.multi_source_results_stream
        print(f"- Graph DB 결과: {len(graph_results)}개")
        print(f"- Multi Source 결과: {len(multi_results)}개")

        evaluation_result = await self._evaluate_sufficiency(
            state.original_query, graph_results, multi_results
        )

        print(f"- 평가 결과: {evaluation_result.get('status', 'insufficient')}")
        print(f"- 평가 신뢰도: {evaluation_result.get('confidence', 0.0)}")
        print(f"- 평가 이유: {evaluation_result.get('reasoning', 'N/A')}")

        state.critic1_result = CriticResult(**evaluation_result)

        # 여기의 조건문을 수정합니다.
        if evaluation_result.get("status") == "sufficient":
            state.info_sufficient = True
            print(
                "- 정보가 충분하여 다음 단계로 진행합니다."
            )  # 수정 후에는 이 메시지가 올바른 상황에만 출력됩니다.
        else:
            state.info_sufficient = False
            print("- 정보가 부족하여 추가 검색을 요청합니다.")
            # 'status'가 'insufficient'일 때만 suggestion을 출력하도록 보장됩니다.
            if evaluation_result.get("status") == "insufficient":
                print(f"- 개선 제안: {evaluation_result.get('suggestion', 'N/A')}")

        memory = state.get_agent_memory(AgentType.CRITIC_1)
        memory.add_finding(f"정보 충분성 평가: {state.info_sufficient}")
        memory.update_metric(
            "confidence_score", evaluation_result.get("confidence", 0.5)
        )
        print(">> CRITIC_1 완료")
        return state

    def _summarize_results(self, results, source_name):
        """검색 결과 요약 헬퍼 함수"""
        if not results:
            return f"{source_name}: 검색 결과 없음\n"

        summary = f"{source_name} ({len(results)}개 결과):\n"
        for r in results[:3]:
            content_preview = r.content[:100].strip().replace("\n", " ")
            summary += f"  - {content_preview}...\n"
        return summary

    async def _evaluate_sufficiency(self, original_query, graph_results, multi_results):
        results_summary = self._summarize_results(
            graph_results, "Graph DB"
        ) + self._summarize_results(multi_results, "Multi-Source (ReAct Agent)")

        prompt = f"""
        당신은 수집된 정보가 사용자의 질문에 답변하기에 "대체로 충분한 수준인지"를 실용적으로 평가하는 수석 분석가입니다. 완벽하지 않더라도, 핵심적인 답변 생성이 가능한지를 판단하는 것이 중요합니다.

        ### [매우 중요한 판단 기준]
        - 정보가 약 80% 이상 포함되어 있고, 질문의 핵심 논지를 파악할 수 있다면 **'sufficient'로 판단**하세요.
        - 일부 정보가 누락되었더라도, 답변의 전체적인 흐름을 만드는 데 지장이 없다면 **'sufficient'로 판단**하세요.
        - 정보가 전혀 없거나, 질문의 주제와 완전히 동떨어진 내용일 경우에만 **'insufficient'로 판단**하세요.
        ---

        <원본 질문>
        "{original_query}"

        <수집된 정보 요약>
        {results_summary}
        ---

        **[평가 결과]** (아래 형식을 반드시 준수하여 답변하세요.)
        STATUS: sufficient 또는 insufficient
        REASONING: [판단 근거를 위 기준에 맞춰 간결하게 작성]
        SUGGESTION: [STATUS가 'insufficient'일 경우에만, 다음 검색에 도움이 될 구체적인 제안을 작성. 'sufficient'일 경우 '없음'으로 작성.]
        CONFIDENCE: [당신의 'STATUS' 판단에 대한 신뢰도를 0.0 에서 1.0 사이의 점수로 표현. 점수가 0.85 이상이면 매우 확신하는 상태임.]
        """
        response = await self.chat.ainvoke(prompt)
        return self._parse_evaluation(response.content)

    def _parse_evaluation(self, response_content):
        try:
            lines = response_content.strip().split("\n")
            result = {}
            for line in lines:
                if line.startswith("STATUS:"):
                    result["status"] = line.split(":", 1)[1].strip()
                elif line.startswith("REASONING:"):
                    result["reasoning"] = line.split(":", 1)[1].strip()
                elif line.startswith("SUGGESTION:"):
                    result["suggestion"] = line.split(":", 1)[1].strip()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        result["confidence"] = float(line.split(":", 1)[1].strip())
                    except:
                        result["confidence"] = 0.5

            if "status" not in result:
                result["status"] = "insufficient"
            if "reasoning" not in result:
                result["reasoning"] = "판단 근거 없음"
            if result.get("status") == "insufficient" and not result.get("suggestion"):
                result["suggestion"] = "내용 보강이 필요합니다."

            return result
        except Exception as e:
            print(f"- 파싱 실패: {e}, 기본값 사용")
            return {
                "status": "insufficient",
                "reasoning": "평가 파싱 실패",
                "suggestion": "맥락 재구성 권장",
                "confidence": 0.5,
            }


# ContextIntegratorAgent: 검색 결과 통합
class ContextIntegratorAgent:
    def __init__(self):
        # 최종 보고서 초안 작성이므로 더 성능 좋은 모델 사용을 고려해볼 수 있음
        self.chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.agent_type = AgentType.CONTEXT_INTEGRATOR

    async def integrate(self, state: StreamingAgentState) -> StreamingAgentState:
        """수집된 모든 정보를 바탕으로 최종 답변의 '초안'을 생성"""
        print(">> CONTEXT_INTEGRATOR 시작 (답변 초안 생성)")

        graph_results = state.graph_results_stream
        multi_results = state.multi_source_results_stream
        all_results = graph_results + multi_results

        if not all_results:
            print("- 통합할 결과가 없음")
            state.integrated_context = (
                "검색된 정보가 없어 답변 초안을 생성할 수 없습니다."
            )
            return state

        print(f"- 총 {len(all_results)}개 검색 결과를 바탕으로 초안 작성 시작")

        # _create_draft 함수를 호출하여 초안을 생성
        draft = await self._create_draft(state.original_query, all_results)

        # 생성된 초안을 integrated_context에 저장
        # 이 초안은 다음 단계인 Critic2가 최종 검수
        state.integrated_context = draft

        print(f"- 답변 초안 생성 완료 (길이: {len(draft)}자)")
        print("\n>> CONTEXT_INTEGRATOR 완료")
        return state

    async def _create_draft(self, original_query: str, all_results: list) -> str:
        """수집된 정보를 바탕으로 자연스러운 문장의 초안을 작성"""

        # 프롬프트에 전달하기 위해 검색 결과를 간결하게 요약
        context_summary = ""
        for result in all_results[:15]:  # 너무 많지 않게 상위 15개 결과만 사용
            context_summary += f"- 출처({result.source}): {result.content}\n"

        prompt = f"""
        당신은 여러 소스에서 수집된 복잡한 정보들을 종합하여, 사용자의 질문에 대한 답변 '초안'을 작성하는 수석 분석가입니다.

        ### 작업 지침:
        1.  **'원본 질문'의 핵심 의도**를 명확히 파악합니다.
        2.  주어진 **'검색 결과 요약'**에 있는 모든 정보를 종합적으로 고려합니다.
        3.  정보들을 논리적인 순서에 맞게 재구성하여, 질문에 대한 답변이 될 수 있는 **하나의 완성된 글(초안)**을 작성합니다.
        4.  서론, 본론, 결론의 구조를 갖춘 자연스러운 설명글 형식으로 작성해주세요.
        5.  각 정보의 출처는 내용 뒤에 `(출처: {result.source})` 와 같은 형식으로 간결하게 언급할 수 있습니다.

        ---
        **[원본 질문]**
        {original_query}

        **[검색 결과 요약]**
        {context_summary}
        ---

        **[답변 초안 작성]**
        """

        response = await self.chat.ainvoke(prompt)
        return response.content



# 차트 생성 지침
CHART_GENERATION_INSTRUCTIONS = """
## 차트 데이터 생성 지침

**중요: 테이블 데이터는 마크다운 테이블로 작성하고, 시각화가 필요한 데이터만 차트로 생성하세요.**

보고서에 차트가 필요한 부분에서는 아래 형식을 정확히 따라 완전한 JSON 데이터를 생성해야 합니다.

### 올바른 차트 형식:
{{CHART_START}}
{"title": "타겟 세그먼트별 관심사 분포", "type": "pie", "data": {"labels": ["환경친화성", "가성비", "브랜드 신뢰도", "건강 기능성", "SNS 트렌드"], "datasets": [{"label": "관심도 (%)", "data": [35, 25, 20, 15, 5]}]}}
{{CHART_END}}
"""

# 보고서 템플릿 정의
REPORT_TEMPLATES = {
    "marketing": {
        "comprehensive": {
            "role_description": "bain_principal_marketing",
            "sections": [
                {
                    "key": "marketing_insights_summary",
                    "words": "450-500",
                    "details": [
                        "core_trends_5",
                        "immediate_opportunities_3",
                        "competitive_advantage",
                        "growth_potential",
                    ],
                },
                {
                    "key": "consumer_behavior_analysis",
                    "words": "500",
                    "details": [
                        "target_segment_profiles",
                        "customer_journey_mapping",
                        "brand_perception_analysis",
                        "lifestyle_changes",
                    ],
                },
                {
                    "key": "competitive_market_opportunities",
                    "words": "450",
                    "details": [
                        "competitor_benchmarking",
                        "new_player_analysis",
                        "whitespace_discovery",
                        "category_expansion",
                    ],
                },
                {
                    "key": "omnichannel_strategy",
                    "words": "400",
                    "details": [
                        "channel_optimization",
                        "new_channel_development",
                        "integrated_brand_experience",
                        "marketing_automation",
                    ],
                },
                {
                    "key": "campaign_strategy",
                    "words": "450",
                    "details": [
                        "short_term_campaigns",
                        "medium_term_campaigns",
                        "long_term_campaigns",
                        "integrated_roadmap",
                    ],
                },
                {
                    "key": "performance_framework",
                    "words": "300",
                    "details": [
                        "kpi_dashboard",
                        "ab_testing",
                        "roi_tracking",
                        "feedback_loop",
                    ],
                },
            ],
            "total_words": "2000-3000",
            "charts": "6-8",
        },
        "detailed": {
            "role_description": "strategic_marketing_analyst",
            "sections": [
                {
                    "key": "market_consumer_analysis",
                    "words": "400",
                    "details": [
                        "market_size_growth",
                        "consumer_segmentation",
                        "journey_analysis",
                        "trend_analysis",
                    ],
                },
                {
                    "key": "competitive_positioning",
                    "words": "400",
                    "details": [
                        "competitor_analysis",
                        "brand_positioning",
                        "differentiation_strategy",
                        "whitespace_opportunities",
                    ],
                },
                {
                    "key": "integrated_marketing",
                    "words": "400",
                    "details": [
                        "channel_strategy",
                        "campaign_strategy",
                        "message_strategy",
                        "budget_optimization",
                    ],
                },
                {
                    "key": "digital_innovation",
                    "words": "300",
                    "details": [
                        "digital_transformation",
                        "data_utilization",
                        "new_technology",
                        "automation",
                    ],
                },
                {
                    "key": "execution_performance",
                    "words": "300",
                    "details": [
                        "execution_roadmap",
                        "kpi_measurement",
                        "risk_management",
                        "continuous_optimization",
                    ],
                },
            ],
            "total_words": "1500-2000",
            "charts": "4-5",
        },
        "standard": {
            "role_description": "marketing_strategist",
            "sections": [
                {
                    "key": "market_consumer_insights",
                    "words": "350",
                    "details": [
                        "market_size_analysis",
                        "target_segments",
                        "competitive_environment",
                        "growth_drivers",
                    ],
                },
                {
                    "key": "brand_positioning_strategy",
                    "words": "350",
                    "details": [
                        "brand_positioning",
                        "differentiation_strategy",
                        "message_strategy",
                        "brand_assets",
                    ],
                },
                {
                    "key": "marketing_mix",
                    "words": "300",
                    "details": [
                        "channel_strategy",
                        "campaign_planning",
                        "budget_allocation",
                        "execution_timeline",
                    ],
                },
                {
                    "key": "performance_execution",
                    "words": "250",
                    "details": [
                        "kpi_setting",
                        "roi_prediction",
                        "risk_management",
                        "next_steps",
                    ],
                },
            ],
            "total_words": "1000-1500",
            "charts": "3",
        },
        "brief": {
            "role_description": "marketing_consultant",
            "sections": [
                {
                    "key": "market_situation_opportunities",
                    "words": "250",
                    "details": [
                        "key_trends",
                        "target_analysis",
                        "competitive_situation",
                    ],
                },
                {
                    "key": "recommended_strategy",
                    "words": "200",
                    "details": ["core_strategy", "priority_tasks", "expected_results"],
                },
                {
                    "key": "execution_plan",
                    "words": "150",
                    "details": [
                        "action_plan",
                        "required_resources",
                        "performance_measurement",
                    ],
                },
            ],
            "total_words": "500-800",
            "charts": "1-2",
        },
    },
    "purchasing": {
        "comprehensive": {
            "role_description": "mckinsey_procurement_partner",
            "sections": [
                {
                    "key": "executive_strategic_recommendations",
                    "words": "500",
                    "details": [
                        "value_opportunities",
                        "strategic_sourcing_insights",
                        "risk_mitigation",
                        "financial_impact",
                    ],
                },
                {
                    "key": "market_intelligence_pricing",
                    "words": "600",
                    "details": [
                        "commodity_price_analysis",
                        "global_supply_demand",
                        "geopolitical_regulatory",
                        "ai_prediction_models",
                    ],
                },
                {
                    "key": "supplier_ecosystem_evaluation",
                    "words": "500",
                    "details": [
                        "supplier_scorecards",
                        "supply_base_optimization",
                        "financial_health_assessment",
                        "emerging_suppliers",
                    ],
                },
                {
                    "key": "procurement_excellence_digital",
                    "words": "450",
                    "details": [
                        "category_strategy_enhancement",
                        "contract_optimization",
                        "digital_procurement_platform",
                        "organizational_capability",
                    ],
                },
                {
                    "key": "advanced_risk_management",
                    "words": "400",
                    "details": [
                        "risk_quantification",
                        "scenario_based_response",
                        "alternative_sourcing",
                        "hedging_strategies",
                    ],
                },
                {
                    "key": "performance_continuous_improvement",
                    "words": "300",
                    "details": [
                        "kpi_dashboard",
                        "benchmarking_framework",
                        "innovation_metrics",
                        "sustainability_matrix",
                    ],
                },
            ],
            "total_words": "2000-3000",
            "charts": "7-8",
        },
        "detailed": {
            "role_description": "procurement_strategist",
            "sections": [
                {
                    "key": "procurement_strategy_optimization",
                    "words": "400",
                    "details": [
                        "procurement_strategy",
                        "supplier_management",
                        "risk_analysis",
                        "cost_optimization",
                    ],
                }
            ],
            "total_words": "1500-2000",
            "charts": "4-5",
        },
        "standard": {
            "role_description": "procurement_analyst",
            "sections": [
                {
                    "key": "procurement_analysis",
                    "words": "350",
                    "details": [
                        "market_analysis",
                        "supplier_evaluation",
                        "cost_strategy",
                    ],
                }
            ],
            "total_words": "1000-1500",
            "charts": "3",
        },
        "brief": {
            "role_description": "procurement_consultant",
            "sections": [
                {
                    "key": "procurement_insights",
                    "words": "250",
                    "details": ["key_findings", "recommendations", "action_items"],
                }
            ],
            "total_words": "500-800",
            "charts": "1-2",
        },
    },
    "development": {
        "comprehensive": {
            "role_description": "innovation_strategist",
            "sections": [
                {
                    "key": "product_innovation_strategy",
                    "words": "500",
                    "details": [
                        "innovation_strategy",
                        "technology_roadmap",
                        "commercialization",
                    ],
                }
            ],
            "total_words": "2000-3000",
            "charts": "6-8",
        }
    },
    "general_affairs": {
        "comprehensive": {
            "role_description": "operations_excellence_expert",
            "sections": [
                {
                    "key": "operational_optimization",
                    "words": "500",
                    "details": [
                        "operations_optimization",
                        "employee_satisfaction",
                        "cost_efficiency",
                    ],
                }
            ],
            "total_words": "2000-3000",
            "charts": "6-8",
        }
    },
    "general": {
        "comprehensive": {
            "role_description": "business_analyst",
            "sections": [
                {
                    "key": "strategic_business_analysis",
                    "words": "500",
                    "details": [
                        "market_analysis",
                        "opportunity_assessment",
                        "strategic_recommendations",
                    ],
                }
            ],
            "total_words": "2000-3000",
            "charts": "6-8",
        }
    },
}

# 언어별 번역
TRANSLATIONS = {
    "korean": {
        # 역할 설명
        "bain_principal_marketing": "당신은 베인앤컴퍼니의 프린시플로서 100개 이상의 브랜드를 성공으로 이끈 마케팅 전략가입니다.",
        "strategic_marketing_analyst": "당신은 전략적 마케팅 분석을 전문으로 하는 시니어 애널리스트입니다.",
        "marketing_strategist": "당신은 마케팅 전략 수립을 전문으로 하는 컨설턴트입니다.",
        "marketing_consultant": "당신은 마케팅 인사이트를 제공하는 전문 컨설턴트입니다.",
        "mckinsey_procurement_partner": "당신은 맥킨지앤컴퍼니의 시니어 파트너로서 Fortune 500 기업의 조달 혁신을 전문으로 합니다.",
        "procurement_strategist": "당신은 구매 전략과 공급망 최적화를 전문으로 하는 컨설턴트입니다.",
        "procurement_analyst": "당신은 구매 분석과 공급업체 관리를 전문으로 하는 애널리스트입니다.",
        "procurement_consultant": "당신은 구매 최적화를 위한 인사이트를 제공하는 컨설턴트입니다.",
        "innovation_strategist": "당신은 제품 혁신과 기술 전략을 전문으로 하는 전략가입니다.",
        "operations_excellence_expert": "당신은 운영 우수성과 직원 경험 최적화를 전문으로 하는 컨설턴트입니다.",
        "business_analyst": "당신은 전략적 비즈니스 분석을 전문으로 하는 컨설턴트입니다.",
        # 섹션 제목
        "marketing_insights_summary": "마케팅 인사이트 종합 요약",
        "consumer_behavior_analysis": "심층 소비자 행동 분석",
        "competitive_market_opportunities": "경쟁 환경 및 시장 기회",
        "omnichannel_strategy": "옴니채널 실행 전략",
        "campaign_strategy": "구체적 캠페인 전략",
        "performance_framework": "성과 측정 프레임워크",
        "market_consumer_analysis": "시장 기회 및 소비자 분석",
        "competitive_positioning": "경쟁 환경 및 포지셔닝",
        "integrated_marketing": "통합 마케팅 전략",
        "digital_innovation": "디지털 마케팅 혁신",
        "execution_performance": "실행 계획 및 성과 관리",
        "market_consumer_insights": "시장 기회 및 소비자 인사이트",
        "brand_positioning_strategy": "브랜드 전략 및 포지셔닝",
        "marketing_mix": "마케팅 믹스 전략",
        "performance_execution": "성과 측정 및 실행 방안",
        "market_situation_opportunities": "시장 현황 및 기회",
        "recommended_strategy": "추천 전략",
        "execution_plan": "실행 방안",
        # 구매 관련
        "executive_strategic_recommendations": "경영진 요약 및 전략적 제안",
        "market_intelligence_pricing": "시장 인텔리전스 및 가격 동향",
        "supplier_ecosystem_evaluation": "공급업체 생태계 종합 평가",
        "procurement_excellence_digital": "조달 우수성 및 디지털 전환",
        "advanced_risk_management": "고급 리스크 관리",
        "performance_continuous_improvement": "성과 측정 및 지속적 개선",
        "procurement_strategy_optimization": "구매 전략 및 공급망 최적화",
        "procurement_analysis": "구매 전략 분석",
        "procurement_insights": "구매 인사이트 요약",
        # 기타
        "product_innovation_strategy": "제품 혁신 및 기술 전략",
        "operational_optimization": "운영 우수성 및 직원 경험 최적화",
        "strategic_business_analysis": "전략적 비즈니스 분석 및 인사이트",
        # 세부 항목들
        "core_trends_5": "핵심 트렌드 5가지: 각 트렌드별 정량적 임팩트 분석",
        "immediate_opportunities_3": "즉시 활용 기회 3가지: ROI 예측치와 구체적 실행 방안",
        "competitive_advantage": "경쟁사 대비 우위: 차별화 전략과 포지셔닝 갭 분석",
        "growth_potential": "성장 잠재력: ROAS, CAC, LTV 기반 정량적 예측",
    },
    "english": {
        # 역할 설명
        "bain_principal_marketing": "You are a Principal at Bain & Company who has led over 100 brands to success as a marketing strategist.",
        "strategic_marketing_analyst": "You are a senior analyst specializing in strategic marketing analysis.",
        "marketing_strategist": "You are a consultant specializing in marketing strategy development.",
        "marketing_consultant": "You are a professional consultant providing marketing insights.",
        "mckinsey_procurement_partner": "You are a Senior Partner at McKinsey & Company specializing in procurement innovation for Fortune 500 companies.",
        "procurement_strategist": "You are a consultant specializing in procurement strategy and supply chain optimization.",
        "procurement_analyst": "You are an analyst specializing in procurement analysis and supplier management.",
        "procurement_consultant": "You are a consultant providing insights for procurement optimization.",
        "innovation_strategist": "You are a strategist specializing in product innovation and technology strategy.",
        "operations_excellence_expert": "You are a consultant specializing in operational excellence and employee experience optimization.",
        "business_analyst": "You are a consultant specializing in strategic business analysis.",
        # 섹션 제목
        "marketing_insights_summary": "Marketing Insights Summary",
        "consumer_behavior_analysis": "Consumer Behavior Analysis",
        "competitive_market_opportunities": "Competitive Environment and Market Opportunities",
        "omnichannel_strategy": "Omnichannel Execution Strategy",
        "campaign_strategy": "Specific Campaign Strategy",
        "performance_framework": "Performance Measurement Framework",
        "market_consumer_analysis": "Market Opportunities and Consumer Analysis",
        "competitive_positioning": "Competitive Environment and Positioning",
        "integrated_marketing": "Integrated Marketing Strategy",
        "digital_innovation": "Digital Marketing Innovation",
        "execution_performance": "Execution Plan and Performance Management",
        "market_consumer_insights": "Market Opportunities and Consumer Insights",
        "brand_positioning_strategy": "Brand Strategy and Positioning",
        "marketing_mix": "Marketing Mix Strategy",
        "performance_execution": "Performance Measurement and Execution Plan",
        "market_situation_opportunities": "Market Situation and Opportunities",
        "recommended_strategy": "Recommended Strategy",
        "execution_plan": "Execution Plan",
        # 구매 관련
        "executive_strategic_recommendations": "Executive Summary and Strategic Recommendations",
        "market_intelligence_pricing": "Market Intelligence and Pricing Trends",
        "supplier_ecosystem_evaluation": "Supplier Ecosystem Comprehensive Evaluation",
        "procurement_excellence_digital": "Procurement Excellence and Digital Transformation",
        "advanced_risk_management": "Advanced Risk Management",
        "performance_continuous_improvement": "Performance Measurement and Continuous Improvement",
        "procurement_strategy_optimization": "Procurement Strategy and Supply Chain Optimization",
        "procurement_analysis": "Procurement Strategy Analysis",
        "procurement_insights": "Procurement Insights Summary",
        # 기타
        "product_innovation_strategy": "Product Innovation and Technology Strategy",
        "operational_optimization": "Operational Excellence and Employee Experience Optimization",
        "strategic_business_analysis": "Strategic Business Analysis and Insights",
        # 세부 항목들
        "core_trends_5": "5 Core Trends: Quantitative impact analysis for each trend",
        "immediate_opportunities_3": "3 Immediate Opportunities: ROI predictions and specific execution plans",
        "competitive_advantage": "Competitive Advantage: Differentiation strategy and positioning gap analysis",
        "growth_potential": "Growth Potential: Quantitative predictions based on ROAS, CAC, LTV",
    },
}


class ReportGeneratorAgent:
    def __init__(self):
        self.streaming_chat = ChatOpenAI(
            model="gpt-4o-mini", temperature=0.9, streaming=True
        )
        self.non_streaming_chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self.agent_type = "REPORT_GENERATOR"
        self.use_plan_first = False

    async def generate_streaming(
        self, state: StreamingAgentState
    ) -> AsyncGenerator[str, None]:
        """실시간 스트리밍으로 답변을 생성 - 메모리 컨텍스트 포함"""
        print("\n>> STREAMING REPORT_GENERATOR 시작")
        integrated_context = state.integrated_context
        original_query = state.original_query

        # 메모리 컨텍스트 추출
        memory_context = getattr(state, "memory_context", "")
        user_context = getattr(state, "user_context", None)

        print(
            f"- 메모리 컨텍스트 사용: {len(memory_context)}자"
            if memory_context
            else "- 메모리 컨텍스트 없음"
        )

        if not integrated_context:
            error_msg = "분석할 충분한 정보가 수집되지 않았습니다."
            state.final_answer = error_msg
            yield error_msg
            return

        # 질문 복잡도 분석 (사용자 전문성 수준 고려)
        complexity_analysis = self._analyze_query_complexity(
            original_query, user_context
        )
        print(f"- 질문 복잡도: {complexity_analysis['report_type']}")
        print(f"- 권장 길이: {complexity_analysis['recommended_length']}")
        print(
            f"- 사용자 전문성: {complexity_analysis.get('user_expertise', 'intermediate')}"
        )

        # 메모리 컨텍스트를 포함한 프롬프트 생성
        prompt = self._create_prompt(
            original_query, integrated_context, memory_context, user_context
        )
        full_response = ""

        try:
            async for chunk in self.streaming_chat.astream(prompt):
                if chunk.content:
                    full_response += chunk.content
                    yield chunk.content
        except Exception as e:
            error_msg = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
            yield error_msg
            full_response = error_msg

        state.final_answer = full_response
        print(f"\n- 실시간 스트리밍 완료")

    def _analyze_query_complexity(self, query: str, user_context=None) -> dict:
        """질문의 복잡도와 요구되는 보고서 길이 분석 - 사용자 컨텍스트 포함"""
        if not query or not isinstance(query, str):
            return {
                "complexity_score": 0,
                "report_type": "standard",
                "recommended_length": "1000-1500단어, 4-5개 섹션, 3-4개 차트",
                "user_expertise": "intermediate",
            }

        query_lower = query.lower().strip()
        complexity_score = 0

        # 복잡한 분석 요구 키워드
        complex_keywords = [
            "보고서",
            "report",
            "전략",
            "strategy",
            "분석",
            "analysis",
            "계획",
            "plan",
            "로드맵",
            "roadmap",
            "컨설팅",
            "consulting",
            "상세",
            "detailed",
            "자세",
            "comprehensive",
            "심층",
            "deep",
            "종합",
            "전체",
            "완전",
            "포괄적",
        ]

        # 간단한 정보 요청 키워드
        simple_keywords = [
            "간단히",
            "briefly",
            "짧게",
            "요약",
            "summary",
            "개요",
            "overview",
            "뭐야",
            "what is",
            "알려줘",
            "tell me",
            "빠르게",
            "quick",
        ]

        # 점수 계산
        for keyword in complex_keywords:
            if keyword in query_lower:
                complexity_score += 1.5

        for keyword in simple_keywords:
            if keyword in query_lower:
                complexity_score -= 1.5

        # 질문 길이도 고려
        if len(query) > 100:
            complexity_score += 1.5
        elif len(query) > 50:
            complexity_score += 0.5

        # 사용자 전문성 수준 고려
        user_expertise = "intermediate"
        if user_context:
            expertise_level = getattr(user_context, "expertise_level", None)
            if expertise_level:
                user_expertise = (
                    expertise_level.value
                    if hasattr(expertise_level, "value")
                    else str(expertise_level)
                )

                # 전문가일수록 더 복잡한 보고서 선호
                if user_expertise == "expert":
                    complexity_score += 1.0
                elif user_expertise == "beginner":
                    complexity_score -= 0.5

        # 보고서 타입 결정
        if complexity_score >= 3:
            report_type = "comprehensive"
            recommended_length = "2000-3000단어, 6-8개 섹션, 5-8개 차트"
        elif complexity_score >= 1.5:
            report_type = "detailed"
            recommended_length = "1500-2000단어, 5-6개 섹션, 4-5개 차트"
        elif complexity_score <= -1.5:
            report_type = "brief"
            recommended_length = "500-800단어, 3개 섹션, 1-2개 차트"
        else:
            report_type = "standard"
            recommended_length = "1000-1500단어, 4-5개 섹션, 3-4개 차트"

        return {
            "complexity_score": complexity_score,
            "report_type": report_type,
            "recommended_length": recommended_length,
            "user_expertise": user_expertise,
        }

    def _detect_language(self, query: str) -> str:
        """질문 언어 감지"""
        if not query or not isinstance(query, str):
            return "korean"

        query = query.strip()
        if not query:
            return "korean"

        korean_chars = sum(1 for char in query if "\uac00" <= char <= "\ud7af")
        total_chars = len([char for char in query if char.isalpha()])

        if total_chars > 0 and korean_chars / total_chars > 0.5:
            return "korean"
        else:
            return "english"

    def _detect_team_type(self, query: str) -> str:
        """질문 내용을 분석하여 어떤 팀용 보고서인지 판단"""
        if not query or not isinstance(query, str):
            return "general"

        query_lower = query.lower().strip()
        if not query_lower:
            return "general"

        # 팀별 키워드 정의
        team_keywords = {
            "purchasing": [
                "가격",
                "시세",
                "공급업체",
                "조달",
                "구매",
                "원가",
                "계약",
                "비용",
                "supplier",
                "procurement",
                "sourcing",
                "vendor",
            ],
            "marketing": [
                "마케팅",
                "브랜드",
                "광고",
                "캠페인",
                "소비자",
                "고객",
                "타겟",
                "sns",
                "전략",
                "marketing",
                "brand",
                "campaign",
                "consumer",
            ],
            "development": [
                "개발",
                "제품",
                "영양",
                "성분",
                "기능성",
                "연구",
                "r&d",
                "신제품",
                "기술",
                "development",
                "nutrition",
                "ingredient",
                "innovation",
            ],
            "general_affairs": [
                "급식",
                "직원",
                "사내",
                "구내식당",
                "메뉴",
                "식단",
                "운영",
                "만족도",
                "cafeteria",
                "employee",
                "facility",
                "office",
            ],
        }

        # 각 팀별 점수 계산
        scores = {}
        for team, keywords in team_keywords.items():
            scores[team] = sum(1 for keyword in keywords if keyword in query_lower)

        # 가장 높은 점수의 팀 반환
        max_score = max(scores.values()) if scores.values() else 0
        if max_score == 0:
            return "general"

        return max(scores, key=scores.get)

    def _create_prompt(
        self, query: str, context: str, memory_context: str = "", user_context=None
    ) -> str:
        """메모리 컨텍스트를 포함한 프롬프트 생성"""
        if not query or not context:
            return "입력 데이터가 부족합니다."

        team_type = self._detect_team_type(query)
        language = self._detect_language(query)
        complexity_analysis = self._analyze_query_complexity(query, user_context)
        report_type = complexity_analysis["report_type"]
        user_expertise = complexity_analysis["user_expertise"]

        # 사용자 정보 추출
        user_name = ""
        user_preferences = {}
        if user_context:
            mentioned_info = getattr(user_context, "mentioned_info", {})
            if mentioned_info and isinstance(mentioned_info, dict):
                user_name = mentioned_info.get("name", "")

            preferences = getattr(user_context, "preferences", {})
            if preferences:
                user_preferences = preferences

        # 메모리 정보 처리
        memory_info = ""
        if memory_context:
            memory_info = f"""
**이전 대화 맥락 및 사용자 정보:**
{memory_context}

중요: 위 정보는 이전 대화에서 나눈 내용입니다. 이 대화를 참고를 참고해서 답변해주세요.(예: 사용자가 이름을 알려줬다면 보고서에서 그 이름으로 언급하고,
이전에 관심을 보인 주제나 전문 분야가 있다면 해당 내용을 보고서에 반영)
"""

        # 개인화 정보
        personalization_info = ""
        if user_name:
            personalization_info += f"사용자 이름: {user_name}님\n"

        personalization_info += f"전문성 수준: {user_expertise}\n"

        if user_preferences:
            pref_items = [f"{k}: {v}" for k, v in user_preferences.items()]
            personalization_info += f"사용자 선호도: {', '.join(pref_items)}\n"

        # 기본 프롬프트 생성
        base_prompt = self._create_base_prompt(
            language,
            complexity_analysis,
            query,
            context,
            memory_info,
            personalization_info,
        )

        # 팀별 템플릿 기반 프롬프트 생성
        team_prompt = self._generate_template_prompt(team_type, report_type, language)

        return base_prompt + team_prompt + CHART_GENERATION_INSTRUCTIONS

    def _create_base_prompt(
        self,
        language: str,
        complexity_analysis: dict,
        query: str,
        context: str,
        memory_info: str,
        personalization_info: str,
    ) -> str:
        """메모리 정보를 포함한 기본 프롬프트 생성"""
        user_expertise = complexity_analysis["user_expertise"]

        # 전문성 수준에 따른 안내
        expertise_guidance = {
            "beginner": "기본 개념부터 차근차근 설명하고, 전문 용어 사용 시 쉬운 설명을 병행해주세요.",
            "intermediate": "실무에 도움이 되는 구체적인 정보와 실용적인 인사이트를 제공해주세요.",
            "expert": "심화된 분석과 전문적인 관점에서의 고급 인사이트를 제공해주세요.",
        }

        expertise_guide = expertise_guidance.get(
            user_expertise, expertise_guidance["intermediate"]
        )

        if language == "korean":
            return f"""
당신은 글로벌 식품회사의 전문 분석가입니다. 주어진 정보를 바탕으로 사용자의 질문에 대한 전문적이고 개인화된 보고서를 한국어로 작성해주세요.

{memory_info}

**개인화 정보:**
{personalization_info}

**보고서 작성 요구사항:**
- 보고서 복잡도: {complexity_analysis['report_type'].upper()}
- 목표 길이: {complexity_analysis['recommended_length']}
- 사용자 전문성 수준: {user_expertise}
- 전문성 가이드: {expertise_guide}
- 모든 답변은 반드시 한국어로 작성
- 마크다운 형식 사용
- 전문적이면서도 읽기 쉬운 한국어 사용
- 차트를 적극적으로 활용(차트에 대한 간단 설명 및 인사이트 제공 필수)
- 사용자의 이름이나 이전 대화 내용을 자연스럽게 참조하여 개인화된 보고서 작성

**개인화 지침:**
- 사용자 이름이 있다면 적절한 위치에서 자연스럽게 언급
- 이전 대화에서 관심을 보인 주제가 있다면 해당 내용을 보고서에 연결
- 사용자의 전문성 수준에 맞는 용어와 설명 깊이 조절
- 개인적인 상황이나 선호도가 파악된다면 그에 맞는 권장사항 제시

**[주어진 핵심 정보]**
{context}

**[사용자의 질문]**
"{query}"

"""
        else:
            return f"""
You are a professional analyst at a global food company. Please create a professional and personalized report based on the given information in English.

{memory_info}

**Personalization Information:**
{personalization_info}

**Report Requirements:**
- Report Complexity: {complexity_analysis['report_type'].upper()}
- Target Length: {complexity_analysis['recommended_length']}
- User Expertise Level: {user_expertise}
- Expertise Guide: {expertise_guide}
- All responses must be written in English
- Use markdown formatting
- Use professional yet accessible English
- Actively utilize charts and visualizations
- Create personalized content by referencing user's name and previous conversations

**Personalization Guidelines:**
- Naturally mention user's name if available
- Connect previous conversation topics to current report
- Adjust terminology and explanation depth to user's expertise level
- Provide recommendations based on user's identified preferences or situations

**[Given Core Information]**
{context}

**[User's Question]**
"{query}"

"""

    def _generate_template_prompt(
        self, team_type: str, report_type: str, language: str
    ) -> str:
        """템플릿 기반 프롬프트 생성"""
        # 해당 팀과 복잡도에 맞는 템플릿 가져오기
        template = REPORT_TEMPLATES.get(team_type, {}).get(report_type)
        if not template:
            # 기본 템플릿 사용
            template = REPORT_TEMPLATES.get("general", {}).get(report_type)
            if not template:
                template = REPORT_TEMPLATES["general"]["comprehensive"]

        # 번역 텍스트 가져오기
        translations = TRANSLATIONS[language]

        # 프롬프트 생성
        role_desc = translations.get(template["role_description"], "")

        prompt = f"""
**[{translations.get('strategic_business_analysis', 'Strategic Analysis Report')}]**

{role_desc}

## {translations.get('strategic_business_analysis', 'Strategic Framework')} ({template['total_words']})

"""

        # 섹션별 프롬프트 생성
        for i, section in enumerate(template["sections"], 1):
            section_title = translations.get(section["key"], section["key"])
            prompt += f"""
### {i}. {section_title} ({section["words"]}단어)
"""

            # 세부 항목들 추가
            if "details" in section:
                for detail in section["details"]:
                    detail_text = translations.get(detail, f"- **{detail}**")
                    if detail_text.startswith("- **"):
                        prompt += f"{detail_text}\n"
                    else:
                        prompt += f"- **{detail_text}**\n"

            prompt += "\n"

        prompt += f"""
## 필수 차트 ({template['charts']}개)
각 섹션에 전략적으로 배치하세요.
"""

        return prompt


# Enhanced SimpleAnswererAgent with Claude-level conversational ability
class SimpleAnswererAgent:
    """단순 질문 전용 Agent - 메모리 컨텍스트 지원"""

    def __init__(self, vector_db=None):
        self.vector_db = vector_db
        # 스트리밍용과 일반 호출용 모델을 분리하여 안정성 확보
        self.streaming_chat = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0.9, streaming=True
        )
        self.non_streaming_chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
        self.agent_type = "SIMPLE_ANSWERER"

    async def answer(self, state: StreamingAgentState) -> StreamingAgentState:
        """기존 방식 (스트리밍 없음) - 메모리 컨텍스트 포함"""
        print("\n>> SIMPLE_ANSWERER 시작")

        if await self._needs_vector_search(state.original_query):
            simple_results = await self._simple_search(state.original_query)
        else:
            simple_results = []

        # 메모리 컨텍스트 추출
        memory_context = getattr(state, "memory_context", "")

        state.final_answer = await self._generate_full_answer(
            state.original_query, simple_results, memory_context
        )
        print(f"- 답변 생성 완료 (길이: {len(state.final_answer)}자)")
        return state

    async def answer_streaming(
        self, state: StreamingAgentState
    ) -> AsyncGenerator[str, None]:
        """스트리밍으로 답변을 생성하는 메서드 - 메모리 컨텍스트 포함"""
        print("\n>> STREAMING SIMPLE_ANSWERER 시작")

        if await self._needs_vector_search(state.original_query):
            simple_results = await self._simple_search(state.original_query)
        else:
            simple_results = []

        # 메모리 컨텍스트 추출
        memory_context = getattr(state, "memory_context", "")
        print(
            f"- 메모리 컨텍스트 사용: {len(memory_context)}자"
            if memory_context
            else "- 메모리 컨텍스트 없음"
        )

        full_response = ""
        prompt = self._create_enhanced_prompt_with_memory(
            state.original_query, simple_results, memory_context
        )

        async for chunk in self.streaming_chat.astream(prompt):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content

        state.final_answer = full_response
        print(f"\n- 스트리밍 답변 생성 완료 (길이: {len(full_response)}자)")

    def _create_enhanced_prompt_with_memory(
        self, query: str, search_results: list, memory_context: str
    ) -> str:
        """메모리 컨텍스트를 포함한 향상된 프롬프트"""
        from datetime import datetime

        current_date_str = datetime.now().strftime("%Y년 %m월 %d일")

        # 검색 결과 요약
        context_summary = ""
        if search_results:
            summary_parts = []
            for result in search_results[:3]:
                if isinstance(result, dict):
                    content = result.get("content", str(result))
                else:
                    content = getattr(result, "content", str(result))
                summary_parts.append(f"- {content[:300]}...")
            context_summary = "\n".join(summary_parts)

        # 메모리 컨텍스트 처리
        memory_info = ""
        if memory_context:
            memory_info = f"""
**이전 대화에서 기억해야 할 정보:**
{memory_context}

중요: 위 정보는 이전 대화에서 나눈 내용입니다. 사용자가 이름을 알려줬다면 그 이름으로 부르고, 이전에 언급된 정보들을 기억하고 있음을 보여주세요.
"""

        return f"""
당신은 농수산물 전문 AI 어시스턴트입니다. 사용자와의 이전 대화를 기억하고 개인화된 답변을 제공합니다.

**오늘 날짜:** {current_date_str}

{memory_info}

**현재 질문:** "{query}"

**관련 데이터베이스 정보:**
{context_summary if context_summary else "관련 데이터 없음"}

**답변 지침:**
1. **이전 대화 활용:**
   - 사용자가 이름을 알려준 적이 있다면 반드시 그 이름으로 부르기
   - 이전에 언급된 정보들을 기억하고 있음을 자연스럽게 보여주기
   - 대화의 연속성을 유지하여 개인화된 답변 제공

2. **질문 유형별 답변:**
   - 개인 정보 확인(이름 등): 이전 대화에서 알려준 정보를 정확히 답변
   - 캐주얼한 인사: 친근하고 따뜻하게, 가능하면 이름 포함
   - 정보성 질문: 철저하고 전문적인 답변

3. **톤과 스타일:**
   - 자연스럽고 친근한 대화체
   - 이전 대화를 기억하고 있다는 느낌 전달
   - 농수산물 전문가로서의 신뢰성 유지

4. **출력 형식 (중요):**
   - 모든 답변은 마크다운 형식으로 작성
   - 제목이 필요한 경우: ## 제목(정말 제목 작성이 필요한 경우만 단순 답변에는 제목 없이 답해도 됨)
   - 강조가 필요한 단어: **강조**
   - 목록이 필요한 경우: - 항목1, - 항목2
   - 긴 답변의 경우 적절한 단락 구분 사용
   - 표가 필요한 경우: | 컬럼1 | 컬럼2 | 형태로 작성
   - 수식은 꼭 Latex문법으로 표현(React에서 렌더링 가능하도록)

**답변 (마크다운 형식으로):**
"""

    def _create_enhanced_prompt(self, query: str, search_results: list) -> str:
        """기존 호환성을 위한 메서드 - 메모리 없이, 마크다운 지침 포함"""
        from datetime import datetime

        current_date_str = datetime.now().strftime("%Y년 %m월 %d일")

        context_summary = ""
        if search_results:
            summary_parts = []
            for result in search_results[:3]:
                if isinstance(result, dict):
                    content = result.get("content", str(result))
                else:
                    content = getattr(result, "content", str(result))
                summary_parts.append(f"- {content[:300]}...")
            context_summary = "\n".join(summary_parts)

        return f"""
당신은 농수산물 전문 AI 어시스턴트입니다. 도움이 되고, 정확하며, 정직한 답변을 제공합니다.

**오늘 날짜:** {current_date_str}

**사용 가능한 컨텍스트 정보:**
{context_summary if context_summary else "내부 자료에서 직접적으로 관련된 정보를 찾지 못했습니다."}

**사용자 질문:** "{query}"

**답변 지침:**
1. **질문 유형 판단:**
   - 캐주얼한 인사(안녕, 하이, 고마워 등): 1-2문장의 따뜻하고 친근한 답변
   - 정보 요청: 철저하고 잘 정리된 답변 제공

2. **정보성 답변의 경우:**
   - 관련성 있는 컨텍스트 정보 활용
   - 논리적 흐름으로 명확하게 구조화
   - 도움이 될 때 구체적 세부사항과 예시 포함
   - 사용 가능한 정보가 불완전한 경우 한계 인정

3. **출력 형식 (필수!):**
   - 모든 답변은 마크다운 형식으로 작성
   - 제목: ## 제목, ### 소제목(정말 필요한 경우에만)
   - 강조: **중요한 내용**
   - 목록: - 항목1, - 항목2
   - 표: | 컬럼1 | 컬럼2 |
   - 적절한 단락 구분 사용
   - 수식은 Latex 문법으로 표현

**답변 (마크다운 형식으로):**
"""

    async def _generate_full_answer(
        self, query: str, search_results: list, memory_context: str = ""
    ) -> str:
        """스트리밍 없이 전체 답변을 한 번에 생성합니다. - 메모리 컨텍스트 포함"""
        prompt = self._create_enhanced_prompt_with_memory(
            query, search_results, memory_context
        )
        try:
            response = await self.non_streaming_chat.ainvoke(prompt)
            return response.content
        except Exception as e:
            return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"

    async def _needs_vector_search(self, query: str) -> bool:
        """Vector DB 검색 필요성 판단 - 향상된 로직"""
        prompt = f"""
Analyze the user's question and determine if it requires searching internal knowledge base.

Guidelines:
- Return "YES" for questions asking for specific information, facts, or explanations
- Return "NO" for casual greetings, thanks, personal questions, or general conversation
- Personal questions like "내 이름이 뭐야?" should return "NO" (use memory instead)

Examples:
- "안녕하세요" → NO
- "고마워요" → NO
- "내 이름이 뭐야?" → NO
- "퀴노아의 영양성분이 궁금해요" → YES
- "가격 정보를 알려주세요" → YES

Question: "{query}"
Decision (YES/NO):
"""
        try:
            response = await self.non_streaming_chat.ainvoke(prompt)
            decision = "yes" in response.content.lower()
            print(f"Vector DB 검색 쿼리: {query}")
            print(f"- 검색 필요성 판단: {'필요' if decision else '불필요'}")
            return decision
        except Exception:
            return True  # 오류 시 안전하게 검색 수행

    async def _simple_search(self, query: str):
        """Vector DB 간단 검색 - 향상된 결과 처리"""
        try:
            if self.vector_db:
                vector_results = self.vector_db.search(query)

                # 검색 결과를 SearchResult 객체로 변환
                search_results = []
                for doc in vector_results[:5]:  # 상위 5개 결과만 사용
                    search_results.append(
                        {
                            "content": doc.get("content", ""),
                            "source": "vector_db",
                            "relevance_score": doc.get("similarity_score", 0.7),
                            "title": doc.get("title", "Unknown"),
                        }
                    )

                print(f"- Vector DB 검색 결과: {len(search_results)}개")
                return search_results
            else:
                print("- Vector DB가 설정되지 않음")
                return []
        except Exception as e:
            print(f"- Vector DB 검색 오류: {e}")
            return []


# CriticAgent2: 컨텍스트 품질/신뢰도 평가
class CriticAgent2:
    def __init__(self):
        self.chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.agent_type = AgentType.CRITIC_2

    async def evaluate(self, state: StreamingAgentState) -> StreamingAgentState:
        print(">> CRITIC_2 시작")
        integrated_context = state.integrated_context
        original_query = state.original_query

        if not integrated_context:
            state.critic2_result = CriticResult(
                status="insufficient",
                suggestion="통합된 맥락이 없어 평가 불가",
                confidence=0.0,
                reasoning="맥락 통합 단계 미완료",
            )
            state.context_sufficient = False
            return state

        print(f"- 통합 맥락 길이: {len(integrated_context)}자")

        evaluation_result = await self._evaluate_context_quality(
            original_query, integrated_context, state.critic1_result
        )

        print(f"- 평가 결과: {evaluation_result.get('status', 'insufficient')}")
        print(f"- 평가 이유: {evaluation_result.get('reasoning', 'N/A')}")

        state.critic2_result = CriticResult(**evaluation_result)

        if evaluation_result.get("status") == "sufficient":
            state.context_sufficient = True
            print("- 맥락 완성도 충분 - 보고서 생성 가능")
        else:
            state.context_sufficient = False
            print("- 맥락 완성도 부족 - 추가 보완 필요")

        memory = state.get_agent_memory(AgentType.CRITIC_2)
        memory.add_finding(f"맥락 완성도 평가: {state.context_sufficient}")
        memory.update_metric(
            "context_quality_score", evaluation_result.get("confidence", 0.5)
        )
        print("\n>> CRITIC_2 완료")
        return state

    async def _evaluate_context_quality(
        self, original_query, integrated_context, critic1_result
    ):
        critic1_summary = "이전 Critic1의 피드백 없음. (1차 검수 통과)"
        if critic1_result and critic1_result.status == "insufficient":
            critic1_summary = f"이전 단계에서 정보 부족 평가가 있었음. (피드백: '{critic1_result.suggestion}')"

        prompt = f"""
        당신은 최종 보고서 작성을 앞두고, 현재까지 수집 및 통합된 정보가 사용자의 질문에 대한 완벽한 답변이 될 수 있는지 최종 검수하는 수석 분석가입니다.

        ### 최종 검수 기준:
        1.  **답변의 완성도:** '통합된 맥락'이 '원본 질문'에 대해 완전하고 명확한 답변을 제공하는가? 모호하거나 빠진 부분은 없는가?
        2.  **논리적 흐름:** 정보들이 자연스럽고 논리적으로 연결되어 있는가? 이야기의 흐름이 매끄러운가?
        3.  **피드백 반영 여부:** (만약 있다면) '이전 단계 피드백'에서 요구한 내용이 '통합된 맥락'에 잘 반영되었는가?
        ---

        <원본 질문>
        "{original_query}"

        <이전 단계 피드백>
        {critic1_summary}

        <최종 보고서의 기반이 될 통합된 맥락>
        {integrated_context}
        ---

        **[최종 검수 결과]** (아래 형식을 반드시 준수하여 답변하세요.)
        STATUS: sufficient 또는 insufficient
        REASONING: [판단 근거를 간결하게 작성. 피드백이 잘 반영되었는지 여부를 반드시 언급.]
        SUGGESTION: [insufficient일 경우, 최종 보고서 생성 전에 무엇을 더 보강해야 할지 구체적으로 제안.]
        CONFIDENCE: [0.0 ~ 1.0 사이의 신뢰도 점수]
        """
        response = await self.chat.ainvoke(prompt)
        return self._parse_evaluation(response.content)

    def _parse_evaluation(self, response_content):
        # Critic1과 동일한 파서를 사용해도 무방
        try:
            lines = response_content.strip().split("\n")
            result = {}
            for line in lines:
                if line.startswith("STATUS:"):
                    result["status"] = line.split(":", 1)[1].strip()
                elif line.startswith("REASONING:"):
                    result["reasoning"] = line.split(":", 1)[1].strip()
                elif line.startswith("SUGGESTION:"):
                    result["suggestion"] = line.split(":", 1)[1].strip()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        result["confidence"] = float(line.split(":", 1)[1].strip())
                    except:
                        result["confidence"] = 0.5

            if "status" not in result:
                result["status"] = "insufficient"
            if "reasoning" not in result:
                result["reasoning"] = "판단 근거 없음"
            if result.get("status") == "insufficient" and not result.get("suggestion"):
                result["suggestion"] = "내용 보강이 필요합니다."

            return result
        except Exception as e:
            print(f"- 파싱 실패: {e}, 기본값 사용")
            return {
                "status": "insufficient",
                "reasoning": "평가 파싱 실패",
                "suggestion": "맥락 재구성 권장",
                "confidence": 0.5,
            }
