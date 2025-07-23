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

import json
from typing import Dict, List, AsyncGenerator
from langchain_openai import ChatOpenAI

# 로컬 imports
from ...core.config.report_config import TeamType, ReportType, Language
from ...services.templates.report_templates import ReportTemplateManager
from ...services.builders.prompt_builder import PromptBuilder
from ...utils.analyzers.query_analyzer import QueryAnalyzer

# 로컬 모듈
from ..models.models import (
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
from ...services.search.search_tools import (
    debug_web_search,
    graph_db_search,
    rdb_search,
    mock_vector_search,
    scrape_and_extract_content,
)
from ...utils.utils import create_agent_message

class DataExtractor:
    """검색 결과에서 실제 수치 데이터를 추출하는 클래스"""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    async def extract_numerical_data(self, search_results: List[SearchResult], query: str) -> Dict[str, Any]:
        """검색 결과에서 수치 데이터를 추출"""


        combined_text = ""
        for result in search_results:
            combined_text += f"{result.content}\n"

        prompt = f"""
        다음 텍스트에서 숫자, 퍼센트, 통계 데이터를 추출하고 JSON 형태로 정리해주세요.

        원본 질문: {query}

        텍스트:
        {combined_text[:]}  # 너무 길면 잘라서

        다음 형식으로 추출해주세요:
        {{
            "extracted_numbers": [
                {{"value": 숫자, "unit": "단위", "context": "설명", "source": "출처"}}
            ],
            "percentages": [
                {{"value": 숫자, "context": "설명"}}
            ],
            "trends": [
                {{"period": "기간", "change": "변화율", "description": "설명"}}
            ],
            "categories": {{
                "category_name": {{"value": 숫자, "description": "설명"}}
            }}
        }}

        실제 숫자가 없으면 빈 배열이나 객체를 반환하세요.
        """

        try:
            response = await self.llm.ainvoke(prompt)
            return json.loads(response.content)
        except:
            return {"extracted_numbers": [], "percentages": [], "trends": [], "categories": {}}



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

        # 매핑된 값들
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
        """COMPLEX 레벨 실행 계획"""

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
            scrape_and_extract_content,
            mock_vector_search,
            rdb_search,
            graph_db_search,
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

        original_query = state.original_query

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

        # 2. RDB 검색
        if self.rdb:
            search_tasks.append(self._async_rdb_search(query))

        # 3. Graph DB 검색
        if self.graph_db:
            search_tasks.append(self._async_graph_search(query))

        # 4. 간단한 웹 검색
        search_tasks.append(self._async_web_search_enhanced(query))


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
        """
        ReAct 에이전트 단독 실행 (COMPLEX 복잡도)
        - 복잡한 질문은 ReAct 에이전트에게 모든 검색 및 추론 과정을 위임
        """

        print("\n>> ReAct 에이전트 단독 실행")

        react_task = self._async_react_search(query)

        try:
            start_time = time.time()

            react_result = await react_task

            execution_time = time.time() - start_time
            print(f"- ReAct 에이전트 실행 완료: {execution_time:.2f}초")

            total_results = 0
            # ReAct 결과 처리
            if not isinstance(react_result, Exception) and react_result:
                state.add_multi_source_result(react_result)
                total_results += 1

            state.add_step_result("full_react_search", {
                "execution_time": execution_time,
                "total_results": total_results,
                "react_included": True
            })

            print(f"- 총 {total_results}개 결과 추가 (ReAct)")

        except Exception as e:
            print(f"- ReAct 에이전트 실행 실패: {e}")
            fallback_result = self._create_fallback_result(query, "react_agent_error")
            state.add_multi_source_result(fallback_result)

        return state

    async def _execute_multi_stage_parallel_search(
        self, state: StreamingAgentState, query: str
    ) -> StreamingAgentState:
        """다단계 병렬 검색 (SUPER_COMPLEX 복잡도)"""
        print("\n>> 다단계 병렬 검색 실행")

        try:
            # 1단계: 초기 정보 수집
            print("- 1단계: 초기 정보 수집")
            await self._execute_full_parallel_search(state, query)

            # 2단계: 키워드 확장 및 심화 검색
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

            # 스레드 풀 사용
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

            loop = asyncio.get_event_loop()
            graph_result_str = await loop.run_in_executor(
                self.thread_pool,
                lambda: graph_db_search.invoke({"query": query})
            )

            results = []
            if isinstance(graph_result_str, str) and "Neo4j" in graph_result_str:
                result = SearchResult(
                    source="graph_db",
                    content=graph_result_str,
                    relevance_score=0.85, # 그래프 DB는 신뢰도가 높음
                    metadata={"search_type": "graph"},
                    search_query=query,
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


            processed_query = self._preprocess_rdb_query(query)
            print(f"    → 전처리된 쿼리: {processed_query}")

            loop = asyncio.get_event_loop()
            rdb_results_content = await loop.run_in_executor(
                self.thread_pool,
                lambda: rdb_search.invoke({"query": processed_query})
            )

            # 반환값이 문자열인지 확인하여 처리하는 로직으로 변경
            if isinstance(rdb_results_content, str) and rdb_results_content:
                # 전체 문자열을 content로 하는 단일 SearchResult 객체 생성
                result = SearchResult(
                    source="rdb",
                    content=rdb_results_content,
                    relevance_score=0.85, # DB에서 직접 온 정보이므로 신뢰도 높게 설정
                    metadata={"search_type": "rdb"},
                    search_query=processed_query,
                )
                print(f"    ✓ RDB: 1개 결과 객체 생성 완료")
                return [result] # 생성된 객체를 리스트에 담아 반환

            # 딕셔너리 형태의 예외적인 경우도 처리
            elif isinstance(rdb_results_content, dict) and "results" in rdb_results_content:
                # 이 로직은 거의 실행되지 않겠지만, 호환성을 위해 유지
                results = []
                for i, doc in enumerate(rdb_results_content["results"][:2]):
                    results.append(SearchResult(
                        source="rdb",
                        content=doc.get("content", ""),
                        relevance_score=0.8,
                        metadata={"search_type": "rdb", "rank": i + 1},
                        search_query=processed_query,
                    ))
                print(f"    ✓ RDB (Dict): {len(results)}개 결과")
                return results

            else:
                print(f"    ✗ RDB: 유효한 결과를 받지 못함 (Type: {type(rdb_results_content)})")
                return []

        except Exception as e:
            print(f"    ✗ RDB 오류: {e}")
            return []

    def _preprocess_rdb_query(self, query: str) -> str:
        """RDB 검색을 위한 쿼리 전처리"""

        # 적절한 길이라면 그대로 사용
        if len(query) <= 50:
            return query

        # 계획서나 분석 문서 감지
        plan_indicators = [
            '실행 계획', '분석 접근법', '결과물 구성', '전략', '마케팅',
            '보고서', '섹션', '시각적 자료', 'SWOT', '접근법', 'MZ세대'
        ]

        if any(indicator in query for indicator in plan_indicators):
            print(f"      → 계획서 문서 감지, 키워드 추출 중...")

            # 농산물 키워드 우선 추출
            import re
            food_keywords = re.findall(
                r'(감자|사과|배|양파|당근|배추|무|고구마|옥수수|쌀|보리|밀|콩|팥|딸기|포도|복숭아|자두|체리|수박|참외|호박|오이|토마토|상추|시금치|깻잎|마늘|생강|파|대파|쪽파|부추|고추|피망|파프리카|감귤|귤|오렌지|바나나|키위|망고)',
                query
            )

            # 검색 의도 키워드 추출
            intent_keywords = re.findall(
                r'(가격|시세|영양|칼로리|비타민|단백질|생산량|수급|소비|트렌드|시장|분석)',
                query
            )

            # 지역 키워드 추출
            region_keywords = re.findall(
                r'(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)',
                query
            )

            # 시간 키워드 추출
            time_keywords = re.findall(
                r'(최근|오늘|어제|이번주|지난주|이번달|지난달|올해|작년|현재|2024|2025)',
                query
            )

            # 키워드 조합하여 간단한 쿼리 생성
            if food_keywords:
                result_query = food_keywords[0]

                if intent_keywords:
                    result_query += f" {intent_keywords[0]}"

                if time_keywords:
                    result_query = f"{time_keywords[0]} {result_query}"

                if region_keywords:
                    result_query += f" {region_keywords[0]}"

                return result_query

            # 농산물이 없으면 일반적인 검색 의도
            elif intent_keywords:
                return f"농산물 {intent_keywords[0]}"

            else:
                return "농산물 시장 정보"

        # 3. 일반적인 긴 쿼리는 첫 번째 문장만
        sentences = query.split('.')
        if sentences and len(sentences[0]) < 100:
            return sentences[0].strip()

        # 4. 너무 길면 처음 50자만
        return query[:50] + "..."


    async def _async_web_search_enhanced(self, query: str) -> List[SearchResult]:
        """비동기 웹 검색 (정찰 -> 선별 -> 분석)"""
        print(f"  └ Enhanced Web 검색 시작: {query[:30]}...")

        # 1. 정찰(Scout): debug_web_search로 URL과 요약 수집
        # (주의: debug_web_search가 List[Dict]를 반환하도록 수정되어 있어야 함)
        loop = asyncio.get_event_loop()
        try:
            scout_results = await loop.run_in_executor(
                self.thread_pool,
                lambda: debug_web_search.invoke({"query": query})
            )
            if not scout_results or isinstance(scout_results[0], dict) and "error" in scout_results[0]:
                print("    ✗ 웹 검색(정찰) 실패 또는 결과 없음")
                return []
            print(f"    ✓ 정찰 완료: {len(scout_results)}개 URL 후보 발견")
        except Exception as e:
            print(f"    ✗ 웹 검색(정찰) 중 예외 발생: {e}")
            return []

        # 2. 선별(Select): LLM을 이용해 스크래핑할 최적의 URL 선택
        best_urls = await self._select_best_urls_for_scraping(query, scout_results)
        if not best_urls:
            print("    ✗ 스크래핑할 유효한 URL을 선택하지 못함")
            # 스크래핑 실패 시, 정찰 결과 요약이라도 반환
            summary_content = "\n".join([f"제목: {r.get('title', '')}\n요약: {r.get('snippet', '')}" for r in scout_results])
            return [SearchResult(source="web_search_summary", content=summary_content, relevance_score=0.6)]


        # 3. 분석(Analyze): 선택된 URL들을 병렬로 스크래핑
        print(f"    → {len(best_urls)}개 URL에 대한 병렬 스크래핑 시작...")
        scraping_tasks = [self._async_scrape_url(url, query) for url in best_urls]
        final_results = await asyncio.gather(*scraping_tasks)

        # None이 아닌 유효한 결과만 필터링
        valid_results = [res for res in final_results if res is not None]

        print(f"    ✓ Enhanced Web 검색 완료: {len(valid_results)}개 상세 정보 획득")
        return valid_results


    async def _async_react_search(self, query: str) -> Optional[SearchResult]:
        """비동기 ReAct 검색 - 모든 내용 다 합치기"""
        try:
            print(f"  └ ReAct 검색: {query[:30]}...")

            if not self.react_agent_executor:
                print(f"    ✗ ReAct 에이전트가 초기화되지 않음")
                return None

            enhanced_prompt = self._create_enhanced_query_prompt(query)

            result = await asyncio.wait_for(
                self.react_agent_executor.ainvoke({"input": enhanced_prompt}),
                timeout=180
            )

            output = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])

            print(f"    → ReAct output: {len(output)}자")
            print(f"    → intermediate_steps: {len(intermediate_steps)}개")


            all_content = ""

            for i, step in enumerate(intermediate_steps):
                if isinstance(step, tuple) and len(step) >= 2:
                    action, observation = step[0], step[1]

                    if isinstance(observation, str) and len(observation) > 10:
                        all_content += f"=== Step {i+1} ===\n{observation}\n\n"
                        print(f"      - Step {i+1}: {len(observation)}자 추가")

            if output:
                all_content += f"=== Final Output ===\n{output}\n"
                print(f"      - Final Output: {len(output)}자 추가")

            print(f"    → 전체 합친 내용: {len(all_content)}자")

            if len(all_content) > 50:
                search_result = SearchResult(
                    source="react_agent",
                    content=all_content,
                    relevance_score=0.9,
                    metadata={
                        "search_type": "react",
                        "steps_count": len(intermediate_steps),
                        "total_length": len(all_content),
                        "output_length": len(output)
                    },
                    search_query=query,
                )
                print(f"    ✓ ReAct 완료: 전체 {len(all_content)}자")
                return search_result
            else:
                print(f"    ✗ 내용이 너무 짧음: {len(all_content)}자")
                return None

        except Exception as e:
            print(f"    ✗ ReAct 오류: {e}")
            return None



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
        """ReAct 에이전트 생성 - 기존 코드 기반으로 최소한만 수정"""
        try:
            # 기본 ReAct 프롬프트 가져오기
            base_prompt = hub.pull("hwchase17/react")

            system_instruction = f"""
    You are a Senior Analyst at a prestigious agricultural-food economic research institute. Your mission is to provide objective, data-driven, and well-structured analysis based on verifiable information.
    Current Date: {self.current_date_str}

    CORE DIRECTIVES:
    1. Prioritize Facts: Always prioritize verifiable data from reliable sources over speculation or opinion.
    2. State Limitations: If you cannot find a satisfactory answer after a thorough search, clearly state that the information is unavailable. Do not invent information.
    3. Follow Protocol: The research protocol below is mandatory.

    AVAILABLE TOOLS:
    1. rdb_search: For specific, structured data (prices, nutrition). Highest priority for quantitative data.
    2. graph_db_search: For relationship data (e.g., origins, suppliers).
    3. debug_web_search: (Scout Tool) Finds candidate URLs. The output is incomplete and for triage purposes only.
    4. scrape_and_extract_content: (Analyst Tool) Extracts the full, detailed information from a URL. This is the only source of valid web data.
    5. mock_vector_search: For finding information within internal research papers.

    MANDATORY RESEARCH PROTOCOL:

    1. Initial Analysis & Internal Search:
    - Analyze the user's query to understand the core objective.
    - First, attempt to find the answer in internal databases using rdb_search or graph_db_search.

    2. Web Research Protocol (STRICT 3-Step Process):
    - Step 2.1 (Scout): If internal data is insufficient, use debug_web_search to get candidate URLs.
    - Step 2.2 (Assess & Select): From the scout results, assess the reliability of the sources. Prioritize official government sites, research institutions (like KREI), and established news media. Select the single most reliable and relevant URL to analyze first.
    - Step 2.3 (Analyze & Verify): Use scrape_and_extract_content on the selected URL. If the scrape fails or the content is not useful, you MUST return to the scout results and try the second-best URL. You may attempt scraping a maximum of two URLs.

    3. Synthesis & Final Answer:
    - After gathering data from 1-2 tools, you MUST provide your Final Answer.
    - Do NOT keep searching endlessly.
    - Use: Final Answer: [comprehensive answer with the information you found]
    - Even partial information is better than no answer.

    FINAL ANSWER DIRECTIVES:
    - When you have gathered enough information, you MUST provide your Final Answer.
    - Use this EXACT format: Final Answer: [your comprehensive answer]
    - Do NOT continue trying to use tools after you have enough information.
    - Structure your final answer with clear headings and citations.
    - If you have ANY relevant information, provide a Final Answer rather than continuing to search.

    TOOL USAGE FORMAT:
    When a tool needs multiple inputs, provide them as a JSON dictionary.

    Action: tool_name
    Action Input: search_query_or_json_string

    CORRECT EXAMPLES:
    Action: debug_web_search
    Action Input: 2025년 사과 마케팅 전략

    Action: scrape_and_extract_content
    Action Input: JSON string with url and query fields

    Remember: Follow the exact Thought/Action/Action Input/Observation format.
    """

            react_prompt = PromptTemplate(
                template=system_instruction + "\n\n" + base_prompt.template,
                input_variables=base_prompt.input_variables
            )

            # ReAct 에이전트 생성
            react_agent_runnable = create_react_agent(
                self.llm, self.available_tools, react_prompt
            )


            return AgentExecutor(
                agent=react_agent_runnable,
                tools=self.available_tools,
                verbose=True,
                handle_parsing_errors="You must provide a Final Answer now. Stop trying to use tools and give your final response using: Final Answer: [your answer]",
                max_iterations=4,
                max_execution_time=120,
                early_stopping_method="generate",  # generate로 변경
                return_intermediate_steps=True,
            )

        except Exception as e:
            print(f"ReAct 에이전트 초기화 실패: {e}")
            return None

    async def _select_best_urls_for_scraping(self, query: str, search_results: List[Dict]) -> List[str]:
        """LLM을 사용하여 스크래핑할 최적의 URL을 선택합니다."""
        if not search_results:
            return []

        # LLM에게 보여줄 컨텍스트 생성
        context = "아래는 웹 검색 결과 요약입니다.\n\n"
        for i, result in enumerate(search_results):
            context += f"{i+1}. URL: {result.get('link', 'N/A')}\n"
            context += f"   제목: {result.get('title', 'N/A')}\n"
            context += f"   요약: {result.get('snippet', 'N/A')}\n\n"

        prompt = f"""당신은 최고의 리서처입니다. 사용자의 질문에 가장 정확한 답변을 제공할 가능성이 높은 웹페이지를 선택해야 합니다.
        아래 검색 결과 목록을 보고, 상세 분석을 위해 방문할 가장 중요한 URL을 1~2개만 선택해주세요.

        사용자 질문: "{query}"

        {context}

        가장 유용해 보이는 URL을 쉼표(,)로 구분하여 정확히 반환해주세요. 다른 설명은 절대 추가하지 마세요.
        예시: https://example.com/news/1,https://another-site.com/article/2
        """

        try:
            response = await self.llm.ainvoke(prompt)
            urls = [url.strip() for url in response.content.split(',') if url.strip().startswith('http')]
            print(f"    → LLM이 스크래핑할 URL 선택: {urls[:2]}")
            return urls[:2] # 최대 2개만 반환
        except Exception as e:
            print(f"    ✗ URL 선택 중 오류: {e}")
            return []


    async def _async_scrape_url(self, url: str, query: str) -> Optional[SearchResult]:
        """단일 URL을 비동기적으로 스크래핑하고 SearchResult로 변환합니다."""
        # 이전에 정의한 scrape_and_extract_content 도구를 사용합니다.
        # 이 도구는 search_tools.py에 정의되어 있어야 합니다.
        try:
            from .search_tools import scrape_and_extract_content # 실제 경로에 맞게 수정

            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                self.thread_pool,
                lambda: scrape_and_extract_content.invoke({"url": url, "query": query})
            )

            if content and "오류" not in content:
                return SearchResult(
                    source="web_scrape",
                    content=content,
                    relevance_score=0.88,
                    metadata={"search_type": "scrape", "source_url": url},
                    search_query=query
                )
            return None
        except Exception as e:
            print(f"    ✗ URL({url}) 스크래핑 실패: {e}")
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


        if evaluation_result.get("status") == "sufficient":
            state.info_sufficient = True
            print(
                "- 정보가 충분하여 다음 단계로 진행합니다."
            )
        else:
            state.info_sufficient = False
            print("- 정보가 부족하여 추가 검색을 요청합니다.")

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
        """검색 결과 요약 헬퍼 함수 - 더 많은 내용 포함"""
        if not results:
            print(f"    - {source_name}: 검색 결과 없음")
            return f"{source_name}: 검색 결과 없음\n"

        print(f"    - {source_name}: {len(results)}개 결과 발견")

        summary = f"{source_name} ({len(results)}개 결과):\n"
        for i, r in enumerate(results[:3]):  # 상위 3개만
            content = r.content if hasattr(r, 'content') else str(r)

            # 훨씬 더 많은 내용 포함 (100자 → 1000자)
            content_preview = content[:10000].strip().replace("\n", " ")

            print(f"      [{i+1}] 길이: {len(content)}자, 미리보기: {content_preview[:100]}...")
            summary += f"  - [{i+1}] 길이: {len(content)}자\n"
            summary += f"    내용: {content_preview}...\n"

        return summary

    async def _evaluate_sufficiency(self, original_query, graph_results, multi_results):
        """정보 충분성 평가 - 더 관대한 기준으로 수정"""

        # 디버깅: 실제 결과 내용 확인
        print(f"  - 평가 대상:")
        print(f"    - Graph 결과: {len(graph_results)}개")
        print(f"    - Multi 결과: {len(multi_results)}개")

        # 모든 결과 합쳐서 총 길이 계산
        total_content = ""
        for result in graph_results + multi_results:
            content = result.content if hasattr(result, 'content') else str(result)
            total_content += content + "\n"

        total_length = len(total_content.strip())
        print(f"    - 총 내용 길이: {total_length}자")

        print(f"    - 실제 내용 확인:")
        for i, result in enumerate(graph_results + multi_results):
            content = result.content if hasattr(result, 'content') else str(result)
            source = result.source if hasattr(result, 'source') else 'unknown'
            print(f"      [{i+1}] {source}: {len(content)}자")
            print(f"          샘플: {content[:100]}...")

            # Final Answer 체크
            if "Final Answer:" in content:
                print(f"          → Final Answer 포함됨")
            if "### Report" in content or "#### " in content:
                print(f"          → 구조화된 보고서 포함됨")


        # 기존 상세 평가 로직
        results_summary = self._summarize_results(
            graph_results, "Graph DB"
        ) + self._summarize_results(multi_results, "Multi-Source")

        # 실제 내용도 포함해서 프롬프트 작성
        content_sample = total_content[:500] if total_content else "내용 없음"

        prompt = f"""
        다음 검색 결과를 바탕으로 정보 충분성을 평가해주세요.

        ### 원본 질문
        "{original_query}"

        ### 수집된 정보 요약
        {results_summary}

        ### 실제 내용 샘플 (처음 500자)
        {content_sample}

        ### 매우 관대한 평가 기준:
        - 질문과 관련된 ANY 정보가 있으면 → sufficient
        - 50자 이상의 관련 내용이 있으면 → sufficient
        - 완전히 빈 결과거나 완전 무관한 내용만 있을 때만 → insufficient

        **[평가 결과]**
        STATUS: sufficient 또는 insufficient
        REASONING: [매우 관대한 기준으로 판단한 근거]
        SUGGESTION: [insufficient일 경우에만]
        CONFIDENCE: [0.0 ~ 1.0]
        """

        response = await self.chat.ainvoke(prompt)
        result = self._parse_evaluation(response.content)

        print(f"  - Critic1 최종 판단: {result.get('status', 'unknown')}")
        return result

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


# ContextIntegratorAgent
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

        # PostgreSQL 결과를 먼저 파싱하여 구조화된 데이터 추출
        structured_data = self._parse_postgresql_results(all_results)
        print(f"- PostgreSQL 구조화 데이터: {len(structured_data.get('nutrition_data', []))}건 영양소, {len(structured_data.get('price_data', []))}건 가격")

        # _create_draft 함수를 호출하여 초안을 생성
        draft = await self._create_draft(state.original_query, all_results, structured_data)

        # 생성된 초안을 integrated_context에 저장
        state.integrated_context = draft

        print(f"- 답변 초안 생성 완료 (길이: {len(draft)}자)")
        print("\n>> CONTEXT_INTEGRATOR 완료")
        return state

    def _parse_postgresql_results(self, all_results: list) -> dict:
        """PostgreSQL 검색 결과에서 구조화된 데이터 추출"""
        structured_data = {
            'nutrition_data': [],
            'price_data': [],
            'other_data': []
        }

        for result in all_results:
            content = result.content

            try:
                # PostgreSQL 결과인지 확인
                if 'PostgreSQL 검색 결과' in content:
                    # 정확한 JSON 블록만 추출
                    json_match = re.search(r'### 상세 데이터 \(JSON\)\s*(\{.*?\n\})', content, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1).strip()
                        data = json.loads(json_content)

                        # 영양소 데이터 추출
                        if 'nutrition_data' in data and data['nutrition_data']:
                            for item in data['nutrition_data']:
                                structured_item = {
                                    '식품명': item.get('식품명', 'N/A'),
                                    '식품군': item.get('식품군', 'N/A'),
                                    '출처': item.get('출처', 'N/A'),
                                    '칼로리': item.get('칼로리', 0),
                                    '단백질': item.get('단백질', 0),
                                    '지방': item.get('지방', 0),
                                    '탄수화물': item.get('탄수화물', 0),
                                    '식이섬유': item.get('식이섬유', 0),
                                    '칼슘': item.get('칼슘', 0),
                                    '철': item.get('철', 0),
                                    '나트륨': item.get('나트륨', 0),
                                    '칼륨': item.get('칼륨', 0),
                                    '마그네슘': item.get('마그네슘', 0),
                                    '비타민b1': item.get('비타민b1', 0),
                                    '비타민b2': item.get('비타민b2', 0),
                                    '비타민b6': item.get('비타민b6', 0),
                                    '비타민c': item.get('비타민c', 0),
                                    '비타민e': item.get('비타민e', 0),
                                    '엽산': item.get('엽산', 0)
                                }
                                structured_data['nutrition_data'].append(structured_item)

                        # 가격 데이터 추출
                        if 'price_data' in data and data['price_data']:
                            for item in data['price_data']:
                                structured_item = {
                                    '품목명': item.get('product_cls_name', 'N/A'),
                                    '카테고리': item.get('category_name', 'N/A'),
                                    '가격': item.get('value', 0),
                                    '단위': item.get('unit', 'kg'),
                                    '날짜': item.get('regday', 'N/A')
                                }
                                structured_data['price_data'].append(structured_item)

            # 디버깅을 위한 로그
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"- PostgreSQL 결과 파싱 오류: {e}")
                # 파싱 실패 시 원본 내용을 other_data에 추가
                structured_data['other_data'].append({
                    'source': result.source,
                    'content': content[:500] + "..." if len(content) > 500 else content
                })

        return structured_data


    async def _create_draft(self, original_query: str, all_results: list, structured_data: dict) -> str:
        """수집된 정보를 바탕으로 자연스러운 문장의 초안을 작성 - PostgreSQL 데이터 우선 활용"""

        # 1. PostgreSQL 구조화 데이터를 우선 처리
        postgresql_summary = self._format_postgresql_data(structured_data)

        # 2. 다른 검색 결과 요약 (PostgreSQL 제외)
        other_results_summary = ""
        non_postgresql_count = 0
        for result in all_results[:10]:  # 최대 10개까지
            if hasattr(result, 'source') and result.source == 'rdb':
                continue  # PostgreSQL 결과는 이미 구조화해서 처리했으므로 스킵

            source_name = getattr(result, 'source', 'Unknown')
            content = getattr(result, 'content', str(result))
            other_results_summary += f"- 출처({source_name}): {content[:300]}...\n"
            non_postgresql_count += 1

        prompt = f"""
        당신은 여러 소스에서 수집된 복잡한 정보들을 종합하여, 사용자의 질문에 대한 답변 '초안'을 작성하는 수석 분석가입니다.

        ### 중요: PostgreSQL 농진청 데이터 최우선 활용

        **[원본 질문]**
        {original_query}

        **[1순위: PostgreSQL 구조화 데이터 - 반드시 우선 활용]**
        {postgresql_summary}

        **[2순위: 기타 검색 결과 - 보완적 활용]**
        {other_results_summary if other_results_summary else "기타 검색 결과 없음"}

        ### 작업 지침:
        1. **PostgreSQL 농진청 데이터를 반드시 최우선으로 활용**하여 답변 작성
        2. 영양소 데이터가 있으면 정확한 수치(칼로리, 단백질 등)를 반드시 포함
        3. 가격 데이터가 있으면 최신 시세 정보를 포함
        4. 농진청 출처 데이터는 반드시 "(출처: 농진청 'XX, RDB)" 형태로 명시
        5. 기타 검색 결과는 PostgreSQL 데이터를 보완하는 용도로만 사용
        6. 서론, 본론, 결론의 구조를 갖춘 자연스러운 설명글 형식 작성
        7. 실제 수치가 있으면 절대 다른 수치로 대체하지 말 것

        ### 금지사항:
        - PostgreSQL에 정확한 농진청 데이터가 있는데 다른 수치 사용 금지
        - 일반적인 해외 데이터(USDA 등)를 농진청 데이터보다 우선 사용 금지
        - 추정치나 임의 수치를 실제 데이터 대신 사용 금지

        **[답변 초안 작성]**
        """

        response = await self.chat.ainvoke(prompt)
        return response.content

    def _format_postgresql_data(self, structured_data: dict) -> str:
        """PostgreSQL 구조화 데이터를 읽기 쉬운 형태로 포맷팅"""
        formatted = ""

        # 영양소 데이터 포맷팅
        nutrition_data = structured_data.get('nutrition_data', [])
        if nutrition_data:
            formatted += "### 농진청 영양소 데이터 (PostgreSQL):\n"
            for item in nutrition_data:
                formatted += f"**{item['식품명']}** ({item['식품군']})\n"
                formatted += f"- 출처: {item['출처']}\n"
                formatted += f"- 칼로리: {item['칼로리']}kcal/100g\n"
                formatted += f"- 단백질: {item['단백질']}g/100g\n"
                formatted += f"- 지방: {item['지방']}g/100g\n"
                formatted += f"- 탄수화물: {item['탄수화물']}g/100g\n"
                formatted += f"- 식이섬유: {item['식이섬유']}g/100g\n"

                # 미네랄 정보가 있으면 추가
                if item['칼슘'] or item['철'] or item['마그네슘']:
                    formatted += f"- 칼슘: {item['칼슘']}mg, 철: {item['철']}mg, 마그네슘: {item['마그네슘']}mg\n"

                # 비타민 정보가 있으면 추가
                if item['비타민b1'] or item['비타민b2'] or item['비타민e']:
                    formatted += f"- 비타민B1: {item['비타민b1']}mg, 비타민B2: {item['비타민b2']}mg, 비타민E: {item['비타민e']}mg\n"

                formatted += "\n"

        # 가격 데이터 포맷팅
        price_data = structured_data.get('price_data', [])
        if price_data:
            formatted += "### 농수산물 가격 데이터 (PostgreSQL):\n"
            for item in price_data:
                formatted += f"**{item['품목명']}** ({item['카테고리']})\n"
                formatted += f"- 가격: {item['가격']}원/{item['단위']}\n"
                formatted += f"- 날짜: {item['날짜']}\n\n"

        # 기타 데이터
        other_data = structured_data.get('other_data', [])
        if other_data:
            formatted += "### 기타 RDB 데이터:\n"
            for item in other_data:
                formatted += f"- {item['content']}\n"

        if not formatted:
            formatted = "PostgreSQL 구조화 데이터 없음\n"

        return formatted



# 리팩토링 관련 임포트(참고용)
from ...core.config.report_config import TeamType, ReportType, Language
from ...services.templates.report_templates import ReportTemplateManager
from ...services.builders.prompt_builder import PromptBuilder
from ...utils.analyzers.query_analyzer import QueryAnalyzer


class ReportGeneratorAgent:
    """보고서 생성 에이전트"""

    def __init__(self):
        self.streaming_chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.9, streaming=True)
        self.non_streaming_chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self.agent_type = "REPORT_GENERATOR"

        self.template_manager = ReportTemplateManager()
        self.prompt_builder = PromptBuilder(self.template_manager)
        self.data_extractor = DataExtractor()

    async def generate_streaming_with_sources(
        self,
        state: StreamingAgentState,
        source_collection_data: Dict = None
    ) -> AsyncGenerator[str, None]:
        """스트리밍 보고서 생성 - 메인 로직"""

        print("\n>> REFACTORED REPORT_GENERATOR 시작")

        # 기본 정보 추출
        integrated_context = state.integrated_context
        original_query = state.original_query
        memory_context = getattr(state, "memory_context", "")
        user_context = getattr(state, "user_context", None)

        print(f"- 쿼리: {original_query[:50]}...")
        print(f"- 컨텍스트: {len(integrated_context)}자")

        if not integrated_context:
            error_msg = "분석할 충분한 정보가 수집되지 않았습니다."
            state.final_answer = error_msg
            yield error_msg
            return

        # 1. 쿼리 분석
        team_type = QueryAnalyzer.detect_team_type(original_query)
        language = QueryAnalyzer.detect_language(original_query)
        complexity_analysis = QueryAnalyzer.analyze_complexity(original_query, user_context)
        report_type = complexity_analysis["report_type"]

        print(f"- 분석 결과: {team_type.value} / {report_type.value} / {language.value}")

        # 2. 실제 데이터 추출
        all_results = getattr(state, 'graph_results_stream', []) + getattr(state, 'multi_source_results_stream', [])
        extracted_data = await self.data_extractor.extract_numerical_data(all_results, original_query)

        print(f"- 추출된 수치: {len(extracted_data.get('extracted_numbers', []))}개")

        # 3. 차트 생성
        real_charts = await self._create_data_driven_charts(extracted_data, original_query)
        print(f"- 생성된 차트: {len(real_charts)}개")

        # 4. 프롬프트 생성
        prompt = self.prompt_builder.build_prompt(
            query=original_query,
            context=integrated_context,
            team_type=team_type,
            report_type=report_type,
            language=language,
            extracted_data=extracted_data,
            real_charts=real_charts,
            source_data=source_collection_data
        )

        # 5. 스트리밍 생성
        full_response = ""
        try:
            print("- 스트리밍 시작...")
            async for chunk in self.streaming_chat.astream(prompt):
                if chunk.content:
                    full_response += chunk.content
                    yield chunk.content
        except Exception as e:
            error_msg = f"답변 생성 중 오류: {str(e)}"
            yield error_msg
            full_response = error_msg

        state.final_answer = full_response
        print(f"- 스트리밍 완료 (총 {len(full_response)}자)")

    async def _create_data_driven_charts(self, extracted_data: Dict, query: str) -> List[Dict]:
        """실제 데이터 기반 차트 생성 - 데이터 검증 로직 강화"""
        charts = []

        print(f"\n>> 차트 생성 시작")
        print(f"- 추출된 데이터: {extracted_data}")

        if not extracted_data:
            print("- 추출된 데이터 없음, 빈 차트 리스트 반환")
            return charts

        # 1. 퍼센트 데이터 -> 파이 차트
        percentages = extracted_data.get('percentages', [])
        if len(percentages) >= 2:
            print(f"- 퍼센트 데이터 발견: {len(percentages)}개")
            # 데이터 검증
            valid_percentages = [
                p for p in percentages
                if isinstance(p.get('value'), (int, float)) and 0 <= p.get('value') <= 100
            ]

            if len(valid_percentages) >= 2:
                labels = []
                values = []

                for p in valid_percentages[:5]:
                    context = p.get('context', '항목')
                    if len(context) > 20:
                        context = context[:20] + "..."
                    labels.append(context)
                    values.append(float(p['value']))

                chart = {
                    "title": f"{query[:30]}... 비율 분석 (실제 데이터)",
                    "type": "pie",
                    "data": {
                        "labels": labels,
                        "datasets": [{"label": "비율 (%)", "data": values}]
                    },
                    "source": "실제 추출 데이터",
                    "data_type": "real"
                }
                charts.append(chart)
                print(f"- 파이 차트 생성 완료: {chart['title']}")

        # 2. 일반 수치 데이터 -> 바 차트
        numbers = extracted_data.get('extracted_numbers', [])
        if len(numbers) >= 2:
            print(f"- 수치 데이터 발견: {len(numbers)}개")
            # 데이터 검증
            valid_numbers = [
                n for n in numbers
                if isinstance(n.get('value'), (int, float))
            ]

            if len(valid_numbers) >= 2:
                labels = []
                values = []
                units = []

                for n in valid_numbers[:5]:
                    context = n.get('context', '항목')
                    if len(context) > 15:
                        context = context[:15] + "..."
                    labels.append(context)
                    values.append(float(n['value']))
                    units.append(n.get('unit', ''))

                # 단위 통일 (첫 번째 단위 사용)
                primary_unit = units[0] if units[0] else '단위'

                chart = {
                    "title": f"{query[:30]}... 주요 수치 (실제 데이터)",
                    "type": "bar",
                    "data": {
                        "labels": labels,
                        "datasets": [{"label": f"수치 ({primary_unit})", "data": values}]
                    },
                    "source": "실제 추출 데이터",
                    "data_type": "real"
                }
                charts.append(chart)
                print(f"- 바 차트 생성 완료: {chart['title']}")

        # 3. 트렌드 데이터 -> 라인 차트
        trends = extracted_data.get('trends', [])
        if len(trends) >= 2:
            print(f"- 트렌드 데이터 발견: {len(trends)}개")

            labels = []
            values = []

            for t in trends[:6]:
                period = t.get('period', '기간')
                labels.append(period)

                # 변화율 추출
                change_str = str(t.get('change', '0'))
                import re
                numbers = re.findall(r'-?\d+\.?\d*', change_str)
                change_value = float(numbers[0]) if numbers else 0
                values.append(change_value)

            if len(values) >= 2:
                chart = {
                    "title": f"{query[:30]}... 시간별 변화 추이 (실제 데이터)",
                    "type": "line",
                    "data": {
                        "labels": labels,
                        "datasets": [{"label": "변화율 (%)", "data": values}]
                    },
                    "source": "실제 추출 데이터",
                    "data_type": "real"
                }
                charts.append(chart)
                print(f"- 라인 차트 생성 완료: {chart['title']}")

        print(f"- 총 {len(charts)}개 차트 생성 완료")
        return charts

    def _validate_chart_data(self, chart: Dict) -> bool:
        """차트 데이터 유효성 검증"""
        try:
            # 필수 필드 확인
            required_fields = ['title', 'type', 'data', 'source', 'data_type']
            for field in required_fields:
                if field not in chart:
                    print(f"- 차트 검증 실패: {field} 필드 없음")
                    return False

            # 데이터 구조 확인
            data = chart['data']
            if 'labels' not in data or 'datasets' not in data:
                print(f"- 차트 검증 실패: 데이터 구조 오류")
                return False

            # 데이터 길이 확인
            labels = data['labels']
            datasets = data['datasets']

            if not labels or not datasets:
                print(f"- 차트 검증 실패: 빈 데이터")
                return False

            # 첫 번째 데이터셋의 데이터 길이와 라벨 길이 일치 확인
            if len(datasets) > 0 and 'data' in datasets[0]:
                if len(labels) != len(datasets[0]['data']):
                    print(f"- 차트 검증 실패: 라벨과 데이터 길이 불일치")
                    return False

            # JSON 직렬화 가능 확인
            json.dumps(chart, ensure_ascii=False)

            print(f"- 차트 검증 성공: {chart['title']}")
            return True

        except Exception as e:
            print(f"- 차트 검증 실패: {str(e)}")
            return False

class SimpleAnswererAgent:
    """단순 질문 전용 Agent - 메모리 컨텍스트 지원"""

    def __init__(self, vector_db=None):
        self.vector_db = vector_db
        self.streaming_chat = ChatOpenAI(
            model="gpt-4o-mini", temperature=0.9, streaming=True
        )
        self.agent_type = "SIMPLE_ANSWERER"

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
   - 차트 생성 후에는 해당 차트에 대한 설명과 주요 내용을 마크다운 ('>')를 사용하여 작성해주세요.(예시: > 이 차트는 각 캠페인의 예상 ROI를 보여줍니다. 추정 데이터 기반으로 경상북도가 집중해야 할 캠페인 전략을 시사합니다.)

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
