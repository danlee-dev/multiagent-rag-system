import os
import re
import requests
import random
import time
import json
import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain import hub

from models import (
    AgentType,
    DatabaseType,
    QueryPlan,
    AgentMessage,
    MessageType,
    SearchResult,
    CriticResult,
    StreamingAgentState,
)
from utils import create_agent_message

from search_tools import (
    debug_web_search,
    mock_rdb_search,
    mock_vector_search,
    mock_graph_db_search,
)


# PlanningAgent: ToT(Tree of Thoughts) 적용된 쿼리 분석 및 작업 플랜 수립 담당
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from models import StreamingAgentState, QueryPlan, AgentType
import re


class PlanningAgent:
    """
    LLM 기반 지능적 질문 분석 및 실행 계획 수립 에이전트
    - 규칙 기반이 아닌 LLM의 추론 능력을 활용한 동적 분석
    - 질문 의도, 복잡도, 작업 범위를 종합적으로 판단
    - ToT(Tree of Thoughts) 원리 적용으로 최적의 실행 계획 수립
    """

    def __init__(self):
        self.chat = ChatOpenAI(model="gpt-4o", temperature=0)
        self.agent_type = AgentType.PLANNING

    async def plan(self, state: StreamingAgentState) -> StreamingAgentState:
        """질문을 분석하고, 지능적으로 최적의 실행 계획을 수립합니다."""
        print(">> PLANNING 단계 시작 (LLM 기반 지능적 분석)")
        query = state.original_query
        print(f"- 원본 쿼리: {query}")

        # 이전 단계 피드백 수집
        feedback_context = self._collect_feedback(state)
        if feedback_context:
            print(f"- 피드백 반영: {feedback_context}")

        # 1. 질문의 종합적 분석 (의도, 복잡도, 작업 범위)
        analysis_result = await self._analyze_query_comprehensively(
            query, feedback_context
        )
        print(f"- 질문 분석 결과: {analysis_result}")

        # 2. 분석 결과에 따른 실행 계획 수립
        if analysis_result["complexity"] == "SIMPLE":
            # 단순 질문: 직접 처리
            state.query_plan = QueryPlan(
                original_query=query, sub_queries=[query], estimated_complexity="low"
            )
            print("- 단순 질문으로 판단: 직접 처리")
        else:
            # 복잡 질문: ToT 기반 계획 수립
            execution_plan = await self._create_execution_plan_with_tot(
                query, analysis_result, feedback_context
            )
            state.query_plan = QueryPlan(
                original_query=query,
                sub_queries=[execution_plan],
                estimated_complexity="high",
            )
            print(f"- 복잡 질문으로 판단: 실행 계획 수립 완료")

        state.planning_complete = True
        print(">> PLANNING 단계 완료")
        return state

    def _collect_feedback(self, state: StreamingAgentState) -> Optional[str]:
        """이전 단계의 피드백을 수집합니다."""
        if state.critic2_result and state.critic2_result.status == "insufficient":
            return f"최종 검수 피드백: {state.critic2_result.suggestion}"
        elif state.critic1_result and state.critic1_result.status == "insufficient":
            return f"초기 수집 피드백: {state.critic1_result.suggestion}"
        return None

    async def _analyze_query_comprehensively(
        self, query: str, feedback: Optional[str] = None
    ) -> Dict:
        """질문을 종합적으로 분석하여 의도, 복잡도, 작업 범위를 판단합니다."""

        feedback_section = ""
        if feedback:
            feedback_section = f"""
## 이전 시도 피드백
{feedback}

위 피드백을 고려하여 분석해주세요.
"""

        prompt = f"""당신은 세계 최고 수준의 AI 시스템 아키텍트입니다. 사용자의 질문을 정확히 분석하여 최적의 처리 방식을 결정해야 합니다.

{{feedback_section}}

## 분석 대상 질문
"{query}"

## 분석 기준

### 1. 질문 의도 (INTENT)
- CAPABILITY_INQUIRY: AI 능력이나 기능에 대한 문의
  예: "뭘 할 수 있어?", "어떤 도움을 줄 수 있나요?", "가능한 서비스는?"

- INFORMATION_REQUEST: 특정 정보나 지식에 대한 단순 요청
  예: "아마란스가 뭐야?", "칼로리 알려줘", "영양성분은?"

- TASK_REQUEST: 구체적인 작업이나 분석 수행 요청
  예: "시장 분석해줘", "전략 수립해줘", "보고서 작성해줘"

### 2. 복잡도 (COMPLEXITY)
- SIMPLE: 단일 정보 제공이나 간단한 설명으로 충분한 질문
  - 사전적 정의, 기본 정보, 단순 계산
  - 1-2개 문장으로 답변 가능
  - 추가 리서치나 다단계 분석 불필요

- COMPLEX: 다단계 분석, 종합적 판단, 전략적 사고가 필요한 질문
  - 시장 분석, 경쟁사 비교, 트렌드 예측
  - 전략 수립, 계획 작성, 종합 보고서
  - 여러 관점의 정보 수집과 통합 분석 필요

### 3. 작업 범위 (SCOPE)
- NARROW: 특정 주제나 영역에 국한된 질문
- BROAD: 여러 영역을 아우르는 종합적 질문
- STRATEGIC: 전략적 의사결정이나 장기적 관점이 필요한 질문

## 출력 형식
다음 JSON 형식으로만 응답하세요:

```json
{{{{
  "intent": "CAPABILITY_INQUIRY|INFORMATION_REQUEST|TASK_REQUEST",
  "complexity": "SIMPLE|COMPLEX",
  "scope": "NARROW|BROAD|STRATEGIC",
  "reasoning": "판단 근거를 2-3문장으로 설명",
  "requires_research": true|false,
  "estimated_response_length": "short|medium|long"
}}}}
```

정확한 JSON 형식을 준수하고, 질문의 진짜 의도를 파악하여 판단하세요.""".format(
            feedback_section=feedback_section
        )

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
            print(f"분석 파싱 오류: {e}")
            # 폴백: 기본값 반환
            return {
                "intent": "INFORMATION_REQUEST",
                "complexity": "SIMPLE",
                "scope": "NARROW",
                "reasoning": "분석 중 오류 발생으로 기본값 적용",
                "requires_research": False,
                "estimated_response_length": "short",
            }

    async def _create_execution_plan_with_tot(
        self, query: str, analysis: Dict, feedback: Optional[str] = None
    ) -> str:
        """ToT 기반으로 복잡한 질문에 대한 최적의 실행 계획을 수립합니다."""

        # 1. 다양한 관점의 실행 계획 생성
        alternative_plans = await self._generate_strategic_plans(
            query, analysis, feedback
        )
        print(f"\n>> 생성된 실행 계획 후보 ({len(alternative_plans)}개):")
        for i, plan in enumerate(alternative_plans, 1):
            print(f"   {i}. {plan[:100]}...")

        # 2. 최적 계획 선택
        if len(alternative_plans) == 1:
            return alternative_plans[0]

        best_plan = await self._select_optimal_plan(query, analysis, alternative_plans)
        print(f"\n- 선택된 최적 계획: {best_plan[:100]}...")

        return best_plan

    async def _generate_strategic_plans(
        self, query: str, analysis: Dict, feedback: Optional[str] = None
    ) -> List[str]:
        """전략적 관점에서 다양한 실행 계획을 생성합니다."""

        feedback_section = ""
        if feedback:
            feedback_section = f"""
## 중요: 이전 시도 피드백
{feedback}

위 피드백을 반드시 해결하는 새로운 접근법을 포함해야 합니다.
"""

        prompt = f"""당신은 McKinsey, BCG, Bain 출신의 세계 최고 전략 컨설턴트입니다.
클라이언트의 요청을 완벽히 해결하기 위한 3가지 차별화된 실행 계획을 수립해야 합니다.

{feedback_section}

## 클라이언트 요청
"{query}"

## 요청 분석 결과
- 의도: {analysis.get('intent', 'TASK_REQUEST')}
- 복잡도: {analysis.get('complexity', 'COMPLEX')}
- 범위: {analysis.get('scope', 'BROAD')}
- 판단 근거: {analysis.get('reasoning', '')}

## 실행 계획 수립 지침

### 품질 기준
1. **완전성**: 클라이언트 요청의 모든 핵심 요소를 빠짐없이 포함
2. **실행 가능성**: 구체적이고 명확한 단계별 실행 방안
3. **차별화**: 각 계획은 서로 다른 접근법과 관점 반영
4. **결과 지향**: 고품질의 최종 산출물 창출 가능

### 접근 관점
- **관점 1**: 데이터 중심 분석적 접근 (정량적 분석, 시장 데이터, 통계적 검증)
- **관점 2**: 전략적 컨설팅 접근 (프레임워크 활용, 경쟁 분석, 포지셔닝)
- **관점 3**: 혁신적 실무 접근 (창의적 솔루션, 실무 경험, 실행 중심)

## 출력 형식
각 계획은 완전한 실행 지시문이 되어야 하며, 다음 형식으로 작성하세요:

**계획 1 (데이터 분석 중심):**
[구체적인 실행 계획 - 200-300자]

**계획 2 (전략 컨설팅 중심):**
[구체적인 실행 계획 - 200-300자]

**계획 3 (혁신 실무 중심):**
[구체적인 실행 계획 - 200-300자]

각 계획은 후속 AI 에이전트가 그대로 실행할 수 있을 정도로 구체적이어야 합니다."""

        try:
            response = await self.chat.ainvoke(prompt)

            # 계획들을 파싱
            content = response.content
            plans = []

            # **계획 X**로 시작하는 섹션들을 찾아서 추출
            import re

            pattern = r"\*\*계획 \d+[^:]*:\*\*\s*\n?(.*?)(?=\*\*계획 \d+|$)"
            matches = re.findall(pattern, content, re.DOTALL)

            for match in matches:
                plan = match.strip()
                if plan and len(plan) > 50:  # 최소 길이 체크
                    plans.append(plan)

            # 파싱 실패 시 폴백
            if not plans:
                lines = [line.strip() for line in content.split("\n") if line.strip()]
                plans = [line for line in lines if len(line) > 100][:3]

            # 최소 1개는 보장
            if not plans:
                plans = [
                    f"클라이언트 요청 '{query}'에 대한 종합적 분석과 실행 방안을 수립하여 완성도 높은 결과물을 제공합니다."
                ]

            return plans[:3]  # 최대 3개만 반환

        except Exception as e:
            print(f"계획 생성 오류: {e}")
            # 폴백 계획
            return [f"'{query}' 요청에 대한 체계적 분석과 전략적 솔루션을 제공합니다."]

    async def _select_optimal_plan(
        self, query: str, analysis: Dict, plans: List[str]
    ) -> str:
        """생성된 실행 계획들 중 최적의 계획을 선택합니다."""

        if not plans:
            return f"'{query}' 요청에 대한 종합적 분석과 솔루션을 제공합니다."

        if len(plans) == 1:
            return plans[0]

        plans_text = "\n\n".join(
            [f"**후보 {i+1}:**\n{plan}" for i, plan in enumerate(plans)]
        )

        prompt = f"""당신은 세계 최고 수준의 AI 시스템 아키텍트이자 의사결정 전문가입니다.
여러 실행 계획 중에서 클라이언트에게 최고의 가치를 제공할 수 있는 최적의 계획을 선택해야 합니다.

## 클라이언트 요청
"{query}"

## 요청 분석 컨텍스트
- 의도: {analysis.get('intent')}
- 복잡도: {analysis.get('complexity')}
- 범위: {analysis.get('scope')}
- 리서치 필요: {analysis.get('requires_research')}

## 실행 계획 후보들
{plans_text}

## 평가 기준 (우선순위 순)

### 1. 요구사항 충족도 (40%)
- 클라이언트 요청의 모든 핵심 요소가 포함되었는가?
- 요청의 진짜 의도와 목적을 정확히 이해했는가?

### 2. 실행 가능성 (30%)
- 계획이 구체적이고 명확한가?
- 후속 AI 에이전트가 바로 실행할 수 있는 수준인가?
- 필요한 정보 수집과 분석 단계가 체계적인가?

### 3. 결과 품질 잠재력 (20%)
- 최종 산출물의 완성도와 통찰력이 높을 것으로 예상되는가?
- 클라이언트에게 실질적 가치를 제공할 수 있는가?

### 4. 효율성 (10%)
- 불필요한 중복이나 비효율적 단계가 없는가?
- 적절한 수준의 깊이와 범위를 가지는가?

## 선택 결과

위 평가 기준에 따라 종합적으로 판단하여, 가장 우수한 계획의 **전체 내용을 그대로** 반환하세요.
선택 이유나 부가 설명 없이, 선택된 계획의 내용만 정확히 출력하세요."""

        try:
            response = await self.chat.ainvoke(prompt)
            selected_plan = response.content.strip()

            # 혹시 "후보 X:" 같은 표시가 있다면 제거
            import re

            selected_plan = re.sub(r"^\*\*후보 \d+:\*\*\s*\n?", "", selected_plan)

            return selected_plan

        except Exception as e:
            print(f"계획 선택 오류: {e}")
            # 폴백: 첫 번째 계획 반환
            return plans[0]


# RetrieverAgentX: Graph DB 중심 검색 + 피드백
class RetrieverAgentX:
    def __init__(self, graph_db):
        self.graph_db = graph_db
        self.chat = ChatOpenAI(model="gpt-3.5-turbo")
        self.agent_type = AgentType.RETRIEVER_X

    async def search(self, state: StreamingAgentState) -> StreamingAgentState:
        print(">> RETRIEVER_X (단일 쿼리 모드) 시작")

        # PlanningAgent가 선택한 단 하나의 최적 쿼리를 가져옵니다.
        if not state.query_plan or not state.query_plan.sub_queries:
            print("- 처리할 쿼리가 없어 RETRIEVER_X를 종료합니다.")
            return state

        single_query = state.query_plan.sub_queries[0]

        all_graph_results = []
        extracted_hints_for_integration = []

        keywords = await self._optimize_keywords(single_query)
        print(f"- Graph DB 키워드: {keywords}")

        for keyword in keywords[:2]:
            graph_result = self.graph_db.search(keyword)

            for node in graph_result["nodes"]:
                search_result = SearchResult(
                    source="graph_db",
                    content=f"{node['properties'].get('name', 'Unknown')}: {str(node['properties'])}",
                    relevance_score=random.uniform(0.7, 0.95),
                    metadata=node,
                    search_query=keyword,
                )
                all_graph_results.append(search_result)

                if "properties" in node:
                    props = node["properties"]
                    hint_data = {
                        "entity": props.get("name", ""),
                        "category": props.get("category", ""),
                        "search_query": keyword,
                    }
                    extracted_hints_for_integration.append(hint_data)

        for result in all_graph_results:
            state.add_graph_result(result)

        state.x_extracted_hints = extracted_hints_for_integration
        print(
            f"- State에 Graph DB 결과 {len(all_graph_results)}개 및 힌트 {len(extracted_hints_for_integration)}개 저장"
        )

        memory = state.get_agent_memory(AgentType.RETRIEVER_X)
        memory.add_finding(f"Graph DB 검색 완료: {len(all_graph_results)}개 결과")

        print(">> RETRIEVER_X 완료")
        return state

    async def _optimize_keywords(self, query):
        prompt = f"""
        당신은 Graph Database 검색을 위한 핵심 키워드를 추출하는 전문가입니다.
        주어진 질문에서 Graph DB에 저장된 식품재료, 가격정보, 트렌드, 뉴스 등의 정보를 검색하는 데 가장 효율적인 키워드를 식별합니다.

        ---
        **키워드 추출 지침:**
        1.  원본 질문의 주요 명사, 동사, 그리고 핵심 개념을 파악합니다.
        2.  Graph DB의 데이터 유형(식품재료, 가격정보, 트렌드, 뉴스)을 고려하여 관련성 높은 키워드를 선택합니다.
        3.  최대 2-3개의 가장 중요한 키워드를 선정합니다.
        ---

        <질문>
        {query}
        </질문>

        **출력 형식:**
        추출된 키워드들을 쉼표(,)로 구분하여 답변해주세요. 다른 설명은 포함하지 마세요.
        예시: `쌀, 영양성분, 효능`

        답변:
        """
        response = await self.chat.ainvoke(prompt)
        keywords = [kw.strip() for kw in response.content.split(",")]
        return keywords[:3]


# RetrieverAgentY: Multi-Source 검색 + 피드백
import asyncio
import time
from typing import Dict, List, Set
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain import hub
from models import StreamingAgentState, SearchResult
from search_tools import (
    debug_web_search,
    mock_vector_search,
    mock_rdb_search,
    mock_graph_db_search,
)


import time
import asyncio
from datetime import datetime
from typing import Set, Dict
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate


import time
import asyncio
from datetime import datetime
from typing import Set, Dict
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate


class RetrieverAgentY:
    def __init__(self, vector_db=None, rdb=None):
        self.vector_db = vector_db
        self.rdb = rdb
        self.available_tools = [
            debug_web_search,
            mock_vector_search,
            mock_rdb_search,
            mock_graph_db_search,
        ]
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 무한루프 방지를 위한 추적 시스템
        self.search_history: Set[str] = set()
        self.tool_usage_count: Dict[str, int] = {}
        self.max_tool_usage = 2
        self.max_total_attempts = 5
        self.current_attempts = 0

        # 날짜 정보 설정
        self.current_date = datetime.now()
        self.current_date_str = self.current_date.strftime("%Y년 %m월 %d일")
        self.current_year = self.current_date.year
        self.current_date_en = self.current_date.strftime("%B %d, %Y")

        # ReAct 에이전트 설정 - 날짜 정보 포함
        self.react_agent_executor = self._create_improved_react_agent()

    def _create_improved_react_agent(self):
        """개선된 ReAct 에이전트 생성 - 날짜 정보 및 무한루프 방지 기능 포함"""
        try:
            base_prompt = hub.pull("hwchase17/react")

            # LLM 기반 지능적 시간 맥락 판단이 포함된 시스템 프롬프트
            system_instruction = f"""
You are an intelligent research assistant with advanced temporal reasoning capabilities and strict anti-loop protection.

AVAILABLE TOOLS - USE EXACT NAMES:
- debug_web_search: For web searches
- mock_vector_search: For vector database searches
- mock_rdb_search: For relational database searches
- mock_graph_db_search: For graph database searches

MANDATORY ACTION FORMAT:
Action: [EXACT_TOOL_NAME]
Action Input: "search query"

CORRECT EXAMPLES:
Action: debug_web_search
Action Input: "Tesla stock price 2024"

Action: mock_vector_search
Action Input: "quinoa market trends"

WRONG EXAMPLES (NEVER DO THIS):
Action: I will search for Tesla stock price
Action Input: "Tesla stock price"

Action: Let me find information about
Action Input: "search query"

TEMPORAL AWARENESS CONTEXT:
- Today's Date: {self.current_date_str} ({self.current_date_en})
- Current Year: {self.current_year}
- Use LLM reasoning to determine temporal intent, not keyword matching

TEMPORAL INTENT ANALYSIS:
Before starting any search, analyze the query to determine:
1. CURRENT/RECENT INTENT: User wants latest, up-to-date information
2. HISTORICAL INTENT: User asks about specific past time periods
3. GENERAL INTENT: User wants comprehensive information regardless of time

CRITICAL RULES:
1. ALWAYS use exact tool names in Action field
2. INTELLIGENT TEMPORAL DETECTION: Use LLM reasoning to understand query intent
3. NO REPETITION: Never use the exact same search query twice
4. TOOL DIVERSITY: If one tool fails, try a different tool type
5. KEYWORD EVOLUTION: Each search must use different keywords/approach
6. 3-STRIKE RULE: After 3 failed attempts, provide answer with available info

SEARCH STRATEGY PROGRESSION:
- Step 1: Determine temporal intent using LLM reasoning
- Step 2: Use debug_web_search with keywords appropriate to detected temporal context
- Step 3: Try mock_vector_search with alternative keywords within same temporal context
- Step 4: Try mock_rdb_search or mock_graph_db_search with broader terms
- Step 5: Synthesize available information and conclude

TEMPORAL CONTEXT EXAMPLES:
For CURRENT Intent queries:
Action: debug_web_search
Action Input: "Tesla stock price July 2025 current"

For HISTORICAL Intent queries:
Action: debug_web_search
Action Input: "Tesla stock price 2020 COVID impact"

For GENERAL Intent queries:
Action: mock_vector_search
Action Input: "Tesla company information general"

EFFICIENCY RULES:
- Use exact tool names only in Action field
- Match search strategy to detected intent context
- If search yields insufficient results, try different tool with different keywords
- Always explain your temporal intent detection and search strategy changes
- Focus on insights appropriate to the detected temporal context
- Clearly state temporal scope and data limitations when relevant

Remember: Always use EXACT tool names in Action field!
System reference date: {self.current_date_str}
"""

            new_prompt_template = system_instruction + "\n\n" + base_prompt.template
            react_prompt = PromptTemplate.from_template(new_prompt_template)

            react_agent_runnable = create_react_agent(
                self.llm, self.available_tools, react_prompt
            )

            return AgentExecutor(
                agent=react_agent_runnable,
                tools=self.available_tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5,
                max_execution_time=60,
            )

        except Exception as e:
            print(f">> ReAct 에이전트 초기화 실패: {e}")
            return None

    async def search(self, state: StreamingAgentState) -> StreamingAgentState:
        """개선된 검색 로직 - 날짜 정보 및 무한루프 방지 포함"""
        print(">> RETRIEVER_Y 시작 (날짜 정보 포함 + 무한루프 방지 버전)")
        print(f"- 현재 날짜: {self.current_date_str}")
        print(f"- 분석 기준년도: {self.current_year}")

        if not self.react_agent_executor:
            print("- ReAct 에이전트가 초기화되지 않아 검색을 건너뜁니다.")
            return state

        if not state.query_plan or not state.query_plan.sub_queries:
            print("- 처리할 쿼리가 없어 RETRIEVER_Y를 종료합니다.")
            return state

        original_query = state.query_plan.sub_queries[0]

        # 무한루프 방지: 타임아웃 설정
        start_time = time.time()
        timeout_seconds = 90

        print(f"- 원본 쿼리: {original_query}")
        print(f"- 최대 시도 횟수: {self.max_total_attempts}")
        print(f"- 타임아웃: {timeout_seconds}초")

        try:
            # LLM 기반 지능적 맥락 판단이 포함된 쿼리 구성
            enhanced_query_prompt = f"""
Research Query: {original_query}

SYSTEM DATE REFERENCE:
- Current Date: {self.current_date_str} ({self.current_date_en})
- Current Year: {self.current_year}

SEARCH INSTRUCTIONS:

1. TEMPORAL INTENT ANALYSIS:
First, use your LLM reasoning to determine the temporal intent of this query:
- Does the user want CURRENT/RECENT information?
- Does the user want HISTORICAL information from a specific time period?
- Does the user want GENERAL information regardless of time?

2. SEARCH STRATEGY BASED ON INTENT:
After determining intent, apply appropriate search strategy:
- For CURRENT intent: Include current year or recent timeframe in searches
- For HISTORICAL intent: Focus on the specific time period the user mentioned
- For GENERAL intent: Search for comprehensive information without temporal bias

3. ITERATION AND OPTIMIZATION:
- Never repeat the exact same search query
- If a search approach fails, try different keywords within the same temporal context
- After 3 search attempts, synthesize available information
- Always explain your temporal intent detection and strategy changes

4. RESULT QUALITY:
- Focus on insights appropriate to the detected temporal context
- Clearly state any temporal limitations of the data you find
- Provide actionable information within the appropriate time scope

Previous search attempts: {len(self.search_history)}
Session start: {self.current_date_str}

Begin by analyzing the temporal intent of the query, then proceed with appropriate searches.
"""

            # 타임아웃과 함께 비동기 실행
            result = await asyncio.wait_for(
                self.react_agent_executor.ainvoke({"input": enhanced_query_prompt}),
                timeout=timeout_seconds,
            )

            output = result.get("output", "No output")

            # 결과가 너무 짧거나 의미없으면 날짜 정보가 포함된 기본 정보 제공
            if len(output) < 100 or "did not yield" in output.lower():
                output = self._generate_fallback_analysis(original_query)
                print("- 검색 결과가 부족하여 날짜 정보가 포함된 기본 분석으로 대체")

            search_result = SearchResult(
                source="react_agent_with_date",
                content=output,
                relevance_score=0.8,
                metadata={
                    "query": original_query,
                    "search_attempts": len(self.search_history),
                    "execution_time": time.time() - start_time,
                    "analysis_date": self.current_date_str,
                    "reference_year": self.current_year,
                },
                search_query=original_query,
            )

            state.add_multi_source_result(search_result)
            print(f"- ReAct 검색 결과 추가: {len(output)}자")
            print(f"- 실행 시간: {time.time() - start_time:.2f}초")

        except asyncio.TimeoutError:
            print(f"- ReAct 검색 타임아웃 ({timeout_seconds}초 초과)")
            fallback_output = self._generate_fallback_analysis(original_query)

            search_result = SearchResult(
                source="timeout_fallback_with_date",
                content=fallback_output,
                relevance_score=0.6,
                metadata={
                    "query": original_query,
                    "reason": "timeout",
                    "analysis_date": self.current_date_str,
                    "reference_year": self.current_year,
                },
                search_query=original_query,
            )
            state.add_multi_source_result(search_result)

        except Exception as e:
            print(f"- ReAct 검색 실패: {e}")
            fallback_output = self._generate_fallback_analysis(original_query)

            search_result = SearchResult(
                source="error_fallback_with_date",
                content=fallback_output,
                relevance_score=0.5,
                metadata={
                    "query": original_query,
                    "error": str(e),
                    "analysis_date": self.current_date_str,
                    "reference_year": self.current_year,
                },
                search_query=original_query,
            )
            state.add_multi_source_result(search_result)

        print(">> RETRIEVER_Y 완료")
        return state

    def _generate_fallback_analysis(self, query: str) -> str:
        """검색 실패 시 날짜 정보가 포함된 기본 분석 제공"""
        return f"""
**분석 기준일: {self.current_date_str}**
**참조 연도: {self.current_year}년**

검색을 통해 구체적인 최신 데이터를 확보하지는 못했지만, {self.current_year}년 현재 시점의 일반적인 시장 동향을 기반으로 분석을 제공합니다:

## 쿼리: {query}

### 시장 현황 분석 ({self.current_year}년 기준)
- 퀴노아 시장은 2024년 111억 달러 규모로 평가되었으며, 2031년까지 268억 달러에 도달할 것으로 예상
- 연평균 성장률(CAGR) 12.90%의 높은 성장세 예상 ({self.current_year}년 현재 기준)
- 영양가 높고 편리한 식품에 대한 소비자 선호도 증가 추세 (최근 트렌드)

### 소비자 트렌드 인사이트 ({self.current_year}년 현재)
- 건강 의식 증가로 인한 슈퍼푸드 수요 확대
- 바로 먹을 수 있는(RTE) 제품에 대한 수요 증가
- 완전 단백질과 필수 아미노산에 대한 관심 증가
- 글루텐 프리 제품에 대한 선호도 상승

### 주요 성장 동력 (현재 시점 기준)
- 영양 밀도가 높은 식품에 대한 수요 증가
- 식물 기반 단백질에 대한 관심 확산
- 건강한 라이프스타일 추구 트렌드
- 아시아 태평양 지역의 급성장

### 마케팅 전략 제안 ({self.current_year}년 하반기 기준)
- MZ세대를 타겟으로 한 소셜미디어 마케팅 강화
- 완전 단백질과 영양 밀도를 강조한 포지셔닝
- 편의성과 건강함을 동시에 어필하는 제품 개발
- 글루텐 프리, 비건 등 특수 니즈 대응 제품 확대

**데이터 한계사항:**
- 위 분석은 {self.current_date_str} 현재 접근 가능한 일반적 시장 정보를 기반으로 함
- 최신 소비자 설문조사나 실시간 시장 데이터는 포함되지 않음
- 보다 정확한 분석을 위해서는 최신 시장 조사 데이터 확보 필요

이러한 {self.current_year}년 현재 기준의 전반적인 트렌드를 바탕으로 마케팅 전략을 수립하는 것을 권장합니다.
"""


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


import re
from typing import AsyncGenerator, Dict, List, Any
from langchain_openai import ChatOpenAI
from models import StreamingAgentState

# 차트 생성 지침
CHART_GENERATION_INSTRUCTIONS = """
## 차트 데이터 생성 지침

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
            model="gpt-4o-mini", temperature=0.6, streaming=True
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

중요: 위 정보는 이전 대화에서 나눈 내용입니다. 사용자가 이름을 알려줬다면 보고서에서 그 이름으로 언급하고,
이전에 관심을 보인 주제나 전문 분야가 있다면 해당 내용을 보고서에 반영해주세요.
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
            model="gpt-3.5-turbo", temperature=0.7, streaming=True
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
