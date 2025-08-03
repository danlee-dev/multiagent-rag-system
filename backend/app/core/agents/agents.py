# 표준 라이브러리
import asyncio
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Literal

# 서드파티 라이브러리
import requests

# LangChain 관련
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field

from typing import Dict, Optional, Literal

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
    vector_db_search,
    scrape_and_extract_content,
)
from ...utils.utils import create_agent_message

class DataExtractor:
    """검색 결과에서 실제 수치 데이터를 추출하는 클래스"""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    async def extract_numerical_data(self, search_results: List[SearchResult], query: str) -> Dict[str, Any]:
        """검색 결과에서 수치 데이터를 추출"""

        print(f"\n>> 수치 데이터 추출 시작")
        print(f"- 검색 결과 개수: {len(search_results)}")

        combined_text = ""
        for result in search_results:
            combined_text += f"{result.content}\n"

        print(f"- 결합된 텍스트 길이: {len(combined_text)}자")

        # 텍스트 길이 제한 (하지만 중요한 부분 보존)
        max_length = 8000  # 더 넉넉하게
        if len(combined_text) > max_length:
            # PostgreSQL 결과 우선 보존
            postgres_parts = []
            other_parts = []

            for result in search_results:
                if "PostgreSQL" in result.content or "가격 데이터" in result.content or "영양 정보" in result.content:
                    postgres_parts.append(result.content)
                else:
                    other_parts.append(result.content)

            # PostgreSQL 결과 전부 + 나머지 일부
            combined_text = "\n".join(postgres_parts)
            remaining_length = max_length - len(combined_text)

            if remaining_length > 0:
                other_text = "\n".join(other_parts)
                combined_text += "\n" + other_text[:remaining_length]

            print(f"- 텍스트 최적화 후 길이: {len(combined_text)}자")

        prompt = f"""
        다음 텍스트에서 숫자, 퍼센트, 통계 데이터를 추출하고 JSON 형태로 정리해주세요.

        원본 질문: {query}

        텍스트:
        {combined_text}

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

        **중요 지침:**
        1. PostgreSQL 검색 결과의 실제 수치를 최우선으로 추출
        2. "가격 데이터", "영양 정보" 섹션의 모든 숫자 추출
        3. 단위도 정확히 추출 (원, g, kcal, mg 등)
        4. 실제 숫자가 없으면 빈 배열이나 객체를 반환

        예시: "1,033원/100g" → {{"value": 1033, "unit": "원", "context": "100g당 가격"}}
        """

        try:
            print("- LLM 수치 추출 시작...")
            response = await self.llm.ainvoke(prompt)
            result = json.loads(response.content)

            print(f"- 추출된 수치: {len(result.get('extracted_numbers', []))}개")
            print(f"- 추출된 퍼센트: {len(result.get('percentages', []))}개")

            return result
        except Exception as e:
            print(f"- 수치 추출 실패: {e}")
            return {"extracted_numbers": [], "percentages": [], "trends": [], "categories": {}}


# ==============================================================================
# Pydantic 모델 정의: 안정적인 구조화를 위해 클래스 외부에 정의
# ==============================================================================
class ResourceRequirements(BaseModel):
    """실행 계획에 필요한 리소스 요구사항을 정의합니다."""
    search_needed: bool = Field(description="정보 검색이 필요한지 여부")
    react_needed: bool = Field(description="자체 추론 루프(Tool Calling)가 필요한지 여부")
    multi_agent_needed: bool = Field(description="다중 에이전트 협업이 필요한지 여부")
    estimated_time: Literal['fast', 'medium', 'slow', 'very_slow'] = Field(description="예상 소요 시간")

class ComplexityAnalysis(BaseModel):
    """사용자 질문의 복잡도 분석 결과를 담는 데이터 구조입니다."""
    complexity_level: Literal['SIMPLE', 'MEDIUM', 'COMPLEX', 'SUPER_COMPLEX'] = Field(description="분석된 복잡도 수준")
    execution_strategy: Literal['direct_answer', 'basic_search', 'full_react', 'multi_agent'] = Field(description="복잡도에 따른 실행 전략")
    reasoning: str = Field(description="복잡도를 판단한 근거를 2-3문장으로 설명")
    resource_requirements: ResourceRequirements = Field(description="필요한 리소스 요구사항")
    expected_output_type: Literal['simple_text', 'analysis', 'report', 'comprehensive_strategy'] = Field(description="예상되는 결과물 유형")

class DecomposedQuery(BaseModel):
    """사용자의 복잡한 질문을 해결 가능한 여러 개의 하위 질문으로 분해한 결과입니다."""
    sub_queries: List[str] = Field(
        description="분해된 2~5개의 간단하고 명확하며 독립적으로 검색 가능한 하위 질문 목록"
    )

class PlanningAgent:
    """
    섹션 기반 보고서 계획 수립 에이전트.
    질문을 분석하여 필요한 섹션을 나누고, 각 섹션별로 필요한 정보 수집 쿼리를 생성합니다.
    """

    def __init__(self):
        self.chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.agent_type = AgentType.PLANNING

    async def plan(self, state: StreamingAgentState) -> StreamingAgentState:
        """질문을 분석하여 섹션 기반 계획을 수립하고 정보 수집 쿼리를 생성"""
        print(">> PLANNING 단계 시작 (섹션 분할 및 쿼리 생성)")
        query = state.original_query
        print(f"- 원본 쿼리: {query}")

        # 1. 보고서 섹션 계획 수립
        print("- 섹션 계획 수립 시작...")
        section_plan = await self._create_section_plan(query)
        print(f"- 섹션 계획: {section_plan}")

        # 2. 섹션별 정보 수집 쿼리 생성
        sub_queries = []
        sections = section_plan.get('sections', [])

        if not sections:  # 섹션이 없으면 단순 질문으로 처리
            print("- 간단한 질문으로 판단, 원본 쿼리만 사용")
            sub_queries = [query]
        else:
            print("- 섹션별 정보 수집 쿼리 생성 시작...")
            for i, section in enumerate(sections):
                section_name = section.get('name', f'섹션 {i+1}')
                section_queries = await self._generate_section_queries(query, section)
                print(f"  * {section_name}: {len(section_queries)}개 쿼리 생성")
                sub_queries.extend(section_queries)

        print(f"- 총 {len(sub_queries)}개의 정보 수집 쿼리 생성 완료")

        # 3. 상태 업데이트
        state.query_plan = QueryPlan(
            original_query=query,
            sub_queries=sub_queries,
            estimated_complexity="adaptive",  # 섹션 기반으로 적응적 처리
            execution_strategy="section_based",
            resource_requirements={
                "sections": sections,
                "total_queries": len(sub_queries),
                "report_structure": section_plan
            },
        )

        state.planning_complete = True
        print(f">> PLANNING 단계 완료 - {len(sections)}개 섹션, {len(sub_queries)}개 쿼리")
        return state

    async def _create_section_plan(self, query: str) -> Dict:
        """질문을 분석하여 보고서에 필요한 섹션을 계획"""
        prompt = f"""당신은 비즈니스 보고서 구조 설계 전문가입니다.
사용자의 질문을 분석하여 완성도 높은 보고서를 작성하기 위해 필요한 섹션들을 계획해주세요.

## 원본 질문
"{query}"

## 섹션 계획 원칙
1. **유연성**: 질문의 복잡도와 범위에 따라 자연스럽게 섹션 수를 결정하고, 섹션이 필요하지 않을 수도 있음
2. **간단한 질문**: 특별한 섹션 구분이 필요 없으면 섹션 없음으로 처리
3. **복잡한 질문**: 논리적이고 체계적인 섹션으로 구성
4. **실용성**: 각 섹션이 명확한 목적과 가치를 가져야 함
5. **차별화**: 각 섹션은 독립적이고 중복되지 않도록 구성

## 질문 유형별 가이드

### 단순 정보 요청 (섹션 불필요)
- 예: "오늘 사과 가격은?", "김치의 영양성분은?"
- 처리: 섹션 없음, 직접 답변

### 분석 요청 (2-4개 섹션)
- 예: "프리미엄 소스 시장 현황 분석해줘"
- 섹션 예시: 시장 규모, 주요 업체, 소비 트렌드, 향후 전망

### 전략/계획 요청 (3-6개 섹션)
- 예: "대체육 시장 진출 전략 수립해줘"
- 섹션 예시: 시장 현황, 경쟁사 분석, 소비자 인사이트, 진출 전략, 위험 요인

### 종합 분석 요청 (4-8개 섹션)
- 예: "글로벌 발효식품 시장 종합 분석과 한국 기업 해외 진출 전략"
- 섹션 예시: 글로벌 시장 현황, 지역별 분석, 규제 환경, 경쟁 구도, 소비 트렌드, 진출 전략, 위험 관리

## 응답 형식
다음 JSON 형식으로만 응답하세요:

```json
{{
  "report_type": "simple|analysis|strategy|comprehensive",
  "sections": [
    {{
      "name": "섹션 이름",
      "purpose": "이 섹션의 목적과 다룰 내용",
      "key_questions": ["이 섹션에서 답해야 할 핵심 질문들"]
    }},
    ...
  ],
  "reasoning": "이렇게 섹션을 나눈 이유"
}}
```

**중요**:
- 간단한 질문(섹션이 필요 없다면)이면 sections를 빈 배열 []로 설정
- 각 섹션은 명확한 목적과 차별화된 내용을 가져야 함
- 불필요한 섹션 생성 금지, 자연스럽고 논리적인 구성만 허용
"""

        try:
            response = await self.chat.ainvoke(prompt)
            import json

            # JSON 추출
            content = response.content
            start = content.find('{')
            end = content.rfind('}') + 1
            json_str = content[start:end]

            section_plan = json.loads(json_str)
            return section_plan

        except Exception as e:
            print(f"섹션 계획 생성 실패: {e}")
            return {"report_type": "simple", "sections": [], "reasoning": "분석 실패로 단순 처리"}

    async def _intention_analysis(self, query: str) -> Dict:
        """질문을 분석하여 사용자의 의도 파악"""
        prompt = f"""당신은 비즈니스 질문 분석 전문가입니다.
        사용자의 질문의도를 파악하여, 보고서 섹션 계획 수립에 도움이 될 수 있는 정보를 제공해주세요.
"""

        try:
            response = await self.chat.ainvoke(prompt)
            return response.content

        except Exception as e:
            print(f"의도 분석 실패: {e}")
            return ""

    async def _generate_section_queries(self, original_query: str, section: Dict) -> List[str]:
        """특정 섹션에 필요한 정보 수집 쿼리들을 생성"""
        section_name = section.get('name', '섹션')
        section_purpose = section.get('purpose', '정보 수집')
        key_questions = section.get('key_questions', [])

        prompt = f"""당신은 정보 검색 쿼리 생성 전문가입니다.
주어진 섹션에 필요한 구체적이고 검색 가능한 쿼리들을 생성해주세요.

## 원본 질문
"{original_query}"

## 대상 섹션
- **섹션명**: {section_name}
- **목적**: {section_purpose}
- **핵심 질문들**: {key_questions}

## 쿼리 생성 원칙
1. **구체성**: 각 쿼리는 독립적으로 검색 가능해야 함
2. **적절성**: 섹션 목적에 맞는 정보만 수집
3. **효율성**: 꼭 필요한 정보만, 중복 최소화
4. **유연성**: 섹션의 복잡도에 따라 1-5개 정도의 쿼리 생성

## 생성 가이드
- **간단한 섹션**: 1-2개의 핵심 쿼리
- **표준 섹션**: 2-3개의 상세 쿼리
- **복잡한 섹션**: 3-5개의 세분화된 쿼리
- **원본 쿼리 맥락 유지**: 연도, 지역, 제품군 등 핵심 맥락을 모든 쿼리에 포함

## 응답 형식
Python 리스트 형태로만 응답하세요:

```
[
"구체적인 검색 쿼리 1",
"구체적인 검색 쿼리 2",
"구체적인 검색 쿼리 3"
]
```

**주의사항**:
- 다른 설명 없이 순수 리스트만 반환
- 각 쿼리는 명확하고 검색 가능한 형태
- 원본 질문의 핵심 맥락(시기, 대상, 범위 등) 반드시 포함
"""

        try:
            response = await self.chat.ainvoke(prompt)
            content = response.content

            # 리스트 추출
            start = content.find('[')
            end = content.rfind(']') + 1
            list_str = content[start:end]

            import ast
            queries = ast.literal_eval(list_str)

            # 빈 쿼리나 너무 짧은 쿼리 필터링
            valid_queries = [q for q in queries if isinstance(q, str) and len(q.strip()) > 10]

            return valid_queries # 최대 5개로 제한

        except Exception as e:
            print(f"섹션 쿼리 생성 실패 ({section_name}): {e}")
            # fallback: 섹션명을 기반으로 기본 쿼리 생성
            return [f"{original_query} {section_name} 관련 정보"]



# ==============================================================================
# Tool Calling 기반의 새로운 Reasoning 에이전트
# ==============================================================================
class ToolCallingAgent:
    """
    LangChain의 Tool Calling 기능을 사용하여 ReAct Loop를 직접 구현한 에이전트.
    LLM이 직접 Tool을 선택하고, 구조화된 응답을 받아 처리합니다.
    ⭐️ 자동 재시도, 다단계 검증, 쿼리 동적 변환 로직 탑재, 도구 병렬 실행, 반복 방지
    """
    def __init__(self, llm, tools: List[Any]):
        self.llm = llm
        self.tools = tools
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_mapping = {tool.name: tool for tool in tools}
        self.current_date_str = datetime.now().strftime("%Y년 %m월 %d일")

    # ⭐️ GraphDB 검색을 위한 쿼리 분해 메소드
    async def _decompose_for_graphdb(self, sub_query: str) -> str:
        """주어진 쿼리에서 GraphDB 검색에 용이한 핵심 키워드를 2~3개 추출합니다."""
        print(f"---> GraphDB용 키워드 추출 시작: {sub_query}")
        prompt = f"""
        사용자의 질문에서 그래프 데이터베이스(GraphDB) 검색에 가장 효과적인 핵심 키워드를 2~3개 추출해줘.
        추출된 키워드들은 쉼표(,)로 구분해서 간결하게 답변해줘.

        예시:
        - 질문: "제주도산 감귤의 원산지 정보 알려줄래?"
        - 답변: 제주도, 감귤, 원산지

        질문: "{sub_query}"
        답변:
        """
        response = await self.llm.ainvoke(prompt)
        keywords = response.content.strip()
        print(f"---> 추출된 GraphDB 키워드: {keywords}")
        return keywords

    # ⭐️ VectorDB 검색을 위한 쿼리 변환 메소드
    async def _transform_for_vectordb(self, sub_query: str) -> str:
        """주어진 쿼리를 벡터 검색에 더 유리하도록 의미적으로 풍부하게 재작성합니다."""
        print(f"---> VectorDB용 쿼리 변환 시작: {sub_query}")
        prompt = f"""
        당신은 벡터 검색을 위한 쿼리 확장 전문가입니다.
        사용자의 원본 질문을 분석하여, 지식 베이스에서 관련 문서를 더 잘 찾을 수 있도록 의미가 풍부하고 상세한 문장으로 재작성해주세요.
        동의어, 관련 개념을 포함하여 완전한 질문 형태로 만들어주세요.

        예시:
        - 원본: "사과 재배 시 병충해 예방"
        - 변환: "사과 과수원 운영 시 주로 발생하는 병충해 종류와 그에 대한 유기농 및 화학적 예방 방법과 관리 지침에 대한 상세 정보"

        - 원본: "스마트팜 기술 동향"
        - 변환: "최신 스마트팜 기술 트렌드, 사물인터넷(IoT) 센서, 자동화 시스템, 데이터 분석 기반의 정밀 농업 기술 적용 사례 및 전망"

        원본 질문: "{sub_query}"
        변환된 질문:
        """
        response = await self.llm.ainvoke(prompt)
        transformed_query = response.content.strip()
        print(f"---> 변환된 VectorDB 쿼리: {transformed_query}")
        return transformed_query

    async def _verify_tool_output(self, query: str, tool_name: str, tool_args: dict, observation: str) -> bool:
        """
        ⭐️ 도구 실행 결과의 유효성과 관련성을 모두 LLM을 통해 검증합니다.
        """
        print(f"---> 결과 검증 시작: {tool_name}")

        verification_prompt = f"""당신은 AI 에이전트의 행동을 감독하는 엄격한 품질 관리자입니다.

        사용자의 원본 질문과, 이 질문을 해결하기 위해 에이전트가 사용한 도구 및 그 결과를 보고 결과가 적절한지 평가해주세요.

        - **원본 질문**: "{query}"
        - **사용된 도구**: `{tool_name}`
        - **도구 입력값**: `{tool_args}`
        - **도구 실행 결과**: "{observation}"

        [평가 기준]
        1. **적절성**: 이 도구와 결과가 원본 질문에 대한 답변을 찾는 데 정말로 도움이 됩니까?
        2. **유효성**: 결과가 명백한 "오류" 메시지이거나, 내용이 완전히 비어있거나, "결과 없음" 같은 무의미한 내용은 아닙니까?

        위 두 기준을 **모두 만족해야 'YES'**입니다. 하나라도 만족하지 못하면 'NO'입니다.

        "YES" 또는 "NO"로만 답변해주세요.
        - **YES**: 결과가 유효하고, 질문과 관련이 있을 때.
        - **NO**: 결과가 유효하지 않거나, 질문과 전혀 관련이 없을 때.

        판단 (YES/NO):
        """

        response = await self.llm.ainvoke(verification_prompt)
        decision = "yes" in response.content.lower()
        print(f"---> 검증 결과: {'통과' if decision else '실패'}")
        return decision

    # ⭐️ 단일 도구 호출의 전체 로직을 담당하는 helper 메소드
    async def _execute_single_tool_call(self, tool_call: dict, query_for_verification: str, is_first_turn: bool, call_history: Set) -> ToolMessage:
        """단일 도구 호출을 처리하고, 반복을 방지하며, 결과를 종합적으로 검증합니다."""
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_function = self.tool_mapping.get(tool_name)

        if not tool_function:
            return ToolMessage(content=f"오류: '{tool_name}'이라는 이름의 도구를 찾을 수 없습니다.", tool_call_id=tool_call["id"])

        # 반복 작업 방지 로직
        call_signature = (tool_name, frozenset(tool_args.items()))
        if call_signature in call_history:
            print(f"!! 반복 작업 감지, 실행 건너뛰기: {call_signature}")
            return ToolMessage(
                content="오류: 이전과 동일한 작업을 반복하고 있습니다. 다른 도구를 사용하거나 다른 방식으로 접근해주세요.",
                tool_call_id=tool_call["id"]
            )
        call_history.add(call_signature)

        # 도구별 쿼리 동적 변환 로직
        try:
            original_sub_query = tool_args.get("query", "")
            if original_sub_query:
                if tool_name == "graph_db_search":
                    tool_args["query"] = await self._decompose_for_graphdb(original_sub_query)
                elif tool_name == "vector_db_search":
                    tool_args["query"] = await self._transform_for_vectordb(original_sub_query)
        except Exception as e:
            print(f"!! 쿼리 변환 중 오류 발생: {e}")

        # 자동 재시도 로직
        observation = None
        retry_attempts = 2
        for attempt in range(retry_attempts):
            try:
                print(f"Executing Tool: {tool_name} with args: {tool_args} (Attempt {attempt + 1}/{retry_attempts})")
                observation = tool_function.invoke(tool_args)
                break
            except Exception as e:
                observation = f"Tool '{tool_name}' 실행 오류: {e}"
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(1)

        # 첫 턴에만 모든 도구에 대해 종합 검증 수행
        is_valid_and_relevant = True
        if is_first_turn:
            is_valid_and_relevant = await self._verify_tool_output(query_for_verification, tool_name, tool_args, str(observation))

        if is_valid_and_relevant:
            return ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        else:
            # 검증 실패 시, VectorDB는 특별 처리
            if tool_name == "vector_db_search":
                print(f"!! VectorDB 결과가 무관하거나 유효하지 않아 폐기합니다.")
                feedback_content = "관련성 없는 문서가 검색되어 결과를 폐기했습니다. 다른 방법을 시도해주세요."
            else:
                feedback_content = f"'{tool_name}' 도구의 결과가 유효하지 않거나 질문과 관련이 없습니다. 다른 도구나 다른 접근 방식을 시도해주세요."

            return ToolMessage(content=feedback_content, tool_call_id=tool_call["id"])


    async def run(self, query: str) -> SearchResult:
        """Tool Calling을 사용한 Reasoning Loop를 실행합니다."""
        print("\n>> Tool Calling Reasoning Loop 시작 (병렬 실행, 반복 방지, 종합 검증 탑재)")

        system_prompt = f"""당신은 농수산물 경제 연구소의 수석 분석가입니다.
오늘 날짜: {self.current_date_str}
사용자의 질문에 답하기 위해 주어진 도구를 체계적으로 사용하여 포괄적이고 데이터 기반의 답변을 찾아야 합니다.

**[업무 가이드라인]**
1.  **RDB 사용**: 정확한 **수치 데이터**(가격, 시세, 영양성분, 품종, 칼로리)가 필요할 때는 **반드시 `rdb_search`를 최우선으로 사용**해야 합니다.
2.  **GraphDB 사용**: 농산물의 **원산지** 정보를 파악해야 할 때는 **반드시 `graph_db_search`를 최우선으로 사용**해야 합니다.
3.  **VectorDB 사용**: 시장 분석 보고서, 정책 문서 등 **설명적이고 긴 텍스트**에서 관련 문맥을 찾아야 할 때는 `vector_db_search`를 사용하세요.
4.  **Web Search 사용**: 내부 데이터베이스 검색 결과가 불충분하거나, 사용자가 '최신', '최근', '트렌드'와 같이 **실시간 정보**를 물어볼 때만 `debug_web_search`를 사용해 외부 정보를 보충하세요.
5.  **Scraping 사용**: `debug_web_search`를 통해 유효한 URL을 확보한 후에만 `scrape_and_extract_content`를 사용하세요. URL 없이 절대 호출하지 마세요.

필요한 모든 정보를 찾았다고 판단되면, 더 이상 도구를 사용하지 말고 최종 답변을 자연스러운 문장으로 작성하세요."""

        history = [
            HumanMessage(content=system_prompt, name="system"),
            HumanMessage(content=query, name="user")
        ]
        max_turns = 5

        # 반복 작업을 감지하기 위한 호출 기록 세트
        call_history = set()

        for i in range(max_turns):
            print(f"--- Turn {i+1}/{max_turns} ---")

            ai_response = await self.llm_with_tools.ainvoke(history)
            history.append(ai_response)
            print(ai_response)

            if not ai_response.tool_calls:
                print(">> 수집된 정보를 바탕으로 최종 답변 생성을 시작합니다.")
                break

            print(f"LLM wants to use {len(ai_response.tool_calls)} tool(s): {[tc['name'] for tc in ai_response.tool_calls]}")

            # 병렬 도구 실행 로직
            is_first_turn = (i == 0)
            tasks = [
                self._execute_single_tool_call(tool_call, query, is_first_turn, call_history)
                for tool_call in ai_response.tool_calls
            ]

            tool_messages = await asyncio.gather(*tasks)

            history.extend(tool_messages)

        final_answer = history[-1].content if isinstance(history[-1], AIMessage) else "답변을 생성하지 못했습니다."

        return SearchResult(
            source="tool_calling_agent",
            content=final_answer,
            relevance_score=0.9,
            metadata={"steps_taken": i + 1},
            search_query=query,
        )



# ==============================================================================
# ⭐️ Pydantic 모델 정의: 안정적인 구조화를 위해 클래스 외부에 정의
# ==============================================================================
class SearchSourceDecision(BaseModel):
    """어떤 데이터 소스를 검색할지에 대한 결정 구조"""
    vector_db: bool = Field(description="일반적인 문서, 가이드, 설명서(농업 기술, 재배법 등) 검색 필요 여부")
    rdb: bool = Field(description="정확한 수치 데이터(가격, 영양성분, 생산량 등) 검색 필요 여부")
    graph_db: bool = Field(description="관계형 데이터(공급망, 지역별 특산품 등) 검색 필요 여부")
    web_search: bool = Field(description="최신 뉴스, 트렌드, 실시간 정보 검색 필요 여부")
    reasoning: str = Field(description="이렇게 판단한 구체적인 근거")


# ==============================================================================
# ⭐️ RetrieverAgent 수정
# ==============================================================================
class RetrieverAgent:
    """통합 검색 에이전트 - 복잡도별 차등 + 병렬 처리 + 자체 Reasoning Loop"""

    def __init__(self):
        """
        ⭐️ 기존 ReAct Executor 대신 ToolCallingAgent를 생성합니다.
        """
        # 사용 가능한 도구들
        self.available_tools = [
            debug_web_search,
            scrape_and_extract_content,
            vector_db_search,
            rdb_search,
            graph_db_search,
        ]

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.chat = ChatOpenAI(model="gpt-4o-mini")

        # 날짜 정보
        self.current_date = datetime.now()
        self.current_date_str = self.current_date.strftime("%Y년 %m월 %d일")
        self.current_year = self.current_date.year

        # vector_db 속성 추가 (기본값으로 True 설정)
        self.vector_db = True
        self.rdb = True
        self.graph_db = True
        self.web_search = True

        # ToolCallingAgent 초기화
        self.tool_calling_agent = ToolCallingAgent(llm=self.llm, tools=self.available_tools)
        self.source_determiner_llm = self.llm.with_structured_output(SearchSourceDecision)

        # 병렬 처리용 스레드 풀
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    async def search(self, state: StreamingAgentState) -> StreamingAgentState:
        """
        호출 방식 변경: 복잡도에 따라 적절한 실행 함수를 호출합니다.
        """
        print(">> 통합 RETRIEVER 시작")

        if not state.query_plan or not state.query_plan.sub_queries:
            print("- 처리할 쿼리가 없어 RETRIEVER를 종료합니다.")
            return state

        # 복잡도 및 실행 전략 결정
        complexity_level = state.get_complexity_level()
        execution_strategy = state.query_plan.execution_strategy
        original_query = state.original_query

        print(f"- 복잡도: {complexity_level}")
        print(f"- 실행 전략: {execution_strategy}")
        print(f"- 원본 쿼리: {original_query}")

        # 실행 전략에 따른 분기
        if execution_strategy == ExecutionStrategy.DIRECT_ANSWER:
            print("- SIMPLE: 검색 생략")
            return state

        elif execution_strategy == ExecutionStrategy.BASIC_SEARCH:
            print("- MEDIUM: 기본 병렬 검색")
            return await self._execute_basic_parallel_search(state, original_query)

        elif execution_strategy == ExecutionStrategy.FULL_REACT:
            # "COMPLEX" 전략일 때, 새로운 reasoning loop 함수를 호출하도록 변경
            print("- COMPLEX: Tool Calling Agent 실행")
            return await self._execute_reasoning_loop(state, original_query)

        elif execution_strategy == ExecutionStrategy.MULTI_AGENT:
            print("- SUPER_COMPLEX: 다단계 병렬 검색")
            # ⭐️ SUPER_COMPLEX의 첫 단계도 새로운 reasoning loop를 사용하도록 변경
            return await self._execute_multi_stage_parallel_search(state, original_query)

        else:
            return await self._execute_basic_parallel_search(state, original_query)

    async def _execute_reasoning_loop(
        self, state: StreamingAgentState, query: str
    ) -> StreamingAgentState:
        """
        섹션 기반 정보 수집: Planning에서 생성한 섹션별로 쿼리를 병렬 처리하고 결과를 수집합니다.
        """
        print("\n>> 섹션 기반 정보 수집 시작 (병렬 처리)")

        # Planning에서 생성한 섹션 정보 가져오기
        sections = state.query_plan.resource_requirements.get("sections", [])
        sub_queries = state.query_plan.sub_queries
        report_structure = state.query_plan.resource_requirements.get("report_structure", {})

        print(f"- 총 {len(sections)}개 섹션, {len(sub_queries)}개 쿼리 처리 예정")

        # 섹션이 없으면 단순 처리
        if not sections:
            print("- 섹션 없음: 단순 질문으로 처리")
            return await self._process_simple_queries(state, query, sub_queries)

        # 전체 처리 시작 시간
        total_start_time = time.time()

        # 섹션별 처리 태스크 생성
        section_tasks = []
        for section_idx, section in enumerate(sections):
            section_name = section.get("name", f"섹션 {section_idx + 1}")
            section_purpose = section.get("purpose", "정보 수집")

            # 이 섹션에 해당하는 쿼리들 찾기 (간단한 방식: 순서대로 배분)
            queries_per_section = len(sub_queries) // len(sections) if len(sections) > 0 else 1
            start_idx = section_idx * queries_per_section
            end_idx = start_idx + queries_per_section

            # 마지막 섹션은 남은 모든 쿼리를 처리
            if section_idx == len(sections) - 1:
                end_idx = len(sub_queries)

            section_queries = sub_queries[start_idx:end_idx]

            print(f"- 섹션 '{section_name}' 준비: {len(section_queries)}개 쿼리")

            # 섹션 처리 태스크 생성
            task = self._process_single_section(
                section_name, section_purpose, section_queries, section_idx
            )
            section_tasks.append(task)

        # 모든 섹션 동시 처리
        print(f">> {len(section_tasks)}개 섹션 병렬 처리 시작")
        section_results_list = await asyncio.gather(*section_tasks, return_exceptions=True)

        total_execution_time = time.time() - total_start_time

        # 결과 정리 및 오류 처리
        section_results = {}
        for idx, result in enumerate(section_results_list):
            if isinstance(result, Exception):
                section_name = sections[idx].get("name", f"섹션 {idx + 1}")
                print(f"   ✗ 섹션 '{section_name}' 처리 실패: {result}")
                section_results[section_name] = {
                    "purpose": sections[idx].get("purpose", "정보 수집"),
                    "queries": [],
                    "content": [{"query": "오류", "result": f"섹션 처리 실패: {str(result)}", "relevance_score": 0.1}],
                    "execution_time": 0,
                    "total_queries": 0,
                    "successful_queries": 0,
                    "error": str(result)
                }
            else:
                section_results.update(result)

        # 각 섹션별로 SearchResult 생성 (Context Integrator가 섹션별로 처리할 수 있도록)
        for section_name, section_data in section_results.items():
            # 오류가 있는 섹션은 건너뛰기
            if "error" in section_data:
                continue

            # 섹션 내 모든 결과를 하나의 content로 통합
            combined_content = f"## {section_name}\n\n"
            combined_content += f"**목적**: {section_data['purpose']}\n\n"

            for content_item in section_data['content']:
                combined_content += f"### 질문: {content_item['query']}\n"
                combined_content += f"{content_item['result']}\n\n"

            # 섹션별 SearchResult 생성
            section_search_result = SearchResult(
                source=f"section_{section_name}",
                content=combined_content,
                relevance_score=0.9,
                metadata={
                    "section_name": section_name,
                    "section_purpose": section_data['purpose'],
                    "total_queries": section_data['total_queries'],
                    "successful_queries": section_data['successful_queries'],
                    "execution_time": section_data['execution_time'],
                    "section_type": "structured_section"
                },
                search_query=query
            )

            state.add_multi_source_result(section_search_result)
            print(f"   섹션 '{section_name}' 결과 저장 완료")

        # 전체 결과 요약
        state.add_step_result("section_based_reasoning", {
            "total_sections": len(sections),
            "total_queries": len(sub_queries),
            "total_execution_time": total_execution_time,
            "section_results": section_results,
            "report_structure": report_structure,
            "parallel_processing": True
        })

        print(f"\n>> 섹션 기반 정보 수집 완료 (병렬 처리)")
        print(f"   - 처리된 섹션: {len(sections)}개")
        print(f"   - 처리된 쿼리: {len(sub_queries)}개")
        print(f"   - 총 소요 시간: {total_execution_time:.2f}초")

        return state

    async def _process_single_section(self, section_name: str, section_purpose: str,
                                    section_queries: List[str], section_idx: int) -> dict:
        """
        단일 섹션을 처리하는 메소드 (병렬 처리용)
        """
        print(f"\n>> 섹션 '{section_name}' 처리 시작 (병렬)")
        print(f"   목적: {section_purpose}")
        print(f"   처리할 쿼리 {len(section_queries)}개: {section_queries}")

        section_content = []
        section_start_time = time.time()

        # 섹션 내 쿼리들을 순차 처리 (향후 이 부분도 병렬 처리 가능)
        for query_idx, section_query in enumerate(section_queries):
            print(f"     섹션 '{section_name}' - 쿼리 {query_idx + 1}/{len(section_queries)}: '{section_query}'")
            try:
                # ToolCallingAgent로 쿼리 처리
                reasoning_result = await self.tool_calling_agent.run(section_query)

                if reasoning_result and reasoning_result.content:
                    section_content.append({
                        "query": section_query,
                        "result": reasoning_result.content,
                        "relevance_score": getattr(reasoning_result, 'relevance_score', 0.8)
                    })
                    print(f"     ✓ 섹션 '{section_name}' - 쿼리 처리 완료")
                else:
                    print(f"     ✗ 섹션 '{section_name}' - 쿼리 처리 결과 없음")

            except Exception as e:
                print(f"     ✗ 섹션 '{section_name}' - 쿼리 처리 실패: {e}")
                section_content.append({
                    "query": section_query,
                    "result": f"오류 발생: {str(e)}",
                    "relevance_score": 0.1
                })

        section_execution_time = time.time() - section_start_time

        # 섹션 결과 반환
        section_result = {
            section_name: {
                "purpose": section_purpose,
                "queries": section_queries,
                "content": section_content,
                "execution_time": section_execution_time,
                "total_queries": len(section_queries),
                "successful_queries": len([c for c in section_content if c.get("relevance_score", 0) > 0.5])
            }
        }

        print(f"   섹션 '{section_name}' 완료: {section_execution_time:.2f}초")
        return section_result

    async def _process_simple_queries(self, state: StreamingAgentState, query: str, sub_queries: List[str]) -> StreamingAgentState:
        """간단한 질문 처리 (섹션이 없는 경우)"""
        print(">> 단순 쿼리 처리 모드")

        all_contents = []
        total_execution_time = 0

        for i, sub_query in enumerate(sub_queries):
            print(f"   쿼리 {i+1}/{len(sub_queries)}: '{sub_query}'")
            try:
                start_time = time.time()
                reasoning_result = await self.tool_calling_agent.run(sub_query)
                execution_time = time.time() - start_time
                total_execution_time += execution_time

                if reasoning_result and reasoning_result.content:
                    all_contents.append(f"### {sub_query}\n{reasoning_result.content}")
                    print(f"   ✓ 처리 완료: {execution_time:.2f}초")

            except Exception as e:
                print(f"   ✗ 처리 실패: {e}")
                all_contents.append(f"### {sub_query}\n오류 발생: {str(e)}")

        # 단순 통합 결과 생성
        final_content = "\n\n---\n\n".join(all_contents)

        simple_search_result = SearchResult(
            source="simple_reasoning",
            content=final_content,
            relevance_score=0.85,
            metadata={
                "total_queries": len(sub_queries),
                "execution_time": total_execution_time,
                "processing_type": "simple"
            },
            search_query=query
        )

        state.add_multi_source_result(simple_search_result)

        state.add_step_result("simple_reasoning", {
            "total_queries": len(sub_queries),
            "execution_time": total_execution_time
        })

        print(f">> 단순 쿼리 처리 완료: {total_execution_time:.2f}초")
        return state

    async def _determine_search_sources(self, query: str) -> Dict[str, bool]:
        """
        ⭐️ LLM이 쿼리를 분석해서 어떤 DB를 검색할지 결정 (Tool Calling 방식)
        """
        print(f"\n>> 검색 소스 결정 시작 (Tool Calling 방식)")
        prompt = f"""다음 질문을 분석해서 어떤 데이터베이스를 검색해야 하는지 판단해라.

        질문: "{query}"

        검색 소스별 특징:
        - vector_db: 일반적인 문서, 가이드, 설명서 (농업 기술, 재배법 등)
        - rdb: 정확한 수치 데이터 (가격, 영양성분, 생산량 등)
        - graph_db: 관계형 데이터 (공급망, 지역별 특산품, 연관 정보)
        - web_search: 최신 뉴스, 트렌드, 실시간 정보
        """
        try:
            # .ainvoke()를 호출하면 파싱이 완료된 Pydantic 모델 객체를 바로 받습니다.
            decision_model = await self.source_determiner_llm.ainvoke(prompt)
            return decision_model.dict()  # Pydantic 모델을 딕셔너리로 변환하여 반환
        except Exception as e:
            print(f"소스 결정 오류: {e}")
            # 오류 시 안전하게 모든 소스를 검색하도록 폴백(fallback)
            return {
                "vector_db": True, "rdb": True, "graph_db": True, "web_search": True,
                "reasoning": "판단 오류로 모든 소스 검색"
            }

    async def _execute_basic_parallel_search(self, state: StreamingAgentState, query: str) -> StreamingAgentState:
        """LLM 판단 기반 선택적 병렬 검색"""
        print("\n>> 지능형 선택적 병렬 검색 실행")

        # DB 상태 확인
        print(f"- DB 상태 확인:")
        print(f"  vector_db: {self.vector_db is not None}")
        print(f"  rdb: {self.rdb is not None}")
        print(f"  graph_db: {self.graph_db is not None}")

        # LLM이 어떤 소스를 검색할지 결정
        search_decision = await self._determine_search_sources(query)
        print(f"- 검색 결정: {search_decision}")

        search_tasks = []

        # LLM 판단에 따라 선택적 검색
        if search_decision.get("vector_db"):
            search_tasks.append(self._async_vector_search(query))
            print("- Vector DB 검색 추가")

        if search_decision.get("rdb"):
            search_tasks.append(self._async_rdb_search(query))
            print("- RDB 검색 추가")

        if search_decision.get("graph_db"):
            search_tasks.append(self._async_graph_search(query))
            print("- Graph DB 검색 추가")

        if search_decision.get("web_search"):
            search_tasks.append(self._async_web_search_enhanced(query))
            print("- 웹 검색 추가")


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
                print(f"- LLM 분석 결과: {analysis_result}")
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



    async def _execute_multi_stage_parallel_search(
        self, state: StreamingAgentState, query: str
    ) -> StreamingAgentState:
        """다단계 병렬 검색 (SUPER_COMPLEX 복잡도)"""
        print("\n>> 다단계 병렬 검색 실행")

        try:
            # ⭐️ 1단계: 초기 정보 수집도 새로운 Reasoning Loop를 사용합니다.
            print("- 1단계: 초기 정보 수집 (by Tool Calling Agent)")
            state = await self._execute_reasoning_loop(state, query)

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
        """비동기 Vector DB 검색 - search_tools 통일"""
        try:
            print(f"  └ Vector DB 검색: {query[:30]}...")

            # search_tools.py의 vector_db_search 함수 사용
            loop = asyncio.get_event_loop()
            vector_results = await loop.run_in_executor(
                self.thread_pool,
                lambda: vector_db_search.invoke({"query": query})
            )

            # 결과를 SearchResult로 변환
            results = []
            if isinstance(vector_results, str) and vector_results:
                result = SearchResult(
                    source="vector_db",
                    content=vector_results,
                    relevance_score=0.7,
                    metadata={"search_type": "vector"},
                    search_query=query,
                )
                results.append(result)

            print(f"    ✓ Vector DB: {len(results)}개 결과")
            return results

        except Exception as e:
            print(f"    ✗ Vector DB 오류: {e}")
            return []


    async def _async_graph_search(self, query: str) -> List[SearchResult]:
        """비동기 Graph DB 검색 - search_tools 통일"""
        try:
            print(f"  └ Graph DB 검색: {query[:30]}...")

            # search_tools.py의 graph_db_search 함수 사용
            loop = asyncio.get_event_loop()
            graph_results = await loop.run_in_executor(
                self.thread_pool,
                lambda: graph_db_search.invoke({"query": query})
            )

            # 결과를 SearchResult로 변환
            results = []
            if isinstance(graph_results, str) and "Neo4j" in graph_results:
                result = SearchResult(
                    source="graph_db",
                    content=graph_results,
                    relevance_score=0.85,
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
        """비동기 RDB 검색 - search_tools 통일"""
        try:
            print(f"  └ RDB 검색: {query[:30]}...")

            processed_query = self._preprocess_rdb_query(query)
            print(f"    → 전처리된 쿼리: {processed_query}")

            # search_tools.py의 rdb_search 함수 사용
            loop = asyncio.get_event_loop()
            rdb_results = await loop.run_in_executor(
                self.thread_pool,
                lambda: rdb_search.invoke({"query": processed_query})
            )

            # 결과를 SearchResult로 변환
            results = []
            if isinstance(rdb_results, str) and rdb_results:
                result = SearchResult(
                    source="rdb",
                    content=rdb_results,
                    relevance_score=0.85,
                    metadata={"search_type": "rdb"},
                    search_query=processed_query,
                )
                results.append(result)

            print(f"    ✓ RDB: {len(results)}개 결과")
            return results

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
        """비동기 웹 검색 - search_tools 통일"""
        try:
            print(f"  └ Enhanced Web 검색 시작: {query[:30]}...")

            # search_tools.py의 debug_web_search 함수 사용
            loop = asyncio.get_event_loop()
            web_results = await loop.run_in_executor(
                self.thread_pool,
                lambda: debug_web_search.invoke({"query": query})
            )

            # 결과를 SearchResult로 변환
            results = []
            if isinstance(web_results, str) and web_results:
                # 문자열에서 링크 추출해보기
                import re
                urls = re.findall(r'https?://[^\s]+', web_results)

                if urls and len(urls) > 0:
                    # 링크가 있으면 첫 번째 링크 스크래핑 시도
                    print(f"    → {len(urls)}개 URL 발견, 첫 번째 스크래핑 시도")
                    scraping_result = await self._async_scrape_url(urls[0], query)
                    if scraping_result:
                        return [scraping_result]

                # 링크 스크래핑 실패하면 원본 문자열을 SearchResult로 반환
                result = SearchResult(
                    source="web_search_summary",
                    content=web_results,
                    relevance_score=0.6,
                    metadata={"search_type": "web_summary"},
                    search_query=query
                )
                results.append(result)

            print(f"    ✓ Web Search: {len(results)}개 결과")
            return results

        except Exception as e:
            print(f"    ✗ 웹 검색 중 예외 발생: {e}")
            return []

    async def _async_scrape_url(self, url: str, query: str) -> Optional[SearchResult]:
        """단일 URL을 비동기적으로 스크래핑 - search_tools 통일"""
        try:
            # search_tools.py의 scrape_and_extract_content 함수 사용
            action_input = json.dumps({"url": url, "query": query})

            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                self.thread_pool,
                lambda: scrape_and_extract_content.invoke({"action_input": action_input})
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

        try:
            from ...services.search.search_tools import scrape_and_extract_content # 실제 경로에 맞게 수정

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
            content_preview = content[:].strip().replace("\n", " ")

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
        content_sample = total_content[:] if total_content else "내용 없음"

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
            if "confidence" not in result:
                result["confidence"] = 0.5
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
import re
from typing import Dict, List, Any

class ContextIntegratorAgent:
    def __init__(self):
        # 최종 보고서 초안의 품질을 높이기 위해 더 강력한 모델을 사용합니다.
        self.chat = ChatOpenAI(model="gpt-4o", temperature=0.1)
        self.agent_type = AgentType.CONTEXT_INTEGRATOR

# ContextIntegratorAgent
import re
from typing import Dict, List, Any

class ContextIntegratorAgent:
    def __init__(self):
        self.chat = ChatOpenAI(model="gpt-4o", temperature=0.1)
        self.agent_type = AgentType.CONTEXT_INTEGRATOR

    async def integrate(self, state: StreamingAgentState) -> StreamingAgentState:
        """수집된 모든 정보를 정확하게 통합하여 보고서 생성을 위한 구조화된 컨텍스트를 생성합니다."""
        print(">> CONTEXT_INTEGRATOR 시작 (데이터 통합 및 구조화)")

        all_results = state.graph_results_stream + state.multi_source_results_stream

        if not all_results:
            print("- 통합할 결과가 없음")
            state.integrated_context = "검색된 정보가 없어 답변을 생성할 수 없습니다."
            return state

        print(f"- 총 {len(all_results)}개 검색 결과를 분석 및 통합")

        # 수집된 데이터를 소스별로 정리
        organized_data = self._organize_data_by_source(all_results)

        # CoT 기반으로 구조화된 컨텍스트 생성
        integrated_context = await self._create_structured_context(
            state.original_query,
            organized_data
        )

        state.integrated_context = integrated_context
        print(f"- 구조화된 컨텍스트 생성 완료 (길이: {len(integrated_context)}자)")
        print(all_results[0].content)
        print(">> CONTEXT_INTEGRATOR 완료")
        return state

    def _organize_data_by_source(self, all_results: list) -> Dict[str, List[str]]:
        """검색 결과를 소스별로 정리"""
        organized = {
            'vector_db': [],
            'graph_db': [],
            'rdb': [],
            'web_search': [],
            'web_scrape': [],
            'react_agent': [],
            'other': []
        }

        for result in all_results:
            source = getattr(result, 'source', 'other').lower()
            content = getattr(result, 'content', '')

            if not content.strip():
                continue

            # 소스 매핑
            if source in ['vector_db', 'vector']:
                organized['vector_db'].append(content)
            elif source == 'graph_db':
                organized['graph_db'].append(content)
            elif source in ['rdb', 'postgresql']:
                organized['rdb'].append(content)
            elif source in ['web_search_summary', 'debug_web_search']:
                organized['web_search'].append(content)
            elif source == 'web_scrape':
                organized['web_scrape'].append(content)
            elif source == 'react_agent':
                organized['react_agent'].append(content)
            else:
                organized['other'].append(content)

        return organized

    async def _create_structured_context(self, original_query: str, organized_data: Dict[str, List[str]]) -> str:
        """CoT 기반으로 구조화된 컨텍스트 생성"""

        # 실제 데이터가 있는 소스만 포함
        available_sources = {k: v for k, v in organized_data.items() if v}

        if not available_sources:
            return "수집된 데이터가 없습니다."

        # 데이터 요약 생성
        data_summary = []
        for source, contents in available_sources.items():
            source_summary = f"**{source.upper()}** ({len(contents)}건):"
            for i, content in enumerate(contents[:2], 1):  # 각 소스당 최대 2건만
                # 내용을 적절히 요약
                summary = content[:300] + "..." if len(content) > 300 else content
                source_summary += f"\n- 결과 {i}: {summary}"
            data_summary.append(source_summary)

        combined_data = "\n\n".join(data_summary)

        system_prompt = """
당신은 데이터 분석 전문가이자 보고서 설계자입니다. 수집된 데이터를 바탕으로 사용자 질문에 대한 정확하고 구조화된 컨텍스트를 생성하되, 각 데이터를 어떤 섹션에서 어떻게 활용할지 구체적으로 명시해야 합니다.

**핵심 원칙:**
1. **정확성 우선**: 제공된 데이터에 없는 내용은 절대 추가하지 마세요
2. **수치 정확성**: 모든 숫자, 퍼센트, 금액 등은 원본 그대로 사용
3. **데이터 활용 계획**: 각 데이터를 어떤 섹션에서 어떻게 사용할지 명시
4. **차트 설계**: 구체적인 수치가 있을 때만 고급 차트 설계 제안
- 단순 수치 → bar, column, horizontalbar
- 비율/퍼센트 → pie, doughnut, donut
- 시계열 → line, area, timeseries
- 트렌드 → line, area (변동성에 따라)
- 다중 시리즈 → groupedbar, stacked, combo
- 성능 지표 → radar
- 분포 → scatter, bubble
- 복합 데이터 → mixed, combo
- 특수 형태 → funnel, waterfall, gauge, heatmap

**작업 과정 (Chain of Thought):**
1. 사용자 질문 분석 → 어떤 정보가 필요한가?
2. 수집된 데이터 검토 → 어떤 데이터가 실제로 있는가?
3. 섹션 구조 설계 → 질문에 답하기 위해 어떤 섹션들이 필요한가?
4. 데이터 매핑 → 각 데이터를 어떤 섹션에서 어떻게 활용할 것인가?
5. 차트 설계 → 어떤 수치 데이터를 어떤 고급 차트로 시각화할 것인가?
   - 데이터 특성 분석: 범위, 분포, 변동성, 음수 포함 여부
   - 최적 차트 타입 선택: 단순형/복합형/특수형
   - 색상 팔레트 선택: modern/corporate/vibrant/warm/pastel/gradient
   - 인터랙션 고려: 다중 축, 스택, 그룹화

**출력 형식:**
# 보고서 구조 설계 및 데이터 활용 계획

## 1. 예상 섹션 구조
[질문에 답하기 위해 필요한 섹션들을 순서대로 나열]

## 2. 섹션별 데이터 활용 계획
### 섹션 A: [섹션명]
**목적:** [이 섹션에서 답하고자 하는 구체적 질문]
**활용 데이터:**
- [출처: XX] 구체적 데이터 내용 → 이 섹션에서 어떻게 활용할지
- [출처: XX] 구체적 수치 (예: 35.2%) → 어떤 맥락에서 사용할지

### 섹션 B: [섹션명]
**목적:** [이 섹션에서 답하고자 하는 구체적 질문]
**활용 데이터:**
- [출처: XX] 구체적 데이터 내용 → 이 섹션에서 어떻게 활용할지

## 3. 고급 차트 설계 계획
### 차트 1: [차트 제목]
**데이터 출처:** [출처: XX]
**구체적 수치:**
- 항목A: XX (정확한 수치와 단위)
- 항목B: XX (정확한 수치와 단위)
**차트 타입:** bar/line/pie/doughnut/radar/scatter/combo/stacked/waterfall/heatmap
**색상 팔레트:** modern/corporate/vibrant/warm/pastel/gradient
**데이터 특성:**
- 데이터 개수: X개
- 값 범위: 최소 XX ~ 최대 XX
- 특이사항: [음수 포함/이상치/변동성 등]
**활용 섹션:** [어느 섹션에서 사용할지]
**시각화 목적:** [이 차트로 무엇을 보여줄 것인가]

## 4. 핵심 인사이트 및 메시지
[각 섹션에서 전달하고자 하는 핵심 메시지들]
"""

        human_prompt = f"""
**사용자 질문:** {original_query}

**수집된 데이터:**
{combined_data}

위 데이터를 바탕으로 다음을 수행해주세요:

1. **질문 분석**: 사용자가 궁금해하는 것이 정확히 무엇인지 파악
2. **데이터 검토**: 수집된 데이터 중에서 실제로 사용 가능한 것들 식별
3. **섹션 설계**: 질문에 체계적으로 답하기 위한 섹션 구조 설계
4. **데이터 매핑**: 각 섹션에서 어떤 데이터를 어떻게 활용할지 구체적 계획
5. **차트 설계**: 수치 데이터가 있다면 어떤 고급 차트로 시각화할지 계획
   - 데이터 특성에 맞는 최적 차트 타입 선택
   - 복잡한 데이터는 복합 차트(combo, mixed) 고려
   - 많은 카테고리는 horizontalbar, 적은 카테고리는 column
   - 시계열은 area/timeseries, 분포는 scatter/bubble
   - 성능 지표는 radar, 비율은 doughnut/pie
   - 색상 팔레트와 인터랙션 요소도 함께 계획

**중요**:
- 제공된 데이터에 없는 내용은 절대 추가하지 마세요
- 모든 수치는 정확히 그대로 인용하세요
- 각 데이터의 출처를 명확히 표기하세요
- 어떤 섹션에서 어떻게 사용할지 구체적으로 설명하세요
"""

        messages = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]

        print("- CoT 기반 컨텍스트 구조화 진행...")
        response = await self.chat.ainvoke(messages)

        return response.content.strip()


# 리팩토링 관련 임포트(참고용)
from ...core.config.report_config import TeamType, ReportType, Language
from ...services.templates.report_templates import ReportTemplateManager
from ...services.builders.prompt_builder import PromptBuilder
from ...utils.analyzers.query_analyzer import QueryAnalyzer


class ReportGeneratorAgent:
    """보고서 생성 에이전트"""

    def __init__(self):
        self.streaming_chat = ChatOpenAI(model="gpt-4o", temperature=0.4, streaming=True)
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
        """스트리밍 보고서 생성 - 마크다운 형식 지원"""

        print("\n>> REFACTORED REPORT_GENERATOR 시작")

        # integrated_context 타입 확인 및 처리
        integrated_context = state.integrated_context
        original_query = state.original_query
        memory_context = getattr(state, "memory_context", "")
        user_context = getattr(state, "user_context", None)

        print(f"- 쿼리: {original_query[:50]}...")
        print(f"- integrated_context 타입: {type(integrated_context)}")

        # integrated_context가 딕셔너리(새로운 방식)인지 문자열(기존 방식)인지 확인
        if isinstance(integrated_context, dict):
            # 새로운 방식: 마크다운 데이터 패키지
            data_package = integrated_context

            # 데이터 패키지 검증
            if not data_package:
                error_msg = "분석할 충분한 정보가 수집되지 않았습니다."
                state.final_answer = error_msg
                yield error_msg
                return

            if data_package.get('error'):
                error_msg = data_package['error']
                state.final_answer = error_msg
                yield error_msg
                return

            # 새로운 방식으로 처리
            full_response = ""
            async for chunk in self._process_markdown_data_package(data_package, original_query, user_context, source_collection_data):
                full_response += chunk
                yield chunk

            state.final_answer = full_response

        elif isinstance(integrated_context, str):
            # 기존 방식: 문자열 컨텍스트
            if not integrated_context:
                error_msg = "분석할 충분한 정보가 수집되지 않았습니다."
                state.final_answer = error_msg
                yield error_msg
                return

            # 기존 방식으로 처리
            full_response = ""
            async for chunk in self._process_string_format(integrated_context, original_query, user_context, source_collection_data):
                full_response += chunk
                yield chunk

            state.final_answer = full_response
        else:
            error_msg = f"지원하지 않는 integrated_context 타입: {type(integrated_context)}"
            state.final_answer = error_msg
            yield error_msg
            return

    async def _process_markdown_data_package(
        self,
        data_package: Dict,
        original_query: str,
        user_context: Any,
        source_collection_data: Dict
    ) -> AsyncGenerator[str, None]:
        """새로운 마크다운 데이터 패키지 형식 처리"""


        print(f"- 데이터 패키지 타입: {type(data_package)}")
        print(f"- 총 결과 개수: {data_package.get('total_results_count', 0)}")

        # 마크다운 content 확인
        markdown_content = data_package.get('markdown_content', '')
        print(f"- 마크다운 컨텐츠: {len(markdown_content)}자")

        # 각 source별 데이터 개수 출력
        data_summary = data_package.get('data_summary', {})
        print(f"- Vector DB: {data_summary.get('vector_db_count', 0)}건")
        print(f"- Graph DB: {data_summary.get('graph_db_count', 0)}건")
        print(f"- PostgreSQL: {data_summary.get('rdb_count', 0)}건")
        print(f"- Web Search: {data_summary.get('web_search_summary_count', 0)}건")
        print(f"- Web Scrape: {data_summary.get('web_scrape_count', 0)}건")
        print(f"- React Agent: {data_summary.get('react_agent_count', 0)}건")

        # 1. 쿼리 분석
        team_type = QueryAnalyzer.detect_team_type(original_query)
        language = QueryAnalyzer.detect_language(original_query)
        complexity_analysis = QueryAnalyzer.analyze_complexity(original_query, user_context)
        report_type = complexity_analysis["report_type"]

        print(f"- 분석 결과: {team_type.value} / {report_type.value} / {language.value}")

        # 2. Source별 content를 SearchResult 형태로 재구성 (DataExtractor 호환)
        all_search_results = []
        source_contents = data_package.get('source_contents', {})

        for source_name, contents in source_contents.items():
            for content in contents:
                if content.strip():  # 빈 내용이 아닌 경우만
                    class SearchResultLike:
                        def __init__(self, source, content, relevance_score=0.0):
                            self.source = source
                            self.content = content
                            self.relevance_score = relevance_score
                            self.metadata = {}

                    search_result = SearchResultLike(
                        source=source_name,
                        content=content,
                        relevance_score=0.8
                    )
                    all_search_results.append(search_result)

        print(f"- 재구성된 SearchResult: {len(all_search_results)}개")

        # 3. 실제 데이터 추출
        extracted_data = await self.data_extractor.extract_numerical_data(all_search_results, original_query)
        print(f"- 추출된 수치: {len(extracted_data.get('extracted_numbers', []))}개")

        # 4. 차트 생성
        real_charts = await self._create_data_driven_charts(extracted_data, original_query)
        print(f"- 생성된 차트: {len(real_charts)}개")

        # 5. 마크다운 content를 메인 컨텍스트로 사용
        enhanced_context = self._create_enhanced_context_with_markdown(data_package, markdown_content)
        print(f"- 향상된 컨텍스트: {len(enhanced_context)}자")

        # 6. 프롬프트 생성
        prompt = self.prompt_builder.build_prompt(
            query=original_query,
            context=enhanced_context,
            team_type=team_type,
            report_type=report_type,
            language=language,
            extracted_data=extracted_data,
            real_charts=real_charts,
            source_data=source_collection_data
        )

        # 7. 스트리밍 생성
        try:
            print("- 스트리밍 시작...")
            async for chunk in self.streaming_chat.astream(prompt):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            error_msg = f"답변 생성 중 오류: {str(e)}"
            yield error_msg

    async def _process_string_format(
        self,
        integrated_context: str,
        original_query: str,
        user_context: Any,
        source_collection_data: Dict
    ) -> AsyncGenerator[str, None]:
        """기존 문자열 형식 처리 (하위 호환성)"""

        print(f"- 기존 문자열 형식으로 처리: {len(integrated_context)}자")

        # 1. 쿼리 분석
        team_type = QueryAnalyzer.detect_team_type(original_query)
        language = QueryAnalyzer.detect_language(original_query)
        complexity_analysis = QueryAnalyzer.analyze_complexity(original_query, user_context)
        report_type = complexity_analysis["report_type"]

        print(f"- 분석 결과: {team_type.value} / {report_type.value} / {language.value}")

        # 2. 데이터 추출 (문자열에서 직접)
        # 임시로 SearchResult 객체 생성
        class TempSearchResult:
            def __init__(self, content):
                self.content = content
                self.source = "integrated_context"
                self.relevance_score = 1.0
                self.metadata = {}

        temp_results = [TempSearchResult(integrated_context)]
        extracted_data = await self.data_extractor.extract_numerical_data(temp_results, original_query)

        # 3. 차트 생성
        real_charts = await self._create_data_driven_charts(extracted_data, original_query)

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
        try:
            print("- 스트리밍 시작...")
            async for chunk in self.streaming_chat.astream(prompt):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            error_msg = f"답변 생성 중 오류: {str(e)}"
            yield error_msg

    def _create_enhanced_context_with_markdown(self, data_package: Dict, markdown_content: str) -> str:
        """마크다운 content를 활용한 향상된 컨텍스트 생성"""


        context_parts = []

        # 1. PostgreSQL 구조화 데이터 (최우선)
        postgresql_structured = data_package.get('postgresql_structured_data', {})
        if postgresql_structured:
            formatted_postgresql = self._format_postgresql_data_for_context(postgresql_structured)
            if formatted_postgresql:
                context_parts.append("## 농진청 공식 구조화 데이터 (PostgreSQL)")
                context_parts.append(formatted_postgresql)
                context_parts.append("")

        # 2. 마크다운 형식의 모든 source content (메인 컨텐츠)
        if markdown_content:
            context_parts.append("## 수집된 정보 (Source별 정리)")
            context_parts.append(markdown_content)
            context_parts.append("")

        # 3. 데이터 요약 정보
        data_summary = data_package.get('data_summary', {})
        summary_info = f"""## 수집 데이터 요약
    - 총 검색 결과: {data_package.get('total_results_count', 0)}건
    - Vector DB: {data_summary.get('vector_db_count', 0)}건
    - Graph DB: {data_summary.get('graph_db_count', 0)}건
    - PostgreSQL: {data_summary.get('rdb_count', 0)}건 (영양소: {data_summary.get('nutrition_data_count', 0)}건, 가격: {data_summary.get('price_data_count', 0)}건)
    - Web Search Summary: {data_summary.get('web_search_summary_count', 0)}건
    - Web Scrape: {data_summary.get('web_scrape_count', 0)}건
    - ReAct Agent: {data_summary.get('react_agent_count', 0)}건
    - LLM Analysis: {data_summary.get('llm_analysis_count', 0)}건
    - Strategic Synthesis: {data_summary.get('strategic_synthesis_count', 0)}건
    - Final Validation: {data_summary.get('final_validation_count', 0)}건
    - 기타: {data_summary.get('other_count', 0)}건
    """
        context_parts.append(summary_info)

        enhanced_context = "\n".join(context_parts)

        print(f"  - 컨텍스트 섹션 수: {len([p for p in context_parts if p.startswith('##')])}")
        print(f"  - 총 컨텍스트 길이: {len(enhanced_context)}자")

        return enhanced_context

    def _format_postgresql_data_for_context(self, structured_data: dict) -> str:
        """PostgreSQL 구조화 데이터를 컨텍스트용으로 포맷팅"""
        formatted_parts = []

        # 영양소 데이터
        nutrition_data = structured_data.get('nutrition_data', [])
        if nutrition_data:
            formatted_parts.append("### 영양성분 정보")
            for item in nutrition_data[:5]:  # 최대 5개
                formatted_parts.append(f"**{item.get('식품명', 'N/A')}** ({item.get('식품군', 'N/A')})")
                formatted_parts.append(f"- 칼로리: {item.get('칼로리', 0)}kcal/100g")
                formatted_parts.append(f"- 단백질: {item.get('단백질', 0)}g, 지방: {item.get('지방', 0)}g, 탄수화물: {item.get('탄수화물', 0)}g")
                if item.get('칼슘') or item.get('철') or item.get('비타민c'):
                    formatted_parts.append(f"- 칼슘: {item.get('칼슘', 0)}mg, 철: {item.get('철', 0)}mg, 비타민C: {item.get('비타민c', 0)}mg")
                formatted_parts.append("")

        # 가격 데이터
        price_data = structured_data.get('price_data', [])
        if price_data:
            formatted_parts.append("### 가격 정보")
            for item in price_data[:5]:  # 최대 5개
                formatted_parts.append(f"**{item.get('품목명', 'N/A')}** ({item.get('카테고리', 'N/A')})")
                formatted_parts.append(f"- 가격: {item.get('가격', 0)}원/{item.get('단위', 'kg')}")
                formatted_parts.append(f"- 기준일: {item.get('날짜', 'N/A')}")
                formatted_parts.append("")

        return "\n".join(formatted_parts) if formatted_parts else ""

    async def _create_data_driven_charts(self, extracted_data: Dict, query: str) -> List[Dict]:
        """실제 데이터 기반 차트 생성 - 복잡한 차트 타입 지원"""
        charts = []

        print(f"\n>> 고급 차트 생성 시작")
        print(f"- 추출된 데이터: {extracted_data}")

        if not extracted_data:
            print("- 추출된 데이터 없음, 빈 차트 리스트 반환")
            return charts

        # 1. 퍼센트 데이터 -> 다양한 원형 차트
        percentages = extracted_data.get('percentages', [])
        if len(percentages) >= 2:
            print(f"- 퍼센트 데이터 발견: {len(percentages)}개")
            valid_percentages = [
                p for p in percentages
                if isinstance(p.get('value'), (int, float)) and 0 <= p.get('value') <= 100
            ]

            if len(valid_percentages) >= 2:
                labels = []
                values = []

                for p in valid_percentages[:6]:  # 더 많은 데이터 지원
                    context = p.get('context', '항목')
                    if len(context) > 20:
                        context = context[:20] + "..."
                    labels.append(context)
                    values.append(float(p['value']))

                # 데이터가 많으면 doughnut, 적으면 pie
                chart_type = "doughnut" if len(values) > 4 else "pie"

                chart = {
                    "title": f"{query[:30]}... 비율 분석",
                    "type": chart_type,
                    "palette": "modern",
                    "data": {
                        "labels": labels,
                        "datasets": [{"label": "비율 (%)", "data": values}]
                    },
                    "source": "실제 추출 데이터",
                    "data_type": "real"
                }
                charts.append(chart)
                print(f"- {chart_type.upper()} 차트 생성: {chart['title']}")

        # 2. 수치 데이터 -> 다양한 바 차트 및 복합 차트
        numbers = extracted_data.get('extracted_numbers', [])
        if len(numbers) >= 2:
            print(f"- 수치 데이터 발견: {len(numbers)}개")
            valid_numbers = [
                n for n in numbers
                if isinstance(n.get('value'), (int, float))
            ]

            if len(valid_numbers) >= 2:
                labels = []
                values = []
                units = []

                for n in valid_numbers[:8]:  # 더 많은 데이터 지원
                    context = n.get('context', '항목')
                    if len(context) > 15:
                        context = context[:15] + "..."
                    labels.append(context)
                    values.append(float(n['value']))
                    units.append(n.get('unit', ''))

                primary_unit = units[0] if units[0] else '단위'

                # 데이터 특성에 따라 차트 타입 선택
                if len(values) > 6:
                    chart_type = "horizontalbar"  # 많은 데이터는 가로 바
                    palette = "corporate"
                elif any(v < 0 for v in values):  # 음수 포함시
                    chart_type = "waterfall"
                    palette = "warm"
                else:
                    chart_type = "bar"
                    palette = "modern"

                chart = {
                    "title": f"{query[:30]}... 주요 수치",
                    "type": chart_type,
                    "palette": palette,
                    "data": {
                        "labels": labels,
                        "datasets": [{"label": f"수치 ({primary_unit})", "data": values}]
                    },
                    "source": "실제 추출 데이터",
                    "data_type": "real"
                }
                charts.append(chart)
                print(f"- {chart_type.upper()} 차트 생성: {chart['title']}")

        # 3. 트렌드 데이터 -> 고급 시계열 차트
        trends = extracted_data.get('trends', [])
        if len(trends) >= 3:  # 최소 3개 이상의 트렌드 데이터
            print(f"- 트렌드 데이터 발견: {len(trends)}개")

            labels = []
            values = []

            for t in trends[:10]:  # 더 많은 트렌드 데이터 지원
                period = t.get('period', '기간')
                labels.append(period)

                change_str = str(t.get('change', '0'))
                import re
                numbers = re.findall(r'-?\d+\.?\d*', change_str)
                change_value = float(numbers[0]) if numbers else 0
                values.append(change_value)

            if len(values) >= 3:
                # 변화 패턴에 따라 차트 타입 선택
                has_negative = any(v < 0 for v in values)
                variance = max(values) - min(values)

                if variance > 50:  # 변동성이 큰 경우
                    chart_type = "area"
                    palette = "vibrant"
                elif has_negative:  # 음수 포함
                    chart_type = "line"
                    palette = "warm"
                else:
                    chart_type = "timeseries"
                    palette = "modern"

                chart = {
                    "title": f"{query[:30]}... 시간별 변화 추이",
                    "type": chart_type,
                    "palette": palette,
                    "data": {
                        "labels": labels,
                        "datasets": [{"label": "변화율 (%)", "data": values}]
                    },
                    "source": "실제 추출 데이터",
                    "data_type": "real"
                }
                charts.append(chart)
                print(f"- {chart_type.upper()} 차트 생성: {chart['title']}")

        # 4. 카테고리별 데이터 -> 그룹화된 차트
        categories = extracted_data.get('categories', [])
        if len(categories) >= 3:
            print(f"- 카테고리 데이터 발견: {len(categories)}개")

            # 카테고리 데이터가 여러 시리즈를 가지는지 확인
            multi_series = any(isinstance(cat.get('values'), list) for cat in categories)

            if multi_series:
                # 다중 시리즈 데이터 -> 스택 차트 또는 그룹 바 차트
                labels = []
                series_data = {}

                for cat in categories[:6]:
                    labels.append(cat.get('name', '카테고리'))
                    values = cat.get('values', [])
                    for i, value in enumerate(values[:3]):  # 최대 3개 시리즈
                        series_name = f"시리즈 {i+1}"
                        if series_name not in series_data:
                            series_data[series_name] = []
                        series_data[series_name].append(value)

                datasets = []
                for series_name, data in series_data.items():
                    datasets.append({"label": series_name, "data": data})

                chart = {
                    "title": f"{query[:30]}... 카테고리별 비교",
                    "type": "groupedbar",
                    "palette": "corporate",
                    "data": {
                        "labels": labels,
                        "datasets": datasets
                    },
                    "source": "실제 추출 데이터",
                    "data_type": "real"
                }
                charts.append(chart)
                print(f"- GROUPEDBAR 차트 생성: {chart['title']}")

        # 5. 성능 지표 데이터 -> 레이더 차트
        metrics = extracted_data.get('metrics', [])
        if len(metrics) >= 3:
            print(f"- 성능 지표 발견: {len(metrics)}개")

            labels = []
            values = []

            for m in metrics[:8]:  # 레이더 차트는 최대 8개 축
                labels.append(m.get('name', '지표'))
                score = m.get('score', m.get('value', 50))
                values.append(float(score))

            chart = {
                "title": f"{query[:30]}... 성능 지표 분석",
                "type": "radar",
                "palette": "vibrant",
                "data": {
                    "labels": labels,
                    "datasets": [{"label": "성능 점수", "data": values}]
                },
                "source": "실제 추출 데이터",
                "data_type": "real"
            }
            charts.append(chart)
            print(f"- RADAR 차트 생성: {chart['title']}")

        # 6. 좌표 데이터 -> 스캐터/버블 차트
        points = extracted_data.get('coordinates', [])
        if len(points) >= 4:
            print(f"- 좌표 데이터 발견: {len(points)}개")

            scatter_data = []
            for p in points[:20]:  # 최대 20개 포인트
                x = p.get('x', 0)
                y = p.get('y', 0)
                size = p.get('size', 10)
                scatter_data.append({"x": x, "y": y, "r": size})

            chart_type = "bubble" if any(p.get('r', 10) != 10 for p in scatter_data) else "scatter"

            chart = {
                "title": f"{query[:30]}... 분포 분석",
                "type": chart_type,
                "palette": "gradient",
                "data": {
                    "datasets": [{"label": "데이터 포인트", "data": scatter_data}]
                },
                "source": "실제 추출 데이터",
                "data_type": "real"
            }
            charts.append(chart)
            print(f"- {chart_type.upper()} 차트 생성: {chart['title']}")

        # 7. 복합 데이터 -> 콤보 차트
        if len(numbers) >= 2 and len(trends) >= 2:
            print("- 복합 데이터 발견, 콤보 차트 생성")

            # 수치 데이터와 트렌드 데이터 결합
            combo_labels = []
            bar_data = []
            line_data = []

            # 라벨 통합 (최대 6개)
            for i in range(min(6, min(len(numbers), len(trends)))):
                combo_labels.append(f"항목 {i+1}")
                bar_data.append(float(numbers[i].get('value', 0)))

                # 트렌드에서 변화율 추출
                change_str = str(trends[i].get('change', '0'))
                import re
                change_numbers = re.findall(r'-?\d+\.?\d*', change_str)
                change_value = float(change_numbers[0]) if change_numbers else 0
                line_data.append(change_value)

            chart = {
                "title": f"{query[:30]}... 복합 분석",
                "type": "combo",
                "palette": "corporate",
                "data": {
                    "labels": combo_labels,
                    "datasets": [
                        {
                            "type": "bar",
                            "label": "수치 데이터",
                            "data": bar_data,
                            "yAxisID": "y"
                        },
                        {
                            "type": "line",
                            "label": "변화율 (%)",
                            "data": line_data,
                            "yAxisID": "y1"
                        }
                    ]
                },
                "source": "실제 추출 데이터",
                "data_type": "real"
            }
            charts.append(chart)
            print(f"- COMBO 차트 생성: {chart['title']}")

        print(f"- 총 {len(charts)}개 고급 차트 생성 완료")
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
당신은 식품 전문 AI 어시스턴트입니다. 사용자와의 이전 대화를 기억하고 개인화된 답변을 제공합니다.

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
   - 수식은 꼭 LaTex문법으로 표현(React에서 렌더링 가능하도록)
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
