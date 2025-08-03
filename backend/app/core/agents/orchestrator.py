import json
import sys
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI

from ..models.models import StreamingAgentState, SearchResult
from .worker_agents import DataGathererAgent, ProcessorAgent


class TriageAgent:
    """요청 분류 및 라우팅 담당 Agent"""

    def __init__(self, model: str = "gemini-2.5-flash-lite", temperature: float = 0.1):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)

    async def classify_request(self, query: str, state: StreamingAgentState) -> StreamingAgentState:
        """요청을 분석하여 flow_type 결정"""
        print(f"\n>> Triage: 요청 분류 시작 - '{query}'")
        sys.stdout.flush()

        classification_prompt = f"""
사용자 요청을 분석하여 적절한 처리 방식을 결정하세요:

사용자 요청: {query}

분류 기준:
1. **chat**: 간단한 질문, 일반적인 대화, 답담, 기존 정보로 답변 가능
   - 예: "안녕하세요", "감사합니다", "간단한 설명 요청"

2. **task**: 복합적인 분석, 데이터 수집, 간단한 웹 검색, 리포트 생성이 필요
   - 예: "~를 분석해줘", "~에 대한 자료를 찾아줘", "보고서 작성"

JSON으로 응답:
{{
    "flow_type": "chat" 또는 "task",
    "reasoning": "분류 근거 설명",
}}
"""

        try:
            response = await self.llm.ainvoke(classification_prompt)
            response_content = response.content.strip()

            # JSON 응답 추출 시도
            classification = None
            try:
                # 직접 파싱 시도
                classification = json.loads(response_content)
            except json.JSONDecodeError:
                # JSON 블록 찾기 시도
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    classification = json.loads(json_match.group())
                else:
                    raise ValueError("Valid JSON not found in response")

            # 필수 필드 확인
            required_fields = ["flow_type", "reasoning"]
            for field in required_fields:
                if field not in classification:
                    raise ValueError(f"Missing required field: {field}")

            # state 업데이트 (딕셔너리 접근 방식)
            state["flow_type"] = classification["flow_type"]
            state["metadata"].update({
                "triage_reasoning": classification["reasoning"],
                "classified_at": datetime.now().isoformat()
            })

            print(f"- 분류 결과: {classification['flow_type']}")
            print(f"- 근거: {classification['reasoning']}")
            sys.stdout.flush()

        except Exception as e:
            print(f"- 분류 실패, 기본값(task) 적용: {e}")
            sys.stdout.flush()
            state["flow_type"] = "task"  # 기본값
            state["metadata"].update({
                "triage_error": str(e),
                "classified_at": datetime.now().isoformat()
            })

        return state


class OrchestratorAgent:
    """전체 워크플로우 조율 담당 Agent"""

    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.2):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)

        # Worker agents 초기화
        self.data_gatherer = DataGathererAgent()
        self.processor = ProcessorAgent()

    async def generate_plan(self, state: StreamingAgentState) -> StreamingAgentState:
        """실행 계획 수립"""
        print(f"\n>> Orchestrator: 실행 계획 수립")

        query = state["original_query"]
        feedback = state.get('replan_feedback', '피드백 없음')

        planning_prompt = f"""
사용자 요청에 대한 구체적이고 실행 가능한 계획을 수립하세요:

요청: {query}

이전 피드백: {feedback}

사용 가능한 도구들:
**tool - DataGathererAgent용:**
- web_search: 웹에서 최신 정보 검색
- vector_db_search: 내부 벡터 DB 검색
- graph_db_search: 그래프 DB 관계 검색
- rdb_search: 관계형 DB 데이터 검색
- scrape_content: 특정 웹페이지 스크래핑

**processor_type - ProcessorAgent용:**
- evaluate_criticism: 정보 충분성 평가
- integrate_context: 다중 소스 정보 통합
- generate_report: 최종 보고서 생성
- create_charts: 데이터 시각화 차트 생성

다음 형태로 구체적인 실행 계획을 JSON으로 생성하세요:
{{
    "title": "보고서/분석 제목",
    "reasoning": "이 계획을 선택한 이유와 접근 방식",
    "steps": [
        {{
            "step_id": 0,
            "description": "단계에 대한 구체적 설명",
            "agent": "DataGathererAgent" 또는 "ProcessorAgent",
            "inputs": {{
                "tool": "사용할 도구명" (DataGatherer인 경우),
                "query": "구체적인 검색 쿼리" (DataGatherer인 경우),
                "processor_type": "처리 타입" (Processor인 경우),
                "source_steps": [의존하는 이전 단계 ID들] (Processor인 경우)
            }}
        }}
    ]
}}

예시 (스마트팜 관련 요청):
- 웹에서 최신 트렌드 검색 → 내부 DB에서 성공 사례 검색 → 정보 통합 → 최종 보고서 생성
- 각 단계는 명확한 목적과 구체적인 쿼리를 가져야 함
- step_id는 0부터 시작하는 순차적 번호
"""
        try:
            response = await self.llm.ainvoke(planning_prompt)

            # JSON 추출
            content = response.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                plan = json.loads(json_str)

                # 기본 필드 검증 및 보완
                if "title" not in plan:
                    plan["title"] = f"{query} 분석 보고서"
                if "reasoning" not in plan:
                    plan["reasoning"] = "사용자 요청에 따른 체계적 분석 계획"
                if "steps" not in plan:
                    plan["steps"] = []

                print(f"  계획 생성 완료: {plan['title']}")
                print(f"  단계 수: {len(plan['steps'])}")
                sys.stdout.flush()

                state["plan"] = plan
                return state
            else:
                raise ValueError("JSON 형식을 찾을 수 없음")

        except Exception as e:
            print(f"계획 생성 실패, 기본 계획 사용: {e}")
            sys.stdout.flush()

            # 기본 계획 생성
            default_plan = {
                "title": f"{query} 기본 분석",
                "reasoning": "계획 생성 실패로 기본 워크플로우 사용",
                "steps": [
                    {
                        "step_id": 0,
                        "description": "웹에서 관련 정보 검색",
                        "agent": "DataGathererAgent",
                        "inputs": {
                            "tool": "web_search",
                            "query": query
                        }
                    },
                    {
                        "step_id": 1,
                        "description": "검색 결과 통합 및 요약",
                        "agent": "ProcessorAgent",
                        "inputs": {
                            "processor_type": "summarize_and_integrate",
                            "source_steps": [0]
                        }
                    },
                    {
                        "step_id": 2,
                        "description": "최종 보고서 생성",
                        "agent": "ProcessorAgent",
                        "inputs": {
                            "processor_type": "generate_report",
                            "source_steps": [1]
                        }
                    }
                ]
            }

            state["plan"] = default_plan
            return state
