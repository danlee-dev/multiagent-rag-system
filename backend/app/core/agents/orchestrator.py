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

        # 환경 변수에서 ReAct 사용 여부 확인
        import os
        use_react = os.getenv('USE_REACT_AGENT', 'ture').lower() == 'true'
        self.processor = ProcessorAgent(use_react=use_react)

    async def generate_plan(self, state: StreamingAgentState) -> StreamingAgentState:
        """실행 계획 수립 - 쿼리 복잡도에 따른 적응적 계획"""
        print(f"\n>> Orchestrator: 실행 계획 수립")

        query = state["original_query"]
        feedback = state.get('replan_feedback', '피드백 없음')

        planning_prompt = f"""
사용자 요청에 대한 최적화된 실행 계획을 수립하세요:

요청: {query}
현재 날짜: 2025년 8월

이전 피드백: {feedback}

사용 가능한 도구들:
**DataGathererAgent 도구:**
- web_search: 웹에서 최신 정보 검색 (2024-2025 최신 데이터 우선)
- vector_db_search: 내부 벡터 DB 검색
- graph_db_search: 그래프 DB 관계 검색
- rdb_search: 관계형 DB 데이터 검색

**ProcessorAgent 처리:**
- integrate_context: 다중 소스 정보 통합 (여러 데이터 소스가 있을 때만)
- generate_report: 최종 보고서 생성
- create_charts: 데이터 시각화 차트 생성

**계획 수립 원칙:**

1. **단순 웹 검색 요청** (예: "최근 트렌드 조사해줘", "~에 대해 알려줘")
   → 2단계: web_search → generate_report

2. **복잡한 분석 요청** (예: "비교 분석", "통합 보고서", "시각화 포함")
   → 3-5단계: 데이터 수집 → 통합/분석 → 시각화/보고서

3. **다중 소스 필요 시에만** integrate_context 사용

쿼리 복잡도에 맞는 **최소한의 효율적인** 실행 계획을 JSON으로 생성하세요:

**단순 검색 예시:**
{{
    "title": "보고서 제목",
    "reasoning": "단순 웹 검색으로 충분한 요청이므로 2단계 계획",
    "steps": [
        {{
            "step_id": 0,
            "description": "웹에서 최신 정보 검색",
            "agent": "DataGathererAgent",
            "inputs": {{
                "tool": "web_search",
                "query": "구체적인 검색 쿼리 2024 2025 최신"
            }}
        }},
        {{
            "step_id": 1,
            "description": "검색 결과 기반 보고서 생성",
            "agent": "ProcessorAgent",
            "inputs": {{
                "processor_type": "generate_report",
                "source_steps": [0]
            }}
        }}
    ]
}}

**복잡 분석 예시:**
{{
    "title": "종합 분석 보고서",
    "reasoning": "다중 소스 통합 및 시각화가 필요한 복잡한 요청",
    "steps": [
        {{
            "step_id": 0,
            "description": "웹 검색",
            "agent": "DataGathererAgent",
            "inputs": {{"tool": "web_search", "query": "..."}}
        }},
        {{
            "step_id": 1,
            "description": "내부 DB 검색",
            "agent": "DataGathererAgent",
            "inputs": {{"tool": "vector_db_search", "query": "..."}}
        }},
        {{
            "step_id": 2,
            "description": "정보 통합",
            "agent": "ProcessorAgent",
            "inputs": {{"processor_type": "integrate_context", "source_steps": [0, 1]}}
        }},
        {{
            "step_id": 3,
            "description": "최종 보고서 생성",
            "agent": "ProcessorAgent",
            "inputs": {{"processor_type": "generate_report", "source_steps": [2]}}
        }}
    ]
}}

**중요:** 요청의 실제 복잡도를 정확히 분석하여 꼭 필요한 단계만 포함하세요!
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

                # 단계 수 제한 (성능 최적화)
                if len(plan["steps"]) > 6:
                    plan["steps"] = plan["steps"][:6]
                    plan["reasoning"] += " (성능 최적화를 위해 6단계로 제한)"

                print(f"  계획 생성 완료: {plan['title']}")
                print(f"  단계 수: {len(plan['steps'])}")

                # 상세 계획 디버깅 출력
                print(f"\n>> 계획 상세 정보:")
                print(f"  제목: {plan['title']}")
                print(f"  근거: {plan['reasoning']}")
                print(f"  전체 단계:")
                for step in plan['steps']:
                    print(f"    단계 {step['step_id']}: {step['description']}")
                    print(f"      에이전트: {step['agent']}")
                    if 'tool' in step['inputs']:
                        print(f"      도구: {step['inputs']['tool']}")
                        print(f"      쿼리: {step['inputs']['query']}")
                    elif 'processor_type' in step['inputs']:
                        print(f"      처리타입: {step['inputs']['processor_type']}")
                        if 'source_steps' in step['inputs']:
                            print(f"      소스단계: {step['inputs']['source_steps']}")

                print(f"  계획 생성 응답: {plan}")
                sys.stdout.flush()

                state["plan"] = plan
                return state
            else:
                raise ValueError("JSON 형식을 찾을 수 없음")

        except Exception as e:
            print(f"계획 생성 실패, 기본 계획 사용: {e}")
            sys.stdout.flush()

            # 간소화된 기본 계획 생성 (단순 웹 검색 → 보고서)
            default_plan = {
                "title": f"{query} 기본 분석",
                "reasoning": "계획 생성 실패로 기본 2단계 워크플로우 사용",
                "steps": [
                    {
                        "step_id": 0,
                        "description": "웹에서 최신 정보 검색",
                        "agent": "DataGathererAgent",
                        "inputs": {
                            "tool": "web_search",
                            "query": f"{query} 2024 2025 최신 현황"
                        }
                    },
                    {
                        "step_id": 1,
                        "description": "검색 결과 기반 보고서 생성",
                        "agent": "ProcessorAgent",
                        "inputs": {
                            "processor_type": "generate_report",
                            "source_steps": [0]
                        }
                    }
                ]
            }

            # 기본 계획 디버깅 출력
            print(f"\n>> 기본 계획 상세 정보:")
            print(f"  제목: {default_plan['title']}")
            print(f"  근거: {default_plan['reasoning']}")
            print(f"  전체 단계:")
            for step in default_plan['steps']:
                print(f"    단계 {step['step_id']}: {step['description']}")
                print(f"      에이전트: {step['agent']}")
                if 'tool' in step['inputs']:
                    print(f"      도구: {step['inputs']['tool']}")
                    print(f"      쿼리: {step['inputs']['query']}")
                elif 'processor_type' in step['inputs']:
                    print(f"      처리타입: {step['inputs']['processor_type']}")
                    if 'source_steps' in step['inputs']:
                        print(f"      소스단계: {step['inputs']['source_steps']}")

            print(f"  기본 계획 전체: {default_plan}")
            sys.stdout.flush()

            state["plan"] = default_plan
            return state
