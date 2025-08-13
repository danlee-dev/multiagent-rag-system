import json
import sys
import asyncio
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import re

from ..models.models import StreamingAgentState, SearchResult
from .worker_agents import DataGathererAgent, ProcessorAgent
from sentence_transformers import SentenceTransformer

# --- 페르소나 프롬프트 로드 ---
PERSONA_PROMPTS = {}
try:
    with open("agents/prompts/persona_prompts.json", "r", encoding="utf-8") as f:
        PERSONA_PROMPTS = json.load(f)
    print("OrchestratorAgent: 페르소나 프롬프트 로드 성공.")
except FileNotFoundError:
    print("경고: persona_prompts.json 파일을 찾을 수 없습니다.")
except json.JSONDecodeError:
    print("경고: persona_prompts.json 파일 파싱에 실패했습니다.")
# -----------------------------

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
1. **chat**: 간단한 질문, 일반적인 대화, 답담, 그리고 간단한 Web Search나, 내부 정보를 조회하여 답변할 수 있는 경우
   - 예: "안녕하세요", "감사합니다", "간단한 설명 요청", "최근 ~ 시세 알려줘", "최근 이슈 Top 10이 뭐야?"

2. **task**: 복합적인 분석, 데이터 수집, 리포트 생성이 필요, 정확히는 여러 섹션에 걸친 보고서 생성이 필요한 질문일 경우 또는, 자세한 영양 정보와 같은 RDB를 조회 해야하는 질문일 경우
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
    """고성능 비동기 스케줄러 및 지능형 계획 수립 Agent"""

    def __init__(self, model: str = "gemini-2.5-flash-lite", temperature: float = 0.2):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        self.llm_openai_mini = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
        self.data_gatherer = DataGathererAgent()
        self.processor = ProcessorAgent()
        self.personas = PERSONA_PROMPTS
        

    # ✅ 추가: 일관된 상태 메시지 생성을 위한 헬퍼 함수
    def _create_status_event(self, stage: str, sub_stage: str, message: str, details: Optional[Dict] = None) -> Dict:
        """표준화된 상태 이벤트 객체를 생성합니다."""
        return {
            "type": "status",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "agent": "OrchestratorAgent",
                "stage": stage,
                "sub_stage": sub_stage,
                "message": message,
                "details": details or {}
            }
        }

    async def _invoke_with_fallback(self, prompt: str, primary_model, fallback_model):
        """Gemini API rate limit 시 OpenAI로 fallback 처리"""
        try:
            result = await primary_model.ainvoke(prompt)
            return result
        except Exception as e:
            error_str = str(e).lower()
            rate_limit_indicators = ['429', 'quota', 'rate limit', 'exceeded', 'resource_exhausted']

            if any(indicator in error_str for indicator in rate_limit_indicators):
                print(f"OrchestratorAgent: Gemini API rate limit 감지, fallback 시도: {e}")
                if fallback_model:
                    try:
                        result = await fallback_model.ainvoke(prompt)
                        print("OrchestratorAgent: fallback 성공")
                        return result
                    except Exception as fallback_error:
                        print(f"OrchestratorAgent: fallback도 실패: {fallback_error}")
                        raise fallback_error
                else:
                    print("OrchestratorAgent: fallback 모델이 초기화되지 않음")
                    raise e
            else:
                raise e

    def _inject_context_into_query(self, query: str, context: Dict[int, str]) -> str:
        """'[step-X의 결과]' 플레이스홀더를 실제 컨텍스트로 교체하는 헬퍼 함수"""
        match = re.search(r"\[step-(\d+)의 결과\]", query)
        if match:
            step_index = int(match.group(1))
            if step_index in context:
                print(f"  - {step_index}단계 컨텍스트 주입: '{context[step_index][:]}...'")
                # 플레이스홀더를 이전 단계의 요약 결과로 치환
                return query.replace(match.group(0), f"이전 단계 분석 결과: '{context[step_index]}'")
        return query

    async def generate_plan(self, state: StreamingAgentState) -> StreamingAgentState:
        """페르소나 관점과 의존성을 반영하여 단계별 실행 계획(Hybrid Model)을 수립합니다."""
        print(f"\n>> Orchestrator: 지능형 단계별 계획 수립 시작")
        query = state["original_query"]
        current_date_str = datetime.now().strftime("%Y년 %m월 %d일")

        # 페르소나 정보 추출
        persona_name = state.get("persona", "기본")
        persona_info = self.personas.get(persona_name, {})
        persona_description = persona_info.get("description", "일반적인 분석가")
        print(f"  - 계획 수립에 '{persona_name}' 페르소나 관점 적용")

        planning_prompt = f"""
당신은 **'{persona_name}'의 유능한 AI 수석 보좌관**이자 **실행 계획 설계 전문가**입니다.
당신의 임무는 **'{persona_name}'의 관점에서 가장 중요한 정보를 찾아내기 위한, 논리적이고 효율적인 실행 계획**을 수립하는 것입니다.

**핵심 임무 (반드시 준수할 것)**:
1.  **페르소나 관점 반영 (What to ask)**: 생성하는 모든 하위 질문은 철저히 '{persona_name}'의 관심사에 부합해야 합니다. 이들의 핵심 니즈(예: 구매-원가/시세, 마케팅-트렌드/소비자, R&D-신원료/성분)를 만족시키는 데 집중하세요.
2.  **논리적 계획 수립 (How to ask)**: 페르소나의 관점에서 도출된 질문들의 선후 관계를 분석하여, 의존성이 있는 작업은 순차적으로, 없는 작업은 병렬로 처리하는 최적의 실행 단계를 설계하세요.

**사용자 원본 요청**: "{query}"
**현재 날짜**: {current_date_str}

---
**## 보유 도구 명세서 및 선택 가이드**

**1. rdb_search (PostgreSQL) - 1순위 활용**
   - **데이터 종류**: 정형 데이터 (테이블 기반: 식자재 **영양성분**, 농·축·수산물 **시세/가격/거래량** 등 수치 데이터).
   - **사용 시점**: 영양성분, 현재가격, 시세변동, 가격비교, 순위/평균/합계 등 **정확한 수치 연산**이 필요할 때.
   - **특징**: 날짜·지역·품목 컬럼으로 **필터/집계** 최적화. 다중 조건(where)과 group by, order by를 통한 **통계/랭킹** 질의에 적합. (관계 그래프 탐색은 비권장)
   - **예시 질의 의도**: "사과 비타민C 함량", "지난달 제주 감귤 평균가", "전복 가격 추이", "영양성분 상위 TOP 10"

**2. vector_db_search (Elasticsearch) - 1순위 활용**
   - **데이터 종류**: 비정형 데이터 (뉴스기사, 논문, 보고서 전문).
   - **사용 시점**: 시장분석, 정책문서, 트렌드분석, 배경정보, 실무가이드 등 서술형 정보나 분석이 필요할 때.
   - **특징**: 의미기반 검색으로 질문의 맥락과 가장 관련성 높은 문서를 찾아줌.

**3. graph_db_search (Neo4j) - 1순위 활용**
   - **데이터 종류**: **관계형(그래프) 데이터**. 노드: 품목(농산물/수산물/축산물), **Origin(원산지: city/region)**, **Nutrient(영양소)**.
     관계: `(품목)-[:isFrom]->(Origin)`, `(품목)-[:hasNutrient]->(Nutrient)`. 수산물은 품목 노드에 `fishState`(활어/선어/냉동/건어) 속성 존재.
   - **사용 시점**: **품목 ↔ 원산지**, **품목 ↔ 영양소**처럼 **엔티티 간 연결**이 핵심일 때. 지역·상태(fishState) 조건을 얹은 **원산지/특산품 탐색**.
   - **특징**: 지식그래프 경로 탐색에 최적화. 키워드는 **품목명/지역명/영양소/수산물 상태(fishState)**로 간결히 표현하고, 질문은 **"A의 원산지", "A의 영양소", "지역 B의 특산품/원산지", "활어 A의 원산지"**처럼 **관계를 명시**할수록 정확도 상승.
   - **예시 질의 의도**: "사과의 원산지", "오렌지의 영양소", "제주도의 감귤 원산지", "활어 문어 원산지", "경상북도 사과 산지 연결"

**4. web_search - 2순위 (최후의 수단)**
   - **데이터 종류**: 실시간 최신 정보, 외부 일반 지식.
   - **사용 조건**: 내부 DB(rdb, vector, graph)의 주제를 벗어나거나, 사용자가 명시적으로 '매우 최신' 정보를 요구할 때만 사용.
   - **사용 금지**: 농축수산물 시세, 영양정보, 원산지 등 내부 DB로 명백히 해결 가능한 질문.

**도구 선택 우선순위:**
1. **수치/통계 데이터 (식자재 영양성분, 농축수산물 시세)** → `rdb_search`
2. **관계/분류 정보 (품목-원산지, 품목-영양소, 지역-특산품, 수산물 상태별 원산지)** → `graph_db_search`
3. **분석/연구 문서 (시장분석, 소비자 조사)** → `vector_db_search`
4. **최신 트렌드/실시간 정보** → `web_search`

**각 도구별 적용 예시:**
- `rdb_search`: "식자재 영양성분", "농축수산물 시세", "가격 추이/비교", "영양성분 상위 TOP"
- `graph_db_search`: "사과의 원산지", "오렌지의 영양소", "제주-감귤 관계", "활어 문어 원산지", "지역별 특산품 연결"
- `vector_db_search`: "시장 분석 보고서", "소비자 행동 연구", "정책 문서"
- `web_search`: "2025년 최신 트렌드", "실시간 업계 동향"

---
**## 계획 수립을 위한 단계별 사고 프로세스 (반드시 준수할 것)**

**1단계: 목표 재해석**
- 사용자의 원본 요청을 분석하여, 최종 목표가 무엇인지 **'{persona_name}'의 입장에서** 명확하게 재정의합니다.
- (예: 원본 요청이 "만두 시장 조사"일 때, 페르소나가 '마케팅 담당자'라면 최종 목표는 '신제품 만두의 성공적인 시장 진입을 위한 마케팅 전략 보고서'로 구체화합니다.)

**2단계: 페르소나 기반 정보 식별 및 질문 구체화**
- 1단계에서 재정의한 목표를 달성하기 위해, **'{persona_name}'가 가장 중요하게 생각할 핵심 키워드(예: 구매 담당자-원가/시세, 마케터-트렌드/소비자, R&D-신원료/성분)를 중심으로** 필요한 정보 조각들을 식별합니다.
- 식별된 정보 조각들을 바탕으로, 각각 완결된 형태의 구체적인 질문으로 분해합니다.
- 생성된 모든 하위 질문은 원본 요청의 핵심 맥락(예: '대한민국', '건강기능식품')을 반드시 포함해야 합니다.

**3단계: 질문 간 의존성 분석 (가장 중요한 단계)**
- 분해된 질문들 간의 선후 관계를 분석합니다.
- **"어떤 질문이 다른 질문의 결과를 반드시 알아야만 제대로 수행될 수 있는가?"**를 판단합니다.
- 예시: `A분야의 시장 규모`를 알아야 `A분야의 주요 경쟁사`를 조사할 수 있으므로, 두 질문은 의존성이 있습니다. 반면, `A분야의 시장 규모`와 `B분야의 시장 규모` 조사는 서로 독립적입니다.

**4단계: 실행 단계 그룹화 (Grouping)**
- **Step 1**: 서로 의존성이 없는, 가장 먼저 수행되어야 할 병렬 실행 가능한 질문들을 배치합니다. (예: 시장 규모 조사, 최신 트렌드 조사)
- **Step 2 이후**: 이전 단계의 결과(`[step-X의 결과]` 플레이스홀더 사용)를 입력으로 사용하는 의존성 있는 질문들을 배치합니다. (예: 1단계에서 찾은 '성장 분야'의 경쟁사 조사)

**5단계: 각 질문에 대한 최적 도구 선택**
- '보유 도구 명세서'를 참고하여 각 하위 질문에 가장 적합한 도구를 **단 하나만** 신중하게 선택합니다.
    - **"성분", "영양", "시세", "가격"** 포함 → `rdb_search`
    - **"원산지", "관계", "제조사", "특산품", "fishState(활어/선어/냉동/건어)"** 포함 → `graph_db_search`
    - **"분석", "연구", "조사", "보고서", "동향"** 포함 → `vector_db_search`
    - **"최신 트렌드", "실시간 정보", "2025년"** 등 최신성 강조 시 → `web_search`

**6단계: 최종 JSON 형식화**
- 위에서 결정된 모든 내용을 아래 '최종 출력 포맷'에 맞춰 JSON으로 작성합니다.
- **중요**: `sub_questions` 키는 반드시 `execution_steps` 배열의 각 요소 안에만 존재해야 합니다.

---
**## 계획 수립 예시**

**요청**: "만두 신제품 개발을 위해, 해외 수출 사례와 최신 식품 트렌드에 맞는 원료를 추천해줘."

**생성된 계획(JSON)**:
{{
    "title": "신제품 만두 개발을 위한 시장 조사 및 원료 추천",
    "reasoning": "시장 조사와 트렌드 분석을 1단계에서 병렬로 수행한 뒤, 그 결과를 바탕으로 2단계에서 구체적인 원료를 추천하는 2단계 계획을 수립합니다.",
    "execution_steps": [
        {{
            "step": 1,
            "reasoning": "기반 정보인 '수출 사례'와 '최신 트렌드'를 병렬로 수집합니다.",
            "sub_questions": [
                {{"question": "대한민국 냉동만두 해외 수출 성공 사례 및 인기 제품 특징 분석", "tool": "vector_db_search"}},
                {{"question": "2025년 최신 글로벌 식품 트렌드 및 소비자 선호 원료", "tool": "web_search"}}
            ]
        }},
        {{
            "step": 2,
            "reasoning": "1단계 결과를 바탕으로 신제품에 적용할 구체적인 원료를 탐색합니다.",
            "sub_questions": [
                {{"question": "[step-1의 결과]를 바탕으로, '건강 및 웰빙' 트렌드에 맞는 식물성 만두 원료 추천", "tool": "vector_db_search"}},
                {{"question": "추천된 신규 원료(예: 대체육)의 영양성분 정보", "tool": "rdb_search"}}
            ]
        }}
    ]
}}

---
**## 최종 출력 포맷**

**중요 규칙**: 내부 DB를 최우선으로 활용하고, web_search는 신중하게 사용하세요. 반드시 아래 JSON 형식으로만 응답해야 합니다.

{{
    "title": "분석 보고서의 전체 제목",
    "reasoning": "이러한 단계별 계획을 수립한 핵심적인 이유.",
    "execution_steps": [
        {{
            "step": 1,
            "reasoning": "1단계 계획에 대한 설명. 병렬 실행될 작업들을 기술.",
            "sub_questions": [
                {{
                    "question": "1단계에서 병렬로 실행할 첫 번째 하위 질문",
                    "tool": "선택된 도구 이름"
                }},
                {{
                    "question": "1단계에서 병렬로 실행할 두 번째 하위 질문",
                    "tool": "선택된 도구 이름"
                }}
            ]
        }},
        {{
            "step": 2,
            "reasoning": "2단계 계획에 대한 설명. 1단계 결과에 의존함을 명시.",
            "sub_questions": [
                {{
                    "question": "2단계에서 실행할 하위 질문 (필요시 '[step-1의 결과]' 포함)",
                    "tool": "선택된 도구 이름"
                }}
            ]
        }}
    ]
}}
"""

        try:
            response = await self.llm.ainvoke(planning_prompt)
            content = response.content.strip()

            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)

            if json_match:
                json_str = json_match.group(1) if '```' in json_match.group(0) else json_match.group(0)
                plan = json.loads(json_str)
            else:
                raise ValueError("Valid JSON plan not found in response")

            print(f"  - 지능형 단계별 계획 생성 완료: {plan.get('title', '제목 없음')}")
            print("  - 계획 JSON:")
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            state["plan"] = plan

        except Exception as e:
            print(f"  - 지능형 계획 생성 실패, 단일 단계로 직접 검색 실행: {e}")
            state["plan"] = {
                "title": f"{query} 분석",
                "reasoning": "지능형 단계별 계획 수립에 실패하여, 사용자 원본 쿼리로 직접 검색을 실행합니다.",
                "execution_steps": [{
                    "step": 1,
                    "reasoning": "Fallback 실행",
                    "sub_questions": [{"question": query, "tool": "vector_db_search"}]
                }]
            }

        return state



    # ⭐ 핵심 수정: 요약된 내용이 아닌 전체 원본 내용을 LLM에게 제공
    async def _select_relevant_data_for_step(self, step_info: Dict, current_collected_data: List[SearchResult], query: str) -> List[int]:
        """현재 단계에서 수집된 데이터 중 관련 있는 것만 LLM이 선택 (전체 내용 기반)"""

        step_title = f"Step {step_info['step']}"
        step_reasoning = step_info.get('reasoning', '')
        sub_questions = step_info.get('sub_questions', [])

        # ⭐ 핵심 개선: 전체 원본 내용을 LLM에게 제공 (요약 없이)
        full_data_context = ""
        for i, res in enumerate(current_collected_data):
            source = getattr(res, 'source', 'Unknown')
            title = getattr(res, 'title', 'No Title')
            content = getattr(res, 'content', '')  # ⭐ 전체 내용 (요약 안함)

            full_data_context += f"""
    --- 데이터 인덱스 [{i}] ---
    출처: {source}
    제목: {title}
    전체 내용: {content}

    """

        # 현재 단계의 질문들
        questions_summary = ""
        for sq in sub_questions:
            questions_summary += f"- {sq.get('question', '')} ({sq.get('tool', '')})\n"

        # ⭐ 컨텍스트 길이 관리 (중요한 부분만 잘라내기)
        # 너무 길면 각 데이터당 최대 1000자로 제한
        if len(full_data_context) > 15000:
            print(f"  - 컨텍스트가 너무 긺 ({len(full_data_context)}자), 데이터별 1000자로 제한")

            truncated_context = ""
            for i, res in enumerate(current_collected_data):
                source = getattr(res, 'source', 'Unknown')
                title = getattr(res, 'title', 'No Title')
                content = getattr(res, 'content', '')[:1000]  # 1000자로 제한

                truncated_context += f"""
    --- 데이터 인덱스 [{i}] ---
    출처: {source}
    제목: {title}
    내용: {content}{"..." if len(getattr(res, 'content', '')) > 1000 else ""}

    """
            full_data_context = truncated_context

        selection_prompt = f"""
    당신은 데이터 분석 전문가입니다.
    현재 단계에서 수집된 데이터 중에서 **다음 단계에서 활용할 가치가 있는 핵심 데이터만** 선택해주세요.

    **전체 사용자 질문**: "{query}"

    **현재 단계 정보**:
    - {step_title}: {step_reasoning}
    - 실행한 질문들:
    {questions_summary}

    **수집된 전체 데이터** (전체 내용 포함):
    {full_data_context[:12000]}

    **선택 기준**:
    1. **내용을 꼼꼼히 읽고** 현재 단계의 목적과 직접적으로 관련된 데이터
    2. 향후 단계에서 참고할 가치가 있는 **실질적인 정보**가 포함된 데이터
    3. 제목만 보고 판단하지 말고 **실제 내용의 질과 관련성** 확인
    4. 중복되거나 관련성이 낮은 데이터는 제외
    5. 최대 10개 이내로 선별 (품질 우선)

    **중요**:
    - 각 데이터의 **전체 내용을 읽고** 관련성을 판단하세요
    - 단순히 제목이나 출처만 보고 결정하지 마세요
    - 실제로 유용한 정보가 담긴 데이터만 선택하세요

    다음 JSON 형식으로만 응답하세요:
    {{
        "selected_indexes": [0, 2, 5, 8],
        "reasoning": "각 선택된 데이터가 왜 중요한지 구체적으로 설명",
        "rejected_reason": "제외된 데이터들의 주요 제외 이유"
    }}
    """

        try:
            response = await self._invoke_with_fallback(
                selection_prompt,
                self.llm,
                self.llm_openai_mini
            )

            # JSON 파싱
            result = json.loads(re.search(r'\{.*\}', response.content, re.DOTALL).group())
            selected_indexes = result.get("selected_indexes", [])
            reasoning = result.get("reasoning", "")
            rejected_reason = result.get("rejected_reason", "")

            # 인덱스 유효성 검증
            max_index = len(current_collected_data) - 1
            valid_indexes = [idx for idx in selected_indexes if isinstance(idx, int) and 0 <= idx <= max_index]

            print(f"  - LLM 데이터 선택 완료 (전체 내용 기반):")
            print(f"    선택된 인덱스: {valid_indexes}")
            print(f"    선택 이유: {reasoning}")
            print(f"    제외 이유: {rejected_reason}")

            # 선택된 데이터 미리보기
            print(f"  - 선택된 데이터 목록:")
            for idx in valid_indexes[:5]:  # 처음 5개만
                data_item = current_collected_data[idx]
                print(f"    [{idx:2d}] {getattr(data_item, 'source', 'Unknown'):10s} | {getattr(data_item, 'title', 'No Title')[:60]}")

            return valid_indexes

        except Exception as e:
            print(f"  - LLM 데이터 선택 실패: {e}")
            # fallback: 현재 단계에서 수집된 모든 데이터 유지
            return list(range(len(current_collected_data)))


    async def _reselect_indexes_after_recollection(self, section_info: Dict, all_data: List[SearchResult], previous_selected: List[int], query: str) -> List[int]:
        """데이터 재수집 후 해당 섹션을 위한 인덱스 재선택"""

        section_title = section_info.get('section_title', '섹션')
        content_type = section_info.get('content_type', 'synthesis')

        # 전체 데이터 요약 (인덱스 포함)
        data_summary = ""
        for i, res in enumerate(all_data):
            source = getattr(res, 'source', 'Unknown')
            title = getattr(res, 'title', 'No Title')
            content = getattr(res, 'content', '')[:150]

            # 새로 추가된 데이터인지 표시
            is_new = i >= len(all_data) - 10  # 마지막 10개는 새 데이터로 가정
            marker = "[NEW]" if is_new else ""

            data_summary += f"[{i:2d}]{marker} [{source}] {title}: {content}...\n"

        reselection_prompt = f"""
    당신은 데이터 분석 전문가입니다.
    데이터 재수집이 완료된 후, **특정 섹션을 위해** 가장 적합한 데이터들을 다시 선택해주세요.

    **전체 사용자 질문**: "{query}"

    **현재 생성할 섹션 정보**:
    - 섹션 제목: "{section_title}"
    - 컨텐츠 타입: "{content_type}"

    **이전에 선택된 인덱스**: {previous_selected}

    **현재 전체 데이터** (인덱스: 0부터 시작, [NEW] = 새로 추가된 데이터):
    {data_summary[:6000]}

    **재선택 기준**:
    1. **"{section_title}" 섹션 주제와 직접 관련된 데이터 우선 선택**
    2. **새로 추가된 데이터([NEW])를 적극적으로 고려** - 이 데이터들은 해당 섹션을 위해 특별히 수집된 것임
    3. **기존 선택된 데이터 중 여전히 관련성 높은 것들도 유지**
    4. **최대 10개 이내로 선별** (품질과 관련성 우선)
    5. **중복되거나 유사한 내용은 제외**

    **특별 고려사항**:
    - content_type이 "full_data_for_chart"인 경우: 수치, 통계, 트렌드 데이터 우선
    - content_type이 "synthesis"인 경우: 다양한 관점의 종합적 정보 우선

    다음 JSON 형식으로만 응답하세요:
    {{
        "reselected_indexes": [2, 5, 8, 12, 15],
        "reasoning": "재선택 이유와 새 데이터 활용 방안",
        "new_data_count": 3,
        "kept_from_previous": 2
    }}
    """

        try:
            response = await self._invoke_with_fallback(
                reselection_prompt,
                self.llm,
                self.llm_openai_mini
            )

            # JSON 파싱
            result = json.loads(re.search(r'\{.*\}', response.content, re.DOTALL).group())
            reselected_indexes = result.get("reselected_indexes", [])
            reasoning = result.get("reasoning", "")
            new_data_count = result.get("new_data_count", 0)
            kept_count = result.get("kept_from_previous", 0)

            # 인덱스 유효성 검증
            max_index = len(all_data) - 1
            valid_indexes = [idx for idx in reselected_indexes if isinstance(idx, int) and 0 <= idx <= max_index]

            print(f"  - 섹션 '{section_title}' 인덱스 재선택 완료:")
            print(f"    재선택된 인덱스: {valid_indexes}")
            print(f"    새 데이터 활용: {new_data_count}개")
            print(f"    기존 데이터 유지: {kept_count}개")
            print(f"    선택 이유: {reasoning}")

            return valid_indexes

        except Exception as e:
            print(f"  - 인덱스 재선택 실패: {e}")
            # fallback: 기존 선택 + 새 데이터 일부
            fallback_indexes = previous_selected.copy()
            new_data_start = max(0, len(all_data) - 10)  # 마지막 10개는 새 데이터
            fallback_indexes.extend(list(range(new_data_start, len(all_data))))
            return list(set(fallback_indexes))  # 중복 제거

    async def execute_report_workflow(self, state: StreamingAgentState) -> AsyncGenerator[str, None]:
        """단계별 계획에 따라 순차적, 병렬적으로 데이터 수집 및 보고서 생성"""
        query = state["original_query"]

        # --- 추가: 페르소나 확인 및 상태 알림 ---
        # 사용자가 선택한 페르소나가 state에 이미 포함되어 있다고 가정합니다.
        # 예: state['persona'] = '구매 담당자'
        selected_persona = state.get("persona")
        if not selected_persona or selected_persona not in self.personas:
            print(f"경고: 유효하지 않거나 지정되지 않은 페르소나 ('{selected_persona}'). '기본'으로 설정합니다.")
            selected_persona = "기본"
            state["persona"] = selected_persona

        yield self._create_status_event("PLANNING", "PERSONA_CONFIRMED", f"'{selected_persona}' 페르소나로 보고서 생성을 시작합니다.")

        # 차트 카운터 초기화
        state['chart_counter'] = 0

        # 1. 단계별 계획 수립
        yield self._create_status_event("PLANNING", "GENERATE_PLAN_START", "분석 계획 수립 중...")
        state_with_plan = await self.generate_plan(state)
        plan = state_with_plan.get("plan", {})

        yield {"type": "plan", "data": {"plan": plan}}

        yield self._create_status_event("PLANNING", "GENERATE_PLAN_COMPLETE", "분석 계획 수립 완료.", details={
            "plan_title": plan.get('title'),
            "plan_reasoning": plan.get('reasoning'),
            "step_count": len(plan.get("execution_steps", []))
        })

        await asyncio.sleep(0.01)

        # 2. 단계별 데이터 수집 실행
        execution_steps = plan.get("execution_steps", [])
        final_collected_data: List[SearchResult] = []
        step_results_context: Dict[int, str] = {}
        cumulative_selected_indexes: List[int] = []

        for i, step_info in enumerate(execution_steps):
            current_step_index = step_info["step"]
            yield self._create_status_event("GATHERING", "STEP_START", f"데이터 수집 ({i + 1}/{len(execution_steps)}) 시작.")

            tasks_for_this_step = []
            for sq in step_info.get("sub_questions", []):
                injected_query = self._inject_context_into_query(sq["question"], step_results_context)
                tasks_for_this_step.append({"tool": sq["tool"], "inputs": {"query": injected_query}})
            if not tasks_for_this_step:
                continue

            step_collected_data: List[SearchResult] = []
            async for event in self.data_gatherer.execute_parallel_streaming(tasks_for_this_step):
                if event["type"] == "search_results":
                    yield event
                elif event["type"] == "collection_complete":
                    step_collected_data = event["data"]["collected_data"]

            summary_of_step = " ".join([res.content for res in step_collected_data])
            step_results_context[current_step_index] = summary_of_step[:2000]
            final_collected_data.extend(step_collected_data)

            print(f">> {current_step_index}단계 완료: {len(step_collected_data)}개 데이터 수집. (총 {len(final_collected_data)}개)")

            if len(final_collected_data) > 0:
                yield self._create_status_event("PROCESSING", "FILTER_DATA_START", "수집 데이터 선별 중...")

                # reasoning을 반환하지 않으므로 selected_indexes만 받습니다.
                selected_indexes = await self._select_relevant_data_for_step(
                    step_info, final_collected_data, state["original_query"]
                )

                yield self._create_status_event("PROCESSING", "FILTER_DATA_COMPLETE", f"핵심 데이터 {len(selected_indexes)}개 선별 완료.", details={
                    "selected_indices": selected_indexes
                })
                cumulative_selected_indexes = sorted(list(set(cumulative_selected_indexes + selected_indexes)))

        # >> 핵심 수정: 전체 데이터 딕셔너리를 프론트로 먼저 전송
        print(f"\n>> 전체 데이터 딕셔너리 생성 및 전송")

        # 전체 데이터를 인덱스:데이터 형태의 딕셔너리로 변환
        full_data_dict = {}
        print(f"\n🔍 === FULL_DATA_DICT 생성 디버깅 ===")
        print(f"final_collected_data 총 개수: {len(final_collected_data)}")

        for idx, data in enumerate(final_collected_data):
            full_data_dict[idx] = {
                "title": getattr(data, 'title', 'No Title'),
                "content": getattr(data, 'content', ''),
                "source": getattr(data, 'source', 'Unknown'),
                "url": getattr(data, 'url', ''),
                "source_url": getattr(data, 'source_url', ''),
                "score": getattr(data, 'score', 0.0),
                "document_type": getattr(data, 'document_type', 'unknown')
            }

            # 첫 5개와 마지막 5개만 상세 로그
            if idx < 5 or idx >= len(final_collected_data) - 5:
                print(f"  [{idx}]: 제목='{getattr(data, 'title', 'No Title')[:50]}...' 출처='{getattr(data, 'source', 'Unknown')}'")

        print(f"전체 데이터 딕셔너리 키들: {list(full_data_dict.keys())}")
        print(f"전체 데이터 딕셔너리 크기: {len(full_data_dict)}개")

        # 전체 데이터 딕셔너리를 프론트로 전송
        print(f"\n🚀 === 프론트엔드로 FULL_DATA_DICT 전송 ===")
        print(f"전송할 데이터 구조:")
        print(f"  type: 'full_data_dict'")
        print(f"  data.data_dict 키들: {list(full_data_dict.keys())}")
        print(f"  data.data_dict 크기: {len(full_data_dict)}")

        # 샘플 데이터 확인 (첫 번째 것만)
        if full_data_dict:
            first_key = list(full_data_dict.keys())[0]
            first_item = full_data_dict[first_key]
            print(f"  샘플 [{first_key}]: 제목='{first_item['title'][:30]}...' 출처='{first_item['source']}'")

        yield {"type": "full_data_dict", "data": {"data_dict": full_data_dict}}

        # 3. 섹션별 데이터 상태 분석 및 보고서 구조 설계
        # 중복된 full_data_dict 생성 및 전송 제거 (이미 위에서 처리됨)

        yield self._create_status_event("PROCESSING", "DESIGN_STRUCTURE_START", "보고서 구조 설계 중...")

        design = None
        async for result in self.processor.process("design_report_structure", final_collected_data, cumulative_selected_indexes, query, state=state):
            if result.get("type") == "result":
                design = result.get("data")
                break

        if not design or "structure" not in design or not design["structure"]:
            yield {"type": "error", "data": {"message": "보고서 구조 설계에 실패했습니다."}}
            return
        
        section_titles = [s.get('section_title', '제목 없음') for s in design.get('structure', [])]
        yield self._create_status_event("PROCESSING", "DESIGN_STRUCTURE_COMPLETE", "보고서 구조 설계 완료.", details={
            "report_title": design.get("title"),
            "section_titles": section_titles
        })

        # 보고서 제목을 가장 먼저 스트리밍
        yield {"type": "content", "data": {"chunk": f"# {design.get('title', query)}\n\n---\n\n"}}

        # 4. 섹션별 생성 루프
        for i, section in enumerate(design.get("structure", [])):
            section_title = section.get('section_title', f'섹션 {i+1}')
            use_contents = section.get("use_contents", [])

            yield self._create_status_event("GENERATING", "GENERATE_SECTION_START", f"'{section_title}' 섹션 생성 중...", details={
                "section_index": i,
                "section_title": section_title,
                "using_indices": use_contents
            })

            # >> 단순화: section_data_list만 생성 (차트용)
            section_data_list = []
            for actual_index in use_contents:
                if 0 <= actual_index < len(final_collected_data):
                    section_data_list.append(final_collected_data[actual_index])

            print(f"\n🔍 === 섹션 '{section_title}' 생성 디버깅 ===")
            print(f"   사용할 실제 인덱스: {use_contents}")
            print(f"   final_collected_data 길이: {len(final_collected_data)}")
            print(f"   full_data_dict 키들: {list(full_data_dict.keys())}")

            # 사용되는 인덱스들의 실제 데이터 확인
            for idx in use_contents[:3]:  # 처음 3개만
                if idx in full_data_dict:
                    print(f"   [{idx}] 제목: '{full_data_dict[idx]['title'][:50]}...'")
                else:
                    print(f"   [❌{idx}] 인덱스가 full_data_dict에 없음!")

            # 섹션 생성 (전체 데이터 딕셔너리와 사용 인덱스 전달)
            buffer = ""
            section_content_generated = False
            try:
                async for chunk in self.processor.generate_section_streaming(
                    section, full_data_dict, query, use_contents, state=state
                ):
                    section_content_generated = True
                    buffer += chunk

                    # 차트 생성 처리 (기존 로직 유지)
                    if "[GENERATE_CHART]" in buffer:
                        parts = buffer.split("[GENERATE_CHART]", 1)

                        if parts[0]:
                            yield {"type": "content", "data": {"chunk": parts[0]}}

                        buffer = parts[1]

                        yield self._create_status_event("GENERATING", "GENERATE_CHART_START", f"'{section_title}' 차트 생성 중...")

                        # 차트 생성 과정의 상태 메시지를 위한 콜백 (상태만 반환, yield 없음)
                        async def chart_yield_callback(event_data):
                            # 차트 생성 상태 로그 출력 (yield 없이 상태만 처리)
                            print(f"차트 생성 상태: {event_data}")
                            return event_data

                        chart_data = None
                        async for result in self.processor.process("create_chart_data", section_data_list, section_title, buffer, "", chart_yield_callback, state=state):
                            if result.get("type") == "chart":
                                chart_data = result.get("data")
                                break

                        if chart_data and "error" not in chart_data:
                            current_chart_index = state.get('chart_counter', 0)
                            chart_placeholder = f"\n\n[CHART-PLACEHOLDER-{current_chart_index}]\n\n"
                            yield {"type": "content", "data": {"chunk": chart_placeholder}}
                            yield {"type": "chart", "data": chart_data}
                            state['chart_counter'] = current_chart_index + 1
                        else:
                            print(f"   차트 생성 실패: {chart_data}")
                            yield {"type": "status", "data": {"message": f"'{section_title}' 차트 생성이 완료되지 못했습니다. 텍스트로 계속 진행합니다."}}
                            yield {"type": "content", "data": {"chunk": "\n\n*[데이터 부족으로 차트 표시가 제한됩니다]*\n\n"}}

                    else:
                        potential_chart_marker = "[GENERATE_CHART]"
                        has_partial_marker = any(potential_chart_marker.startswith(buffer[-i:]) for i in range(1, min(len(buffer)+1, len(potential_chart_marker)+1)) if buffer[-i:])

                        should_flush = (
                            not has_partial_marker and (
                                len(buffer) >= 120 or
                                buffer.endswith(('.', '!', '?', '\n', '다.', '요.', '니다.', '습니다.', '됩니다.', '있습니다.')) or
                                '\n\n' in buffer
                            )
                        )

                        if should_flush:
                            yield {"type": "content", "data": {"chunk": buffer}}
                            buffer = ""

                if buffer.strip():
                    yield {"type": "content", "data": {"chunk": buffer}}

                if not section_content_generated:
                    print(f">> 경고: 섹션 '{section_title}' 내용 생성 실패")
                    yield {"type": "content", "data": {"chunk": f"*'{section_title}' 섹션 생성 중 문제가 발생했습니다.*\n\n"}}

            except Exception as e:
                print(f">> 섹션 생성 중 오류 발생: {e}")
                yield {"type": "content", "data": {"chunk": f"*'{section_title}' 섹션 생성 중 오류가 발생했습니다: {str(e)}*\n\n"}}

            yield {"type": "content", "data": {"chunk": "\n\n"}}

        # 워크플로우 완료 후 출처 정보 설정 (실제 사용된 인덱스만)
        # 모든 섹션에서 사용된 인덱스들을 수집
        used_indexes = set()
        for section in design.get("structure", []):
            use_contents = section.get("use_contents", [])
            used_indexes.update(use_contents)

        print(f">> 실제 사용된 인덱스들: {sorted(used_indexes)}")
        print(f">> final_collected_data 길이: {len(final_collected_data) if final_collected_data else 0}")
        print(f">> used_indexes 길이: {len(used_indexes)}")

        # sources 이벤트는 더 이상 보내지 않음 (full_data_dict만 사용)
        # 대신 사용된 인덱스 정보만 로깅
        if final_collected_data and used_indexes:
            print(f">> 보고서에서 실제 사용된 인덱스들: {sorted(used_indexes)}")
            print(f">> 총 {len(used_indexes)}개 출처 사용 (전체 {len(final_collected_data)}개 중)")

        yield {"type": "complete", "data": {
            "message": "보고서 생성 완료"
        }}

    async def _update_use_contents_after_recollection(
    self,
    section_info: Dict,
    all_data: List[SearchResult],
    original_indexes: List[int],
    new_data_indexes: List[int],
    query: str
    ) -> List[int]:
        """보강 후 해당 섹션의 use_contents를 LLM이 업데이트 (전체 내용 기반)"""

        section_title = section_info.get('section_title', '섹션')

        # ⭐ 핵심 개선: 전체 내용을 LLM에게 제공
        data_summary = ""

        # 기존 데이터 (전체 내용)
        data_summary += "=== 기존 선택된 데이터 (전체 내용) ===\n"
        for idx in original_indexes[:3]:  # 처음 3개만 (길이 제한)
            if 0 <= idx < len(all_data):
                res = all_data[idx]
                content = getattr(res, 'content', '')[:800]  # 800자로 제한
                data_summary += f"""
    [{idx:2d}] [{getattr(res, 'source', 'Unknown')}] {getattr(res, 'title', 'No Title')}
    내용: {content}{"..." if len(getattr(res, 'content', '')) > 800 else ""}

    """

        # 새 데이터 (전체 내용)
        data_summary += "=== 새로 추가된 데이터 (전체 내용) ===\n"
        for idx in new_data_indexes:
            if 0 <= idx < len(all_data):
                res = all_data[idx]
                content = getattr(res, 'content', '')[:800]  # 800자로 제한
                data_summary += f"""
    [{idx:2d}] [NEW] [{getattr(res, 'source', 'Unknown')}] {getattr(res, 'title', 'No Title')}
    내용: {content}{"..." if len(getattr(res, 'content', '')) > 800 else ""}

    """

        update_prompt = f"""
    "{section_title}" 섹션을 위해 기존 데이터와 새로 추가된 데이터의 **전체 내용을 읽고** 가장 적합한 데이터들을 선택해주세요.

    **섹션**: "{section_title}"
    **전체 질문**: "{query}"

    {data_summary[:8000]}

    **선택 기준**:
    1. **각 데이터의 전체 내용을 읽고** 섹션 주제와의 관련성 판단
    2. 제목만 보고 결정하지 말고 **실제 내용의 질과 관련성** 확인
    3. 새 데이터는 해당 섹션을 위해 특별히 수집된 것이므로 적극 고려
    4. 실제로 유용한 정보가 담긴 데이터만 최대 8개 선별

    **원본**: {original_indexes}
    **새 데이터**: {new_data_indexes}

    JSON으로만 응답:
    {{
        "updated_use_contents": [0, 2, 5, 8],
        "reasoning": "각 데이터를 선택/제외한 구체적 이유 (내용 기반)"
    }}
    """

        try:
            response = await self._invoke_with_fallback(update_prompt, self.llm, self.llm_openai_mini)
            result = json.loads(re.search(r'\{.*\}', response.content, re.DOTALL).group())

            updated_indexes = result.get("updated_use_contents", [])
            reasoning = result.get("reasoning", "")

            # 유효성 검증
            max_index = len(all_data) - 1
            valid_indexes = [idx for idx in updated_indexes if isinstance(idx, int) and 0 <= idx <= max_index]

            print(f"  - use_contents 업데이트 완료 (전체 내용 기반):")
            print(f"    최종 선택: {valid_indexes}")
            print(f"    선택 이유: {reasoning}")

            return valid_indexes

        except Exception as e:
            print(f"  - use_contents 업데이트 실패: {e}")
            # fallback: 원본 + 새 데이터 합치기 (최대 8개)
            combined = original_indexes + new_data_indexes
            return combined[:8]

