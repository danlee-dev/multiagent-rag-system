import json
import sys
import asyncio
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
import re

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
        self.data_gatherer = DataGathererAgent()
        self.processor = ProcessorAgent()

    async def generate_plan(self, state: StreamingAgentState) -> StreamingAgentState:
        """맥락을 유지하며 쿼리를 분해하고 초기 데이터 수집 계획 수립"""
        print(f"\n>> Orchestrator: 지능형 계획 수립 시작")
        query = state["original_query"]
        current_date_str = datetime.now().strftime("%Y년 %m월 %d일")

        planning_prompt = f"""
당신은 사용자의 복잡한 요청을 명확한 하위 질문으로 분해하고, 각 질문을 해결하기 위한 최적의 데이터 수집 도구를 선택하여 실행 계획을 수립하는 AI 수석 아키텍트입니다.

**사용자 원본 요청**: "{query}"
**현재 날짜**: {current_date_str}

---
**## 보유 도구 명세서 및 선택 가이드**

**1. rdb_search (PostgreSQL) - 1순위 활용**
   - **데이터 종류**: 정형 데이터 (식자재 영양성분, 농축수산물 시세).
   - **사용 시점**: 영양성분, 현재가격, 시세변동, 가격비교, 영양비교 등 정확한 수치가 필요할 때.
   - **특징**: 실시간 데이터이며 가장 신뢰도 높은 수치를 제공. 검색어는 한국어 필수.

**2. vector_db_search (Elasticsearch) - 1순위 활용**
   - **데이터 종류**: 비정형 데이터 (뉴스기사, 논문, 보고서 전문).
   - **사용 시점**: 시장분석, 정책문서, 트렌드분석, 배경정보, 실무가이드 등 서술형 정보나 분석이 필요할 때.
   - **특징**: 의미기반 검색으로 질문의 맥락과 가장 관련성 높은 문서를 찾아줌.

**3. graph_db_search (Neo4j) - 1순위 활용**
   - **데이터 종류**: 관계형 데이터 (개체간 연결관계, 소속정보).
   - **사용 시점**: 원산지정보, 특산품조회, 공급망관계, 유통경로 등 개체 간의 관계 파악이 필요할 때.
   - **특징**: '제주도'와 '감귤'의 관계처럼 지식그래프를 탐색함.

**4. web_search - 2순위 (최후의 수단)**
   - **데이터 종류**: 실시간 최신 정보, 외부 일반 지식.
   - **사용 조건**: 내부 DB(rdb, vector, graph)의 주제를 벗어나거나, 사용자가 명시적으로 '매우 최신' 정보를 요구할 때만 사용.
   - **사용 금지**: 농축수산물 시세, 영양정보, 원산지 등 내부 DB로 명백히 해결 가능한 질문.

**도구 선택 우선순위:**
1. **수치/통계 데이터 (매출, 판매량, 가격)** → `rdb_search`
2. **관계/분류 정보 (업체-제품, 지역-특산품)** → `graph_db_search`
3. **분석/연구 문서 (시장분석, 소비자 조사)** → `vector_db_search`
4. **최신 트렌드/실시간 정보** → `web_search`

**각 도구별 적용 예시:**
- `rdb_search`: "건강기능식품 매출 현황", "제품별 판매량", "가격 정보"
- `graph_db_search`: "건강기능식품 제조업체", "제품 분류 체계", "지역별 특산품"
- `vector_db_search`: "시장 분석 보고서", "소비자 행동 연구", "정책 문서"
- `web_search`: "2025년 최신 트렌드", "실시간 업계 동향"

---
**## 계획 수립을 위한 단계별 사고 프로세스**

당신은 다음 4단계 사고 프로세스를 반드시 따라야 합니다.

* **1단계: 요청 의도 분석**: 사용자의 최종 목표가 무엇인지 파악합니다. (예: 시장 분석 보고서 작성)
* **2단계: 하위 질문 분해**: 최종 목표를 달성하기 위해 필요한 논리적인 하위 질문들을 3~5개로 분해합니다. **이때, 모든 하위 질문은 원본 요청의 핵심 맥락(예: '대한민국', '건강기능식품')을 반드시 포함해야 합니다.**
* **3단계: 도구 선택**: 각 하위 질문에 대해, 위 '보유 도구 명세서'를 참고하여 **가장 적합한 도구를 단 하나만** 선택합니다.
  - **질문에 "매출", "판매량", "가격", "수치"가 포함** → `rdb_search`
  - **질문에 "업체", "제조사", "분류", "관계"가 포함** → `graph_db_search`
  - **질문에 "분석", "연구", "조사", "보고서"가 포함** → `vector_db_search`
  - **질문에 "최신", "트렌드", "2024-2025"가 포함** → `web_search`
* **4단계: JSON 형식화**: 위에서 결정된 내용을 바탕으로 아래 JSON 출력 포맷에 맞게 최종 계획을 작성합니다.

**예시 검색 전략:**
- "제주도 감귤 영양성분과 가격" → `rdb_search`("제주도 감귤 영양성분"), `rdb_search`("제주도 감귤 가격"), `graph_db_search`("제주도 감귤 원산지 관계")로 분해
- "건강기능식품 시장 동향 분석" → `rdb_search`("건강기능식품 매출 데이터"), `vector_db_search`("건강기능식품 시장 트렌드 분석"), `graph_db_search`("건강기능식품 제조업체 관계"), `web_search`("2025년 건강기능식품 최신 동향")로 분해
- "소비자 구매 패턴 조사" → `vector_db_search`("소비자 건강기능식품 구매 행동 연구"), `rdb_search`("건강기능식품 판매량 통계"), `web_search`("2025년 소비자 건강기능식품 선호도")로 분해

**도구 선택 세부 기준:**
- **판매량, 매출, 가격, 수치 통계** → `rdb_search`
- **업체 관계, 제품 분류, 원산지 정보** → `graph_db_search`
- **시장 분석, 연구 보고서, 정책 문서** → `vector_db_search`
- **2024-2025년 최신 트렌드, 실시간 정보** → `web_search`

---
**## 최종 출력 포맷**

**중요 규칙**: 내부 DB를 최우선으로 활용하고, web_search는 신중하게 사용하세요. 반드시 아래 JSON 형식으로만 응답해야 합니다.

{{
    "title": "분석 보고서의 전체 제목",
    "reasoning": "이러한 계획을 전체적으로 수립한 핵심적인 이유.",
    "sub_questions": [
        {{
            "question": "맥락이 완벽하게 유지된 첫 번째 하위 질문",
            "tool": "선택된 도구 이름"
        }},
        {{
            "question": "맥락이 완벽하게 유지된 두 번째 하위 질문",
            "tool": "선택된 도구 이름"
        }}
    ]
}}
"""
        try:
            response = await self.llm.ainvoke(planning_prompt)
            content = response.content
            plan = json.loads(re.search(r'\{.*\}', content, re.DOTALL).group())

            print(f"  - 지능형 계획 생성 완료: {plan.get('title', '제목 없음')}")
            state["plan"] = plan
        except Exception as e:
            print(f"  - 지능형 계획 생성 실패, 원본 쿼리 직접 검색 실행: {e}")
            state["plan"] = {
                "title": f"{query} 원본 검색",
                "reasoning": "지능형 계획 수립에 실패하여, 사용자 원본 쿼리로 직접 검색을 실행합니다.",
                "sub_questions": [{"question": query, "tool": "vector_db_search"}]
            }

        return state

    async def execute_report_workflow(self, state: StreamingAgentState) -> AsyncGenerator[str, None]:
        """비동기 제어를 통해 순서가 보장되는 보고서 생성 워크플로우"""
        query = state["original_query"]

        # 차트 카운터 초기화
        state['chart_counter'] = 0

        yield {"type": "status", "data": {"message": "분석 계획 수립 중..."}}
        state_with_plan = await self.generate_plan(state)
        plan = state_with_plan.get("plan", {})
        yield {"type": "plan", "data": {"plan": plan}}

        # --- 데이터 수집 로직 변경 ---

        collected_data = []
        all_sub_questions = plan.get("sub_questions", [])

        # 1. 태스크를 '순차 처리'와 '병렬 처리'로 분리
        vector_db_tasks = []
        parallel_tasks = []
        for sq in all_sub_questions:
            task = {"tool": sq["tool"], "inputs": {"query": sq["question"]}}
            if sq["tool"] == "vector_db_search":
                vector_db_tasks.append(task)
            else:
                parallel_tasks.append(task)

        # 2. 병렬 처리 태스크 먼저 실행 (vector_db_search 제외)
        if parallel_tasks:
            yield {"type": "status", "data": {"message": f"{len(parallel_tasks)}개 작업 병렬 수집 시작..."}}
            parallel_results_dict = await self.data_gatherer.execute_parallel(parallel_tasks)
            parallel_results_list = [item for sublist in parallel_results_dict.values() for item in sublist]
            collected_data.extend(parallel_results_list)
            yield {"type": "status", "data": {"message": f"병렬 작업 완료. 현재까지 {len(collected_data)}개 정보 수집."}}

        # 3. vector_db_search 태스크를 순차적으로 실행
        if vector_db_tasks:
            yield {"type": "status", "data": {"message": f"{len(vector_db_tasks)}개 벡터 DB 작업 순차 수집 시작..."}}
            for i, task in enumerate(vector_db_tasks):
                yield {"type": "status", "data": {"message": f"벡터 DB 작업 ({i+1}/{len(vector_db_tasks)}) 실행 중: '{task['inputs']['query']}'"}}
                sequential_result = await self.data_gatherer.execute(task["tool"], task["inputs"])
                collected_data.extend(sequential_result)
            yield {"type": "status", "data": {"message": f"벡터 DB 작업 완료. 현재까지 {len(collected_data)}개 정보 수집."}}

        print(f"\n>> 총 {len(collected_data)}개의 초기 데이터 수집 완료.")
        yield {"type": "status", "data": {"message": f"총 {len(collected_data)}개의 정보를 수집했습니다."}}

        # ------------------

        # 3. 섹션별 데이터 상태 분석 및 보고서 구조 설계
        yield {"type": "status", "data": {"message": "수집된 정보를 바탕으로 보고서 목차를 설계합니다..."}}
        design = await self.processor.process("design_report_structure", collected_data, query)

        if not design or "structure" not in design or not design["structure"]:
            yield {"type": "error", "data": {"message": "보고서 구조를 설계하는 데 실패했습니다. 수집된 데이터가 부족하거나 분석할 수 없는 형식일 수 있습니다."}}
            return

        # 보고서 제목을 가장 먼저 스트리밍
        yield {"type": "content_chunk", "data": {"chunk": f"# {design.get('title', query)}\n\n"}}

        # 4. 데이터 재수집이 필요한 섹션에 대해 백그라운드 작업 실행
        recollection_tasks: Dict[int, asyncio.Task] = {}
        processed_recollection_indices = set()
        for i, section in enumerate(design.get("structure", [])):
            if not section.get("is_sufficient", True):
                feedback = section.get("feedback_for_gatherer", "")
                if feedback:
                    # 상태: 특정 섹션 데이터 보강 시작
                    yield {"type": "status", "data": {"message": f"'{section.get('section_title')}' 섹션의 데이터 보강을 시작합니다..."}}
                    recollection_tasks[i] = asyncio.create_task(
                        self.data_gatherer.execute("vector_db_search", {"query": feedback})
                    )


        # 5. 순차적 생성 및 비동기 대기 루프
        final_data_pool = list(collected_data) # [개선점 2] 재사용을 위해 데이터 풀을 한 번만 생성
        for i, section in enumerate(design.get("structure", [])):
            # [개선점 1] 섹션 내용 생성 전, 필요한 모든 데이터가 준비될 때까지 먼저 대기
            pending_tasks_for_this_step = {
                idx: task for idx, task in recollection_tasks.items() if idx <= i and not task.done()
            }
            if pending_tasks_for_this_step:
                # 상태: 데이터 재수집 대기
                yield {"type": "status", "data": {"message": f"'{section.get('section_title')}' 섹션 생성을 위해 데이터 보강 완료를 기다립니다..."}}
                print(f">> {i+1}번 섹션 생성 전, 재수집 작업 완료 대기...")
                await asyncio.gather(*pending_tasks_for_this_step.values())

            # [개선점 3] 완료된 작업 결과 수집 및 안정성 강화
            for idx, task in recollection_tasks.items():
                if idx == i and task.done(): # 현재 섹션에 해당하는 재수집 작업만 처리
                    try:
                        new_data = task.result()
                        final_data_pool.extend(new_data) # 데이터 풀에 추가
                        print(f">> 백그라운드 재수집 완료 (섹션 {idx+1}): {len(new_data)}개 데이터 추가")
                        # 상태: 데이터 보강 완료
                        yield {"type": "status", "data": {"message": f"'{section.get('section_title')}' 섹션의 데이터 보강이 완료되었습니다."}}
                    except Exception as e:
                        print(f">> 백그라운드 재수집 실패 (섹션 {idx+1}): {e}")
                        # 상태: 데이터 보강 실패
                        yield {"type": "status", "data": {"message": f"'{section.get('section_title')}' 섹션의 데이터 보강에 실패했습니다."}}

            # [개선점 1] 데이터가 모두 준비된 후, 섹션 제목 출력
            yield {"type": "content_chunk", "data": {"chunk": f"## {section.get('section_title', '제목 없음')}\n"}}

            # [개선점 2] 섹션 생성 시, 전체 데이터 풀을 참조로 전달하여 효율성 증대
            buffer = ""

            async for chunk in self.processor.generate_section_streaming(section, final_data_pool, query):
                buffer += chunk
                if "[GENERATE_CHART]" in buffer:
                    parts = buffer.split("[GENERATE_CHART]", 1)

                    # [수정됨] yield하는 모든 데이터를 딕셔너리 형식으로 통일
                    yield {"type": "content_chunk", "data": {"chunk": parts[0]}}
                    buffer = parts[1]

                    # [추가됨] 차트 생성 시작 상태 메시지
                    yield {"type": "status", "data": {"message": f"'{section.get('section_title')}' 섹션의 차트를 생성합니다..."}}
                    chart_description = f"Chart for section titled '{section.get('section_title')}': {section.get('description')}"
                    chart_data = await self.processor.process("create_chart_data", final_data_pool, chart_description)

                    if "error" not in chart_data:
                        # 차트 플레이스홀더를 콘텐츠에 삽입
                        current_chart_index = state.get('chart_counter', 0)
                        chart_placeholder = f"\n\n[CHART-PLACEHOLDER-{current_chart_index}]\n\n"
                        yield {"type": "content_chunk", "data": {"chunk": chart_placeholder}}

                        # 차트 데이터 전송
                        yield {"type": "chart", "data": chart_data}

                        # 차트 카운터 증가
                        state['chart_counter'] = current_chart_index + 1

            if buffer:
                # [수정됨] yield하는 모든 데이터를 딕셔너리 형식으로 통일
                yield {"type": "content_chunk", "data": {"chunk": buffer}}

            # [수정됨] yield하는 모든 데이터를 딕셔너리 형식으로 통일
            yield {"type": "content_chunk", "data": {"chunk": "\n\n"}}
