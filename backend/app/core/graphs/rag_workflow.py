import json
import asyncio
from typing import Dict, Any, List, Optional, Annotated, TypedDict
from datetime import datetime

try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    # LangGraph 0.5.2 버전에서 다른 import 경로 시도
    try:
        from langgraph import StateGraph, START, END
        from langgraph.graph.message import add_messages
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:
        # 최후의 수단으로 기본 클래스 정의
        class StateGraph:
            def __init__(self, state_schema): pass
            def add_node(self, name, func): pass
            def add_edge(self, from_node, to_node): pass
            def add_conditional_edges(self, node, condition, mapping): pass
            def compile(self, checkpointer=None): return self
        START, END = "START", "END"
        def add_messages(x, y): return x + y
        class MemorySaver: pass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from ..models.models import SearchResult, StreamingAgentState
from ..agents.worker_agents import DataGathererAgent, ProcessorAgent
from ...services.search.search_tools import vector_db_search, debug_web_search
from sentence_transformers import SentenceTransformer


class RAGState(TypedDict):
    """LangGraph 상태 정의"""
    # 기본 정보
    original_query: str
    session_id: str
    flow_type: Optional[str]

    # 메시지 기록
    messages: Annotated[List[BaseMessage], add_messages]

    # 워크플로우 상태
    plan: Optional[Dict[str, Any]]
    design: Optional[Dict[str, Any]]
    collected_data: List[SearchResult]
    current_step: int

    # 출력 관련
    final_answer: Optional[str]
    chart_counter: int

    # 메타데이터
    metadata: Dict[str, Any]

    # 스트리밍 이벤트
    streaming_events: List[Dict[str, Any]]


class RAGWorkflow:
    """LangGraph 기반 RAG 워크플로우"""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1)
        self.llm_pro = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
        self.data_gatherer = DataGathererAgent()
        self.processor = ProcessorAgent()

        # LangGraph 워크플로우 구성
        self.workflow = self._create_workflow()
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)

    def _create_workflow(self) -> StateGraph:
        """LangGraph 워크플로우 생성"""
        workflow = StateGraph(RAGState)

        # 노드 추가
        workflow.add_node("triage", self._triage_node)
        workflow.add_node("simple_answer", self._simple_answer_node)
        workflow.add_node("plan_generation", self._plan_generation_node)
        workflow.add_node("data_collection", self._data_collection_node)
        workflow.add_node("report_design", self._report_design_node)
        workflow.add_node("report_generation", self._report_generation_node)

        # 엣지 추가
        workflow.add_edge(START, "triage")
        workflow.add_conditional_edges(
            "triage",
            self._route_after_triage,
            {
                "simple": "simple_answer",
                "complex": "plan_generation"
            }
        )
        workflow.add_edge("simple_answer", END)
        workflow.add_edge("plan_generation", "data_collection")
        workflow.add_edge("data_collection", "report_design")
        workflow.add_edge("report_design", "report_generation")
        workflow.add_edge("report_generation", END)

        return workflow

    async def _triage_node(self, state: RAGState) -> RAGState:
        """요청 분류 노드"""
        print(f"\n>> Triage: 요청 분류 시작 - '{state['original_query']}'")

        classification_prompt = f"""
사용자 요청을 분석하여 적절한 처리 방식을 결정하세요:

사용자 요청: {state['original_query']}

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

            classification = None
            try:
                classification = json.loads(response_content)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    classification = json.loads(json_match.group())
                else:
                    raise ValueError("Valid JSON not found in response")

            flow_type = classification.get("flow_type", "task")
            reasoning = classification.get("reasoning", "분류 실패")

            print(f"- 분류 결과: {flow_type}")
            print(f"- 근거: {reasoning}")

            # 스트리밍 이벤트 추가
            state["streaming_events"].append({
                "type": "status",
                "data": {"message": f"요청 유형 분석 완료: {flow_type}", "session_id": state["session_id"]}
            })

            state["flow_type"] = flow_type
            state["metadata"]["triage_reasoning"] = reasoning
            state["metadata"]["classified_at"] = datetime.now().isoformat()

        except Exception as e:
            print(f"- 분류 실패, 기본값(task) 적용: {e}")
            state["flow_type"] = "task"
            state["metadata"]["triage_error"] = str(e)
            state["metadata"]["classified_at"] = datetime.now().isoformat()

        return state

    def _route_after_triage(self, state: RAGState) -> str:
        """분류 후 라우팅"""
        if state.get("flow_type") == "chat":
            return "simple"
        else:
            return "complex"

    async def _simple_answer_node(self, state: RAGState) -> RAGState:
        """간단한 답변 생성 노드"""
        print("\n>> SimpleAnswer: 간단한 답변 생성 시작")

        query = state["original_query"]

        # 간단한 검색 수행
        search_results = []
        try:
            # 웹 검색 또는 벡터 검색 수행
            if "최신" in query or "2024" in query or "2025" in query:
                web_results = await asyncio.get_event_loop().run_in_executor(
                    None, debug_web_search, query
                )
                if web_results:
                    search_results.append(web_results)
            else:
                vector_results = await asyncio.get_event_loop().run_in_executor(
                    None, vector_db_search, query
                )
                search_results.extend(vector_results[:3])
        except Exception as e:
            print(f"검색 오류: {e}")

        # 답변 생성
        context = ""
        if search_results:
            context = "\n".join([str(result)[:200] for result in search_results[:3]])

        prompt = f"""
당신은 친근하고 도움이 되는 AI 어시스턴트입니다.
현재 날짜: {datetime.now().strftime('%Y년 %m월 %d일')}

참고 정보:
{context if context else "추가 참고 정보 없음"}

사용자 질문: {query}

자연스럽고 친근한 톤으로 답변해주세요. 참고 정보가 있으면 이를 활용하되, 정확한 정보만 사용하세요.
불확실한 내용은 명시적으로 표현하고, 간결하면서도 도움이 되는 답변을 제공해주세요.

답변:
"""

        try:
            response = await self.llm.ainvoke(prompt)
            final_answer = response.content

            state["final_answer"] = final_answer
            state["streaming_events"].append({
                "type": "content",
                "data": {"chunk": final_answer, "session_id": state["session_id"]}
            })

            print(f"- 간단한 답변 생성 완료 (길이: {len(final_answer)}자)")

        except Exception as e:
            print(f"- 답변 생성 오류: {e}")
            fallback_answer = f"죄송합니다. 현재 시스템에 일시적인 문제가 있어 답변을 생성할 수 없습니다. 다시 시도해 주세요."
            state["final_answer"] = fallback_answer
            state["streaming_events"].append({
                "type": "content",
                "data": {"chunk": fallback_answer, "session_id": state["session_id"]}
            })

        return state

    async def _plan_generation_node(self, state: RAGState) -> RAGState:
        """계획 생성 노드"""
        print(f"\n>> PlanGeneration: 지능형 계획 수립 시작")

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

---

반드시 아래 JSON 형식으로만 응답해야 합니다.

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
            import re
            plan = json.loads(re.search(r'\{.*\}', content, re.DOTALL).group())

            print(f"  - 지능형 계획 생성 완료: {plan.get('title', '제목 없음')}")
            state["plan"] = plan

            # 스트리밍 이벤트 추가
            state["streaming_events"].append({
                "type": "plan",
                "data": {"plan": plan, "session_id": state["session_id"]}
            })

        except Exception as e:
            print(f"  - 지능형 계획 생성 실패, 원본 쿼리 직접 검색 실행: {e}")
            state["plan"] = {
                "title": f"{query} 원본 검색",
                "reasoning": "지능형 계획 수립에 실패하여, 사용자 원본 쿼리로 직접 검색을 실행합니다.",
                "sub_questions": [{"question": query, "tool": "vector_db_search"}]
            }

        return state

    async def _data_collection_node(self, state: RAGState) -> RAGState:
        """데이터 수집 노드"""
        print("\n>> DataCollection: 데이터 수집 시작")

        plan = state.get("plan", {})
        sub_questions = plan.get("sub_questions", [])

        if not sub_questions:
            print("- 수집할 질문이 없습니다.")
            return state

        # SentenceTransformer 모델 로딩
        print(">> SentenceTransformer 모델 로딩 시작...")
        hf_model = SentenceTransformer("dragonkue/bge-m3-ko", device="cpu", trust_remote_code=True)
        print(">> SentenceTransformer 모델 로딩 완료.")

        # 병렬 데이터 수집 작업 구성
        collection_tasks = []
        for sq in sub_questions:
            task = {"tool": sq["tool"], "inputs": {"query": sq["question"]}}
            if sq["tool"] == "vector_db_search":
                task["inputs"]["hf_model"] = hf_model
            collection_tasks.append(task)

        # 스트리밍 이벤트
        state["streaming_events"].append({
            "type": "status",
            "data": {"message": f"{len(collection_tasks)}개 작업으로 정보 수집을 시작합니다...", "session_id": state["session_id"]}
        })

        # 병렬 실행
        parallel_results = await self.data_gatherer.execute_parallel(collection_tasks)
        collected_data = [item for sublist in parallel_results.values() for item in sublist]

        print(f"\n>> 총 {len(collected_data)}개의 초기 데이터 수집 완료.")

        state["collected_data"] = collected_data
        state["streaming_events"].append({
            "type": "status",
            "data": {"message": f"총 {len(collected_data)}개의 정보를 수집했습니다.", "session_id": state["session_id"]}
        })

        return state

    async def _report_design_node(self, state: RAGState) -> RAGState:
        """보고서 구조 설계 노드"""
        print("\n>> ReportDesign: 보고서 구조 설계 시작")

        collected_data = state.get("collected_data", [])
        query = state["original_query"]

        design = await self.processor.process("design_report_structure", collected_data, query)

        if not design or "structure" not in design or not design["structure"]:
            print("- 보고서 구조 설계 실패")
            design = {
                "title": f"{query} - 통합 정보 요약",
                "structure": [{
                    "section_title": "수집된 정보 전체 요약",
                    "content_type": "synthesis",
                    "is_sufficient": True,
                    "feedback_for_gatherer": ""
                }]
            }

        state["design"] = design
        state["streaming_events"].append({
            "type": "status",
            "data": {"message": "수집된 정보를 바탕으로 보고서 목차를 설계했습니다.", "session_id": state["session_id"]}
        })

        return state

    async def _report_generation_node(self, state: RAGState) -> RAGState:
        """보고서 생성 노드"""
        print("\n>> ReportGeneration: 보고서 생성 시작")

        design = state.get("design", {})
        collected_data = state.get("collected_data", [])
        query = state["original_query"]

        # 보고서 제목 먼저 출력
        title = design.get("title", query)
        state["streaming_events"].append({
            "type": "content_chunk",
            "data": {"chunk": f"# {title}\n\n", "session_id": state["session_id"]}
        })

        # 각 섹션 생성
        for i, section in enumerate(design.get("structure", [])):
            section_title = section.get("section_title", "제목 없음")

            # 섹션 제목 출력
            state["streaming_events"].append({
                "type": "content_chunk",
                "data": {"chunk": f"## {section_title}\n", "session_id": state["session_id"]}
            })

            # 섹션 내용 생성
            content_buffer = ""
            async for chunk in self.processor.generate_section_streaming(section, collected_data, query):
                content_buffer += chunk

                # 차트 생성 마커 처리
                if "[GENERATE_CHART]" in content_buffer:
                    parts = content_buffer.split("[GENERATE_CHART]", 1)

                    # 텍스트 부분 출력
                    if parts[0]:
                        state["streaming_events"].append({
                            "type": "content_chunk",
                            "data": {"chunk": parts[0], "session_id": state["session_id"]}
                        })

                    content_buffer = parts[1]

                    # 차트 생성
                    chart_description = f"Chart for section titled '{section_title}': {section.get('description', '')}"
                    chart_data = await self.processor.process("create_chart_data", collected_data, chart_description)

                    if "error" not in chart_data:
                        # 차트 플레이스홀더
                        current_chart_index = state.get('chart_counter', 0)
                        chart_placeholder = f"\n\n[CHART-PLACEHOLDER-{current_chart_index}]\n\n"
                        state["streaming_events"].append({
                            "type": "content_chunk",
                            "data": {"chunk": chart_placeholder, "session_id": state["session_id"]}
                        })

                        # 차트 데이터 전송
                        state["streaming_events"].append({
                            "type": "chart",
                            "data": chart_data
                        })

                        state['chart_counter'] = current_chart_index + 1

            # 남은 내용 출력
            if content_buffer:
                state["streaming_events"].append({
                    "type": "content_chunk",
                    "data": {"chunk": content_buffer, "session_id": state["session_id"]}
                })

            # 섹션 구분
            state["streaming_events"].append({
                "type": "content_chunk",
                "data": {"chunk": "\n\n", "session_id": state["session_id"]}
            })

        return state

    async def stream_workflow(self, query: str, session_id: str):
        """워크플로우를 실행하고 스트리밍 이벤트를 생성"""
        config = {"configurable": {"thread_id": session_id}}

        initial_state = RAGState(
            original_query=query,
            session_id=session_id,
            flow_type=None,
            messages=[HumanMessage(content=query)],
            plan=None,
            design=None,
            collected_data=[],
            current_step=0,
            final_answer=None,
            chart_counter=0,
            metadata={},
            streaming_events=[]
        )

        try:
            # 워크플로우 실행
            final_state = await self.app.ainvoke(initial_state, config=config)

            # 스트리밍 이벤트 반환
            for event in final_state.get("streaming_events", []):
                yield event

        except Exception as e:
            print(f"워크플로우 실행 오류: {e}")
            yield {
                "type": "error",
                "data": {"message": f"오류가 발생했습니다: {str(e)}", "session_id": session_id}
            }
