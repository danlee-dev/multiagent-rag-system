import json
import asyncio
from typing import Dict, Any, List, Optional, TypedDict, Callable
from datetime import datetime
from enum import Enum

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from ..models.models import SearchResult, StreamingAgentState
from ..agents.worker_agents import DataGathererAgent, ProcessorAgent
from ...services.search.search_tools import vector_db_search, debug_web_search
from sentence_transformers import SentenceTransformer


class NodeType(Enum):
    """노드 타입 정의"""
    TRIAGE = "triage"
    SIMPLE_ANSWER = "simple_answer"
    PLAN_GENERATION = "plan_generation"
    DATA_COLLECTION = "data_collection"
    REPORT_DESIGN = "report_design"
    REPORT_GENERATION = "report_generation"


class WorkflowState(TypedDict):
    """워크플로우 상태 정의"""
    # 기본 정보
    original_query: str
    session_id: str
    flow_type: Optional[str]

    # 메시지 기록
    messages: List[Dict[str, Any]]

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


class WorkflowNode:
    """워크플로우 노드 기본 클래스"""

    def __init__(self, name: str, node_type: NodeType):
        self.name = name
        self.node_type = node_type

    async def execute(self, state: WorkflowState) -> WorkflowState:
        """노드 실행"""
        raise NotImplementedError


class ConditionalEdge:
    """조건부 엣지"""

    def __init__(self, condition_func: Callable[[WorkflowState], str], routes: Dict[str, str]):
        self.condition_func = condition_func
        self.routes = routes

    def route(self, state: WorkflowState) -> str:
        """라우팅 결정"""
        condition_result = self.condition_func(state)
        return self.routes.get(condition_result, "END")


class CustomWorkflowGraph:
    """커스텀 워크플로우 그래프"""

    def __init__(self):
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: Dict[str, str] = {}
        self.conditional_edges: Dict[str, ConditionalEdge] = {}
        self.start_node: Optional[str] = None

    def add_node(self, name: str, node: WorkflowNode):
        """노드 추가"""
        self.nodes[name] = node

    def add_edge(self, from_node: str, to_node: str):
        """엣지 추가"""
        self.edges[from_node] = to_node

    def add_conditional_edges(self, from_node: str, condition_func: Callable, routes: Dict[str, str]):
        """조건부 엣지 추가"""
        self.conditional_edges[from_node] = ConditionalEdge(condition_func, routes)

    def set_start(self, node_name: str):
        """시작 노드 설정"""
        self.start_node = node_name

    async def execute(self, initial_state: WorkflowState) -> WorkflowState:
        """워크플로우 실행"""
        current_state = initial_state
        current_node = self.start_node

        while current_node and current_node != "END":
            if current_node in self.nodes:
                print(f">> 실행 중인 노드: {current_node}")
                # 노드 실행
                current_state = await self.nodes[current_node].execute(current_state)

                # 다음 노드 결정
                if current_node in self.conditional_edges:
                    current_node = self.conditional_edges[current_node].route(current_state)
                elif current_node in self.edges:
                    current_node = self.edges[current_node]
                else:
                    current_node = "END"
            else:
                break

        return current_state


class TriageNode(WorkflowNode):
    """요청 분류 노드"""

    def __init__(self):
        super().__init__("triage", NodeType.TRIAGE)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1)

    async def execute(self, state: WorkflowState) -> WorkflowState:
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


class SimpleAnswerNode(WorkflowNode):
    """간단한 답변 생성 노드"""

    def __init__(self):
        super().__init__("simple_answer", NodeType.SIMPLE_ANSWER)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

    async def execute(self, state: WorkflowState) -> WorkflowState:
        print("\n>> SimpleAnswer: 간단한 답변 생성 시작")

        query = state["original_query"]

        # 간단한 검색 수행
        search_results = []
        try:
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


class PlanGenerationNode(WorkflowNode):
    """계획 생성 노드"""

    def __init__(self):
        super().__init__("plan_generation", NodeType.PLAN_GENERATION)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)

    async def execute(self, state: WorkflowState) -> WorkflowState:
        print(f"\n>> PlanGeneration: 지능형 계획 수립 시작")

        query = state["original_query"]
        current_date_str = datetime.now().strftime("%Y년 %m월 %d일")

        planning_prompt = f"""
당신은 사용자의 복잡한 요청을 명확한 하위 질문으로 분해하고, 각 질문을 해결하기 위한 최적의 데이터 수집 도구를 선택하여 실행 계획을 수립하는 AI 수석 아키텍트입니다.

**사용자 원본 요청**: "{query}"
**현재 날짜**: {current_date_str}

반드시 아래 JSON 형식으로만 응답해야 합니다.

{{
    "title": "분석 보고서의 전체 제목",
    "reasoning": "이러한 계획을 전체적으로 수립한 핵심적인 이유.",
    "sub_questions": [
        {{
            "question": "맥락이 완벽하게 유지된 첫 번째 하위 질문",
            "tool": "vector_db_search"
        }},
        {{
            "question": "맥락이 완벽하게 유지된 두 번째 하위 질문",
            "tool": "rdb_search"
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


class DataCollectionNode(WorkflowNode):
    """데이터 수집 노드"""

    def __init__(self):
        super().__init__("data_collection", NodeType.DATA_COLLECTION)
        self.data_gatherer = DataGathererAgent()

    async def execute(self, state: WorkflowState) -> WorkflowState:
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


class ReportDesignNode(WorkflowNode):
    """보고서 구조 설계 노드"""

    def __init__(self):
        super().__init__("report_design", NodeType.REPORT_DESIGN)
        self.processor = ProcessorAgent()

    async def execute(self, state: WorkflowState) -> WorkflowState:
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


class ReportGenerationNode(WorkflowNode):
    """보고서 생성 노드"""

    def __init__(self):
        super().__init__("report_generation", NodeType.REPORT_GENERATION)
        self.processor = ProcessorAgent()

    async def execute(self, state: WorkflowState) -> WorkflowState:
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


class RAGWorkflow:
    """커스텀 RAG 워크플로우"""

    def __init__(self):
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> CustomWorkflowGraph:
        """워크플로우 생성"""
        workflow = CustomWorkflowGraph()

        # 노드 추가
        workflow.add_node("triage", TriageNode())
        workflow.add_node("simple_answer", SimpleAnswerNode())
        workflow.add_node("plan_generation", PlanGenerationNode())
        workflow.add_node("data_collection", DataCollectionNode())
        workflow.add_node("report_design", ReportDesignNode())
        workflow.add_node("report_generation", ReportGenerationNode())

        # 엣지 추가
        workflow.set_start("triage")

        # 조건부 라우팅
        def route_after_triage(state: WorkflowState) -> str:
            if state.get("flow_type") == "chat":
                return "simple"
            else:
                return "complex"

        workflow.add_conditional_edges(
            "triage",
            route_after_triage,
            {
                "simple": "simple_answer",
                "complex": "plan_generation"
            }
        )

        # 순차적 엣지
        workflow.add_edge("simple_answer", "END")
        workflow.add_edge("plan_generation", "data_collection")
        workflow.add_edge("data_collection", "report_design")
        workflow.add_edge("report_design", "report_generation")
        workflow.add_edge("report_generation", "END")

        return workflow

    async def stream_workflow(self, query: str, session_id: str):
        """워크플로우를 실행하고 스트리밍 이벤트를 생성"""
        initial_state = WorkflowState(
            original_query=query,
            session_id=session_id,
            flow_type=None,
            messages=[{"role": "user", "content": query}],
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
            final_state = await self.workflow.execute(initial_state)

            # 스트리밍 이벤트 반환
            for event in final_state.get("streaming_events", []):
                yield event

        except Exception as e:
            print(f"워크플로우 실행 오류: {e}")
            yield {
                "type": "error",
                "data": {"message": f"오류가 발생했습니다: {str(e)}", "session_id": session_id}
            }
