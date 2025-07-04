import os
from dotenv import load_dotenv
from mock_databases import create_mock_databases
from models import StreamingAgentState
from agents import (
    PlanningAgent,
    RetrieverAgentXWithFeedback,
    RetrieverAgentYWithFeedback,
    CriticAgent1,
    CriticAgent2,
    ContextIntegratorAgent,
    ReportGeneratorAgent,
    SimpleAnswererAgent,
    RealTimeFeedbackChannel,
)
from langgraph.graph import StateGraph, END
import asyncio
import datetime
from pathlib import Path

# 환경 변수 로딩
load_dotenv()

# OPENAI_API_KEY 확인
if not os.environ.get("OPENAI_API_KEY"):
    print(
        ">> 환경 변수 OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일 또는 환경 변수로 설정해주세요."
    )
else:
    print(">> OPENAI_API_KEY 설정 완료")


# RAGWorkflow 전체 파이프라인 구현
class RAGWorkflow:
    def __init__(self):
        # Mock DB 초기화
        self.graph_db, self.vector_db, self.rdb, self.web_search = (
            create_mock_databases()
        )
        # 에이전트 초기화
        self.planning_agent = PlanningAgent()
        self.simple_answerer = SimpleAnswererAgent(self.vector_db)
        self.critic1 = CriticAgent1()
        self.context_integrator = ContextIntegratorAgent()
        self.critic2 = CriticAgent2()
        self.report_generator = ReportGeneratorAgent()
        # 워크플로우 그래프 생성
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        graph = StateGraph(StreamingAgentState)
        # 노드 추가
        graph.add_node("planning", self.planning_node)
        graph.add_node("simple_answer", self.simple_answer_node)
        graph.add_node("parallel_retrieval", self.parallel_retrieval_node)
        graph.add_node("critic1", self.critic1_node)
        graph.add_node("context_integration", self.context_integration_node)
        graph.add_node("critic2", self.critic2_node)
        graph.add_node("report_generation", self.report_generation_node)
        # 엣지 정의
        graph.set_entry_point("planning")
        graph.add_conditional_edges(
            "planning",
            self.route_by_complexity,
            {"simple": "simple_answer", "complex": "parallel_retrieval"},
        )
        graph.add_edge("simple_answer", END)
        graph.add_edge("parallel_retrieval", "critic1")
        graph.add_conditional_edges(
            "critic1",
            self.check_info_sufficient,
            {"sufficient": "context_integration", "insufficient": "planning"},
        )
        graph.add_edge("context_integration", "critic2")
        graph.add_conditional_edges(
            "critic2",
            self.check_context_sufficient,
            {"sufficient": "report_generation", "insufficient": "planning"},
        )
        graph.add_edge("report_generation", END)
        return graph.compile()

    # 노드 함수들
    async def planning_node(self, state: StreamingAgentState) -> StreamingAgentState:
        print("\n>>> PLANNING 단계 시작")
        return await self.planning_agent.plan(state)

    async def simple_answer_node(
        self, state: StreamingAgentState
    ) -> StreamingAgentState:
        print("\n>>> SIMPLE ANSWER 단계 시작")
        return await self.simple_answerer.answer(state)

    async def parallel_retrieval_node(
        self, state: StreamingAgentState
    ) -> StreamingAgentState:
        print("\n>>> PARALLEL RETRIEVAL 단계 시작")
        feedback_channel = RealTimeFeedbackChannel()
        try:
            retriever_x = RetrieverAgentXWithFeedback(self.graph_db, feedback_channel)
            retriever_y = RetrieverAgentYWithFeedback(
                self.vector_db, self.rdb, self.web_search, feedback_channel
            )
            state.x_active = True
            state.y_active = True
            # 병렬 검색 실행
            x_task = asyncio.create_task(retriever_x.search_with_feedback(state))
            y_task = asyncio.create_task(retriever_y.search_with_feedback(state))
            await asyncio.gather(x_task, y_task)
            state.search_complete = True
            state.x_active = False
            state.y_active = False
            stats = feedback_channel.get_message_count()
            state.total_messages_exchanged = stats["total"]
            print(
                f"- 검색 완료: Graph {len(state.graph_results_stream)}개, Multi {len(state.multi_source_results_stream)}개"
            )
            print(f"- 피드백 메시지: {stats['total']}개")
        finally:
            feedback_channel.stop()
        return state

    async def critic1_node(self, state: StreamingAgentState) -> StreamingAgentState:
        updated_state = await self.critic1.evaluate(state)
        if not updated_state.info_sufficient:
            if updated_state.should_terminate():
                print(">>> 최대 반복 도달(루프 강제 종료)")
                updated_state.info_sufficient = True
            else:
                updated_state.reset_for_new_iteration()
        return updated_state

    async def context_integration_node(
        self, state: StreamingAgentState
    ) -> StreamingAgentState:
        print("\n>>> CONTEXT INTEGRATION 시작")
        return await self.context_integrator.integrate(state)

    async def critic2_node(self, state: StreamingAgentState) -> StreamingAgentState:
        updated_state = await self.critic2.evaluate(state)
        if not updated_state.context_sufficient:
            if updated_state.should_terminate():
                print(">>> 최대 반복 도달(루프 강제 종료)")
                updated_state.context_sufficient = True
            else:
                updated_state.reset_for_new_iteration()
        return updated_state

    async def report_generation_node(
        self, state: StreamingAgentState
    ) -> StreamingAgentState:
        print("\n>>> REPORT GENERATION 시작")
        return await self.report_generator.generate(state)

    # 조건 함수들
    def route_by_complexity(self, state: StreamingAgentState) -> str:
        if not state.query_plan:
            return "complex"
        complexity = state.query_plan.estimated_complexity
        print(f"- 복잡도 판단: {complexity}")
        if complexity == "low":
            return "simple"
        else:
            return "complex"

    def check_info_sufficient(self, state: StreamingAgentState) -> str:
        if state.info_sufficient:
            print("- 정보 충분 >> 통합 단계로")
            return "sufficient"
        else:
            print("- 정보 부족 >> 재검색")
            return "insufficient"

    def check_context_sufficient(self, state: StreamingAgentState) -> str:
        if state.context_sufficient:
            print("- 맥락 충분 >> 보고서 생성")
            return "sufficient"
        else:
            print("- 맥락 부족 >> 재검색")
            return "insufficient"

    # 전체 워크플로우 실행 함수
    async def run(self, query: str, max_iterations: int = 3):
        print(f"\n{'-'*60}")
        print(f">> RAG 워크플로우 시작: {query}")
        print(f"{'-'*60}")

        initial_state = StreamingAgentState(
            original_query=query, max_iterations=max_iterations
        )

        # LangGraph 실행
        final_state = await self.workflow.ainvoke(initial_state)

        print(f"\n{'-'*60}")
        print(f">> RAG 워크플로우 완료")
        print(f"{'-'*60}")

        # final_state는 딕셔너리로 반환됨
        print(f">> 결과 타입: {type(final_state)}")

        # 딕셔너리에서 안전하게 접근
        if isinstance(final_state, dict):
            final_answer = final_state.get(
                "final_answer", "답변을 생성하지 못했습니다."
            )
            print(f"\n[최종 답변]\n{final_answer}")
        else:
            # 만약 객체라면
            final_answer = getattr(
                final_state, "final_answer", "답변을 생성하지 못했습니다."
            )
            print(f"\n[최종 답변]\n{final_answer}")

        return final_state


def save_result_as_markdown(
    query: str, final_answer: str, output_dir: str = "test-report"
):
    """
    >> 결과를 마크다운 파일로 저장
    - query: 사용자가 입력한 질문
    - final_answer: RAG 시스템이 생성한 최종 답변
    - output_dir: 저장할 디렉토리명
    """

    # 디렉토리 생성 (없으면 만들기)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 파일명 생성 (현재 시간 기준)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.md"
    file_path = output_path / filename

    # 마크다운 내용 구성
    markdown_content = f"""

## 생성 정보
- 사용자 쿼리: "{query}"
- 생성 시간: {datetime.datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분 %S초")}
- 파일명: {filename}

---

## Final Result

{final_answer}

---
"""

    # 파일 저장
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print(f"\n>> 마크다운 파일 저장 완료: {file_path}")
        return str(file_path)

    except Exception as e:
        print(f"\n>> 파일 저장 실패: {e}")
        return None


if __name__ == "__main__":
    query = input("\n질문을 입력하세요: ")
    rag = RAGWorkflow()
    final_state = asyncio.run(rag.run(query))

    # 최종 답변 추출
    if isinstance(final_state, dict):
        final_answer = final_state.get("final_answer", "답변을 생성하지 못했습니다.")
    else:
        final_answer = getattr(
            final_state, "final_answer", "답변을 생성하지 못했습니다."
        )

    # 마크다운 파일로 저장
    saved_file = save_result_as_markdown(query, final_answer)

    if saved_file:
        print(f">> 저장된 파일: {saved_file}")

    print("\n>> 실행 완료")
