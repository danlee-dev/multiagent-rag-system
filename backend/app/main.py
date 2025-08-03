import os
import sys
import uuid
import json
import re
import logging
from dotenv import load_dotenv
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, AsyncGenerator, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# 출력 버퍼링 비활성화
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

from .core.config.env_checker import check_api_keys
check_api_keys()

from .core.models.models import (
    StreamingAgentState,
    SearchResult,
)
# 새로운 모듈화된 agent 시스템
from .core.agents.orchestrator import TriageAgent, OrchestratorAgent
from .core.agents.worker_agents import DataGathererAgent, ProcessorAgent
from .core.agents.conversational_agent import SimpleAnswererAgent

from langgraph.graph import StateGraph, END
from .utils.memory.hierarchical_memory import HierarchicalMemorySystem, ConversationMemory


# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    session_id: str
    status: str
    response: str


class RAGWorkflow:
    """최종 아키텍처: 계층적 동적 계획 실행 워크플로우"""

    def __init__(self):
        print("\n>> 최종 아키텍처 RAG 워크플로우 초기화 시작")
        # 1. 역할에 따라 재편성된 에이전트들을 초기화합니다.
        self.triage_agent = TriageAgent()
        self.orchestrator_agent = OrchestratorAgent()
        self.data_gatherer_agent = DataGathererAgent()
        self.processor_agent = ProcessorAgent()
        self.simple_answerer_agent = SimpleAnswererAgent()
        self.workflow = self._create_workflow()
        print(">> 최종 아키텍처 RAG 워크플로우 초기화 완료")

    def _create_workflow(self):
        """LangGraph의 모든 기능을 활용하는 최종 워크플로우 그래프를 생성합니다."""
        graph = StateGraph(StreamingAgentState)

        # 2. 역할에 맞는 노드들을 정의합니다.
        graph.add_node("triage", self.triage_node)
        graph.add_node("simple_chat", self.simple_chat_node)
        graph.add_node("orchestrator", self.orchestrator_node)
        graph.add_node("executor", self.executor_node)

        # 3. 워크플로우의 흐름(엣지)을 정의합니다.
        graph.set_entry_point("triage")

        # Triage 노드 실행 후, State의 'flow_type'에 따라 분기합니다.
        graph.add_conditional_edges(
            "triage",
            lambda state: state.get("flow_type"),
            {"chat": "simple_chat", "task": "orchestrator"}
        )

        # Orchestrator 노드는 항상 Executor 노드로 연결됩니다.
        graph.add_edge("orchestrator", "executor")

        # Executor 노드 실행 후에는 router를 통해 다음 경로를 동적으로 결정합니다.
        graph.add_conditional_edges(
            "executor",
            self.router,
            {
                "continue": "executor",      # 루프: 다음 단계를 위해 다시 executor 호출
                "replan": "orchestrator",    # 재계획: critique 실패 시 orchestrator 호출
                "__end__": "__end__"         # 종료: 모든 계획이 끝났을 때
            }
        )

        # simple_chat 경로는 바로 종료됩니다.
        graph.add_edge("simple_chat", END)

        return graph.compile()

    # 4. 각 노드가 수행할 함수들을 정의합니다.
    async def triage_node(self, state: StreamingAgentState) -> Dict[str, Any]:
        """TriageAgent를 호출하여 State를 업데이트합니다."""
        return await self.triage_agent.classify_request(state["original_query"], state)

    async def orchestrator_node(self, state: StreamingAgentState) -> Dict[str, Any]:
        """OrchestratorAgent를 호출하여 State를 업데이트합니다."""
        return await self.orchestrator_agent.generate_plan(state)

    async def simple_chat_node(self, state: StreamingAgentState) -> Dict[str, Any]:
        """SimpleAnswererAgent를 호출하기 위해 State를 준비합니다."""
        # 실제 스트리밍은 API 계층에서 이 노드의 결과를 보고 처리합니다.
        return state

    async def executor_node(self, state: StreamingAgentState) -> Dict[str, Any]:
        """계획의 '단 한 단계'만 실행하는 핵심 실행 엔진입니다."""
        plan = state.get("plan", {})
        steps = plan.get("steps", [])
        index = state.get("current_step_index", 0)

        if index >= len(steps):
            return {"step_results": state.get("step_results", []), "current_step_index": index}

        step_to_execute = steps[index]
        agent_name = step_to_execute.get("agent")
        inputs = step_to_execute.get("inputs", {})

        print(f"\n>> Executor: 단계 {index} 실행 - {step_to_execute.get('description', '')}")
        sys.stdout.flush()

        result = None
        if agent_name == "DataGathererAgent":
            result = await self.data_gatherer_agent.execute(inputs.get("tool"), inputs)
        elif agent_name == "ProcessorAgent":
            processor_type = inputs.get("processor_type")
            source_steps = inputs.get("source_steps", [])
            data = [state["step_results"][i] for i in source_steps if i < len(state["step_results"])]
            data = data[0] if len(data) == 1 else data

            result = self.processor_agent.process(processor_type, data, state['original_query'])
            if asyncio.iscoroutine(result):
                result = await result

        step_results = state.get("step_results", [])
        step_results.append(result)

        # CriticResult인 경우에만 status 확인
        if hasattr(result, 'status') and result.status == 'fail_with_feedback':
            return {
                "step_results": step_results, "current_step_index": index + 1,
                "needs_replan": True, "replan_feedback": result.feedback
            }

        return {"step_results": step_results, "current_step_index": index + 1}

    def router(self, state: StreamingAgentState) -> str:
        """다음 경로를 결정하는 교통경찰 역할을 합니다."""
        if state.get("needs_replan", False):
            # 재계획이 필요하면 실행 상태를 초기화하고 orchestrator로 보냅니다.
            return "replan"

        if state.get("current_step_index", 0) >= len(state.get("plan", {}).get("steps", [])):
            return "__end__"

        return "continue"

    # 5. API와 통신하는 스트리밍 중계기를 정의합니다.
    async def stream_workflow_events(self, session_id: str, query: str) -> AsyncGenerator[str, None]:
        """LangGraph astream_events를 사용하여 전체 워크플로우를 스트리밍합니다."""
        initial_state = {
            "original_query": query, "session_id": session_id, "user_id": session_id,
            "start_time": datetime.now().isoformat(), "flow_type": None, "plan": None,
            "current_step_index": 0, "step_results": [], "execution_log": [],
            "is_sufficient": False, "needs_replan": False, "replan_feedback": None,
            "final_answer": None, "metadata": {}
        }

        # 선제적 응답 (빠른 TTFT)
        yield json.dumps({"type": "status", "message": "분석을 시작하겠습니다..."})

        async for event in self.workflow.astream_events(initial_state, version="v1"):
            kind = event["event"]

            if kind == "on_chain_end":
                node_name = event["name"]
                outputs = event["data"].get("output", {})

                # Triage가 끝나고 chat 경로로 결정되었을 때
                if node_name == "simple_chat":
                    async for chunk in self.simple_answerer_agent.answer_streaming(outputs):
                        yield json.dumps({"type": "content", "chunk": chunk})

                # Orchestrator가 계획을 생성했을 때
                elif node_name == "orchestrator" and outputs.get("plan"):
                    yield json.dumps({"type": "plan", "plan": outputs["plan"]})

                # Executor가 한 단계를 끝냈을 때
                elif node_name == "executor":
                    last_result = outputs.get("step_results", [])[-1] if outputs.get("step_results") else None

                    if last_result:
                        # 최종 보고서 생성 결과 처리 (문자열)
                        if isinstance(last_result, str):
                            yield json.dumps({"type": "content", "chunk": last_result})

                        # DataGatherer의 결과(SearchResult)를 스트리밍
                        elif isinstance(last_result, list) and all(isinstance(item, SearchResult) for item in last_result):
                            results_for_ui = [res.model_dump() for res in last_result]
                            yield json.dumps({"type": "search_results", "results": results_for_ui})

                        # 기타 결과 (딕셔너리 등)
                        elif isinstance(last_result, dict):
                            yield json.dumps({"type": "result", "data": last_result})

        yield json.dumps({"type": "complete", "message": "모든 작업이 완료되었습니다."})

# 전역 워크플로우 인스턴스
workflow_instance = None


def get_workflow():
    """전역 워크플로우 인스턴스를 반환하거나 생성"""
    global workflow_instance
    if workflow_instance is None:
        workflow_instance = RAGWorkflow()
    return workflow_instance


# FastAPI 앱 설정
app = FastAPI(title="MultiAgent RAG System", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "MultiAgent RAG System v2.0 - 새로운 모듈화된 아키텍처"}


@app.post("/query/stream")
async def stream_query(request: QueryRequest):
    """실시간 스트리밍 쿼리 엔드포인트 - Claude 스타일"""
    session_id = request.session_id or str(uuid.uuid4())

    print(f"\n=== 실시간 스트리밍 쿼리 시작 ===")
    print(f"세션 ID: {session_id}")
    print(f"쿼리: {request.query}")
    sys.stdout.flush()

    workflow = RAGWorkflow()

    async def generate():
        try:
            async for json_data in workflow.stream_workflow_events(session_id, request.query):
                # JSON 데이터를 파싱하여 session_id 추가
                try:
                    data = json.loads(json_data)
                    data["session_id"] = session_id
                    yield f"data: {json.dumps(data)}\n\n"
                except json.JSONDecodeError:
                    # 이전 방식의 텍스트 chunk인 경우
                    yield f"data: {json.dumps({'type': 'content', 'chunk': json_data, 'session_id': session_id})}\n\n"

                await asyncio.sleep(0.01)

            # 최종 완료 신호
            yield f"data: {json.dumps({'type': 'final_complete', 'session_id': session_id})}\n\n"

        except Exception as e:
            error_msg = f"스트리밍 중 오류 발생: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg, 'session_id': session_id})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.get("/memory/stats")
async def get_memory_stats(user_id: str = None):
    """메모리 통계 조회"""
    # 메모리 시스템이 구현되면 활성화
    return {"error": "메모리 시스템이 아직 구현되지 않았습니다"}


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0"
    }
