import os
import sys
import uuid
import json
import asyncio
from datetime import datetime
from typing import AsyncGenerator, Dict, Any

# Pydantic과 FastAPI는 웹 서버 구성을 위해 필요합니다.
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# 시스템 경로 설정을 통해 다른 폴더의 모듈을 임포트합니다.
# 실제 프로젝트 구조에 맞게 이 부분은 조정될 수 있습니다.
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 모델 및 에이전트 클래스 임포트 ---
# StreamingAgentState는 이제 Pydantic 모델로 관리하는 것이 더 효율적입니다.
class StreamingAgentState(BaseModel):
    original_query: str
    session_id: str
    flow_type: str | None = None
    plan: dict | None = None
    design: dict | None = None
    metadata: dict = Field(default_factory=dict)

from .core.agents.orchestrator import TriageAgent, OrchestratorAgent
from .core.agents.conversational_agent import SimpleAnswererAgent

# --- Pydantic 모델 정의 ---
class QueryRequest(BaseModel):
    query: str
    session_id: str | None = Field(default_factory=lambda: str(uuid.uuid4()))

# --- FastAPI 애플리케이션 설정 ---
app = FastAPI(
    title="Intelligent RAG Agent System",
    description="A sophisticated, multi-agent system for handling complex queries.",
    version="3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 에이전트 인스턴스 초기화 ---
triage_agent = TriageAgent()
orchestrator_agent = OrchestratorAgent()
simple_answerer_agent = SimpleAnswererAgent()

# --- API 엔드포인트 정의 ---
@app.get("/")
async def root():
    return {"message": "Intelligent RAG Agent System is running."}

@app.post("/query/stream")
async def stream_query(request: QueryRequest):
    """
    사용자 쿼리를 받아, 유형에 따라 적절한 에이전트 워크플로우를 실행하고
    그 결과를 실시간으로 스트리밍하는 메인 엔드포인트입니다.
    """
    print(f"\n{'='*20} New Query Received {'='*20}")
    print(f"Session ID: {request.session_id}")
    print(f"Query: {request.query}")

    async def event_stream_generator() -> AsyncGenerator[str, None]:
        """쿼리 처리 및 결과 스트리밍을 위한 비동기 생성기"""

        state = StreamingAgentState(
            original_query=request.query,
            session_id=request.session_id,
        )

        try:
            # 1. Triage Agent 실행
            yield server_sent_event("status", {"message": "요청 유형 분석 중...", "session_id": state.session_id})
            state_dict = state.model_dump()
            updated_state_dict = await triage_agent.classify_request(request.query, state_dict)
            state = StreamingAgentState(**updated_state_dict)
            flow_type = state.flow_type or "task"

            # 2. 분류된 유형에 따라 다른 워크플로우 실행
            if flow_type == "chat":
                print(">> Flow type: 'chat'. Starting SimpleAnswererAgent.")
                yield server_sent_event("status", {"message": "간단한 답변 생성 중...", "session_id": state.session_id})

                async for chunk in simple_answerer_agent.answer_streaming(state.model_dump()):
                    yield server_sent_event("content", {"chunk": chunk, "session_id": state.session_id})

            else: # flow_type == "task"
                print(">> Flow type: 'task'. Starting OrchestratorAgent workflow.")

                # OrchestratorAgent의 워크플로우를 스트리밍하면서 상세한 상태 메시지 처리
                chart_index = 1
                # execute_report_workflow는 이제 텍스트/차트 외에 상태 정보도 함께 yield 합니다.

                async for event in orchestrator_agent.execute_report_workflow(state.model_dump()):

                    event_type = event.get("type")
                    data = event.get("data")

                    # session_id를 모든 이벤트에 추가
                    if isinstance(data, dict):
                        data["session_id"] = state.session_id

                    if event_type == "chart":
                        print(f"Chart event received: {data}")
                        # 프론트엔드가 인식할 수 있는 최종 차트 객체로 변환하여 전송
                        chart_payload = {
                            "chart_data": data,
                            "session_id": state.session_id
                        }
                        yield server_sent_event("chart", chart_payload)
                        chart_index += 1
                    else:
                        # status, plan, content_chunk 등 다른 모든 이벤트를 그대로 전달
                        yield server_sent_event(event_type, data if isinstance(data, dict) else {"data": data, "session_id": state.session_id})

        except Exception as e:
            print(f"!! 스트리밍 중 심각한 오류 발생: {e}", file=sys.stderr)
            error_payload = {"message": f"오류가 발생했습니다: {str(e)}", "session_id": state.session_id}
            yield server_sent_event("error", error_payload)

        finally:
            print(f"Query processing finished for session: {request.session_id}\n{'='*57}")
            yield server_sent_event("final_complete", {"message": "모든 작업이 완료되었습니다.", "session_id": state.session_id})

    return StreamingResponse(event_stream_generator(), media_type="text/event-stream")

def server_sent_event(event_type: str, data: Dict[str, Any]) -> str:
    """Server-Sent Events (SSE) 형식에 맞는 문자열을 생성합니다."""
    # 프론트엔드가 기대하는 형식에 맞춰 type을 data에 포함
    data_with_type = {"type": event_type, "session_id": data.get("session_id"), **data}
    payload = json.dumps(data_with_type, ensure_ascii=False)
    return f"data: {payload}\n\n"


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
