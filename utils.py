from models import (
    AgentType,
    MessageType,
    DatabaseType,
    AgentMessage,
    SearchResult,
    QueryPlan,
    CriticResult,
    StreamingAgentState,
)
from datetime import datetime
from typing import Dict, Any, List, Literal

# Agent 메시지 생성 헬퍼


def create_agent_message(
    from_agent: AgentType,
    to_agent: AgentType,
    message_type: MessageType,
    content: str,
    data: Dict[str, Any] = None,
    priority: int = 1,
) -> AgentMessage:
    return AgentMessage(
        from_agent=from_agent,
        to_agent=to_agent,
        message_type=message_type,
        content=content,
        data=data or {},
        priority=priority,
    )


# 검색 결과 생성 헬퍼
def create_search_result(
    source: str,
    content: str,
    relevance_score: float,
    metadata: Dict[str, Any] = None,
    search_query: str = "",
) -> SearchResult:
    return SearchResult(
        source=source,
        content=content,
        relevance_score=relevance_score,
        metadata=metadata or {},
        search_query=search_query,
    )


# Critic 결과 생성 헬퍼
def create_critic_result(
    status: Literal["sufficient", "insufficient"],
    suggestion: str,
    confidence: float,
    reasoning: str,
) -> CriticResult:
    return CriticResult(
        status=status, suggestion=suggestion, confidence=confidence, reasoning=reasoning
    )


# 쿼리 계획 생성 헬퍼
def create_query_plan(
    original_query: str,
    sub_queries: List[str],
    required_databases: List[DatabaseType],
    reasoning: str = "",
    priority: int = 1,
    estimated_complexity: str = "medium",
) -> QueryPlan:
    return QueryPlan(
        original_query=original_query,
        sub_queries=sub_queries,
        required_databases=required_databases,
        reasoning=reasoning,
        priority=priority,
        estimated_complexity=estimated_complexity,
    )


# 초기 상태 + 피드백 채널 생성
def create_initial_state(query: str, max_iterations: int = 3):
    state = StreamingAgentState(original_query=query, max_iterations=max_iterations)
    print(f"- 초기 상태 생성 완료")
    print(f"- 쿼리: {query}")
    print(f"- 최대 반복: {max_iterations}")
    return state


# 상태 유효성 검증
def validate_state(state: StreamingAgentState) -> bool:
    try:
        assert state.original_query, "원본 쿼리가 비어있음"
        assert (
            0 <= state.current_iteration <= state.max_iterations
        ), "반복 횟수 범위 오류"
        assert len(state.agent_memories) == 8, "Agent 메모리 개수 불일치"
        print(f"- 상태 검증 통과")
        return True
    except AssertionError as e:
        print(f"\n>> 상태 검증 실패: {e}")
        return False
    except Exception as e:
        print(f"\n>> 상태 검증 오류: {e}")
        return False


# 샘플 데이터 생성
def create_sample_data():
    sample_plan = create_query_plan(
        original_query="쌀 가격 상승 원인 분석",
        sub_queries=["쌀 가격 상승 현황은?", "주요 원인 요소들은?", "향후 전망은?"],
        required_databases=[DatabaseType.GRAPH_DB, DatabaseType.RDB, DatabaseType.WEB],
        reasoning="다각도 분석을 위해 그래프, 정형데이터, 웹 정보 필요",
    )
    sample_result = create_search_result(
        source="graph_db",
        content="쌀-가격상승-기후요인 관계 발견",
        relevance_score=0.85,
        metadata={"entity": "쌀", "relations": ["가격상승", "기후요인"]},
        search_query="쌀 가격 관계",
    )
    sample_message = create_agent_message(
        from_agent=AgentType.RETRIEVER_X,
        to_agent=AgentType.RETRIEVER_Y,
        message_type=MessageType.REAL_TIME_HINT,
        content="쌀-기후요인 관계 발견, 관련 최신 데이터 검색 필요",
        data={"entities": ["쌀", "기후요인"], "priority": "high"},
    )
    return {
        "query_plan": sample_plan,
        "search_result": sample_result,
        "agent_message": sample_message,
    }
