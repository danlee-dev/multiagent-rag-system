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
    priority: int = 1
) -> AgentMessage: # RetrieverAgentWithFeedback에서 피드백 전송시 사용

    """Agent 메시지 생성 헬퍼"""
    return AgentMessage(
        from_agent=from_agent,
        to_agent=to_agent,
        message_type=message_type,
        content=content,
        data=data or {},
        priority=priority
    )

def create_search_result(
    source: str,
    content: str,
    relevance_score: float,
    metadata: Dict[str, Any] = None,
    search_query: str = ""
) -> SearchResult: # DB 검색 결과를 Search Result로 포맷팅
    """검색 결과 생성 헬퍼"""
    return SearchResult(
        source=source,
        content=content,
        relevance_score=relevance_score,
        metadata=metadata or {},
        search_query=search_query
    )

def create_critic_result(
    status: Literal["sufficient", "insufficient"],
    suggestions: str,
    confidence: float,
    reasoning: str,
    improvement_areas: List[str] = None
) -> CriticResult:
    """Critic 결과 생성 헬퍼"""
    return CriticResult(
        status=status,
        suggestions=suggestions,
        confidence=confidence,
        reasoning=reasoning,
        improvement_areas=improvement_areas or []
    )

def create_query_plan(
    original_query: str,
    sub_queries: List[str],
    required_databases: List[DatabaseType],
    reasoning: str = "",
    priority: int = 1,
    estimated_complexity: str = "medium"
) -> QueryPlan: # Planning Agent의 Plan 생성
    """쿼리 계획 생성 헬퍼"""
    return QueryPlan(
        original_query=original_query,
        sub_queries=sub_queries,
        required_databases=required_databases,
        reasoning=reasoning,
        priority=priority,
        estimated_complexity=estimated_complexity
    )

# 초기 상태 생성 함수
def create_initial_state(query: str, max_iterations: int = 3) -> StreamingAgentState:
    """초기 상태 + 피드백 채널 생성"""
    state = StreamingAgentState(
        original_query=query,
        max_iterations=max_iterations
    ) # Workflow 실행 시 & Test 함수 실행 시 State 초기화

    print(f"- 초기 상태 생성 완료")
    print(f"- 쿼리: {query}")
    print(f"- 최대 반복: {max_iterations}")
    print(f"- 피드백 채널: 활성화")

    return state

def validate_state(state: StreamingAgentState) -> bool: # State가 정상적으로 생성됐는지 여부를 검증
    """상태 유효성 검증"""
    try:
        # 기본 필드 검증
        assert state.original_query, "원본 쿼리가 비어있음" # 값이 없으면 오류 메시지 "원본 쿼리가 비어있음" 반환
        assert 0 <= state.current_iteration <= state.max_iterations, "반복 횟수 범위 오류"

        # Agent 메모리 검증
        assert len(state.agent_memories) == len(AgentType), "Agent 메모리 개수 불일치"

        print(f"- 상태 검증 통과")
        return True

    except AssertionError as e:
        print(f"\n>> 상태 검증 실패: {e}")
        return False
    except Exception as e:
        print(f"\n>> 상태 검증 오류: {e}")
        return False


# Pydantic 모델 테스트용 샘플 데이터 생성
def create_sample_data():
    """테스트용 샘플 데이터"""

    # 샘플 쿼리 계획
    sample_plan = create_query_plan(
        original_query="쌀 가격 상승 원인 분석",
        sub_queries=[
            "쌀 가격 상승 현황은?",
            "주요 원인 요소들은?",
            "향후 전망은?"
        ],
        required_databases=[DatabaseType.GRAPH_DB, DatabaseType.RDB, DatabaseType.WEB],
        reasoning="다각도 분석을 위해 그래프, 정형데이터, 웹 정보 필요"
    )

    # 샘플 검색 결과
    sample_result = create_search_result(
        source="graph_db",
        content="쌀-가격상승-기후요인 관계 발견",
        relevance_score=0.85,
        metadata={"entity": "쌀", "relations": ["가격상승", "기후요인"]},
        search_query="쌀 가격 관계"
    )

    # 샘플 메시지
    sample_message = create_agent_message(
        from_agent=AgentType.RETRIEVER_X,
        to_agent=AgentType.RETRIEVER_Y,
        message_type=MessageType.REAL_TIME_HINT,
        content="쌀-기후요인 관계 발견, 관련 최신 데이터 검색 필요",
        data={"entities": ["쌀", "기후요인"], "priority": "high"}
    )

    return {
        "query_plan": sample_plan,
        "search_result": sample_result,
        "agent_message": sample_message
    }
