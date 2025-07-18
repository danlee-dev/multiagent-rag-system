from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Literal, Union
from enum import Enum
from datetime import datetime


class AgentType(str, Enum):
    """Agent Type 정의"""

    PLANNING = "planning"  # 쿼리 분석 및 작업 플랜 수립
    RETRIEVER_X = "retriever_x"  # Graph DB 중심의 관계 탐색 담당 검색 에이전트
    RETRIEVER_Y = "retriever_y"  # Multi-source 검색 담당 (Vector, RDB, Web 등)
    CRITIC_1 = "critic_1"  # 정보량 충분성 평가 담당
    CRITIC_2 = "critic_2"  # 컨텍스트 품질 및 신뢰도 평가 담당
    CONTEXT_INTEGRATOR = (
        "context_integrator"  # 모든 검색 결과를 통합해 구조화된 문서 생성
    )
    REPORT_GENERATOR = "report_generator"  # 최종 보고서 및 사용자 응답 문서 생성
    SIMPLE_ANSWERER = "simple_answerer"  # 단순 쿼리에 대해 빠른 응답 생성


class MessageType(str, Enum):
    """Agent 간 주고 받는 메세지 타입 정의"""

    REAL_TIME_HINT = "real_time_hint"
    SEARCH_REQUEST = "search_request"
    INTERESTING_FINDING = "interesting_finding"
    FEEDBACK = "feedback"
    RESULT = "result"
    MEMORY_RETRIEVAL = "memory_retrieval"  # 메모리 검색 요청
    MEMORY_STORAGE = "memory_storage"  # 메모리 저장 요청


class DatabaseType(str, Enum):
    """데이터베이스 타입"""

    GRAPH_DB = "graph_db"
    VECTOR_DB = "vector_db"
    RDB = "rdb"
    API = "api"
    WEB = "web"
    MEMORY = "memory"  # 계층 메모리 시스템


class MemoryType(str, Enum):
    """메모리 타입 정의"""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"
    USER_PROFILE = "user_profile"
    KNOWLEDGE_BASE = "knowledge_base"


class ExpertiseLevel(str, Enum):
    """사용자 전문성 수준"""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


class ComplexityLevel(str, Enum):
    """질문 복잡도 레벨"""

    SIMPLE = "simple"  # 직접 답변
    MEDIUM = "medium"  # 기본 검색 + 간단 분석
    COMPLEX = "complex"  # 풀 ReAct 에이전트
    SUPER_COMPLEX = "super_complex"  # 다중 에이전트 협업


class ExecutionStrategy(str, Enum):
    """실행 전략 - 4단계 복잡도 대응"""

    DIRECT_ANSWER = "direct_answer"  # 직접 답변
    BASIC_SEARCH = "basic_search"  # 기본 검색
    FULL_REACT = "full_react"  # 풀 ReAct
    MULTI_AGENT = "multi_agent"  # 다중 에이전트


class AgentMessage(BaseModel):
    """Agent 간 실시간 메시지"""

    from_agent: AgentType
    to_agent: AgentType
    message_type: MessageType
    content: str  # 주요 텍스트 내용
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    priority: int = Field(default=1, description="1=highest, 5=lowest")


class SearchResult(BaseModel):
    """검색 결과 표준 형태"""

    source: str  # 데이터 소스 이름(graph_db, vector_db, memory, ...)
    content: str  # 검색 결과 내용
    relevance_score: float = Field(ge=0.0, le=1.0)  # 검색 결과의 관련도
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    search_query: str = ""  # 검색한 쿼리 그 자체
    memory_type: Optional[MemoryType] = None  # 메모리 기반 검색일 경우


class QueryPlan(BaseModel):
    """향상된 쿼리 계획 - 4단계 복잡도 지원"""

    original_query: str
    sub_queries: List[str] = Field(default_factory=list)
    required_databases: List[DatabaseType] = Field(default_factory=list)
    priority: int = Field(default=1, ge=1, le=5)
    reasoning: str = ""

    # 4단계 복잡도 관련 필드
    estimated_complexity: ComplexityLevel = Field(default=ComplexityLevel.MEDIUM)
    execution_strategy: ExecutionStrategy = Field(default=ExecutionStrategy.BASIC_SEARCH)

    # 리소스 요구사항
    resource_requirements: Dict[str, Any] = Field(default_factory=dict)
    expected_output_type: str = Field(default="analysis")
    estimated_processing_time: str = Field(default="medium")

    # 메모리 및 사용자 컨텍스트
    memory_context_needed: bool = Field(default=True)
    expected_expertise_level: ExpertiseLevel = Field(default=ExpertiseLevel.INTERMEDIATE)

    # 실행 단계 정보
    execution_steps: List[str] = Field(default_factory=list)
    fallback_strategy: Optional[str] = None


class CriticResult(BaseModel):
    """Critic Agent의 평가 결과"""

    status: Literal["sufficient", "insufficient"]
    suggestion: str  # 부족한 부분에 대한 보완 제안
    confidence: float  # 신뢰도 점수
    reasoning: str  # 충분/불충분에 대한 논리적 이유
    memory_recommendation: Optional[str] = None  # 메모리 관련 권장 사항


class AgentMemory(BaseModel):
    """각 Agent의 개별 메모리"""

    agent_type: AgentType
    internal_state: Dict[str, Any] = Field(default_factory=dict)
    message_history: List[AgentMessage] = Field(default_factory=list)
    findings: List[str] = Field(default_factory=list)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    memory_usage_stats: Dict[str, int] = Field(default_factory=dict)

    def add_finding(self, finding: str):
        """새로운 발견 추가"""
        self.findings.append(f"[{datetime.now().strftime('%H:%M:%S')}] {finding}")

    def update_metric(self, metric_name: str, value: float):
        """성능 지표 업데이트"""
        self.performance_metrics[metric_name] = value

    def update_memory_stat(self, stat_name: str, value: int):
        """메모리 통계 업데이트"""
        self.memory_usage_stats[stat_name] = value


class UserContext(BaseModel):
    """사용자 컨텍스트 정보"""

    user_id: str = Field(default="default_user")
    expertise_level: ExpertiseLevel = Field(default=ExpertiseLevel.INTERMEDIATE)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    interaction_patterns: Dict[str, Any] = Field(default_factory=dict)
    mentioned_info: Dict[str, Any] = Field(default_factory=dict)
    session_start: str = Field(default_factory=lambda: datetime.now().isoformat())
    total_interactions: int = Field(default=0)


class MemoryRetrievalResult(BaseModel):
    """메모리 검색 결과"""

    memories: List[SearchResult] = Field(default_factory=list)
    retrieval_time: float = Field(default=0.0)
    total_found: int = Field(default=0)
    relevance_threshold: float = Field(default=0.5)
    context_summary: str = Field(default="")


class SimpleAgentMemory:
    """간단한 에이전트별 메모리 클래스 (Pydantic 없이)"""

    def __init__(self):
        self.findings: List[str] = []
        self.metrics: Dict[str, float] = {}
        self.context: Dict[str, Any] = {}

    def add_finding(self, finding: str):
        """발견사항 추가"""
        self.findings.append(finding)

    def update_metric(self, name: str, value: float):
        """메트릭 업데이트"""
        self.metrics[name] = value

    def set_context(self, key: str, value: Any):
        """컨텍스트 설정"""
        self.context[key] = value


class StreamingAgentState(BaseModel):
    """스트리밍 에이전트 상태 - 4단계 복잡도 지원"""

    # 기본 상태
    original_query: str = ""
    user_id: str = "default_user"

    # 계획 및 실행 (Optional로 변경!)
    query_plan: Optional[QueryPlan] = None
    execution_mode: Optional[ExecutionStrategy] = None

    # 결과 상태
    final_answer: str = ""
    info_sufficient: bool = False
    context_sufficient: bool = False
    search_complete: bool = False
    planning_complete: bool = False

    # 검색 결과
    graph_results_stream: List[SearchResult] = Field(default_factory=list)
    multi_source_results_stream: List[SearchResult] = Field(default_factory=list)

    # 통합 결과
    integrated_context: str = ""

    # 반복 제어
    current_iteration: int = 0
    max_iterations: int = 3

    # 메모리 컨텍스트
    memory_context: str = ""
    additional_context: str = ""

    # Critic 결과
    critic1_result: Optional[CriticResult] = None
    critic2_result: Optional[CriticResult] = None

    # 단계별 결과 추적
    step_results: Dict[str, Any] = Field(default_factory=dict)

    # 에이전트 메모리 (선택적 - Simple 클래스 사용)
    agent_memories: Dict[str, Any] = Field(default_factory=dict)

    # X 에이전트 힌트
    x_extracted_hints: List[Dict[str, Any]] = Field(default_factory=list)

    def get_complexity_level(self) -> str:
        """복잡도 레벨 반환"""
        if self.query_plan and hasattr(self.query_plan, 'estimated_complexity'):
            return str(self.query_plan.estimated_complexity)
        return "medium"

    def add_graph_result(self, result: SearchResult):
        """Graph DB 결과 추가"""
        self.graph_results_stream.append(result)

    def add_multi_source_result(self, result: SearchResult):
        """Multi Source 결과 추가"""
        self.multi_source_results_stream.append(result)

    def add_step_result(self, step_name: str, result: Any):
        """단계별 결과 추가"""
        self.step_results[step_name] = result

    def get_agent_memory(self, agent_type: AgentType):
        """에이전트별 메모리 반환"""
        agent_key = str(agent_type)
        if agent_key not in self.agent_memories:
            self.agent_memories[agent_key] = SimpleAgentMemory()
        return self.agent_memories[agent_key]

    def should_terminate(self) -> bool:
        """종료 조건 확인"""
        return self.current_iteration >= self.max_iterations

    def reset_for_new_iteration(self):
        """새로운 반복을 위한 상태 리셋"""
        self.current_iteration += 1
        self.search_complete = False


class ChartData(BaseModel):
    """차트 데이터"""

    chart_type: str  # "line", "bar", "pie", "scatter", etc.
    title: str
    data: List[Dict[str, Any]]
    x_axis: str
    y_axis: str
    colors: Optional[List[str]] = None
    options: Optional[Dict[str, Any]] = None


class StreamingResponse(BaseModel):
    """스트리밍 응답"""

    chunk_type: str  # "text", "chart", "table", "status", "complete", "error"
    content: str
    metadata: Optional[Dict[str, Any]] = None
    chart_data: Optional[ChartData] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class FeedbackData(BaseModel):
    """피드백 데이터"""

    user_id: str
    conversation_id: str
    query: str
    response: str
    rating: int = Field(ge=1, le=5)  # 1-5
    feedback_text: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    memory_helpful: Optional[bool] = None  # 메모리가 도움이 되었는지


class WorkflowMetrics(BaseModel):
    """워크플로우 성능 메트릭"""

    total_processing_time: float
    node_processing_times: Dict[str, float] = Field(default_factory=dict)
    memory_performance: Dict[str, float] = Field(default_factory=dict)
    search_performance: Dict[str, float] = Field(default_factory=dict)
    user_satisfaction_score: Optional[float] = None


# 내보낼 모델들
__all__ = [
    "AgentType",
    "MessageType",
    "DatabaseType",
    "MemoryType",
    "ExpertiseLevel",
    "ComplexityLevel",
    "ExecutionStrategy",
    "AgentMessage",
    "SearchResult",
    "QueryPlan",
    "CriticResult",
    "AgentMemory",
    "SimpleAgentMemory",
    "UserContext",
    "MemoryRetrievalResult",
    "StreamingAgentState",
    "ChartData",
    "StreamingResponse",
    "FeedbackData",
    "WorkflowMetrics",
]
