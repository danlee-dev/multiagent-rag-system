from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Literal
from enum import Enum
from datetime import datetime


class AgentType(str, Enum):
    """Agent Type 정의"""
    PLANNING = "planning" # 쿼리 분석 및 작업 플랜 수립
    RETRIEVER_X = "retriever_x" # Graph DB 중심의 관계 탐색 담당 검색 에이전트
    RETRIEVER_Y = "retriever_y" # Multi-source 검색 담당 (Vector, RDB, Web 등)
    CRITIC_1 = "critic_1" # 정보량 충분성 평가 담당
    CRITIC_2 = "critic_2" # 컨텍스트 품질 및 신뢰도 평가 담당
    CONTEXT_INTEGRATOR = "context_integrator" # 모든 검색 결과를 통합해 구조화된 문서 생성
    REPORT_GENERATOR = "report_generator" # 최종 보고서 및 사용자 응답 문서 생성
    SIMPLE_ANSWERER = "simple_answerer" # 단순 쿼리에 대해 빠른 응답 생성


class MessageType(str, Enum):
    """Agent 간 주고 받는 메세지 타입 정의"""
    REAL_TIME_HINT = "real_time_hint"
    SEARCH_REQUEST = "search_request"
    INTERESTING_FINDING = "interesting_finding"
    FEEDBACK = "feedback"
    RESULT = "result"
    # 코드 상에서 해당 메시지가 어떤 종류의 메시지 Type인지를 나타냄

class DatabaseType(str, Enum):
    """데이터베이스 타입"""
    GRAPH_DB = "graph_db"
    VECTOR_DB = "vector_db"
    RDB = "rdb"
    API = "api"
    WEB = "web"


class AgentMessage(BaseModel):
    """Agent 간 실시간 메시지"""
    from_agent: AgentType
    to_agent: AgentType
    message_type: MessageType
    content: str # 주요 텍스트 내용
    data: Dict[str, Any] = Field(default_factory=dict) # 부가 데이터(힌트, 키워드, 메타 정보, ...)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    priority: int = Field(default=1, description="1=highest, 5=lowest")
    # 에이전트 간 데이터를 전달하는 구조 그 자체


class SearchResult(BaseModel):
    """검색 결과 표준 형태"""
    source: str # 데이터 소스 이름(graph_db, vector_db, ...)
    content: str # 검색 결과 내용
    relevance_score: float = Field(ge=0.0, le=1.0) # 검색 결과의 관련도
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    search_query: str = "" # 검색한 쿼리 그 자체


class QueryPlan(BaseModel):
    """Planning Agent가 생성하는 쿼리 계획"""
    original_query: str
    sub_queries: List[str] = Field(default_factory=list)
    required_databases: List[DatabaseType] = Field(default_factory=list)
    priority: int = Field(default=1, ge=1, le=5)
    reasoning: str = "" # 현재 계획 수립 기준
    estimated_complexity: str = Field(default="medium")  # low, medium, high


class CriticResult(BaseModel):
    """Critic Agent의 평가 결과"""
    status: Literal["sufficient", "insufficient"]
    suggestion: str # 부족한 부분에 대한 보완 제안
    confidence: float # 신뢰도 점수
    reasoning: str # 충분/불충분에 대한 논리적 이유


class AgentMemory(BaseModel):
    """각 Agent의 개별 메모리"""
    agent_type: AgentType
    internal_state: Dict[str, Any] = Field(default_factory=dict) # 임시 변수나 상태 저장용
    message_history: List[AgentMessage] = Field(default_factory=list)
    findings: List[str] = Field(default_factory=list) # 에이전트가 과정 중 발견한 주요 인사이트/패턴/의미 있는 로그
    performance_metrics: Dict[str, float] = Field(default_factory=dict) # 성능 지표 (검색 속도, 중복 제거 비율, 응답 성공률, ...)

    def add_finding(self, finding: str):
        """새로운 발견 추가"""
        self.findings.append(f"[{datetime.now().strftime('%H:%M:%S')}] {finding}")

    def update_metric(self, metric_name: str, value: float):
        """성능 지표 업데이트"""
        self.performance_metrics[metric_name] = value


class StreamingAgentState(BaseModel):
    """실시간 스트리밍 Multi-Agent 상태"""

    # 메인 정보
    original_query: str
    current_iteration: int = Field(default=0, ge=0) # 현재 반복 횟수
    max_iterations: int = Field(default=2, ge=1, le=5) # 최대 반복 횟수
    start_time: str = Field(default_factory=lambda: datetime.now().isoformat()) # 세션 시작 시간(질의 처리 소요 시간)

    # Planning 결과
    query_plan: Optional[QueryPlan] = None
    planning_complete: bool = False

    # 실시간 검색 결과 스트림
    graph_results_stream: List[SearchResult] = Field(default_factory=list)
    multi_source_results_stream: List[SearchResult] = Field(default_factory=list)

    # Retriever 활성 상태
    x_active: bool = False
    y_active: bool = False
    search_complete: bool = False

    # Agent 메모리
    agent_memories: Dict[str, AgentMemory] = Field(default_factory=dict)

    # Critic 평가 결과
    critic1_result: Optional[CriticResult] = None
    critic2_result: Optional[CriticResult] = None
    info_sufficient: bool = False
    context_sufficient: bool = False

    # 통합 및 최종 생성 결과
    integrated_context: str = ""
    final_answer: str = ""

    # 성능 및 통계
    total_search_results: int = 0
    total_messages_exchanged: int = 0
    processing_time_seconds: float = 0.0

    class Config:
        arbitrary_types_allowed = True # 비정형 객체 허용 설정

    def __init__(self, **data):
        super().__init__(**data)
        # Agent 메모리 초기화(모든 Agent에 대해 AgentMemory를 생성)
        for agent_type in AgentType:
            agent_key = agent_type.value
            if agent_key not in self.agent_memories:
                self.agent_memories[agent_key] = AgentMemory(agent_type=agent_type)

    def get_agent_memory(self, agent_type: AgentType) -> AgentMemory:
        """특정 Agent 메모리 반환"""
        return self.agent_memories[agent_type.value]

    def add_graph_result(self, result: SearchResult):
        """Graph 검색 결과 추가"""
        self.graph_results_stream.append(result)
        self.total_search_results += 1

    def add_multi_source_result(self, result: SearchResult):
        """Multi-source 검색 결과 추가"""
        self.multi_source_results_stream.append(result)
        self.total_search_results += 1

    def get_all_results(self) -> List[SearchResult]: # 현재는 사용 안함(Context integrator에서 사용 가능)
        """모든 검색 결과 반환"""
        return self.graph_results_stream + self.multi_source_results_stream

    def get_latest_results(self, limit: int = 10) -> List[SearchResult]: # 현재는 사용 안함
        """최신 검색 결과 반환"""
        all_results = self.get_all_results()
        sorted_results = sorted(all_results, key=lambda x: x.timestamp, reverse=True)
        return sorted_results[:limit]

    def get_performance_summary(self) -> Dict[str, Any]: # 평가 필요시 사용
        """성능 요약 반환"""
        return {
            "total_iterations": self.current_iteration,
            "total_search_results": self.total_search_results,
            "graph_results": len(self.graph_results_stream),
            "multi_source_results": len(self.multi_source_results_stream),
            "messages_exchanged": self.total_messages_exchanged,
            "processing_time": self.processing_time_seconds,
            "info_sufficient": self.info_sufficient,
            "context_sufficient": self.context_sufficient,
            "planning_complete": self.planning_complete
        }

    def reset_for_new_iteration(self): # Critic에서 정보 불충분으로 인해 새로운 반복이 진행될 때 이전 정보를 리셋
        """새 반복을 위한 상태 리셋"""
        self.x_active = False
        self.y_active = False
        self.search_complete = False
        self.current_iteration += 1

    def should_terminate(self) -> bool:
        """최대 반복 도달 여부"""
        return self.current_iteration >= self.max_iterations
