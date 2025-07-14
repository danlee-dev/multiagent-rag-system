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
    # 코드 상에서 해당 메시지가 어떤 종류의 메시지 Type인지를 나타냄


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


class AgentMessage(BaseModel):
    """Agent 간 실시간 메시지"""

    from_agent: AgentType
    to_agent: AgentType
    message_type: MessageType
    content: str  # 주요 텍스트 내용
    data: Dict[str, Any] = Field(
        default_factory=dict
    )  # 부가 데이터(힌트, 키워드, 메타 정보, ...)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    priority: int = Field(default=1, description="1=highest, 5=lowest")
    # 에이전트 간 데이터를 전달하는 구조 그 자체


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
    """Planning Agent가 생성하는 쿼리 계획"""

    original_query: str
    sub_queries: List[str] = Field(default_factory=list)
    required_databases: List[DatabaseType] = Field(default_factory=list)
    priority: int = Field(default=1, ge=1, le=5)
    reasoning: str = ""  # 현재 계획 수립 기준
    estimated_complexity: str = Field(default="medium")  # low, medium, high
    memory_context_needed: bool = Field(default=True)  # 메모리 컨텍스트 필요 여부
    expected_expertise_level: ExpertiseLevel = Field(default=ExpertiseLevel.INTERMEDIATE)


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
    internal_state: Dict[str, Any] = Field(
        default_factory=dict
    )  # 임시 변수나 상태 저장용
    message_history: List[AgentMessage] = Field(default_factory=list)
    findings: List[str] = Field(
        default_factory=list
    )  # 에이전트가 과정 중 발견한 주요 인사이트/패턴/의미 있는 로그
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict
    )  # 성능 지표 (검색 속도, 중복 제거 비율, 응답 성공률, ...)
    memory_usage_stats: Dict[str, int] = Field(
        default_factory=dict
    )  # 메모리 사용 통계

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


class StreamingAgentState(BaseModel):
    """MemorySaver 호환 실시간 스트리밍 Multi-Agent 상태 - 계층 메모리 통합"""

    # 메인 정보
    original_query: str
    current_iteration: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=2, ge=1, le=5)
    start_time: str = Field(default_factory=lambda: datetime.now().isoformat())

    # ===== MemorySaver가 자동으로 관리할 대화 기록 =====
    # MemorySaver는 전체 상태를 thread_id별로 저장하므로
    # 이전 쿼리들과 답변들이 자동으로 유지됨
    previous_queries: List[str] = Field(default_factory=list)
    previous_answers: List[str] = Field(default_factory=list)
    user_mentioned_info: Dict[str, Any] = Field(default_factory=dict)
    # ==============================================

    # ===== 계층 메모리 관련 필드 =====
    user_context: UserContext = Field(default_factory=UserContext)
    memory_retrieval_result: Optional[MemoryRetrievalResult] = None
    memory_context: str = Field(default="")  # 메모리에서 가져온 컨텍스트
    additional_context: str = Field(default="")  # 작업 메모리 컨텍스트
    memory_processing_time: float = Field(default=0.0)
    # ================================

    # Planning 결과
    query_plan: Optional[QueryPlan] = None
    planning_complete: bool = False

    # 실시간 검색 결과 스트림
    graph_results_stream: List[SearchResult] = Field(default_factory=list)
    multi_source_results_stream: List[SearchResult] = Field(default_factory=list)
    memory_results_stream: List[SearchResult] = Field(default_factory=list)  # 메모리 검색 결과

    # Retriever 힌트 및 키워드
    x_extracted_hints: List[Dict] = Field(default_factory=list)
    y_suggested_keywords: List[str] = Field(default_factory=list)

    # Retriever 활성 상태
    x_active: bool = False
    y_active: bool = False
    memory_active: bool = False  # 메모리 검색 활성 상태
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
    memory_hits: int = Field(default=0)  # 메모리 히트 수
    memory_misses: int = Field(default=0)  # 메모리 미스 수

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        # Agent 메모리 초기화
        for agent_type in AgentType:
            agent_key = agent_type.value
            if agent_key not in self.agent_memories:
                self.agent_memories[agent_key] = AgentMemory(agent_type=agent_type)

    # ===== MemorySaver와 함께 작동하는 메서드들 =====
    def update_conversation_history(self, new_query: str, new_answer: str = ""):
        """새로운 대화를 기록에 추가 (MemorySaver가 자동 저장)"""
        # 현재 쿼리를 이전 쿼리 목록에 추가
        if new_query and new_query not in self.previous_queries:
            self.previous_queries.append(new_query)

        # 답변이 있으면 추가
        if new_answer:
            self.previous_answers.append(new_answer)

        # 사용자 정보 추출
        self.extract_user_info(new_query)

        # 사용자 컨텍스트 업데이트
        self.user_context.total_interactions += 1
        self.user_context.mentioned_info.update(self.user_mentioned_info)

    def extract_user_info(self, query: str):
        """쿼리에서 사용자 정보 추출 및 기록"""
        query_lower = query.lower()

        # 이름 추출
        if "내 이름은" in query:
            try:
                name_part = query.split("내 이름은")[1].strip()
                name = name_part.split()[0].replace("이야", "").replace("야", "")
                self.user_mentioned_info["name"] = name
                print(f"\n>> 사용자 이름 추출됨: {name}")
            except:
                pass

        # 생일 정보 추출
        if "생일" in query and ("오늘" in query or "내" in query):
            today = datetime.now().strftime("%Y-%m-%d")
            self.user_mentioned_info["birthday"] = today
            self.user_mentioned_info["birthday_mentioned"] = True
            print(f"\n>> 생일 정보 추출됨: {today}")

        # 전문성 수준 추정
        complexity_indicators = {
            "시세": ExpertiseLevel.INTERMEDIATE,
            "트렌드": ExpertiseLevel.INTERMEDIATE,
            "분석": ExpertiseLevel.EXPERT,
            "예측": ExpertiseLevel.EXPERT,
            "기본": ExpertiseLevel.BEGINNER,
            "초보": ExpertiseLevel.BEGINNER,
        }

        for keyword, level in complexity_indicators.items():
            if keyword in query:
                self.user_context.expertise_level = level
                break

    def get_conversation_context(self) -> str:
        """이전 대화들의 맥락을 문자열로 반환"""
        context_parts = []

        # 사용자 정보
        if self.user_mentioned_info:
            info_str = ", ".join(
                [f"{k}: {v}" for k, v in self.user_mentioned_info.items()]
            )
            context_parts.append(f"사용자가 언급한 정보: {info_str}")

        # 사용자 전문성 수준
        context_parts.append(f"사용자 전문성 수준: {self.user_context.expertise_level.value}")

        # 최근 3개 대화
        recent_count = min(3, len(self.previous_queries))
        if recent_count > 0:
            context_parts.append("최근 대화:")
            for i in range(
                max(0, len(self.previous_queries) - recent_count),
                len(self.previous_queries),
            ):
                if i < len(self.previous_queries):
                    context_parts.append(f"  사용자: {self.previous_queries[i]}")
                if i < len(self.previous_answers):
                    context_parts.append(f"  어시스턴트: {self.previous_answers[i]}")

        # 메모리 컨텍스트 추가
        if self.memory_context:
            context_parts.append(f"관련 메모리 컨텍스트:\n{self.memory_context}")

        return "\n".join(context_parts)

    def has_user_info(self, key: str) -> bool:
        """특정 사용자 정보가 있는지 확인"""
        return key in self.user_mentioned_info

    def get_expertise_prompt(self) -> str:
        """사용자 전문성 수준에 따른 프롬프트 반환"""
        expertise_prompts = {
            ExpertiseLevel.BEGINNER: "초보자가 이해하기 쉽게 기본 개념부터 설명해주세요.",
            ExpertiseLevel.INTERMEDIATE: "실무진에게 유용한 구체적인 정보와 함께 설명해주세요.",
            ExpertiseLevel.EXPERT: "전문가 수준의 심화된 분석과 인사이트를 제공해주세요."
        }
        return expertise_prompts.get(self.user_context.expertise_level, expertise_prompts[ExpertiseLevel.INTERMEDIATE])

    # ==========================================

    def get_agent_memory(self, agent_type: AgentType) -> AgentMemory:
        return self.agent_memories[agent_type.value]

    def add_graph_result(self, result: SearchResult):
        self.graph_results_stream.append(result)
        self.total_search_results += 1

    def add_multi_source_result(self, result: SearchResult):
        self.multi_source_results_stream.append(result)
        self.total_search_results += 1

    def add_memory_result(self, result: SearchResult):
        """메모리 검색 결과 추가"""
        result.source = "memory"
        self.memory_results_stream.append(result)
        self.total_search_results += 1
        self.memory_hits += 1

    def record_memory_miss(self):
        """메모리 미스 기록"""
        self.memory_misses += 1

    def get_all_results(self) -> List[SearchResult]:
        return self.graph_results_stream + self.multi_source_results_stream + self.memory_results_stream

    def get_memory_hit_ratio(self) -> float:
        """메모리 히트율 계산"""
        total_memory_queries = self.memory_hits + self.memory_misses
        if total_memory_queries == 0:
            return 0.0
        return self.memory_hits / total_memory_queries

    def reset_for_new_iteration(self):
        """새 반복을 위한 상태 리셋 (대화 기록은 MemorySaver가 유지)"""
        self.x_active = False
        self.y_active = False
        self.memory_active = False
        self.search_complete = False
        self.current_iteration += 1
        print(f"\n>> 새 반복 시작: {self.current_iteration}/{self.max_iterations}")
        # 주의: previous_queries, previous_answers, user_mentioned_info는 리셋하지 않음!

    def should_terminate(self) -> bool:
        return self.current_iteration >= self.max_iterations

    def add_processing_time(self, phase: str, time_taken: float):
        """단계별 처리 시간 기록"""
        if phase == "memory":
            self.memory_processing_time += time_taken
        self.processing_time_seconds += time_taken

    def format_for_llm(self) -> str:
        """LLM에 전달할 형태로 상태 포맷팅"""
        context_parts = [
            f"사용자 질문: {self.original_query}",
            f"사용자 전문성: {self.user_context.expertise_level.value}",
            f"반복 횟수: {self.current_iteration}/{self.max_iterations}",
            self.get_expertise_prompt()
        ]

        # 대화 기록 컨텍스트
        conversation_context = self.get_conversation_context()
        if conversation_context:
            context_parts.append(f"대화 맥락:\n{conversation_context}")

        # 추가 컨텍스트
        if self.additional_context:
            context_parts.append(f"추가 정보:\n{self.additional_context}")

        return "\n\n".join(context_parts)

    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 관련 통계 반환"""
        return {
            "memory_hits": self.memory_hits,
            "memory_misses": self.memory_misses,
            "memory_hit_ratio": self.get_memory_hit_ratio(),
            "memory_results_count": len(self.memory_results_stream),
            "memory_processing_time": self.memory_processing_time,
            "total_interactions": self.user_context.total_interactions,
            "expertise_level": self.user_context.expertise_level.value
        }


# 추가 모델들
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
    'AgentType',
    'MessageType',
    'DatabaseType',
    'MemoryType',
    'ExpertiseLevel',
    'AgentMessage',
    'SearchResult',
    'QueryPlan',
    'CriticResult',
    'AgentMemory',
    'UserContext',
    'MemoryRetrievalResult',
    'StreamingAgentState',
    'ChartData',
    'StreamingResponse',
    'FeedbackData',
    'WorkflowMetrics'
]
