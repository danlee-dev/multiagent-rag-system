from pydantic import BaseModel, Field, HttpUrl
from typing import Dict, List, Any, Optional, Literal, Union, TypedDict
from enum import Enum
from datetime import datetime

class ScrapeInput(BaseModel):
    url: str = Field(description="스크래핑할 웹페이지의 전체 URL")
    query: str = Field(description="추출의 맥락을 제공하는 원본 사용자 질문")

class AgentType(str, Enum):
    """Agent Type 정의"""
    PLANNING = "planning"
    RETRIEVER = "retriever"
    CRITIC_1 = "critic_1"
    CRITIC_2 = "critic_2"
    CONTEXT_INTEGRATOR = "context_integrator"
    REPORT_GENERATOR = "report_generator"
    SIMPLE_ANSWERER = "simple_answerer"

class MessageType(str, Enum):
    """Agent 간 주고 받는 메세지 타입 정의"""
    REAL_TIME_HINT = "real_time_hint"
    SEARCH_REQUEST = "search_request"
    INTERESTING_FINDING = "interesting_finding"
    FEEDBACK = "feedback"
    RESULT = "result"
    MEMORY_RETRIEVAL = "memory_retrieval"
    MEMORY_STORAGE = "memory_storage"

class DatabaseType(str, Enum):
    """데이터베이스 타입"""
    GRAPH_DB = "graph_db"
    VECTOR_DB = "vector_db"
    RDB = "rdb"
    API = "api"
    WEB = "web"
    MEMORY = "memory"

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
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    SUPER_COMPLEX = "super_complex"

class ExecutionStrategy(str, Enum):
    """실행 전략 - 4단계 복잡도 대응"""
    DIRECT_ANSWER = "direct_answer"
    BASIC_SEARCH = "basic_search"
    FULL_REACT = "full_react"
    MULTI_AGENT = "multi_agent"

class AgentMessage(BaseModel):
    """Agent 간 실시간 메시지"""
    from_agent: AgentType
    to_agent: AgentType
    message_type: MessageType
    content: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    priority: int = Field(default=1, description="1=highest, 5=lowest")

class SearchResult(BaseModel):
    """검색 결과 표준 형태"""
    source: str
    content: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    search_query: str = ""
    memory_type: Optional[MemoryType] = None

class QueryPlan(BaseModel):
    """섹션 기반 쿼리 계획 - 유연한 보고서 구조 지원"""
    original_query: str
    sub_queries: List[str] = Field(default_factory=list)
    estimated_complexity: str = Field(default="adaptive")
    execution_strategy: str = Field(default="section_based")
    resource_requirements: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""
    priority: int = Field(default=1, ge=1, le=5)
    expected_output_type: str = Field(default="report")
    estimated_processing_time: str = Field(default="adaptive")
    required_databases: List[DatabaseType] = Field(default_factory=list)
    memory_context_needed: bool = Field(default=True)
    execution_steps: List[str] = Field(default_factory=list)
    fallback_strategy: Optional[str] = None

class CriticResult(BaseModel):
    """Critic Agent의 평가 결과"""
    status: Literal["sufficient", "insufficient"]
    suggestion: str
    confidence: float
    reasoning: str
    memory_recommendation: Optional[str] = None

class AgentMemory(BaseModel):
    """각 Agent의 개별 메모리"""
    agent_type: AgentType
    internal_state: Dict[str, Any] = Field(default_factory=dict)
    message_history: List[AgentMessage] = Field(default_factory=list)
    findings: List[str] = Field(default_factory=list)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    memory_usage_stats: Dict[str, int] = Field(default_factory=dict)

    def add_finding(self, finding: str):
        self.findings.append(f"[{datetime.now().strftime('%H:%M:%S')}] {finding}")

    def update_metric(self, metric_name: str, value: float):
        self.performance_metrics[metric_name] = value

    def update_memory_stat(self, stat_name: str, value: int):
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
        self.findings.append(finding)

    def update_metric(self, name: str, value: float):
        self.metrics[name] = value

    def set_context(self, key: str, value: Any):
        self.context[key] = value

class StreamingAgentState(BaseModel):
    """스트리밍 에이전트 상태"""
    model_config = {"extra": "allow", "arbitrary_types_allowed": True}
    
    original_query: str = ""
    conversation_id: str = Field(default_factory=lambda: f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    user_id: str = Field(default="default_user")
    start_time: str = Field(default_factory=lambda: datetime.now().isoformat())
    flow_type: Optional[Literal['chat', 'task']] = None
    plan: Optional[Dict[str, Any]] = None
    current_step_index: int = 0
    step_results: Dict[str, Any] = Field(default_factory=dict)
    execution_log: List[str] = Field(default_factory=list)
    needs_replan: bool = False
    replan_feedback: Optional[str] = None
    final_answer: Optional[str] = None

    # 추가 필드들
    memory_context: Optional[str] = ""
    query_plan: Optional[Any] = None
    execution_mode: Optional[str] = None
    graph_results_stream: List[Any] = Field(default_factory=list)
    multi_source_results_stream: List[Any] = Field(default_factory=list)
    info_sufficient: bool = False
    context_sufficient: bool = False
    search_complete: bool = False
    planning_complete: bool = False
    critic1_result: Optional[Dict[str, Any]] = None
    current_iteration: int = 0
    max_iterations: int = 3
    integrated_context: str = ""
    additional_context: str = ""
    
    def add_step_result(self, key: str, value: Any):
        """step_results에 결과 추가 (딕셔너리 형태로 관리)"""
        self.step_results[key] = value
    
    def get_step_result(self, key: str, default=None):
        """step_results에서 특정 키의 값 조회"""
        return self.step_results.get(key, default)
    
    def add_multi_source_result(self, result: 'SearchResult'):
        """multi_source_results_stream에 검색 결과 추가"""
        if not hasattr(self, 'multi_source_results_stream') or self.multi_source_results_stream is None:
            self.multi_source_results_stream = []
        self.multi_source_results_stream.append(result)
    
    def add_graph_result(self, result: 'SearchResult'):
        """graph_results_stream에 검색 결과 추가"""
        if not hasattr(self, 'graph_results_stream') or self.graph_results_stream is None:
            self.graph_results_stream = []
        self.graph_results_stream.append(result)
    
    def add_execution_log(self, message: str):
        """실행 로그 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.execution_log.append(log_entry)
    
    def get_all_search_results(self) -> List['SearchResult']:
        """모든 검색 결과를 통합하여 반환"""
        all_results = []
        if hasattr(self, 'graph_results_stream') and self.graph_results_stream:
            all_results.extend(self.graph_results_stream)
        if hasattr(self, 'multi_source_results_stream') and self.multi_source_results_stream:
            all_results.extend(self.multi_source_results_stream)
        return all_results
    
    def get_complexity_level(self) -> str:
        """복잡도 레벨 반환"""
        if hasattr(self, 'query_plan') and self.query_plan:
            if hasattr(self.query_plan, 'execution_strategy'):
                strategy = self.query_plan.execution_strategy
            elif isinstance(self.query_plan, dict):
                strategy = self.query_plan.get('execution_strategy', 'basic_search')
            else:
                strategy = 'basic_search'
                
            if strategy == 'direct_answer':
                return 'simple'
            elif strategy == 'basic_search':
                return 'medium'
            elif strategy == 'full_react':
                return 'complex'
            elif strategy == 'multi_agent':
                return 'super_complex'
            else:
                return 'medium'
        return 'medium'

class ChartData(BaseModel):
    """차트 데이터"""
    chart_type: str
    title: str
    data: List[Dict[str, Any]]
    x_axis: str
    y_axis: str
    colors: Optional[List[str]] = None
    options: Optional[Dict[str, Any]] = None

class StreamingResponse(BaseModel):
    """스트리밍 응답"""
    chunk_type: str
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
    rating: int = Field(ge=1, le=5)
    feedback_text: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    memory_helpful: Optional[bool] = None

class WorkflowMetrics(BaseModel):
    """워크플로우 성능 메트릭"""
    total_processing_time: float
    node_processing_times: Dict[str, float] = Field(default_factory=dict)
    memory_performance: Dict[str, float] = Field(default_factory=dict)
    search_performance: Dict[str, float] = Field(default_factory=dict)
    user_satisfaction_score: Optional[float] = None

class SourceInfo(BaseModel):
    """출처 정보 상세 모델 - 기존 SearchResult의 metadata에 들어갈 정보"""
    title: str = Field(description="문서 제목")
    url: Optional[str] = Field(default=None, description="원본 URL (하이퍼링크 가능)")
    author: Optional[str] = Field(default=None, description="작성자")
    organization: Optional[str] = Field(default=None, description="기관/회사명")
    published_date: Optional[str] = Field(default=None, description="발행일")
    access_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"), description="접근일")
    document_type: str = Field(default="web", description="문서 타입 (web, pdf, report, api 등)")
    page_number: Optional[int] = Field(default=None, description="페이지 번호 (PDF 등)")
    section: Optional[str] = Field(default=None, description="섹션/챕터명")
    reliability_score: float = Field(default=0.8, ge=0.0, le=1.0, description="신뢰도 점수")

    def to_citation(self) -> str:
        """출처 인용 형식 자동 생성"""
        citation_parts = []
        if self.author:
            citation_parts.append(f"{self.author}")
        if self.title:
            citation_parts.append(f'"{self.title}"')
        if self.organization:
            citation_parts.append(f"{self.organization}")
        if self.published_date:
            citation_parts.append(f"({self.published_date})")
        if self.url:
            citation_parts.append(f"Retrieved from {self.url}")
        return ", ".join(citation_parts)

class SourceCollectionData(BaseModel):
    """출처 모음 데이터 - StreamingAgentState의 step_results에 저장될 데이터"""
    primary_sources: List[Dict[str, Any]] = Field(default_factory=list, description="주요 출처")
    supporting_sources: List[Dict[str, Any]] = Field(default_factory=list, description="보조 출처")
    total_count: int = Field(default=0, description="총 출처 개수")
    credibility_summary: Dict[str, Any] = Field(default_factory=dict, description="신뢰도 요약")

    def add_source_from_search_result(self, search_result: 'SearchResult', is_primary: bool = True):
        """기존 SearchResult에서 출처 정보 추출해서 추가"""
        source_data = {
            "source": search_result.source,
            "content_preview": search_result.content[:200] + "..." if len(search_result.content) > 200 else search_result.content,
            "relevance_score": search_result.relevance_score,
            "metadata": search_result.metadata,
            "timestamp": search_result.timestamp,
            "search_query": search_result.search_query
        }
        if "source_info" in search_result.metadata:
            source_data["source_info"] = search_result.metadata["source_info"]
        if is_primary:
            self.primary_sources.append(source_data)
        else:
            self.supporting_sources.append(source_data)
        self.total_count = len(self.primary_sources) + len(self.supporting_sources)
        self._update_credibility_summary()

    def _update_credibility_summary(self):
        """신뢰도 요약 업데이트"""
        all_sources = self.primary_sources + self.supporting_sources
        if not all_sources:
            self.credibility_summary = {"average_reliability": 0, "source_types": [], "total_sources": 0}
            return
        reliabilities = []
        source_types = set()
        for source in all_sources:
            if "source_info" in source and isinstance(source["source_info"], dict):
                reliabilities.append(source["source_info"].get("reliability_score", 0.5))
                source_types.add(source["source_info"].get("document_type", "unknown"))
        avg_reliability = sum(reliabilities) / len(reliabilities) if reliabilities else 0
        self.credibility_summary = {
            "average_reliability": round(avg_reliability, 2),
            "source_types": list(source_types),
            "total_sources": len(all_sources),
            "high_reliability_count": len([r for r in reliabilities if r >= 0.8])
        }

def create_source_info(title: str, url: str = None, **kwargs) -> Dict[str, Any]:
    """SourceInfo 생성 헬퍼 함수"""
    source_info = SourceInfo(title=title, url=url, **kwargs)
    return source_info.dict()

def enhance_search_result_with_source(search_result: 'SearchResult', source_info: Dict[str, Any]) -> None:
    """기존 SearchResult에 출처 정보 추가 (인플레이스 수정)"""
    if not hasattr(search_result, 'metadata') or search_result.metadata is None:
        search_result.metadata = {}
    search_result.metadata["source_info"] = source_info
    search_result.metadata["has_enhanced_source"] = True

def extract_sources_from_state(state: 'StreamingAgentState') -> SourceCollectionData:
    """StreamingAgentState에서 모든 출처 정보 추출"""
    print("\n>> extract_sources_from_state 시작")
    source_collection = SourceCollectionData()
    for result in getattr(state, 'graph_results_stream', []):
        source_collection.add_source_from_search_result(result, is_primary=True)
        print(f"- Graph DB 출처 추가: {result.source}")
    for result in getattr(state, 'multi_source_results_stream', []):
        source_collection.add_source_from_search_result(result, is_primary=False)
        print(f"- Multi Source 출처 추가: {result.source}")
    print(f"- 총 출처 개수: {source_collection.total_count}")
    return source_collection

__all__ = [
    "AgentType", "MessageType", "DatabaseType", "MemoryType", "ExpertiseLevel",
    "ComplexityLevel", "ExecutionStrategy", "AgentMessage", "SearchResult",
    "QueryPlan", "CriticResult", "AgentMemory", "SimpleAgentMemory",
    "UserContext", "MemoryRetrievalResult", "StreamingAgentState",
    "ChartData", "StreamingResponse", "FeedbackData", "WorkflowMetrics",
    "SourceInfo", "SourceCollectionData", "create_source_info",
    "enhance_search_result_with_source", "extract_sources_from_state"
]
