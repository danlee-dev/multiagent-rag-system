from pydantic import BaseModel, Field, HttpUrl
from typing import Dict, List, Any, Optional, Literal, Union, TypedDict
from enum import Enum
from datetime import datetime

class ScrapeInput(BaseModel):
    url: str = Field(description="스크래핑할 웹페이지의 전체 URL")
    query: str = Field(description="추출의 맥락을 제공하는 원본 사용자 질문")

class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    DATA_GATHERER = "data_gatherer"
    PROCESSOR = "processor"
    SIMPLE_ANSWERER = "simple_answerer"

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

class SearchResult(BaseModel):
    """검색 결과 표준 형태 - Claude 스타일 UI를 위한 확장된 정보"""
    source: str  # 데이터 소스 이름(graph_db, vector_db, memory, web_search, ...)
    content: str  # 검색 결과 내용
    search_query: str = ""  # 검색한 쿼리 그 자체

    # Claude 스타일 UI를 위한 추가 필드들
    title: str = Field(default="", description="문서 제목 또는 결과 제목")
    url: Optional[str] = Field(default=None, description="원본 URL")
    score: float = Field(default=0.7, description="관련성 점수")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


    # 메타데이터는 호환성을 위해 유지
    @property
    def metadata(self) -> Dict[str, Any]:
        """호환성을 위한 metadata 프로퍼티"""
        return {
            "title": self.title,
            "url": self.url,
            "document_type": self.document_type,
            "relevance_score": self.relevance_score,
            "timestamp": self.timestamp,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "similarity_score": self.similarity_score
        }

class CriticResult(BaseModel):
    # 'pass' 또는 'fail'로 더 명확하게
    status: Literal["pass", "fail_with_feedback"]
    # 재계획에 직접 사용할 피드백
    feedback: str = Field(description="Orchestrator가 재계획에 사용할 구체적인 피드백")
    confidence: float

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

class StreamingAgentState(TypedDict):
    """스트리밍 에이전트 상태"""
    # 기본 정보 필드
    original_query: str
    conversation_id: str
    user_id: str
    start_time: str

    # Planning ('chat' || 'task)
    flow_type: Optional[Literal['chat', 'task']]

    # Orchestration (Workflow 설계 - json 형태로 저장)
    plan: Optional[Dict[str, Any]]

    # Execution (실행 상태 추적)
    current_step_index: int
    step_results: List[Any]
    execution_log: List[str]

    # 재계획 및 분기 여부
    needs_replan: bool
    replan_feedback: Optional[str]

    # 최종 결과
    final_answer: Optional[str]

    # 추가 필드들
    session_id: str
    metadata: Dict[str, Any]

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

# 헬퍼 함수들
def create_source_info(title: str, url: str = None, **kwargs) -> Dict[str, Any]:
    """SourceInfo 생성 헬퍼 함수"""
    source_info = SourceInfo(title=title, url=url, **kwargs)
    return source_info.model_dump()

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
    # 새로운 State 구조에 맞춰 step_results에서 SearchResult 인스턴스를 찾습니다.
    for result in state.get('step_results', []):
        if isinstance(result, SearchResult):
             source_collection.add_source_from_search_result(result, is_primary=True)
             print(f"- SearchResult 출처 추가: {result.source}")
    print(f"- 총 출처 개수: {source_collection.total_count}")
    return source_collection

# 내보낼 모델들
__all__ = [
    "ScrapeInput",
    "AgentRole",
    "DatabaseType",
    "MemoryType",
    "ExpertiseLevel",
    "SearchResult",
    "CriticResult",
    "UserContext",
    "MemoryRetrievalResult",
    "StreamingAgentState",
    "ChartData",
    "StreamingResponse",
    "FeedbackData",
    "WorkflowMetrics",
    "SourceInfo",
    "SourceCollectionData",
    "create_source_info",
    "enhance_search_result_with_source",
    "extract_sources_from_state"
]
