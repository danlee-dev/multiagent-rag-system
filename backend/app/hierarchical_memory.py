import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, asdict

@dataclass
class MemoryEntry:
    """메모리 엔트리 기본 클래스"""
    id: str
    timestamp: datetime
    content: str
    importance: float  
    access_count: int = 0
    last_accessed: Optional[datetime] = None

@dataclass
class ConversationMemory:
    """대화 메모리"""
    id: str
    timestamp: datetime
    content: str
    importance: float
    user_id: str
    query: str
    response: str
    context_used: List[str]
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    feedback: Optional[str] = None

@dataclass
class KnowledgeMemory:
    """지식 메모리 - 자주 검색되는 정보"""
    id: str
    timestamp: datetime
    content: str
    importance: float
    topic: str
    key_facts: List[str]
    sources: List[str]
    related_queries: List[str]
    access_count: int = 0
    last_accessed: Optional[datetime] = None

@dataclass
class UserProfileMemory:
    """사용자 프로필 메모리"""
    id: str
    timestamp: datetime
    content: str
    importance: float
    user_id: str
    preferences: Dict[str, Any]
    expertise_areas: List[str]
    interaction_patterns: Dict[str, Any]
    access_count: int = 0
    last_accessed: Optional[datetime] = None

class HierarchicalMemorySystem:
    """계층적 메모리 시스템"""

    def __init__(self, max_short_term: int = 100, max_long_term: int = 1000):
        print("\n>> 계층적 메모리 시스템 초기화")

        # LLM 요약기 초기화
        try:
            from langchain_openai import ChatOpenAI
            self.summarizer_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
            self.llm_available = True
            print("- LLM 요약기 초기화 완료")
        except Exception as e:
            print(f"- LLM 초기화 실패: {e}")
            self.summarizer_llm = None
            self.llm_available = False

        # 단기 메모리 (현재 세션)
        self.short_term_memory: List[Any] = []
        self.max_short_term = max_short_term

        # 장기 메모리 (영구 저장)
        self.long_term_memory: Dict[str, List[Any]] = defaultdict(list)
        self.max_long_term = max_long_term

        # 작업 메모리 (현재 태스크 관련)
        self.working_memory: Dict[str, Any] = {}

        # 사용자별 메모리
        self.user_memories: Dict[str, List[UserProfileMemory]] = defaultdict(list)

        # 지식 메모리 (도메인별)
        self.knowledge_base: Dict[str, List[KnowledgeMemory]] = defaultdict(list)

        # 통합 설정
        self.consolidation_threshold = 15  # 15개 쌓이면 통합

        print("- 단기 메모리 용량:", max_short_term)
        print("- 장기 메모리 용량:", max_long_term)

    def add_conversation_memory(self, user_id: str, query: str, response: str,
                              context_used: List[str], importance: float = 0.5):
        """대화 메모리 추가 - 기본 방식 (LLM 없이)"""
        memory = ConversationMemory(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            content=f"Q: {query}\nA: {response}",
            importance=importance,
            user_id=user_id,
            query=query,
            response=response,
            context_used=context_used
        )

        self.short_term_memory.append(memory)
        print(f"- 대화 메모리 추가: {query[:50]}...")

        # 단기 메모리 용량 초과 시 중요도 기반 정리
        if len(self.short_term_memory) > self.max_short_term:
            self._consolidate_memories()

    async def add_conversation_memory_smart(self, user_id: str, query: str,
                                          response: str, context_used: List[str]):
        """지능적 대화 메모리 추가 - LLM 요약 포함"""
        if not self.llm_available:
            # LLM 없으면 기본 방식 사용
            self.add_conversation_memory(user_id, query, response, context_used, 0.5)
            return

        try:
            # LLM으로 대화 분석
            summary_data = await self._summarize_conversation(query, response, user_id, context_used)

            # 기억할 가치가 없으면 저장 안 함
            if not summary_data.get('should_remember', True):
                print("- 기억할 필요 없는 대화로 판단, 저장 생략")
                return

            # 요약된 내용으로 메모리 생성
            memory = ConversationMemory(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                content=summary_data['summary'],
                importance=summary_data['importance_score'],
                user_id=user_id,
                query=query[:100] + "..." if len(query) > 100 else query,
                response=summary_data['summary'],
                context_used=context_used[:3]
            )

            self.short_term_memory.append(memory)

            # 사용자 정보 업데이트
            if summary_data.get('user_info'):
                self._update_user_info_from_summary(user_id, summary_data['user_info'])

            # 지식 베이스 업데이트
            if summary_data.get('key_facts'):
                self.add_knowledge_memory(
                    topic=summary_data['topic'],
                    key_facts=summary_data['key_facts'],
                    sources=['conversation'],
                    importance=summary_data['importance_score']
                )

            print(f"- 지능적 대화 메모리 추가: {summary_data['summary']}")

            # 통합 임계점 확인
            if len(self.short_term_memory) >= self.consolidation_threshold:
                await self._intelligent_consolidation()

        except Exception as e:
            print(f"- 지능적 메모리 저장 실패, 기본 방식 사용: {e}")
            self.add_conversation_memory(user_id, query, response, context_used, 0.5)

    async def _summarize_conversation(self, query: str, response: str,
                                    user_id: str, context_used: List[str]) -> Dict[str, Any]:
        """LLM으로 대화 요약 및 분석"""
        prompt = f"""
다음 대화를 분석하고 메모리에 저장할 핵심 정보를 추출해주세요.

**대화:**
사용자: {query}
어시스턴트: {response}

다음 JSON 형식으로 반환하세요:

{{
    "importance_score": 0.0-1.0,
    "summary": "핵심 요약 (50자 이내)",
    "key_facts": ["주요 사실들"],
    "user_info": {{"이름": "값"}},
    "topic": "주제",
    "should_remember": true/false,
    "memory_type": "personal_info|factual|casual"
}}

**중요도 기준:**
- 1.0: 사용자 개인정보 (이름 등)
- 0.8: 중요한 정보 질의
- 0.5: 일반 질의
- 0.2: 단순 인사

유효한 JSON만 반환:
"""

        try:
            response_obj = await self.summarizer_llm.ainvoke(prompt)
            summary_data = json.loads(response_obj.content)
            return summary_data
        except Exception as e:
            print(f"- 대화 요약 실패: {e}")
            return {
                "importance_score": 0.5,
                "summary": f"{query[:30]}...",
                "key_facts": [],
                "user_info": {},
                "topic": "일반",
                "should_remember": True,
                "memory_type": "casual"
            }

    def _update_user_info_from_summary(self, user_id: str, user_info: Dict[str, Any]):
        """요약에서 추출된 사용자 정보 업데이트"""
        existing_profile = self._get_user_profile(user_id)

        if existing_profile:
            existing_profile.preferences.update(user_info)
            existing_profile.last_accessed = datetime.now()
            existing_profile.access_count += 1
        else:
            profile = UserProfileMemory(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                content=f"User profile for {user_id}",
                importance=0.9,
                user_id=user_id,
                preferences=user_info,
                expertise_areas=[],
                interaction_patterns={}
            )
            self.user_memories[user_id].append(profile)

        print(f"- 사용자 정보 업데이트: {user_info}")

    async def _intelligent_consolidation(self):
        """지능적 메모리 통합"""
        print("\n>> 지능적 메모리 통합 시작")

        # 대화 메모리만 선별
        conversation_memories = [
            m for m in self.short_term_memory
            if hasattr(m, 'query') and hasattr(m, 'response')
        ]

        if len(conversation_memories) < 5:
            return

        # 중요도 낮은 메모리들 장기 메모리로 이동
        low_importance = [m for m in conversation_memories if m.importance < 0.6]
        high_importance = [m for m in conversation_memories if m.importance >= 0.6]

        # 중요도 낮은 것들은 장기 메모리로
        for memory in low_importance:
            self.long_term_memory['ConversationMemory'].append(memory)

        # 중요한 것들만 단기 메모리에 유지
        self.short_term_memory = [
            m for m in self.short_term_memory
            if not hasattr(m, 'query') or m.importance >= 0.6
        ]

        print(f"- 메모리 통합 완료: {len(low_importance)}개 장기 메모리로 이동")

    def add_knowledge_memory(self, topic: str, key_facts: List[str],
                           sources: List[str], importance: float = 0.7):
        """지식 메모리 추가"""
        memory = KnowledgeMemory(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            content=f"Topic: {topic}",
            importance=importance,
            topic=topic,
            key_facts=key_facts,
            sources=sources,
            related_queries=[]
        )

        self.knowledge_base[topic].append(memory)
        print(f"- 지식 메모리 추가: {topic}")

    def update_user_profile(self, user_id: str, preferences: Dict[str, Any],
                          expertise_areas: List[str]):
        """사용자 프로필 업데이트"""
        existing_profile = self._get_user_profile(user_id)

        if existing_profile:
            # 기존 프로필 업데이트
            existing_profile.preferences.update(preferences)
            existing_profile.expertise_areas = list(set(existing_profile.expertise_areas + expertise_areas))
            existing_profile.last_accessed = datetime.now()
            existing_profile.access_count += 1
        else:
            # 새 프로필 생성
            profile = UserProfileMemory(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                content=f"User profile for {user_id}",
                importance=0.8,
                user_id=user_id,
                preferences=preferences,
                expertise_areas=expertise_areas,
                interaction_patterns={}
            )
            self.user_memories[user_id].append(profile)

        print(f"- 사용자 프로필 업데이트: {user_id}")

    def retrieve_relevant_memories(self, query: str, user_id: str,
                                 top_k: int = 5) -> List[Any]:
        """관련 메모리 검색"""
        all_memories = []

        # 단기 메모리에서 검색
        all_memories.extend(self.short_term_memory)

        # 장기 메모리에서 검색
        for memories in self.long_term_memory.values():
            all_memories.extend(memories)

        # 지식 베이스에서 검색
        for memories in self.knowledge_base.values():
            all_memories.extend(memories)

        # 관련도 계산 및 정렬
        scored_memories = []
        for memory in all_memories:
            relevance_score = self._calculate_relevance(query, memory)
            scored_memories.append((relevance_score, memory))

        # 관련도 순으로 정렬하고 상위 k개 반환
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        relevant_memories = [memory for _, memory in scored_memories[:top_k]]

        # 접근 횟수 및 시간 업데이트
        for memory in relevant_memories:
            memory.access_count += 1
            memory.last_accessed = datetime.now()

        print(f"- 관련 메모리 {len(relevant_memories)}개 검색됨")
        return relevant_memories

    def get_working_memory_context(self) -> str:
        """현재 작업 메모리 컨텍스트 반환"""
        if not self.working_memory:
            return ""

        context_parts = []
        for key, value in self.working_memory.items():
            context_parts.append(f"{key}: {value}")

        return "\n".join(context_parts)

    def update_working_memory(self, key: str, value: Any):
        """작업 메모리 업데이트"""
        self.working_memory[key] = value
        print(f"- 작업 메모리 업데이트: {key}")

    def _consolidate_memories(self):
        """메모리 통합 - 중요도 기반으로 장기 메모리로 이동"""
        print("\n>> 메모리 통합 시작")

        # 중요도와 접근 빈도를 고려한 점수 계산
        scored_memories = []
        for memory in self.short_term_memory:
            consolidation_score = self._calculate_consolidation_score(memory)
            scored_memories.append((consolidation_score, memory))

        # 점수 순으로 정렬
        scored_memories.sort(key=lambda x: x[0], reverse=True)

        # 상위 50%는 장기 메모리로 이동
        cutoff = len(scored_memories) // 2
        for score, memory in scored_memories[:cutoff]:
            memory_type = type(memory).__name__
            self.long_term_memory[memory_type].append(memory)

        # 단기 메모리는 최근 것들만 유지
        self.short_term_memory = [memory for _, memory in scored_memories[cutoff:]]

        print(f"- {cutoff}개 메모리를 장기 메모리로 이동")
        print(f"- 단기 메모리 현재 크기: {len(self.short_term_memory)}")

    def _calculate_relevance(self, query: str, memory: Any) -> float:
        """쿼리와 메모리 간 관련도 계산"""
        query_words = set(query.lower().split())
        memory_words = set(memory.content.lower().split())

        # 단어 겹침도
        overlap = len(query_words.intersection(memory_words))
        total_words = len(query_words.union(memory_words))

        if total_words == 0:
            word_similarity = 0.0
        else:
            word_similarity = overlap / total_words

        # 시간 가중치 (최근 것일수록 높은 점수)
        time_weight = self._calculate_time_weight(memory.timestamp)

        # 접근 빈도 가중치
        access_weight = min(memory.access_count / 10.0, 1.0)

        # 중요도 가중치
        importance_weight = memory.importance

        # 최종 관련도 점수
        relevance = (word_similarity * 0.4 +
                    time_weight * 0.2 +
                    access_weight * 0.2 +
                    importance_weight * 0.2)

        return relevance

    def _calculate_consolidation_score(self, memory: Any) -> float:
        """메모리 통합 점수 계산"""
        # 중요도
        importance_score = memory.importance

        # 접근 빈도 점수
        access_score = min(memory.access_count / 5.0, 1.0)

        # 시간 점수 (너무 오래된 것은 낮은 점수)
        time_score = self._calculate_time_weight(memory.timestamp)

        return importance_score * 0.5 + access_score * 0.3 + time_score * 0.2

    def _calculate_time_weight(self, timestamp: datetime) -> float:
        """시간 가중치 계산"""
        now = datetime.now()
        time_diff = now - timestamp

        # 24시간 이내: 1.0, 일주일 이내: 0.7, 한 달 이내: 0.3, 그 이후: 0.1
        if time_diff <= timedelta(days=1):
            return 1.0
        elif time_diff <= timedelta(days=7):
            return 0.7
        elif time_diff <= timedelta(days=30):
            return 0.3
        else:
            return 0.1

    def _get_user_profile(self, user_id: str) -> Optional[UserProfileMemory]:
        """사용자 프로필 조회"""
        profiles = self.user_memories.get(user_id, [])
        return profiles[-1] if profiles else None

    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계 반환"""
        return {
            "short_term_count": len(self.short_term_memory),
            "long_term_count": sum(len(memories) for memories in self.long_term_memory.values()),
            "knowledge_base_count": sum(len(memories) for memories in self.knowledge_base.values()),
            "user_profiles_count": sum(len(profiles) for profiles in self.user_memories.values()),
            "working_memory_keys": list(self.working_memory.keys())
        }

    def save_memory_state(self, filepath: str):
        """메모리 상태 저장"""
        state = {
            "short_term_memory": [asdict(m) for m in self.short_term_memory],
            "long_term_memory": {k: [asdict(m) for m in v] for k, v in self.long_term_memory.items()},
            "knowledge_base": {k: [asdict(m) for m in v] for k, v in self.knowledge_base.items()},
            "user_memories": {k: [asdict(m) for m in v] for k, v in self.user_memories.items()},
            "working_memory": self.working_memory
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2, default=str)

        print(f"- 메모리 상태 저장됨: {filepath}")

    def load_memory_state(self, filepath: str):
        """메모리 상태 로드"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)

            # 메모리 복원 로직 구현 필요
            # (datetime 객체 복원 등)

            print(f"- 메모리 상태 로드됨: {filepath}")
        except FileNotFoundError:
            print(f"- 메모리 파일이 없어서 새로 시작: {filepath}")
