import asyncio
from typing import AsyncGenerator, List
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI

from ..models.models import StreamingAgentState, SearchResult
from ...services.search.search_tools import vector_db_search


class SimpleAnswererAgent:
    """단순 질문 전용 Agent - 새로운 아키텍처에 맞게 최적화"""

    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.7):
        self.streaming_chat = ChatGoogleGenerativeAI(
            model=model, temperature=temperature
        )
        self.agent_type = "SIMPLE_ANSWERER"

    async def answer_streaming(
        self, state: StreamingAgentState
    ) -> AsyncGenerator[str, None]:
        """스트리밍으로 답변을 생성하는 메서드"""
        print("\n>> SimpleAnswerer: 스트리밍 답변 시작")

        query = state["original_query"]  # 딕셔너리 접근 방식 사용

        # 간단한 벡터 검색 수행 (필요시)
        search_results = []
        if self._needs_vector_search(query):
            print("- 벡터 검색 수행")
            search_results = await self._simple_vector_search(query)

        # 메모리 컨텍스트 추출
        memory_context = state.get("metadata", {}).get("memory_context", "")
        if memory_context:
            print(f"- 메모리 컨텍스트 사용: {len(memory_context)}자")

        full_response = ""
        prompt = self._create_enhanced_prompt_with_memory(
            query, search_results, memory_context
        )

        try:
            async for chunk in self.streaming_chat.astream(prompt):
                if chunk.content:
                    full_response += chunk.content
                    yield chunk.content

        except Exception as e:
            print(f"- LLM 스트리밍 오류: {e}")
            # fallback 응답
            fallback_response = f"""죄송합니다. 현재 시스템에 일시적인 문제가 있어 답변을 생성할 수 없습니다.

**사용자 질문**: {query}

다시 시도해 주시거나, 잠시 후에 다시 문의해 주세요."""

            yield fallback_response
            full_response = fallback_response

        state["final_answer"] = full_response
        state["metadata"]["simple_answer_completed"] = True
        print(f"- 스트리밍 답변 생성 완료 (길이: {len(full_response)}자)")

    async def _simple_vector_search(self, query: str) -> List[SearchResult]:
        """간단한 벡터 검색"""
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None, vector_db_search, query
            )

            search_results = []
            for result in results[:3]:  # 상위 3개만
                if isinstance(result, dict):
                    search_result = SearchResult(
                        source="vector_db",
                        content=result.get("content", ""),
                        search_query=query,
                        title=result.get("title", "벡터 DB 문서"),
                        url=None,
                        relevance_score=result.get("similarity_score", 0.7),
                        timestamp=datetime.now().isoformat(),
                        document_type="database",
                        similarity_score=result.get("similarity_score", 0.7),
                        metadata=result
                    )
                    search_results.append(search_result)

            return search_results
        except Exception as e:
            print(f"벡터 검색 오류: {e}")
            return []

    def _needs_vector_search(self, query: str) -> bool:
        """Vector 검색이 필요한지 간단 판단"""
        # 간단한 인사말이나 감사 표현은 검색 불필요
        simple_phrases = [
            "안녕", "고마워", "감사", "고맙", "좋아", "괜찮", "네", "예", "응"
        ]

        if any(phrase in query for phrase in simple_phrases) and len(query) < 20:
            return False

        # 질문이나 정보 요청은 검색 필요
        search_indicators = [
            "뭐", "무엇", "어떤", "얼마", "언제", "어디", "누구", "왜", "어떻게",
            "알려줘", "설명", "정보", "가격", "현황"
        ]

        return any(indicator in query for indicator in search_indicators)

    def _create_enhanced_prompt_with_memory(
        self, query: str, search_results: List[SearchResult], memory_context: str
    ) -> str:
        """메모리 컨텍스트를 포함한 향상된 프롬프트"""
        current_date_str = datetime.now().strftime("%Y년 %m월 %d일")

        # 검색 결과 요약
        context_summary = ""
        if search_results:
            summary_parts = []
            for result in search_results[:3]:
                content = result.content
                title = getattr(result, 'metadata', {}).get("title", result.title or "자료")
                summary_parts.append(f"- **{title}**: {content[:200]}...")
            context_summary = "\n".join(summary_parts)

        # 메모리 컨텍스트 처리
        memory_info = ""
        if memory_context:
            memory_info = f"\n## 대화 맥락\n{memory_context[:500]}...\n"

        return f"""당신은 친근하고 도움이 되는 AI 어시스턴트입니다.
현재 날짜: {current_date_str}

{memory_info}

## 참고 정보
{context_summary if context_summary else "추가 참고 정보 없음"}

## 사용자 질문
{query}

## 응답 가이드
- 자연스럽고 친근한 톤으로 답변
- 참고 정보가 있으면 이를 활용하되, 정확한 정보만 사용
- 불확실한 내용은 명시적으로 표현
- 간결하면서도 도움이 되는 답변 제공
- 필요시 추가 질문을 권유
- 마크다운 형식으로 답변 작성

답변:"""
