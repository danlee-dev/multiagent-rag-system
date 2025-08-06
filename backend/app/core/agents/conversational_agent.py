import json
import asyncio
from typing import AsyncGenerator, List
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI

from ..models.models import StreamingAgentState, SearchResult
from ...services.search.search_tools import vector_db_search
from ...services.search.search_tools import debug_web_search


class SimpleAnswererAgent:
    """단순 질문 전용 Agent - 새로운 아키텍처에 맞게 최적화"""

    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.7):
        self.streaming_chat = ChatGoogleGenerativeAI(
            model=model, temperature=temperature
        )
        self.llm_lite = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", temperature=temperature
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
        need_web_search, web_search_query, need_vector_search, vector_search_query = await self._needs_search(query)

        print(f"- 검색 필요 여부: 웹={need_web_search}, 벡터={need_vector_search}")

        if need_web_search:
            print(f"- 웹 검색 필요: {web_search_query}")
            search_results.extend(await self._simple_web_search(web_search_query))
            print(f"- 웹 검색 결과: {(search_results)}")

        if need_vector_search:
            print(f"- 벡터 검색 필요: {vector_search_query}")
            search_results.extend(await self._simple_vector_search(vector_search_query))


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

    async def _simple_web_search(self, query: str) -> List[SearchResult]:
        """간단한 웹 검색"""
        try:
            result_text = await asyncio.get_event_loop().run_in_executor(
                None, debug_web_search, query
            )

            # 결과가 문자열인 경우 파싱
            search_results = []
            if result_text and isinstance(result_text, str):
                # 간단한 파싱으로 SearchResult 객체 생성
                lines = result_text.split('\n')
                current_result = {}

                for line in lines:
                    line = line.strip()
                    if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                        # 이전 결과 저장
                        if current_result:
                            search_result = SearchResult(
                                source="web_search",
                                content=current_result.get("snippet", ""),
                                search_query=query,
                                title=current_result.get("title", "웹 검색 결과"),
                                url=current_result.get("link"),
                                relevance_score=0.9,  # 웹검색 결과는 높은 점수
                                timestamp=datetime.now().isoformat(),
                                document_type="web",
                                metadata={"original_query": query, **current_result}
                            )
                            search_results.append(search_result)

                        # 새 결과 시작
                        current_result = {"title": line[3:].strip()}  # 번호 제거
                    elif line.startswith("링크:"):
                        current_result["link"] = line[3:].strip()
                    elif line.startswith("요약:"):
                        current_result["snippet"] = line[3:].strip()

                # 마지막 결과 저장
                if current_result:
                    search_result = SearchResult(
                        source="web_search",
                        content=current_result.get("snippet", ""),
                        search_query=query,
                        title=current_result.get("title", "웹 검색 결과"),
                        url=current_result.get("link"),
                        relevance_score=0.9,
                        timestamp=datetime.now().isoformat(),
                        document_type="web",
                        metadata={"original_query": query, **current_result}
                    )
                    search_results.append(search_result)

            print(f"- 웹 검색 완료: {len(search_results)}개 결과")
            return search_results[:3]  # 상위 3개 결과만
        except Exception as e:
            print(f"웹 검색 오류: {e}")
            return []

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

    async def _needs_search(self, query: str):
        """질문에 대한 검색이 필요한지 여부를 판단"""
        try:
            prompt = f"""
당신은 AI 어시스턴트입니다. 사용자의 질문에 답변하기 위해 검색이 필요한지 판단하세요.
질문: {query}
오늘 날짜 : {datetime.now().strftime('%Y년 %m월 %d일')}
Web 검색이 필요하면 True, 아니면 False를 반환하세요.
Vector DB 검색이 필요하면 True, 아니면 False를 반환하세요.
- Web 검색은 최근 정보, 이슈, 간단한 정보가 필요할 때 사용
- Vector DB 검색은 특정 데이터, 문서, 현황, 통계, 내부 정보가 필요할 때 사용

다음과 같은 순서/형식으로 응답하세요:
{{
            "needs_web_search": false,
            "web_search_query": "웹 검색 쿼리",
            "needs_vector_search": false,
            "vector_search_query": "벡터 DB 검색 쿼리"
}}

웹 검색 쿼리 예시
- "2025년 최신 건강기능식품 트렌드"
벡터 검색 쿼리 예시
- "2025년 유행하는 건강식품이 뭐가 있나요?"

웹 검색 쿼리는 키워드 기반 문장으로
벡터 검색 쿼리는 질문형식으로 작성하세요
        """
            response = await self.llm_lite.ainvoke(prompt)
            response_content = response.content.strip()

            # JSON 파싱 시도
            try:
                response_json = json.loads(response_content)
                needs_web_search = response_json.get("needs_web_search", False)
                web_search_query = response_json.get("web_search_query", "")
                needs_vector_search = response_json.get("needs_vector_search", False)
                vector_search_query = response_json.get("vector_search_query", "")

                print(f"- 검색 판단 완료: 웹={needs_web_search}, 벡터={needs_vector_search}")
                return needs_web_search, web_search_query, needs_vector_search, vector_search_query

            except json.JSONDecodeError as e:
                print(f"- JSON 파싱 오류: {e}")
                print(f"- LLM 응답: {response_content[:200]}...")
                # 기본값 반환 (간단한 인사는 검색 불필요)
                return False, "", False, ""

        except Exception as e:
            print(f"- _needs_search 오류: {e}")
            # 오류 시 기본값 반환
            return False, "", False, ""


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
- 수집된 문서 바탕으로 작성된 부분이 있을 경우, 해당 부분에 마크다운 형식의 하이퍼링크 출처를 명시

답변:"""
