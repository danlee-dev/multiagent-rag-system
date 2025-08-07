import json
import asyncio
import os
from typing import AsyncGenerator, List
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

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

        # OpenAI fallback 모델들
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.llm_openai_mini = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=temperature,
                api_key=self.openai_api_key
            )
            print("SimpleAnswererAgent: OpenAI fallback 모델 초기화 완료")
        else:
            self.llm_openai_mini = None
            print("SimpleAnswererAgent: 경고: OPENAI_API_KEY가 설정되지 않음")

        self.agent_type = "SIMPLE_ANSWERER"

    async def _astream_with_fallback(self, prompt, primary_model, fallback_model):
        """
        스트리밍을 위한 Gemini API rate limit 시 OpenAI로 fallback 처리
        """
        try:
            async for chunk in primary_model.astream(prompt):
                yield chunk
        except Exception as e:
            error_str = str(e).lower()
            rate_limit_indicators = ['429', 'quota', 'rate limit', 'exceeded', 'resource_exhausted']

            if any(indicator in error_str for indicator in rate_limit_indicators):
                print(f"SimpleAnswererAgent: Gemini API rate limit 감지, OpenAI로 fallback 시도: {e}")
                if fallback_model:
                    try:
                        print("SimpleAnswererAgent: OpenAI fallback으로 스트리밍 시작")
                        async for chunk in fallback_model.astream(prompt):
                            yield chunk
                    except Exception as fallback_error:
                        print(f"SimpleAnswererAgent: OpenAI fallback도 실패: {fallback_error}")
                        raise fallback_error
                else:
                    print("SimpleAnswererAgent: OpenAI 모델이 초기화되지 않음")
                    raise e
            else:
                raise e

    async def _invoke_with_fallback(self, prompt, primary_model, fallback_model):
        """
        Gemini API rate limit 시 OpenAI로 fallback 처리
        """
        try:
            result = await primary_model.ainvoke(prompt)
            return result
        except Exception as e:
            error_str = str(e).lower()
            rate_limit_indicators = ['429', 'quota', 'rate limit', 'exceeded', 'resource_exhausted']

            if any(indicator in error_str for indicator in rate_limit_indicators):
                print(f"SimpleAnswererAgent: Gemini API rate limit 감지, OpenAI로 fallback 시도: {e}")
                if fallback_model:
                    try:
                        result = await fallback_model.ainvoke(prompt)
                        print("SimpleAnswererAgent: OpenAI fallback 성공")
                        return result
                    except Exception as fallback_error:
                        print(f"SimpleAnswererAgent: OpenAI fallback도 실패: {fallback_error}")
                        raise fallback_error
                else:
                    print("SimpleAnswererAgent: OpenAI 모델이 초기화되지 않음")
                    raise e
            else:
                raise e

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

        # 웹 검색 수행 및 결과 스트리밍
        if need_web_search:
            print(f"- 웹 검색 필요: {web_search_query}")
            web_results = await self._simple_web_search(web_search_query)
            if web_results:
                search_results.extend(web_results)
                # 웹 검색 결과를 프론트엔드로 스트리밍 (JSON 이벤트로)
                search_event = {
                    "type": "search_results",
                    "step": 1,
                    "tool_name": "Web Search",
                    "query": web_search_query,
                    "results": [
                        {
                            "title": result.title,
                            "content_preview": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                            "url": result.url if hasattr(result, 'url') else None,
                            "source": result.source,
                            "relevance_score": result.relevance_score,
                            "document_type": result.document_type
                        }
                        for result in web_results
                    ]
                }
                yield json.dumps(search_event)
                print(f"- 웹 검색 결과 스트리밍 완료: {len(web_results)}개 결과")

        # 벡터 검색 수행 및 결과 스트리밍
        if need_vector_search:
            print(f"- 벡터 검색 필요: {vector_search_query}")
            vector_results = await self._simple_vector_search(vector_search_query)
            if vector_results:
                search_results.extend(vector_results)
                # 벡터 검색 결과를 프론트엔드로 스트리밍 (JSON 이벤트로)
                search_event = {
                    "type": "search_results",
                    "step": 2,
                    "tool_name": "Vector Database",
                    "query": vector_search_query,
                    "results": [
                        {
                            "title": result.title,
                            "content_preview": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                            "url": result.url if hasattr(result, 'url') else None,
                            "source": result.source,
                            "relevance_score": result.relevance_score,
                            "document_type": result.document_type
                        }
                        for result in vector_results
                    ]
                }
                yield json.dumps(search_event)
                print(f"- 벡터 검색 결과 스트리밍 완료: {len(vector_results)}개 결과")

        # 메모리 컨텍스트 추출
        memory_context = state.get("metadata", {}).get("memory_context", "")
        if memory_context:
            print(f"- 메모리 컨텍스트 사용: {len(memory_context)}자")

        full_response = ""
        prompt = self._create_enhanced_prompt_with_memory(
            query, search_results, memory_context
        )

        try:
            chunk_count = 0
            content_generated = False

            async for chunk in self._astream_with_fallback(
                prompt,
                self.streaming_chat,
                self.llm_openai_mini
            ):
                chunk_count += 1
                if hasattr(chunk, 'content') and chunk.content:
                    content_generated = True
                    full_response += chunk.content
                    yield chunk.content
                    print(f">> SimpleAnswerer 청크 {chunk_count}: {len(chunk.content)} 문자")

            print(f">> SimpleAnswerer 완료: 총 {chunk_count}개 청크, {len(full_response)} 문자")

            # 내용이 전혀 생성되지 않은 경우 fallback 처리
            if not content_generated or not full_response.strip():
                print(">> 경고: SimpleAnswerer에서 내용이 생성되지 않음, fallback 실행")
                fallback_response = f"""죄송합니다. 현재 시스템에 일시적인 문제가 있어 답변을 생성할 수 없습니다.

**사용자 질문**: {query}

다시 시도해 주시거나, 잠시 후에 다시 문의해 주세요."""
                yield fallback_response
                full_response = fallback_response

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

        # 출처 정보 저장 (프론트엔드에서 사용)
        if search_results:
            sources_data = []
            for i, result in enumerate(search_results[:10], 1):  # 최대 10개로 증가
                source_data = {
                    "id": i,
                    "title": getattr(result, 'metadata', {}).get("title", result.title or "자료"),
                    "content": result.content[:300] + "..." if len(result.content) > 300 else result.content,
                    "url": result.url if hasattr(result, 'url') else None,
                    "source_url": result.source_url if hasattr(result, 'source_url') else None,
                    "source_type": result.source if hasattr(result, 'source') else "unknown"
                }
                sources_data.append(source_data)
            state["metadata"]["sources"] = sources_data

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
                                metadata={"original_query": query, **current_result},
                                source_url=current_result.get("link", "웹 검색 결과")
                            )
                            search_results.append(search_result)

                        # 새 결과 시작
                        current_result = {"title": line[3:].strip()}  # 번호 제거
                    elif line.startswith("출처 링크:"):
                        current_result["link"] = line[7:].strip()  # "출처 링크:" 제거
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
                        metadata={"original_query": query, **current_result},
                        source_url=current_result.get("link", "웹 검색 결과")
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
            response = await self._invoke_with_fallback(
                prompt,
                self.llm_lite,
                self.llm_openai_mini
            )
            response_content = response.content.strip()

            # JSON 파싱 시도 - 개선된 파싱 로직
            try:
                # 코드 블록 제거
                clean_response = response_content
                if "```json" in response_content:
                    clean_response = response_content.split("```json")[1].split("```")[0].strip()
                elif "```" in response_content:
                    clean_response = response_content.split("```")[1].split("```")[0].strip()

                # JSON 파싱
                response_json = json.loads(clean_response)
                needs_web_search = response_json.get("needs_web_search", False)
                web_search_query = response_json.get("web_search_query", "")
                needs_vector_search = response_json.get("needs_vector_search", False)
                vector_search_query = response_json.get("vector_search_query", "")

                print(f"- 검색 판단 완료: 웹={needs_web_search}, 벡터={needs_vector_search}")
                return needs_web_search, web_search_query, needs_vector_search, vector_search_query

            except json.JSONDecodeError as e:
                print(f"- JSON 파싱 오류: {e}")
                print(f"- LLM 응답: {response_content[:200]}...")

                # 문자열 패턴 매칭으로 fallback 파�ing
                needs_web_search = False
                needs_vector_search = False

                # 응답에서 키워드 기반으로 판단
                if "needs_web_search" in response_content:
                    if "needs_web_search\": true" in response_content or "needs_web_search\":true" in response_content:
                        needs_web_search = True

                if "needs_vector_search" in response_content:
                    if "needs_vector_search\": true" in response_content or "needs_vector_search\":true" in response_content:
                        needs_vector_search = True

                print(f"- Fallback 파싱 결과: 웹={needs_web_search}, 벡터={needs_vector_search}")
                # 기본값 반환 (간단한 인사는 검색 불필요)
                return needs_web_search, "", needs_vector_search, ""

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
            for i, result in enumerate(search_results[:3], 1):
                content = result.content
                title = getattr(result, 'metadata', {}).get("title", result.title or "자료")

                # URL 정보 추가 (웹 검색 결과인 경우)
                url_info = ""
                if hasattr(result, 'url') and result.url:
                    url_info = f"\n  **출처 링크**: {result.url}"
                elif hasattr(result, 'source_url') and result.source_url and not result.source_url.startswith(('웹 검색', 'Vector DB')):
                    url_info = f"\n  **출처 링크**: {result.source_url}"

                summary_parts.append(f"**[참고자료 {i}]** **{title}**: {content[:200]}...{url_info}")
            context_summary = "\n\n".join(summary_parts)

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
- **중요**: 참고 정보를 사용할 때는 다음 형식으로 출처를 표기하세요:
  * 문장 끝에 [SOURCE:번호] 형식으로 출처 번호를 표기
  * 예시: "건강기능식품 시장 규모는 6조 440억 원입니다 [SOURCE:1]"
  * 참고 정보의 순서대로 1, 2, 3... 번호를 사용하세요

답변:"""
