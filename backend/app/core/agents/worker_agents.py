from langchain_core.tools import tool
import re
import sys
import asyncio
import json
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from ..models.models import SearchResult, CriticResult, StreamingAgentState
from ...services.search.search_tools import (
    debug_web_search,
    rdb_search,
    vector_db_search,
    graph_db_search,
    scrape_and_extract_content,
)


class DataGathererAgent:
    """데이터 수집 전담 Agent - 다양한 검색 도구를 실행"""

    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        # 도구 매핑 설정 - 이름 통일
        self.tool_mapping = {
            "web_search": self._web_search,
            "vector_db_search": self._vector_db_search,  # 이름 수정
            "graph_db_search": self._graph_db_search,    # 이름 수정
            "rdb_search": self._rdb_search,
            "scrape_content": self._scrape_content,
        }

    async def execute(self, tool: str, inputs: Dict[str, Any]) -> List[SearchResult]:
        """단일 진입점 - 지정된 도구를 실행하고 SearchResult 리스트 반환"""
        print(f"\n>> DataGatherer 실행: {tool} - '{inputs.get('query', '')}'")
        sys.stdout.flush()

        if tool not in self.tool_mapping:
            print(f"- 알 수 없는 도구: {tool}")
            return []

        try:
            results = await self.tool_mapping[tool](**inputs)
            print(f"- {tool} 결과: {len(results)}개")
            return results
        except Exception as e:
            print(f"- {tool} 실행 오류: {e}")
            return []

    async def _web_search(self, query: str, **kwargs) -> List[SearchResult]:
        """웹 검색 실행 - 2024-2025 최신 정보 강조"""
        try:
            # 2024-2025 최신 정보를 강조하는 쿼리 수정
            enhanced_query = f"{query} 2024 2025 최신 현황"
            print(f"- 강화된 검색 쿼리: {enhanced_query}")

            # 기존 debug_web_search 함수 활용
            result_text = await asyncio.get_event_loop().run_in_executor(
                None, debug_web_search, enhanced_query
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
                                search_query=enhanced_query,
                                title=current_result.get("title", "웹 검색 결과"),
                                url=current_result.get("link"),
                                relevance_score=0.9,  # 웹검색 결과는 높은 점수
                                timestamp=datetime.now().isoformat(),
                                document_type="web",
                                metadata={"original_query": query, "enhanced_query": enhanced_query, **current_result}
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
                        search_query=enhanced_query,
                        title=current_result.get("title", "웹 검색 결과"),
                        url=current_result.get("link"),
                        relevance_score=0.9,
                        timestamp=datetime.now().isoformat(),
                        document_type="web",
                        metadata={"original_query": query, "enhanced_query": enhanced_query, **current_result}
                    )
                    search_results.append(search_result)

            print(f"- 웹 검색 완료: {len(search_results)}개 결과")
            return search_results[:5]  # 상위 5개 결과만
        except Exception as e:
            print(f"웹 검색 오류: {e}")
            return []

    async def _vector_db_search(self, query: str, **kwargs) -> List[SearchResult]:
        """Vector DB 검색 실행"""
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None, vector_db_search, query
            )

            search_results = []
            for result in results:
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

            return search_results[:5]
        except Exception as e:
            print(f"Vector DB 검색 오류: {e}")
            return []

    async def _graph_db_search(self, query: str, **kwargs) -> List[SearchResult]:
        """Graph DB 검색 실행"""
        graph_query = await self._decompose_for_graphdb(query)
        print(f"  - 변환된 GraphDB 쿼리: {graph_query}")
        raw_results = await asyncio.to_thread(graph_db_search.invoke, {"query": graph_query})

        search_results = [
            SearchResult(
                source="graph_db", content=res.get("content", ""), search_query=query,
                title=f"그래프 정보: {res.get('entity', '알 수 없음')}", relevance_score=res.get("confidence", 0.8), metadata=res
            ) for res in raw_results[:5] if isinstance(res, dict)
        ]
        return search_results

    async def _rdb_search(self, query: str, **kwargs) -> List[SearchResult]:
        """RDB 검색 실행"""
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None, rdb_search, query
            )

            search_results = []
            for result in results:
                if isinstance(result, dict):
                    search_result = SearchResult(
                        source="rdb_search",
                        content=result.get("content", ""),
                        search_query=query,
                        title=result.get("title", "RDB 데이터"),
                        url=None,
                        relevance_score=0.8,
                        timestamp=datetime.now().isoformat(),
                        document_type="database",
                        metadata=result
                    )
                    search_results.append(search_result)

            return search_results[:5]
        except Exception as e:
            print(f"RDB 검색 오류: {e}")
            return []

    async def _scrape_content(self, url: str, query: str = "", **kwargs) -> List[SearchResult]:
        """웹페이지 스크래핑 실행"""
        try:
            content = await asyncio.get_event_loop().run_in_executor(
                None, scrape_and_extract_content, url, query
            )

            if content:
                search_result = SearchResult(
                    source="scrape_content",
                    content=content,
                    search_query=query,
                    title=f"스크래핑된 콘텐츠: {url}",
                    url=url,
                    relevance_score=0.9,
                    timestamp=datetime.now().isoformat(),
                    document_type="web",
                    metadata={"scraped_url": url}
                )
                return [search_result]

            return []
        except Exception as e:
            print(f"웹 스크래핑 오류: {e}")
            return []

    async def _decompose_for_graphdb(self, query: str) -> str:
        """LLM을 사용하여 쿼리를 GraphDB용 키워드로 변환"""
        prompt = f"다음 사용자 질문에서 그래프 데이터베이스(GraphDB) 검색에 가장 효과적인 핵심 키워드를 2~3개만 쉼표(,)로 구분해서 추출해줘. 다른 설명은 절대 추가하지 마. 예: '제주도 감귤 가격' -> '제주도, 감귤, 가격'. 질문: \"{query}\""
        response = await self.llm.ainvoke(prompt)
        return response.content.strip()


# 차트 생성을 위한 간단한 함수 (tool 데코레이터 제거)
def create_simple_chart(query: str, context: str) -> Dict[str, Any]:
    """차트 생성을 위한 간단한 함수"""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        prompt = f"""
        다음 컨텍스트와 요청을 바탕으로 Chart.js에서 사용할 수 있는 차트 데이터를 JSON으로 생성해줘.
        요청: {query}
        컨텍스트: {context[:2000]}

        JSON 형식:
        {{"type": "bar" or "line" or "pie", "data": {{"labels": ["항목1", "항목2"], "datasets": [{{"label": "데이터셋 이름", "data": [숫자1, 숫자2]}}]}}, "options": {{"responsive": true, "plugins": {{"title": {{"display": true, "text": "차트 제목"}}}}}}}}
        다른 설명 없이 JSON 객체만 반환해.
        """
        response_str = llm.invoke(prompt).content

        # LLM 응답에서 JSON만 추출
        json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "차트 JSON 생성 실패"}
    except Exception as e:
        print(f"차트 생성 오류: {e}")
        return {"error": f"차트 데이터 파싱 실패: {str(e)}"}


# 개선된 프롬프트 - LangChain create_react_agent 호환
IMPROVED_REACT_PROMPT_TEMPLATE = """
당신은 수석 데이터 분석가입니다. 주어진 컨텍스트를 최대한 활용하여 사용자 질문에 대한 전문적인 보고서를 작성하세요.

**중요 규칙:**
1. 주어진 컨텍스트를 우선 활용하세요
2. 컨텍스트만으로 부족한 경우에만 도구를 사용하세요
3. 최신 정보(2024-2025)를 우선시하세요
4. 간결하고 논리적으로 사고하세요

**사용 가능한 도구:** {tool_names}

**도구 목록:**
{tools}

**형식:**
Thought: (현재 상황과 계획)
Action: (도구 이름)
Action Input: (검색어)
Observation: (결과)
...
Thought: 이제 보고서를 작성하겠습니다.
Final Answer: (최종 보고서)

**컨텍스트:**
{context}

**사용자 질문:**
{input}

{agent_scratchpad}
"""

IMPROVED_REACT_PROMPT = PromptTemplate.from_template(IMPROVED_REACT_PROMPT_TEMPLATE)


class ProcessorAgent:
    """데이터 가공 전담 Agent - 다양한 처리 작업을 수행"""

    def __init__(self, model_pro: str = "gemini-2.5-pro", model_flash: str = "gemini-2.5-flash", temperature: float = 0.2, use_react: bool = False):
        self.llm_pro = ChatGoogleGenerativeAI(model=model_pro, temperature=temperature)
        self.llm_flash = ChatGoogleGenerativeAI(model=model_flash, temperature=0)
        self.use_react = use_react  # ReAct 사용 여부 제어 - 기본값을 False로 변경

        # 처리 타입과 해당 메서드를 매핑
        self.processor_mapping = {
            "summarize_and_integrate": self._summarize_and_integrate,
            "integrate_context": self._summarize_and_integrate,  # 동일한 기능
            "critique": self._evaluate_criticism,
            "generate_report": self._generate_report,
            "create_charts": self._create_charts,
        }

    async def process(self, processor_type: str, data: Any, original_query: str) -> Any:
        """단일 진입점 - 지정된 처리를 수행하고 결과 반환"""
        print(f"\n>> Processor 실행: {processor_type}")
        sys.stdout.flush()

        if processor_type not in self.processor_mapping:
            error_message = f"알 수 없는 처리 타입: {processor_type}"
            print(f"- {error_message}")
            return {"error": error_message}

        try:
            # 모든 메서드를 await으로 결과 반환
            result = await self.processor_mapping[processor_type](data=data, query=original_query)
            print(f"- {processor_type} 완료")
            return result
        except Exception as e:
            error_message = f"{processor_type} 처리 오류: {e}"
            print(f"- {error_message}")
            return {"error": error_message}

    async def process_streaming(self, processor_type: str, data: Any, original_query: str):
        """스트리밍 지원 처리 메서드"""
        print(f"\n>> Processor 스트리밍 실행: {processor_type}")
        sys.stdout.flush()

        if processor_type == "generate_report":
            # 보고서 생성은 스트리밍으로 처리
            async for chunk in self._generate_report_streaming(data, original_query):
                yield chunk
        else:
            # 다른 처리는 기존 방식으로 실행 후 결과 반환
            result = await self.process(processor_type, data, original_query)
            yield str(result)

    async def _generate_report_streaming(self, data: Any, query: str):
        """보고서 생성 - 스트리밍 버전"""
        # ReAct 사용이 비활성화된 경우 직접 폴백 사용
        if not self.use_react:
            print("  - ReAct 비활성화됨, 직접 LLM 보고서 생성...")
            context = str(data) if not isinstance(data, str) else data
            async for chunk in self._fallback_report_generation_streaming_chunks(context, query):
                yield chunk
            return

        print("  - ReAct Agent를 사용한 실시간 보고서 생성 시작...")

        try:
            # 데이터를 문자열로 변환
            if isinstance(data, list):
                context = "\n".join([str(item) for item in data])
            else:
                context = str(data)

            # ReAct Agent 도구들 정의
            tools = [debug_web_search, vector_db_search, graph_db_search, rdb_search, scrape_and_extract_content]

            # 더 안정적인 LLM 설정 (온도 낮춤)
            stable_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.1,  # 온도를 낮춰서 더 예측 가능한 출력
                max_output_tokens=2000
            )

            # ReAct Agent 생성 - hub에서 기본 프롬프트 사용
            react_prompt = hub.pull("hwchase17/react")
            react_agent = create_react_agent(
                llm=stable_llm,
                tools=tools,
                prompt=react_prompt
            )

            # Agent Executor 생성 - ReAct 적극 활용 설정
            agent_executor = AgentExecutor(
                agent=react_agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,  # 파싱 오류 자동 처리
                max_iterations=5,  # ReAct 반복 횟수 증가
                return_intermediate_steps=True,  # 중간 단계 반환 활성화
                max_execution_time=60.0  # 실행 시간 제한 확대
            )

            try:
                async for chunk in self._react_agent_streaming_chunks(agent_executor, context, query):
                    yield chunk

            except Exception as e:
                error_message = str(e)
                print(f"  - ReAct Agent 실행 오류: {error_message}")

                # 심각한 ReAct 오류만 폴백 처리, 나머지는 재시도
                critical_errors = [
                    "Invalid Format", "Missing 'Action:'", "JSON decode",
                    "파싱 오류", "Expecting value"
                ]

                if any(keyword in error_message for keyword in critical_errors):
                    print(f"  - 심각한 ReAct 오류 감지, 폴백 사용: {error_message[:100]}...")
                    async for chunk in self._fallback_report_generation_streaming_chunks(context, query):
                        yield chunk
                else:
                    # 시간 초과나 반복 제한은 부분 결과라도 사용 시도
                    print("  - 경미한 오류, 폴백 사용")
                    async for chunk in self._fallback_report_generation_streaming_chunks(context, query):
                        yield chunk

        except Exception as e:
            print(f"  - ReAct Agent 초기화 오류: {e}")
            context = str(data) if not isinstance(data, str) else data
            async for chunk in self._fallback_report_generation_streaming_chunks(context, query):
                yield chunk

    async def _react_agent_streaming_chunks(self, agent_executor: AgentExecutor, context: str, query: str):
        """ReAct Agent를 스트리밍으로 실행하여 청크 단위로 전송"""
        print("  - ReAct Agent 스트리밍 실행 시작...")

        # ReAct Agent가 도구를 적극 활용하도록 유도하는 쿼리
        enhanced_query = f"""
다음 질문에 대해 포괄적이고 상세한 분석을 제공하세요.
필요한 경우 도구를 사용하여 추가 정보를 수집하고 분석하세요.

질문: {query}

기본 컨텍스트: {context[:500] if len(context) > 500 else context}

단계별로 사고하고, 필요한 도구를 사용하여 최상의 답변을 제공하세요.
"""

        try:
            # ReAct Agent를 astream_events로 실행하여 스트리밍 처리
            final_output = ""
            agent_completed = False

            async for event in agent_executor.astream_events(
                {"input": enhanced_query},
                version="v1"
            ):
                kind = event["event"]

                # LLM의 실제 출력을 캐치하여 실시간 전송
                if kind == "on_llm_stream" and event["name"] == "ChatGoogleGenerativeAI":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, 'content') and chunk.content:
                        final_output += chunk.content
                        yield chunk.content

                # Agent의 최종 답변 확인
                elif kind == "on_chain_end" and event["name"] == "AgentExecutor":
                    agent_completed = True
                    output = event["data"].get("output", {})
                    if isinstance(output, dict) and "output" in output:
                        agent_answer = output["output"]
                        # 아직 전송되지 않은 부분이 있다면 전송
                        if agent_answer and len(agent_answer) > len(final_output):
                            remaining = agent_answer[len(final_output):]
                            final_output += remaining
                            yield remaining

            # ReAct Agent가 정상 완료된 경우, 결과의 길이와 상관없이 성공으로 처리
            if agent_completed:
                print(f"\n  - ReAct Agent 성공적으로 완료 (출력 길이: {len(final_output)}자)")
                # 출력이 매우 짧거나 비어있는 경우에만 경고 표시 (하지만 폴백 사용하지 않음)
                if len(final_output.strip()) < 5:
                    print("  - 경고: ReAct Agent 출력이 매우 짧습니다 (하지만 결과를 신뢰합니다)")
            else:
                # Agent가 완료되지 않은 경우에만 폴백 사용
                print("  - ReAct Agent가 완료되지 않음, 폴백 사용")
                async for chunk in self._fallback_report_generation_streaming_chunks(context, query):
                    yield chunk

        except Exception as e:
            print(f"  - ReAct Agent 스트리밍 오류: {e}")
            async for chunk in self._fallback_report_generation_streaming_chunks(context, query):
                yield chunk

    async def _fallback_report_generation_streaming_chunks(self, context: str, query: str):
        """폴백 보고서 생성 - 스트리밍 청크 버전"""
        print("  - 폴백 보고서 스트리밍 생성 시작...")

        try:
            # 안정적인 LLM으로 폴백 보고서 생성
            fallback_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.3,
                max_output_tokens=3000
            )

            fallback_prompt = f"""
당신은 전문적인 시장 분석가입니다. 다음 정보를 바탕으로 간결하고 유용한 보고서를 작성하세요.

사용자 질문: "{query}"
현재 날짜: 2025년 8월

수집된 데이터:
{context[:6000]}

다음 구조로 보고서를 작성하세요:

## 주요 발견사항
[핵심 내용 3-4개 항목으로 정리]

## 상세 분석
[구체적인 데이터와 분석 내용]

## 실행 가능한 제안
[2-3개의 구체적인 제안사항]

**주의사항:**
- 마크다운 형식 사용
- 구체적인 수치나 데이터가 있다면 반드시 포함
- 실용적이고 행동 가능한 내용 위주
- 2000자 이내로 간결하게 작성
"""

            async for chunk in fallback_llm.astream(fallback_prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content

            print(f"\n  - 폴백 보고서 스트리밍 완료")

        except Exception as e:
            print(f"  - 폴백 보고서 스트리밍 실패: {e}")

            # API 할당량 초과 등의 경우 수집된 데이터만으로 간단한 요약 제공
            error_type = str(e)
            if "quota" in error_type.lower() or "429" in error_type:
                yield "\n\n# API 할당량 초과로 인한 간단 요약\n\n"
                yield f"**질문:** {query}\n\n"
                yield "**수집된 정보 요약:**\n"

                # 수집된 컨텍스트에서 핵심 정보 추출
                context_lines = context.split('\n')[:20]  # 처음 20줄만
                for i, line in enumerate(context_lines):
                    if line.strip() and len(line.strip()) > 10:
                        yield f"- {line.strip()}\n"
                        if i >= 10:  # 최대 10개 항목만
                            break

                yield "\n**참고:** API 할당량 제한으로 상세한 분석 대신 수집된 데이터의 요약만 제공됩니다.\n"
                yield "더 자세한 분석을 원하시면 잠시 후 다시 시도해 주세요."
            else:
                # 다른 오류의 경우 기본 메시지
                yield f"""
# 질문에 대한 답변

**질문:** {query}

**수집된 정보 요약:**
{context[:1000]}

**참고:** 시스템 오류로 인해 간단한 요약만 제공됩니다.
더 자세한 정보가 필요하시면 다시 문의해 주세요.
"""

    async def _summarize_and_integrate(self, data: Any, query: str) -> str:
        """여러 검색 결과를 통합하고 요약합니다."""
        print("  - 데이터 통합 및 요약 작업 수행...")

        if not data:
            return "수집된 데이터가 없습니다."

        # data가 리스트인지 확인하고, 각 요소가 SearchResult인지 확인
        search_results = []
        if isinstance(data, list):
            # 리스트의 각 요소를 처리
            for item in data:
                if isinstance(item, list):
                    # 중첩된 리스트인 경우 (DataGatherer에서 반환된 결과)
                    search_results.extend(item)
                elif hasattr(item, 'source') and hasattr(item, 'content'):
                    # SearchResult 유사 객체인 경우
                    search_results.append(item)
                else:
                    # 기타 형태의 데이터는 스킵하고 로그만 출력
                    print(f"  - 알 수 없는 데이터 형태 무시: {type(item)}")
        elif hasattr(data, 'source') and hasattr(data, 'content'):
            # 단일 SearchResult인 경우
            search_results.append(data)
        else:
            # 완전히 다른 형태의 데이터인 경우
            print(f"  - 예상되지 않은 데이터 형태: {type(data)}")
            return f"데이터 통합 실패: 예상되지 않은 데이터 형태 {type(data)}"

        if not search_results:
            return "처리 가능한 검색 결과가 없습니다."

        # 최신 데이터 우선 정렬 (웹 검색 결과 우선)
        sorted_data = sorted(search_results, key=lambda x: (
            x.source == "web_search",  # 웹 검색 우선
            getattr(x, 'relevance_score', 0.7)
        ), reverse=True)

        content_summary = "\n\n".join([
            f"## 출처: {res.source} (관련성: {getattr(res, 'relevance_score', 0.7):.2f})\n"
            f"제목: {getattr(res, 'title', '제목 없음')}\n"
            f"내용: {getattr(res, 'content', '내용 없음')}"
            for res in sorted_data[:10]  # 상위 10개만
        ])

        prompt = f"""
        사용자 질문: "{query}"
        현재 날짜: 2025년 8월

        아래는 여러 소스에서 수집된 정보입니다. 이 정보들을 종합하여 사용자의 질문에 답변하기 위한 일관된 컨텍스트로 통합하고 요약해주세요.

        **중요:** 2024-2025년 최신 정보를 우선시하고, 오래된 정보는 참고용으로만 활용하세요.

        [수집된 정보]
        {content_summary}

        [작업 지침]
        1. 최신 정보(2024-2025)를 중심으로 구성
        2. 중복 정보는 통합하고, 상충 정보는 명시
        3. 논리적 구조로 재구성 (서론-본론-결론)
        4. 보고서 작성에 적합한 형태로 정리

        통합된 컨텍스트:
        """
        response = await self.llm_pro.ainvoke(prompt)
        return response.content

    async def _evaluate_criticism(self, data: str, query: str) -> CriticResult:
        """결과물을 비평합니다."""
        print("  - 결과물 비평 작업 수행...")

        prompt = f"""
        사용자 원본 질문: "{query}"

        아래는 수집 및 통합된 정보입니다:
        {data[:4000]}

        이 정보가 사용자 질문에 대한 완전한 답변을 위해 충분한지 평가하세요.

        JSON 형식으로만 응답:
        {{
            "status": "pass" 또는 "fail_with_feedback",
            "feedback": "부족한 경우 구체적 피드백",
            "confidence": 0.0-1.0
        }}
        """
        response = await self.llm_flash.ainvoke(prompt)
        try:
            # JSON 추출
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group())
                return CriticResult(**result_data)
            else:
                return CriticResult(status="pass", feedback="평가 완료", confidence=0.8)
        except Exception as e:
            print(f"비평 결과 파싱 오류: {e}")
            return CriticResult(status="pass", feedback="평가 오류", confidence=0.5)

    async def _create_charts(self, data: Any, query: str) -> Dict[str, Any]:
        """데이터를 바탕으로 차트 데이터를 생성합니다."""
        print("  - 차트 데이터 생성 작업 수행...")

        try:
            # 데이터를 문자열로 변환
            if isinstance(data, list):
                context = "\n".join([str(item) for item in data])
            else:
                context = str(data)

            # 차트 생성 프롬프트 개선
            chart_prompt = f"""
            다음 데이터를 바탕으로 Chart.js 형식의 차트를 생성해주세요.

            요청: {query}
            데이터: {context[:1500]}

            다음 JSON 형식으로만 응답해주세요. 다른 설명은 절대 추가하지 마세요:
            {{
                "type": "bar",
                "data": {{
                    "labels": ["항목1", "항목2", "항목3"],
                    "datasets": [{{
                        "label": "데이터셋 이름",
                        "data": [10, 20, 30],
                        "backgroundColor": "rgba(75, 192, 192, 0.6)",
                        "borderColor": "rgba(75, 192, 192, 1)",
                        "borderWidth": 1
                    }}]
                }},
                "options": {{
                    "responsive": true,
                    "plugins": {{
                        "title": {{
                            "display": true,
                            "text": "차트 제목"
                        }}
                    }},
                    "scales": {{
                        "y": {{
                            "beginAtZero": true
                        }}
                    }}
                }}
            }}
            """

            response = await self.llm_flash.ainvoke(chart_prompt)
            response_text = response.content.strip()

            # JSON 추출 개선
            try:
                # 코드 블록 제거
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()

                # JSON 파싱
                chart_data = json.loads(response_text)

                # 필수 필드 검증
                if "type" not in chart_data or "data" not in chart_data:
                    raise ValueError("필수 필드 누락")

                print(f"  - 차트 생성 성공: {chart_data['type']} 타입")
                return chart_data

            except (json.JSONDecodeError, ValueError) as e:
                print(f"  - 차트 JSON 파싱 실패: {e}")
                # 기본 차트 데이터 반환
                return {
                    "type": "bar",
                    "data": {
                        "labels": ["2019년", "2023년"],
                        "datasets": [{
                            "label": "시장 점유율 (%)",
                            "data": [25, 35],
                            "backgroundColor": "rgba(75, 192, 192, 0.6)",
                            "borderColor": "rgba(75, 192, 192, 1)",
                            "borderWidth": 1
                        }]
                    },
                    "options": {
                        "responsive": True,
                        "plugins": {
                            "title": {
                                "display": True,
                                "text": "건강기능식품 시장 트렌드"
                            }
                        },
                        "scales": {
                            "y": {
                                "beginAtZero": True
                            }
                        }
                    }
                }

        except Exception as e:
            print(f"  - 차트 생성 전체 오류: {e}")
            return {
                "type": "bar",
                "data": {
                    "labels": ["데이터 없음"],
                    "datasets": [{
                        "label": "오류",
                        "data": [0],
                        "backgroundColor": "rgba(255, 99, 132, 0.6)"
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": "차트 생성 오류"
                        }
                    }
                }
            }

    async def _generate_report(self, data: Any, query: str) -> str:
        """보고서 생성 - ReAct 사용 옵션에 따라 처리 방식 결정"""

        # ReAct 사용이 비활성화된 경우 직접 폴백 사용
        if not self.use_react:
            print("  - ReAct 비활성화됨, 직접 LLM 보고서 생성...")
            context = str(data) if not isinstance(data, str) else data
            return await self._fallback_report_generation_streaming(context, query)

        print("  - ReAct Agent를 사용한 실시간 보고서 생성 시작...")

        try:
            # 데이터를 문자열로 변환
            if isinstance(data, list):
                context = "\n".join([str(item) for item in data])
            else:
                context = str(data)

            # ReAct Agent 도구들 정의
            tools = [debug_web_search, vector_db_search, graph_db_search, rdb_search, scrape_and_extract_content]

            # 더 안정적인 LLM 설정 (온도 낮춤)
            stable_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.1,  # 온도를 낮춰서 더 예측 가능한 출력
                max_output_tokens=2000
            )

            # ReAct Agent 생성 - hub에서 기본 프롬프트 사용
            react_prompt = hub.pull("hwchase17/react")
            react_agent = create_react_agent(
                llm=stable_llm,
                tools=tools,
                prompt=react_prompt
            )

            # Agent Executor 생성 - ReAct 적극 활용 설정
            agent_executor = AgentExecutor(
                agent=react_agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,  # 파싱 오류 자동 처리
                max_iterations=5,  # ReAct 반복 횟수 증가
                return_intermediate_steps=True,  # 중간 단계 반환 활성화
            )

            try:
                # ReAct Agent 실행을 스트리밍으로 처리
                return await self._react_agent_streaming(agent_executor, context, query)

            except Exception as e:
                error_message = str(e)
                print(f"  - ReAct Agent 실행 오류: {error_message}")

                # 심각한 ReAct 오류만 폴백 처리, 나머지는 재시도
                critical_errors = [
                    "Invalid Format", "Missing 'Action:'", "JSON decode",
                    "파싱 오류", "Expecting value"
                ]

                if any(keyword in error_message for keyword in critical_errors):
                    print(f"  - 심각한 ReAct 오류 감지, 폴백 사용: {error_message[:100]}...")
                    return await self._fallback_report_generation_streaming(context, query)
                else:
                    # 시간 초과나 반복 제한은 부분 결과라도 사용 시도
                    print("  - 경미한 오류, 폴백 사용")
                    return await self._fallback_report_generation_streaming(context, query)

        except Exception as e:
            print(f"  - ReAct Agent 초기화 오류: {e}")
            context = str(data) if not isinstance(data, str) else data
            return await self._fallback_report_generation_streaming(context, query)

    async def _react_agent_streaming(self, agent_executor: AgentExecutor, context: str, query: str) -> str:
        """ReAct Agent를 스트리밍으로 실행"""
        print("  - ReAct Agent 스트리밍 실행 시작...")

        # ReAct Agent가 도구를 적극 활용하도록 유도하는 쿼리
        enhanced_query = f"""
다음 질문에 대해 포괄적이고 상세한 분석을 제공하세요.
필요한 경우 도구를 사용하여 추가 정보를 수집하고 분석하세요.

질문: {query}

기본 컨텍스트: {context[:500] if len(context) > 500 else context}

- 단계별로 사고하고, 필요한 도구를 사용하여 최상의 답변을 제공하세요.
- 최종 답변을 제공할 때는, 복잡한 쿼리일 수록, 수집된 정보를 바탕으로 섹션을 잘 나누어서 자세히 답변하세요.
- 간단한 쿼리는 간단하게 답하되 수집된 정보를 최대한 활용해서 최종 답변을 생성하세요.
- 이미 이전 Agent 들이 정보를 많이 수집했기 때문에 최종답변을 생성할 때까지 너무 많은 정보를 또 수집하려고 하지는 마세요. 정말 정보가 부족하거나, 실시간 정보가 부족한 경우에만 추가 정보를 수집하세요.

"""

        try:
            # ReAct Agent를 astream_events로 실행하여 스트리밍 처리
            full_response = ""
            agent_completed = False

            async for event in agent_executor.astream_events(
                {"input": enhanced_query},
                version="v1"
            ):
                kind = event["event"]

                # LLM의 실제 출력을 캐치
                if kind == "on_llm_stream" and event["name"] == "ChatGoogleGenerativeAI":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, 'content') and chunk.content:
                        full_response += chunk.content
                        print(chunk.content, end="", flush=True)

                # Agent의 최종 답변을 확인
                elif kind == "on_chain_end" and event["name"] == "AgentExecutor":
                    agent_completed = True
                    final_output = event["data"].get("output", {})
                    if isinstance(final_output, dict) and "output" in final_output:
                        agent_answer = final_output["output"]
                        if agent_answer and len(agent_answer.strip()) > len(full_response.strip()):
                            full_response = agent_answer

            # ReAct Agent가 정상 완료되었다면 결과를 신뢰
            if agent_completed:
                print(f"\n  - ReAct Agent 성공적으로 완료 (출력 길이: {len(full_response)}자)")
                return full_response
            else:
                # Agent가 완료되지 않은 경우에만 폴백 사용
                print("  - ReAct Agent가 완료되지 않음, 폴백 사용")
                return await self._fallback_report_generation_streaming(context, query)

        except Exception as e:
            print(f"  - ReAct Agent 스트리밍 오류: {e}")
            return await self._fallback_report_generation_streaming(context, query)

    async def _fallback_report_generation_streaming(self, context: str, query: str) -> str:
        """폴백 보고서 생성 - 스트리밍 버전"""
        print("  - 폴백 보고서 스트리밍 생성 시작...")

        try:
            # 안정적인 LLM으로 폴백 보고서 생성
            fallback_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.3,
                max_output_tokens=3000
            )

            fallback_prompt = f"""
당신은 전문적인 시장 분석가입니다. 다음 정보를 바탕으로 간결하고 유용한 보고서를 작성하세요.

사용자 질문: "{query}"
현재 날짜: 2025년 8월

수집된 데이터:
{context[:]}

다음 구조로 보고서를 작성하세요:

## 주요 발견사항
[핵심 내용 3-4개 항목으로 정리]

## 상세 분석
[구체적인 데이터와 분석 내용]

## 실행 가능한 제안
[2-3개의 구체적인 제안사항]

**주의사항:**
- 마크다운 형식 사용
- 구체적인 수치나 데이터가 있다면 반드시 포함
- 실용적이고 행동 가능한 내용 위주
- 2000자 이내로 간결하게 작성
"""

            full_response = ""
            async for chunk in fallback_llm.astream(fallback_prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    print(chunk.content, end="", flush=True)

            print(f"\n  - 폴백 보고서 스트리밍 완료 (길이: {len(full_response)}자)")
            return full_response

        except Exception as e:
            print(f"  - 폴백 보고서 스트리밍 실패: {e}")

            # API 할당량 초과 등의 경우 수집된 데이터만으로 간단한 요약 제공
            error_type = str(e)
            if "quota" in error_type.lower() or "429" in error_type:
                response = f"""
# API 할당량 초과로 인한 간단 요약

**질문:** {query}

**수집된 정보 요약:**
"""
                # 수집된 컨텍스트에서 핵심 정보 추출
                context_lines = context.split('\n')[:15]  # 처음 15줄만
                for line in context_lines:
                    if line.strip() and len(line.strip()) > 10:
                        response += f"- {line.strip()}\n"

                response += "\n**참고:** API 할당량 제한으로 상세한 분석 대신 수집된 데이터의 요약만 제공됩니다.\n"
                response += "더 자세한 분석을 원하시면 잠시 후 다시 시도해 주세요."
                return response
            else:
                # 다른 오류의 경우 기본 메시지
                return f"""
# 질문에 대한 답변

**질문:** {query}

**수집된 정보 요약:**
{context[:1000]}

**참고:** 시스템 오류로 인해 간단한 요약만 제공됩니다.
더 자세한 정보가 필요하시면 다시 문의해 주세요.
"""


