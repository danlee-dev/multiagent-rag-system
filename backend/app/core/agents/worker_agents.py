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
        # 도구 매핑 설정
        self.tool_mapping = {
            "web_search": self._web_search,
            "vector_db": self._vector_db_search,
            "graph_db": self._graph_db_search,
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
        """웹 검색 실행"""
        try:
            # 기존 debug_web_search 함수 활용
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
                                relevance_score=0.8,
                                timestamp=datetime.now().isoformat(),
                                document_type="web",
                                metadata=current_result
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
                        relevance_score=0.8,
                        timestamp=datetime.now().isoformat(),
                        document_type="web",
                        metadata=current_result
                    )
                    search_results.append(search_result)

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

@tool
def create_chart(query: str, context: str) -> Dict[str, Any]:
    """주어진 컨텍스트와 자연어 요청을 바탕으로 시각화에 필요한 차트 데이터를 JSON 형식으로 생성합니다."""
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
    try:
        # LLM 응답에서 JSON만 추출
        json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
        return json.loads(json_match.group()) if json_match else {"error": "차트 JSON 생성 실패"}
    except json.JSONDecodeError:
        return {"error": "차트 데이터 파싱 실패"}


CUSTOM_REACT_PROMPT_TEMPLATE = """
당신은 수석 데이터 분석가로서, 주어진 정보를 바탕으로 최종 보고서를 작성하는 임무를 받았습니다.

[**매우 중요한 행동 규칙**]
1.  **컨텍스트 우선**: 당신의 최우선 임무는 주어진 '종합 컨텍스트'를 최대한 활용하는 것입니다.
2.  **최후의 수단**: 컨텍스트만으로 논리 전개가 불가능한 명백한 정보 공백(예: 특정 연도의 최신 통계)이 발생할 때만, 최후의 수단으로 도구를 사용하세요.
3.  **중복 금지**: 컨텍스트에 이미 있는 내용을 확인하기 위해 도구를 사용하는 것은 절대 금지됩니다.

[사용 가능한 도구]
{tools}

[응답 형식]
당신은 Thought, Action, Action Input, Observation의 순서로 생각하고 행동해야 합니다. 모든 정보가 충분해지면 최종 답변을 생성하세요.

Thought: (현재 상황 분석 및 다음 행동 계획)
Action: (사용할 도구 이름: [{tool_names}])
Action Input: (도구에 전달할 검색어)
Observation: (도구 실행 결과)
... (이 과정을 필요한 만큼 반복) ...
Thought: 이제 최종 답변을 작성할 준비가 되었습니다.
Final Answer: (최종 보고서 내용)

--- 지금부터 시작 ---

[종합 컨텍스트]
{context}

[사용자 원본 질문]
{input}

{agent_scratchpad}
"""

CUSTOM_REACT_PROMPT = PromptTemplate.from_template(
    CUSTOM_REACT_PROMPT_TEMPLATE
)


class ProcessorAgent:
    """데이터 가공 전담 Agent - 다양한 처리 작업을 수행"""

    def __init__(self, model_pro: str = "gemini-2.5-pro", model_flash: str = "gemini-2.5-flash", temperature: float = 0.2):
        self.llm_pro = ChatGoogleGenerativeAI(model=model_pro, temperature=temperature)
        self.llm_flash = ChatGoogleGenerativeAI(model=model_flash, temperature=0)

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

    async def _summarize_and_integrate(self, data: List[SearchResult], query: str) -> str:
        """(구) ContextIntegrator의 핵심 로직. 여러 검색 결과를 통합하고 요약합니다."""
        print("  - 데이터 통합 및 요약 작업 수행...")

        content_summary = "\n\n".join([f"## 출처: {res.source} (관련성: {res.relevance_score:.2f})\n{res.content}" for res in data])

        prompt = f"""
        사용자 질문: "{query}"

        아래는 여러 소스에서 수집된 정보입니다. 이 정보들을 종합하여 사용자의 질문에 답변하기 위한 하나의 일관된 컨텍스트로 통합하고 요약해주세요.

        [수집된 정보]
        {content_summary[:]}

        [작업 지침]
        1. 중복되는 정보는 합치고, 상충되는 정보가 있다면 명시하세요.
        2. 사용자 질문과 가장 관련 높은 내용을 중심으로 논리적으로 재구성하세요.
        3. 최종적으로 생성될 보고서의 '핵심 재료'가 될 수 있도록, 서론-본론-결론 구조를 갖춘 자연스러운 문단으로 작성해주세요.

        통합된 컨텍스트:
        """
        response = await self.llm_pro.ainvoke(prompt)
        return response.content

    async def _evaluate_criticism(self, data: str, query: str) -> CriticResult:
        """(구) CriticAgent의 핵심 로직. 결과물을 비평합니다."""
        print("  - 결과물 비평 작업 수행...")

        prompt = f"""
        사용자 원본 질문: "{query}"

        아래는 이 질문에 답변하기 위해 1차적으로 수집 및 통합된 정보입니다.
        [통합된 정보]
        {data[:4000]}

        이 정보가 사용자의 질문에 답변하는 최종 보고서를 작성하기에 충분한지, 품질이 좋은지 평가해주세요.

        [평가 기준]
        - 정보의 충분성: 질문의 모든 측면에 답변할 수 있는가?
        - 정보의 관련성: 수집된 정보가 질문의 핵심과 직접적으로 관련이 있는가?
        - 논리적 결함: 정보에 명백한 논리적 오류나 모순이 없는가?

        아래 JSON 형식으로만 응답해주세요:
        {{
            "status": "pass" 또는 "fail_with_feedback",
            "feedback": "만약 'fail'이라면, 재계획을 위해 Orchestrator에게 전달할 구체적인 피드백 (예: '경쟁사 분석에 대한 데이터가 부족하므로 추가 검색이 필요합니다.')",
            "confidence": 0.0 에서 1.0 사이의 평가 신뢰도
        }}
        """
        response = await self.llm_flash.ainvoke(prompt)
        try:
            result_data = json.loads(response.content)
            return CriticResult(**result_data)
        except Exception:
            return CriticResult(status="fail_with_feedback", feedback="평가 결과 파싱 오류", confidence=0.5)

    async def _create_charts(self, data: Any, query: str) -> Dict[str, Any]:
        """데이터를 바탕으로 차트 데이터를 생성합니다."""
        print("  - 차트 데이터 생성 작업 수행...")

        # 데이터를 문자열로 변환
        if isinstance(data, list):
            context = "\n".join([str(item) for item in data])
        else:
            context = str(data)

        # create_chart 도구 사용
        chart_result = create_chart(query, context)
        return chart_result

    async def _generate_report(self, data: Any, query: str) -> str:
        """ReAct Agent를 사용하여 최종 보고서를 생성합니다."""
        print("  - ReAct Agent를 사용한 최종 보고서 생성 시작...")

        # 데이터를 문자열로 변환
        if isinstance(data, list):
            context = "\n".join([str(item) for item in data])
        else:
            context = str(data)

        # ReAct Agent 도구들 정의
        tools = [debug_web_search, vector_db_search, graph_db_search, rdb_search, scrape_and_extract_content]

        # ReAct Agent 생성
        react_agent = create_react_agent(
            llm=self.llm_pro,
            tools=tools,
            prompt=CUSTOM_REACT_PROMPT
        )

        # Agent Executor 생성
        agent_executor = AgentExecutor(
            agent=react_agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            early_stopping_method="generate"
        )

        try:
            # ReAct Agent 실행
            result = await agent_executor.ainvoke({
                "input": query,
                "context": context
            })

            # 최종 답변 반환
            final_answer = result.get("output", "보고서 생성에 실패했습니다.")
            print("  - ReAct Agent 보고서 생성 완료")
            return final_answer

        except Exception as e:
            print(f"  - ReAct Agent 실행 오류: {e}")
            # 폴백: 기본 LLM으로 보고서 생성
            fallback_prompt = f"""
            사용자 원본 질문: "{query}"

            아래는 수집되고 통합된 컨텍스트입니다:
            {context}

            위 정보를 바탕으로 사용자의 질문에 대한 포괄적이고 구체적인 보고서를 작성해주세요.

            보고서 구성:
            1. 요약 (Executive Summary)
            2. 시장 현황 분석
            3. 소비자 분석
            4. 추천 사항
            5. 전략 제안
            6. 결론

            각 섹션은 명확한 근거와 함께 구체적인 내용을 포함해야 합니다.
            마크다운 형식으로 작성해주세요.
            """

            response = await self.llm_pro.ainvoke(fallback_prompt)
            return response.content
