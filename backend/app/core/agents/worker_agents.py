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

from ..models.models import SearchResult
from ...services.search.search_tools import (
    debug_web_search,
    rdb_search,
    vector_db_search,
    graph_db_search,
    scrape_and_extract_content,
)


class DataGathererAgent:
    """데이터 수집 및 쿼리 최적화 전담 Agent"""

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

    async def _optimize_query_for_tool(self, query: str, tool: str) -> str:
        """각 도구의 특성에 맞게 자연어 쿼리를 최적화합니다."""

        # RDB와 GraphDB는 키워드 기반 검색에 더 효과적입니다.
        if tool in ["rdb_search", "graph_db_search"]:
            print(f"  - {tool} 쿼리 최적화 시작: '{query}'")
            prompt = f"""
다음 사용자 질문을 {tool} 검색에 가장 효과적인 핵심 키워드 2~3개만 쉼표(,)로 구분해서 추출해줘.
다른 설명은 절대 추가하지 말고 키워드만 반환해.

지역 정보 변환 규칙:
- "국내" → "대한민국"
- "우리나라" → "대한민국"
- "한국" → "대한민국"
- 지역 언급이 없으면 "대한민국" 자동 추가

예시:
- 질문: '국내 감귤의 최신 가격과 영양성분 알려줘'
- 키워드: '대한민국, 감귤, 가격, 영양성분'

- 질문: '우리나라 특산품인 전복의 유통 구조가 궁금해'
- 키워드: '대한민국, 전복, 특산품, 유통'

- 질문: '건강기능식품 시장 현황을 알려줘' (지역 언급 없음)
- 키워드: '대한민국, 건강기능식품, 시장현황'

질문: "{query}"
키워드:
"""
            try:
                response = await self.llm.ainvoke(prompt)
                optimized_query = response.content.strip()
                print(f"  - 최적화된 키워드: '{optimized_query}'")
                return optimized_query
            except Exception as e:
                print(f"  - 쿼리 최적화 실패, 원본 쿼리 사용: {e}")
                return query # 실패 시 원본 쿼리 반환

        # Vector DB는 구체적인 정보 검색 질문으로 변환
        elif tool == "vector_db_search":
            print(f"  - {tool} 쿼리 최적화 시작 (정보 검색 질문 변환): '{query}'")
            prompt = f"""
다음 요청을 Vector DB에서 구체적인 정보를 검색할 수 있는 **단일 질문**으로 변환해주세요.
Vector DB는 실제 문서나 데이터에서 정보를 찾는 용도이므로, 검색 가능한 구체적인 질문이 필요합니다.

변환 규칙:
1. **반드시 하나의 간단한 질문만 생성** - 번호 매기기나 복수 질문 절대 금지
2. 추상적 질문 금지: "인사이트가 무엇인가요?", "전망은 어떤가요?" 같은 추상적 표현 사용 금지
3. 구체적 정보 요청: "매출액은 얼마인가요?", "점유율은 몇 퍼센트인가요?", "주요 기업은 어디인가요?" 형태로 변환
4. 지역 정보 명확화:
   - "국내" → "대한민국"
   - "우리나라" → "대한민국"
   - "한국" → "대한민국"
5. 모호한 시간 표현을 구체적 연도로 변환:
   - "최근" → "2024년, 2025년"
   - "요즘" → "2024년, 2025년" 
   - "현재" → "2025년"
   - "작년" → "2024년"
6. 핵심 키워드(국가, 분야, 기업명 등) 반드시 유지
7. 검색 가능한 구체적 데이터를 요청하는 질문으로 변환

예시:
- 원본: "국내 건강기능식품 시장의 최근 트렌드를 분석합니다"
- 변환: "대한민국 건강기능식품 시장의 2024년 매출액과 주요 기업 점유율은 어떻게 되나요?"

- 원본: "우리나라 식자재 시장의 인사이트를 알려주세요"
- 변환: "2024년 대한민국 식자재 시장 규모와 주요 유통업체별 매출 현황은 어떻게 되나요?"

- 원본: "한국의 유망한 건강기능식품 분야를 추천합니다"
- 변환: "2024년-2025년 대한민국 건강기능식품 분야별 성장률과 시장 규모 데이터가 있나요?"

원본 요청: "{query}"
변환된 단일 질문:
"""
            try:
                response = await self.llm.ainvoke(prompt)
                optimized_query = response.content.strip()
                print(f"  - 최적화된 질문: '{optimized_query}'")
                return optimized_query
            except Exception as e:
                print(f"  - 쿼리 최적화 실패, 원본 쿼리 사용: {e}")
                return query

        # Web Search는 맥락 정보를 포함한 검색 키워드로 최적화
        elif tool == "web_search":
            print(f"  - {tool} 쿼리 최적화 시작 (맥락 강화): '{query}'")
            prompt = f"""
다음 질문을 웹 검색에 최적화된 키워드로 변환해주세요.
검색 효과를 높이기 위해 중요한 맥락 정보(국가, 연도, 대상 등)를 포함해야 합니다.

최적화 규칙:
1. 지역 정보 명확화 및 기본값 설정:
   - "국내" → "대한민국"
   - "우리나라" → "대한민국"  
   - "한국" → "대한민국"
   - 지역 언급이 없으면 → "대한민국" 자동 추가
2. 구체적인 연도 명시:
   - "최근" → "2024년 2025년"
   - "요즘" → "2024년 2025년" 
   - "현재" → "2025년"
   - "작년" → "2024년"
3. 구체적인 분야나 대상 명시
4. 검색 의도에 맞는 키워드 조합

예시:
- 원본: "국내 건강기능식품 시장 현황을 조사합니다"
- 최적화: "2024년 2025년 대한민국 건강기능식품 시장 현황 트렌드"

- 원본: "우리나라 MZ세대 소비 패턴을 분석합니다"
- 최적화: "2024년 2025년 대한민국 MZ세대 소비 트렌드 패턴"

- 원본: "건강기능식품 시장 현황을 조사합니다" (지역 언급 없음)
- 최적화: "2024년 2025년 대한민국 건강기능식품 시장 현황 트렌드"

- 원본: "한국의 유망한 건강기능식품 분야를 추천합니다"
- 최적화: "2024년 2025년 대한민국 건강기능식품 유망 분야 시장 전망"

- 원본: "최근 식자재 시장 트렌드를 조사합니다"
- 최적화: "2024년 2025년 대한민국 식자재 시장 트렌드 동향 분석"

원본 질문: "{query}"
최적화된 검색 키워드:
"""
            try:
                response = await self.llm.ainvoke(prompt)
                optimized_query = response.content.strip()
                print(f"  - 최적화된 검색 키워드: '{optimized_query}'")
                return optimized_query
            except Exception as e:
                print(f"  - 쿼리 최적화 실패, 원본 쿼리 사용: {e}")
                return query

        # 기타 도구는 원본 쿼리 그대로 사용
        return query


    async def execute(self, tool: str, inputs: Dict[str, Any]) -> List[SearchResult]:
        """단일 도구를 비동기적으로 실행하며, 실행 전 쿼리를 최적화합니다."""
        if tool not in self.tool_mapping:
            print(f"- 알 수 없는 도구: {tool}")
            return []

        original_query = inputs.get("query", "")
        # [수정됨] 실제 도구 실행 전, 쿼리 최적화 단계 추가
        optimized_query = await self._optimize_query_for_tool(original_query, tool)

        # 최적화된 쿼리로 새로운 inputs 딕셔너리 생성
        optimized_inputs = inputs.copy()
        optimized_inputs["query"] = optimized_query

        try:
            print(f"\n>> DataGatherer: '{tool}' 도구 실행 (쿼리: '{optimized_query}')")
            return await self.tool_mapping[tool](**optimized_inputs)
        except Exception as e:
            print(f"- {tool} 실행 오류: {e}")
            return []


    async def execute_parallel(self, tasks: List[Dict[str, Any]]) -> Dict[str, List[SearchResult]]:
        """여러 데이터 수집 작업을 병렬로 실행합니다."""
        print(f"\n>> DataGatherer: {len(tasks)}개 작업 병렬 실행 시작")

        # 각 작업에 대해 execute 코루틴을 생성합니다. execute 내부에서 쿼리 최적화가 자동으로 일어납니다.
        coroutines = [self.execute(task.get("tool"), task.get("inputs", {})) for task in tasks]

        # asyncio.gather를 사용하여 모든 작업을 동시에 실행하고 결과를 받습니다.
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        organized_results = {}
        for i, task in enumerate(tasks):
            tool_name = task.get("tool", f"unknown_tool_{i}")
            result = results[i]

            # 작업 실행 중 예외가 발생했는지 확인합니다.
            if isinstance(result, Exception):
                print(f"  - {tool_name} 병렬 실행 오류: {result}")
                organized_results[f"{tool_name}_{i}"] = []
            else:
                print(f"  - {tool_name} 병렬 실행 완료: {len(result)}개 결과")
                organized_results[f"{tool_name}_{i}"] = result

        return organized_results


    async def _web_search(self, query: str, **kwargs) -> List[SearchResult]:
        """웹 검색 실행"""
        try:
            # 최적화된 쿼리 사용 (이미 _optimize_query_for_tool에서 처리됨)
            print(f"- 웹 검색 실행 쿼리: {query}")

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
                                relevance_score=0.9,  # 웹검색 결과는 높은 점수
                                timestamp=datetime.now().isoformat(),
                                document_type="web",
                                metadata={"optimized_query": query, **current_result}
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
                        metadata={"optimized_query": query, **current_result}
                    )
                    search_results.append(search_result)

            print(f"- 웹 검색 완료: {len(search_results)}개 결과")
            return search_results[:5]  # 상위 5개 결과만
        except Exception as e:
            print(f"웹 검색 오류: {e}")
            return []

    async def _vector_db_search(self, query: str, hf_model = None, **kwargs) -> List[SearchResult]:
        """Vector DB 검색 실행"""
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None, vector_db_search, query, hf_model
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
        print(f"  - 변환된 GraphDB 쿼리: {query}")
        raw_results = await asyncio.to_thread(graph_db_search.invoke, {"query": query})

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



class ProcessorAgent:
    """데이터 가공 및 생성 전담 Agent (ReAct 제거, 순차 생성 지원)"""

    def __init__(self, model_pro: str = "gemini-2.5-pro", model_flash: str = "gemini-2.5-flash", temperature: float = 0.3):
        # 보고서 최종 생성을 위한 고품질 모델
        self.llm_pro = ChatGoogleGenerativeAI(model=model_pro, temperature=temperature)
        # 구조 설계, 요약 등 빠른 작업에 사용할 경량 모델
        self.llm_flash = ChatGoogleGenerativeAI(model=model_flash, temperature=0.1)

        # Orchestrator가 호출할 수 있는 작업 목록 정의
        self.processor_mapping = {
            "design_report_structure": self._design_report_structure,
            "create_chart_data": self._create_charts,
        }

    async def process(self, processor_type: str, data: Any, original_query: str) -> Any:
        """Orchestrator로부터 동기식 작업을 받아 처리합니다."""
        print(f"\n>> Processor 실행: {processor_type}")
        if processor_type not in self.processor_mapping:
            return {"error": f"알 수 없는 처리 타입: {processor_type}"}
        try:
            return await self.processor_mapping[processor_type](data, original_query)
        except Exception as e:
            print(f"  - {processor_type} 처리 오류: {e}")
            return {"error": f"{processor_type} 처리 중 오류 발생"}

    async def _design_report_structure(self, data: List[SearchResult], query: str) -> Dict[str, Any]:
        """보고서 구조 설계 및 **섹션별** 데이터 충분성 검증 (Critic 역할 포함)"""
        context_summary = "\n".join([f"- [{res.source}] {res.title}: {res.content[:150]}..." for res in data[:20]])

        limited_context_summary = context_summary[:8000]

        prompt = f"""
당신은 데이터 분석가로서, 주어진 데이터와 사용자 질문을 바탕으로 보고서의 목차를 설계하고 **각 목차별로 데이터가 충분한지 개별적으로 검증**하는 역할을 합니다.

**사용자 질문**: "{query}"
**수집된 데이터 요약**:
{limited_context_summary}

**작업 지침**:
1.  **보고서 목차 설계**: 위 데이터를 바탕으로 사용자 질문에 완벽하게 답할 수 있는 논리적인 보고서 목차를 3~5개의 섹션으로 구성하세요.

2.  **데이터 필요 유형 명시 (차트 적극 활용 전략)**:
    - **다음 키워드가 포함된 섹션은 반드시 'full_data_for_chart'로 설정**:
      * 매출, 판매량, 시장규모, 점유율, 가격, 수치, 통계, 데이터
      * 트렌드, 성장률, 변화, 증감, 비교, 현황, 추이
      * 분석, 조사, 연구, 통계, 수치적, 정량적
      * 연도별, 시기별, 기간별, 월별, 년도별
    - **일반적인 설명이나 개요 섹션만 'synthesis'로 설정**

3.  **섹션별 데이터 충분성 검증 (매우 관대한 기준 적용)**:
    - **기본적으로 모든 섹션의 `is_sufficient`는 `true`라고 가정하세요.**
    - **오직, 해당 섹션의 주제와 관련된 내용이 수집된 데이터 요약에 단 한 줄도 없거나, 명백히 주제와 무관한 내용만 있는 '치명적인 경우'에만 `is_sufficient`를 `false`로 설정하세요.**
    - 약간의 정보라도 있다면, 일단은 충분하다고 판단하고 넘어가야 합니다.

**차트 생성 우선 전략**:
- 시장/업계 분석 → 'full_data_for_chart' (차트 필수)
- 수치 비교/통계 → 'full_data_for_chart' (차트 필수)  
- 트렌드/변화 분석 → 'full_data_for_chart' (차트 필수)
- 단순 설명/개요 → 'synthesis' (텍스트만)

**출력 포맷 (반드시 JSON 형식으로만 응답):**
{{
    "title": "보고서의 최종 제목",
    "structure": [
        {{
            "section_title": "1. 시장 개요 및 현황",
            "content_type": "synthesis",
            "is_sufficient": true,
            "feedback_for_gatherer": ""
        }},
        {{
            "section_title": "2. 매출 및 성장률 분석",
            "content_type": "full_data_for_chart",
            "is_sufficient": true,
            "feedback_for_gatherer": ""
        }},
        {{
            "section_title": "3. 시장 점유율 및 트렌드",
            "content_type": "full_data_for_chart",
            "is_sufficient": true,
            "feedback_for_gatherer": ""
        }}
    ]
}}
"""
        response = await self.llm_flash.ainvoke(prompt)
        print(f"  - 보고서 구조 설계 결과: {response.content}")
        try:
            return json.loads(re.search(r'\{.*\}', response.content, re.DOTALL).group())
        except Exception as e:
            print(f"  - 보고서 구조 설계 실패, 안전 모드 구조 생성: {e}")
            return {
                "title": f"{query} - 통합 정보 요약",
                "structure": [{
                    "section_title": "수집된 정보 전체 요약",
                    "content_type": "synthesis",
                    "is_sufficient": True,
                    "feedback_for_gatherer": ""
                }]
            }

    async def _synthesize_data_for_section(self, section_title: str, all_data: List[SearchResult]) -> str:
        """[고도화됨] 특정 섹션 주제에 맞게 전체 데이터를 지능적으로 '종합'하여 하나의 완성된 글로 작성합니다."""
        context = "\n\n".join([f"--- 문서 ID {i}: [{res.source}] ---\n제목: {res.title}\n내용: {res.content}" for i, res in enumerate(all_data)])

        prompt = f"""
당신은 여러 데이터 소스를 종합하여 특정 주제에 대한 분석 보고서의 한 섹션을 저술하는 주제 전문가입니다.

**작성할 섹션의 주제**: "{section_title}"

**참고할 전체 데이터**:
{context[:8000]}

**작성 지침**:
1. **핵심 정보 추출**: '{section_title}' 주제와 직접적으로 관련된 핵심 사실, 수치, 통계 위주로 정보를 추출하세요.
2. **간결한 요약**: 정보를 단순히 나열하지 말고, 1~2 문단 이내의 간결하고 논리적인 핵심 요약문으로 재구성해주세요.
3. **중복 제거**: 여러 문서에 걸쳐 반복되는 내용은 하나로 통합하여 제거하세요.
4.  **객관성 유지**: 데이터에 기반하여 객관적인 사실만을 전달해주세요.

**결과물 (핵심 요약본)**:
"""
        response = await self.llm_flash.ainvoke(prompt)
        return response.content

    async def generate_section_streaming(self, section: Dict[str, Any], all_context_data: List[SearchResult], original_query: str) -> AsyncGenerator[str, None]:
        """하나의 섹션 내용만 생성하여 스트리밍하며, 필요시 차트 생성 마커를 포함합니다."""
        section_title = section.get("section_title", "제목 없음")
        content_type = section.get("content_type", "synthesis")
        description = section.get("description", "")

        if content_type == "synthesis":
            section_data = await self._synthesize_data_for_section(section_title, all_context_data)
            prompt_template = """
당신은 주어진 데이터를 바탕으로 전문가 수준의 보고서의 한 섹션을 작성하는 AI입니다.

**사용자의 전체 질문**: "{original_query}"
---
**현재 작성할 섹션 제목**: "{section_title}"
**섹션 목표**: "{description}"
---
**참고 데이터 (핵심 요약된 내용)**:
{section_data}
---
**작성 지침 (매우 중요)**:
1.  **간결성 유지**: **반드시 1~2 문단 이내로, 가장 핵심적인 내용만 간결하게 요약하여 작성하세요. 절대 길게 서술하지 마세요.**
2.  **제목 반복 금지**: **주어진 섹션 제목을 절대 반복해서 출력하지 마세요. 바로 본문 내용으로 시작해야 합니다.**
3.  **데이터 기반**: 참고 데이터에 있는 구체적인 수치, 사실, 인용구를 적극적으로 활용하여 내용을 구성하세요. 절대 데이터를 창작하거나 추측하지 마세요
4.  **전문가적 문체**: 명확하고 간결하며 논리적인 전문가의 톤으로 글을 작성하세요.
5.  **가독성**: 마크다운 문법(예: `*` 글머리 기호, `**` 굵은 글씨)을 적절히 사용하여 읽기 쉽게 구성하세요.

**보고서 섹션 내용**:
"""
        else:  # "full_data_for_chart"
            section_data = "\n\n".join([f"**출처: {res.source}**\n- **제목**: {res.title}\n- **내용**: {res.content}" for res in all_context_data])
            # [수정됨] 차트 생성 마커 삽입 지침이 포함된 프롬프트
            prompt_template = """
당신은 데이터 분석가이자 보고서 작성가입니다. 주어진 원본 데이터를 분석하여, 텍스트 설명과 시각적 차트를 결합한 전문가 수준의 보고서 섹션을 작성합니다.

**사용자의 전체 질문**: "{original_query}"
---
**현재 작성할 섹션 제목**: "{section_title}"
**섹션 목표**: "{description}"
---
**참고 데이터 (전체 원본 데이터)**:
{section_data}
---
**작성 지침 (매우 중요)**:
1.  **간결성 유지**: **반드시 1~2 문단 이내로, 데이터에서 가장 중요한 인사이트와 분석 내용만 간결하게 요약하여 작성하세요. 절대 길게 서술하지 마세요.**
2.  **제목 반복 금지**: **주어진 섹션 제목을 절대 반복해서 출력하지 마세요. 바로 본문 내용으로 시작해야 합니다.**
3.  **데이터 기반**: 설명에 구체적인 수치, 사실, 통계 자료를 적극적으로 인용하여 신뢰도를 높이세요.  절대 데이터를 창작하거나 추측하지 마세요.
4.  **차트 마커 삽입 (가장 중요)**: 텍스트 설명의 흐름 상, 시각적 데이터가 필요한 가장 적절한 위치에 `[GENERATE_CHART]` 마커를 한 줄에 단독으로 삽입하세요.
5.  **서술 계속**: 마커를 삽입한 후, 이어서 나머지 텍스트 설명을 자연스럽게 계속 작성할 수 있습니다.
6.  **전문가적 문체 및 가독성**: 명확하고 논리적인 문체로 작성하고, 마크다운 문법(`*`, `**`)을 활용하여 가독성을 높이세요.

**보고서 섹션 본문**:
"""

        prompt = prompt_template.format(
            original_query=original_query,
            section_title=section_title,
            description=description,
            section_data=section_data
        )

        async for chunk in self.llm_pro.astream(prompt):
            if chunk.content:
                yield chunk.content


    async def _create_charts(self, context: Any, query: str) -> Dict[str, Any]:
        """데이터를 바탕으로 차트 데이터를 생성합니다. 할루시네이션 방지 안전장치 포함."""
        print("  - 차트 데이터 생성 작업 수행...")

        try:
            # 실제 데이터에서 수치 추출 시도
            data_summary = ""
            if isinstance(context, list):
                # SearchResult 객체들에서 실제 수치 데이터 추출
                for item in context[:10]:  # 상위 10개만 사용
                    if hasattr(item, 'content'):
                        data_summary += f"- {item.content[:200]}\n"
                    elif isinstance(item, dict):
                        data_summary += f"- {item.get('content', '')[:200]}\n"
            else:
                data_summary = str(context)[:2000]

            chart_prompt = f"""
다음 **실제 수집된 데이터**를 바탕으로만 Chart.js 형식의 차트를 생성해주세요.

**중요한 제약사항**:
1. **절대 수치를 임의로 생성하지 마세요** - 반드시 아래 데이터에서 명시된 수치만 사용
2. **데이터가 불충분하면 "데이터 부족" 차트를 생성하세요**
3. **추측이나 가정으로 수치를 만들지 마세요**

**요청**: {query}
**실제 수집된 데이터**:
{data_summary}

**차트 생성 규칙**:
- 위 데이터에서 명확한 수치(매출액, %, 개수 등)가 있을 때만 해당 수치 사용
- 수치가 불분명하거나 없으면 "데이터 부족" 또는 "정보 없음"으로 표시
- 절대 임의의 숫자나 예시 값을 생성하지 말 것

다음 JSON 형식으로만 응답해주세요. 다른 설명은 절대 추가하지 마세요:
{{
    "type": "bar",
    "data": {{
        "labels": ["실제 데이터 기반 라벨"],
        "datasets": [{{
            "label": "실제 데이터셋 이름",
            "data": [실제_수치만_사용],
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
                "text": "실제 데이터 기반 제목"
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
                # 안전한 기본 차트 (실제 데이터 없음을 명시)
                return {
                    "type": "bar",
                    "data": {
                        "labels": ["데이터 수집 중"],
                        "datasets": [{
                            "label": "정보 수집 상태",
                            "data": [1],
                            "backgroundColor": "rgba(255, 193, 7, 0.6)",
                            "borderColor": "rgba(255, 193, 7, 1)",
                            "borderWidth": 1
                        }]
                    },
                    "options": {
                        "responsive": True,
                        "plugins": {
                            "title": {
                                "display": True,
                                "text": "데이터 수집 중 - 차후 업데이트 예정"
                            }
                        },
                        "scales": {
                            "y": {
                                "beginAtZero": True,
                                "max": 2
                            }
                        }
                    }
                }

        except Exception as e:
            print(f"  - 차트 생성 전체 오류: {e}")
            # 오류 상황에서도 안전한 차트 반환
            return {
                "type": "bar",
                "data": {
                    "labels": ["시스템 오류"],
                    "datasets": [{
                        "label": "처리 상태",
                        "data": [0],
                        "backgroundColor": "rgba(220, 53, 69, 0.6)",
                        "borderColor": "rgba(220, 53, 69, 1)",
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": "차트 생성 중 오류 발생"
                        }
                    },
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "max": 1
                        }
                    }
                }
            }
