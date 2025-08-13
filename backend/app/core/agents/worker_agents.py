from langchain_core.tools import tool
import re
import sys
import asyncio
import json
import concurrent.futures
import os
from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from ..models.models import SearchResult
from ...services.search.search_tools import (
    debug_web_search,
    rdb_search,
    vector_db_search,
    graph_db_search,
    scrape_and_extract_content,
)

# 전역 ThreadPoolExecutor 생성 (재사용으로 성능 향상)
_global_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="search_worker")

# 추가: 페르소나 프롬프트 로드
PERSONA_PROMPTS = {}
try:
    with open("agents/prompts/persona_prompts.json", "r", encoding="utf-8") as f:
        PERSONA_PROMPTS = json.load(f)
    print("ProcessorAgent: 페르소나 프롬프트 로드 성공.")
except Exception as e:
    print(f"ProcessorAgent: 페르소나 프롬프트 로드 실패 - {e}")


class DataGathererAgent:
    """데이터 수집 및 쿼리 최적화 전담 Agent"""

    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0):
        # Gemini 모델 (기본)
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)

        # OpenAI fallback 모델들
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.llm_openai_mini = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=temperature,
                api_key=self.openai_api_key
            )
            self.llm_openai_4o = ChatOpenAI(
                model="gpt-4o",
                temperature=temperature,
                api_key=self.openai_api_key
            )
            print("OpenAI fallback 모델 초기화 완료")
        else:
            self.llm_openai_mini = None
            self.llm_openai_4o = None
            print("경고: OPENAI_API_KEY가 설정되지 않음. Gemini 오류 시 fallback 불가")

        # 도구 매핑 설정 - 이름 통일
        self.tool_mapping = {
            "web_search": self._web_search,
            "vector_db_search": self._vector_db_search,
            "graph_db_search": self._graph_db_search,
            "rdb_search": self._rdb_search,
            "scrape_content": self._scrape_content,
        }

    async def _invoke_with_fallback(self, prompt: str, use_4o: bool = False) -> str:
        """Gemini API 실패 시 OpenAI로 fallback하는 메서드"""
        try:
            # 1차 시도: Gemini
            print("  - Gemini API 시도 중...")
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            error_msg = str(e)
            print(f"  - Gemini API 실패: {error_msg}")

            # Rate limit 또는 quota 오류 체크
            if any(keyword in error_msg.lower() for keyword in ['429', 'quota', 'rate limit', 'exceeded']):
                print("  - Rate limit 감지, OpenAI로 fallback 시도...")

                if self.llm_openai_mini is None:
                    print("  - OpenAI API 키가 없어 fallback 불가")
                    raise e

                try:
                    # 2차 시도: OpenAI
                    fallback_model = self.llm_openai_4o if use_4o else self.llm_openai_mini
                    model_name = "gpt-4o" if use_4o else "gpt-4o-mini"
                    print(f"  - {model_name} API 시도 중...")
                    response = await fallback_model.ainvoke(prompt)
                    print(f"  - {model_name} API 성공!")
                    return response.content.strip()
                except Exception as openai_error:
                    print(f"  - OpenAI API도 실패: {openai_error}")
                    raise openai_error
            else:
                # Rate limit이 아닌 다른 오류는 그대로 발생
                raise e

    async def _optimize_query_for_tool(self, query: str, tool: str) -> str:
        """각 도구의 특성에 맞게 자연어 쿼리를 최적화합니다."""

        # RDB와 GraphDB는 키워드 기반 검색에 더 효과적입니다.
        if tool == "rdb_search":
            print(f"  - {tool} 쿼리 최적화 시작: '{query}'")
            prompt = f"""
        너는 PostgreSQL RDB에 질의할 '요약 검색문'을 만드는 어시스턴트다.
        사용자 질문을 다음 규칙으로 1줄짜리 한국어 문장으로 재작성해라. (추가 설명/따옴표/코드블록 금지)

        [규칙]
        1) 포함 항목: 지역, 품목(들), 의도(가격|시세|영양|칼로리|비타민|무역|수출|수입|시장현황 등), 기간.
        2) 지역 정규화:
        - "국내", "우리나라", "한국" → "대한민국".
        - 지역 미언급 시 "대한민국"을 기본으로 넣는다.
        3) 기간 정규화:
        - 오늘/현재 → today
        - 이번주 → this_week
        - 이번달/당월/최근 1달 → this_month 또는 recent(모호하면 recent)
        - 특정 연도/날짜가 있으면 숫자를 그대로 포함 (예: 2023년 → 2023)
        4) 품목은 질문에 나온 원문 그대로 적되 불필요한 어미/조사는 제거(예: "국내산 사과" → "사과").
        5) 의도는 질문에서 요구한 것을 가능한 한 구체적으로 나열(예: 가격·시세, 영양, 칼로리, 비타민C, 수출액, 성장률 등).
        6) 출력 형식(딱 한 줄):
        "지역=..., 품목=..., 의도=..., 기간=..."

        [예시]
        - 질문: 국내 감귤의 최신 가격과 영양성분 알려줘
        - 출력: 지역=대한민국, 품목=감귤, 의도=가격·시세·영양, 기간=today

        - 질문: 우리나라 특산품인 전복의 유통 구조가 궁금해
        - 출력: 지역=대한민국, 품목=전복, 의도=유통·시장현황, 기간=recent

        - 질문: 건강기능식품 시장 현황을 알려줘
        - 출력: 지역=대한민국, 품목=건강기능식품, 의도=시장현황, 기간=recent

        - 질문: 2023년 만두의 주요 수출국별 수출액과 성장률
        - 출력: 지역=대한민국, 품목=만두, 의도=수출액·성장률, 기간=2023

        사용자 질문: "{query}"
        출력:
        """
            try:
                response_content = await self._invoke_with_fallback(prompt)
                optimized_query = response_content.strip()
                print(f"  - 최적화된 키워드: '{optimized_query}'")
                return optimized_query
            except Exception as e:
                print(f"  - 쿼리 최적화 실패, 원본 쿼리 사용: {e}")
                return query

        elif tool == "graph_db_search":
            print(f"  - {tool} 쿼리 최적화 시작: '{query}'")

            # 그래프 스키마 기준: (품목)-[:isFrom]->(Origin), (품목)-[:hasNutrient]->(Nutrient)
            # 우리가 원하는 정규 문구:
            #   - "<품목>의 원산지"            -> isFrom
            #   - "<품목>의 영양소"            -> hasNutrient
            #   - "<지역>의 <품목> 원산지"     -> isFrom + region filter
            #   - "(활어|선어|냉동|건어) <수산물> 원산지" -> isFrom + fishState filter
            prompt = f"""
        다음 사용자 질문을, Neo4j 그래프 검색에 바로 넣을 수 있는 **정규 질의 문구**로 변환하세요.
        그래프 스키마:
        - 품목 노드 라벨: 농산물 | 수산물 | 축산물  (공통 속성: product)
        - 원산지 노드 라벨: Origin (속성: city, region)
        - 관계: (품목)-[:isFrom]->(Origin), (품목)-[:hasNutrient]->(Nutrient)
        - 수산물 상태: fishState ∈ {{활어, 선어, 냉동, 건어}}

        규칙(반드시 준수):
        1) 결과는 **한 줄당 하나의 질의**로 출력하고, 아래 4가지 패턴만 사용하세요.
        - "<품목>의 원산지"
        - "<품목>의 영양소"
        - "<지역>의 <품목> 원산지"
        - "<상태> <수산물> 원산지"   (상태=활어|선어|냉동|건어 중 하나)
        2) 질문에 해당되지 않는 패턴은 만들지 마세요. 추측 금지.
        3) 불필요한 접두사/설명/따옴표/번호/Bullet 금지. 텍스트만.
        4) 품목, 지역, 상태는 사용자 질문에서 **그대로 발췌**하세요(동의어 치환 금지).
        5) 결과가 없으면 **빈 문자열**만 반환.

        예시:
        - 질문: "사과 어디서 나와?"  ->  사과의 원산지
        - 질문: "오렌지 영양 성분 알려줘" -> 오렌지의 영양소
        - 질문: "경상북도 사과 산지" -> 경상북도의 사과 원산지
        - 질문: "활어 문어 산지" -> 활어 문어 원산지

        질문: "{query}"
        출력:
        """.strip()

            try:
                response_content = await self._invoke_with_fallback(prompt)
                # 파싱: 줄 단위로 정리, 허용 패턴만 통과
                allowed_prefixes = ("활어 ", "선어 ", "냉동 ", "건어 ")
                def _is_allowed(line: str) -> bool:
                    if not line: return False
                    # 패턴 1/2: "<품목>의 (원산지|영양소)"
                    if "의 원산지" in line or "의 영양소" in line:
                        return True
                    # 패턴 3: "<지역>의 <품목> 원산지"
                    if line.endswith(" 원산지") and "의 " in line and not any(line.startswith(p) for p in allowed_prefixes):
                        return True
                    # 패턴 4: "<상태> <수산물> 원산지"
                    if line.endswith(" 원산지") and any(line.startswith(p) for p in allowed_prefixes):
                        return True
                    return False

                lines = [l.strip().lstrip("-•").strip().strip('"').strip("'") for l in response_content.splitlines()]
                lines = [l for l in lines if _is_allowed(l)]
                optimized_query = "\n".join(dict.fromkeys(lines))  # 중복 제거, 순서 유지

                print(f"  - 최적화된 키워드(정규 질의):\n{optimized_query or '(empty)'}")
                return optimized_query if optimized_query else query  # 비면 원본 질문 전달

            except Exception as e:
                print(f"  - 쿼리 최적화 실패, 원본 쿼리 사용: {e}")
                return query

        # Vector DB는 구체적인 정보 검색 질문으로 변환
        elif tool == "vector_db_search":
            print(f"  - {tool} 쿼리 최적화 시작 (정보 검색 질문 변환): '{query}'")
            prompt = f"""
다음 요청을 Vector DB에서 구체적인 정보를 검색할 수 있는 **단일 질문**으로 변환해주세요.

**중요 규칙 (반드시 준수)**:
1. **오직 하나의 간단한 질문만 생성** - 절대 여러 질문 생성 금지
2. **번호 매기기 절대 금지** (1., 2., 3. 등 사용 금지)
3. **목록이나 리스트 형태 금지**
4. **"원본 요청:", "변환된 질문:" 등 부가 설명 금지**
5. **질문 하나만 간결하게 출력**

변환 규칙:
- 추상적 질문 금지: "인사이트", "전망", "트렌드" 등 → 구체적 수치 요청으로 변환
- 지역 정보 명확화: "국내"→"대한민국", "우리나라"→"대한민국", "한국"→"대한민국"
- 시간 표현 구체화: "최근"→"2024년, 2025년", "요즘"→"2024년, 2025년", "현재"→"2025년"
- 구체적 정보 요청: "매출액은 얼마인가요?", "점유율은 몇 퍼센트인가요?" 형태

변환 예시:
입력: "국내 건강기능식품 시장의 최근 트렌드를 분석합니다"
출력: 대한민국 건강기능식품 시장의 2024년 매출액과 주요 기업 점유율은 어떻게 되나요?

입력: "우리나라 식자재 시장의 인사이트를 알려주세요"
출력: 2024년 대한민국 식자재 시장 규모와 주요 유통업체별 매출 현황은 어떻게 되나요?

**원본 요청**: "{query}"

**변환된 단일 질문** (질문 하나만 출력):
"""
            try:
                raw_query = await self._invoke_with_fallback(prompt)

                # [추가됨] 강력한 후처리로 단일 질문만 추출
                optimized_query = self._extract_single_question(raw_query)

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
                optimized_query = await self._invoke_with_fallback(prompt)
                print(f"  - 최적화된 검색 키워드: '{optimized_query}'")
                return optimized_query
            except Exception as e:
                print(f"  - 쿼리 최적화 실패, 원본 쿼리 사용: {e}")
                return query

        # 기타 도구는 원본 쿼리 그대로 사용
        return query

    def _extract_single_question(self, raw_response: str) -> str:
        """LLM 응답에서 단일 질문만 추출하는 헬퍼 메서드"""
        lines = raw_response.strip().split('\n')

        # 불필요한 텍스트 제거
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 번호 매기기나 부가 설명 제거
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '-', '*')):
                continue
            if '원본' in line or '변환' in line or '질문:' in line or '출력:' in line:
                continue
            if '**' in line:  # 마크다운 헤더 제거
                continue
            cleaned_lines.append(line)

        # 첫 번째 질문 문장만 반환
        for line in cleaned_lines:
            if line.endswith('?') or line.endswith('요') or line.endswith('까'):
                return line

        # 적절한 질문이 없으면 첫 번째 유효한 라인 반환
        if cleaned_lines:
            return cleaned_lines[0]

        # 모든 처리가 실패하면 원본 응답 반환
        return raw_response.strip()


    async def execute(self, tool: str, inputs: Dict[str, Any]) -> Tuple[List[SearchResult], str]:
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
            result = await self.tool_mapping[tool](**optimized_inputs)
            return result, optimized_query
        except Exception as e:
            print(f"- {tool} 실행 오류: {e}")
            return [], optimized_query


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
                print(f"  - {tool_name} 병렬 실행 결과: {result[:3]}...")  # 처음 3개 결과만 출력
                search_results, optimized_query = result  # 튜플 언패킹
                organized_results[f"{tool_name}_{i}"] = search_results

        return organized_results

    async def execute_parallel_streaming(self, tasks: List[Dict[str, Any]], state: Dict[str, Any] = None):
        """여러 데이터 수집 작업을 병렬로 실행하되, 각 작업이 완료될 때마다 실시간으로 yield합니다."""
        print(f"\n>> DataGatherer: {len(tasks)}개 작업 스트리밍 병렬 실행 시작")

        # 디버깅
        import pprint
        print("\n-- Tasks to be executed --")
        pprint.pprint(tasks, width=100, depth=2)
        print("\n-- Tasks to be executed --")

        # 각 태스크에 인덱스를 할당하여 순서를 추적
        async def execute_with_callback(task_index: int, task: Dict[str, Any]):
            tool_name = task.get("tool", f"unknown_tool_{task_index}")
            inputs = task.get("inputs", {})
            query = inputs.get("query", "")

            try:
                print(f"  - {tool_name} 시작: {query}")
                result, optimized_query = await self.execute(tool_name, inputs)
                print(f"  - {tool_name} 완료: {len(result)}개 결과")

                # 프론트엔드가 기대하는 형식으로 변환
                formatted_results = []
                for search_result in result:
                    result_dict = search_result.model_dump()
                    formatted_result = {
                        "title": result_dict.get("title", "제목 없음"),
                        "content": result_dict.get("content", "content 없음"),
                        "url": result_dict.get("url", "url 없음"),
                        "source": result_dict.get("source", tool_name),
                        "score": result_dict.get("score", 0.0),
                    }
                    formatted_results.append(formatted_result)

                    print(f"  - {tool_name} 결과 포맷 완료: {formatted_result}")

                return {
                    "step": task_index + 1,
                    "tool_name": tool_name,
                    "query": optimized_query,
                    "results": formatted_results,
                    "original_results": result  # 원본 SearchResult 객체들도 보존
                }

            except Exception as e:
                print(f"  - {tool_name} 실행 오류: {e}")
                return {
                    "step": task_index + 1,
                    "tool_name": tool_name,
                    "query": optimized_query,
                    "results": [],
                    "error": str(e),
                    "original_results": []
                }

        # 모든 작업을 비동기로 시작하고, 완료되는 대로 yield
        tasks_coroutines = [execute_with_callback(i, task) for i, task in enumerate(tasks)]

        # asyncio.as_completed를 사용하여 완료되는 순서대로 결과 처리
        collected_data = []

        for coro in asyncio.as_completed(tasks_coroutines):
            result = await coro
            collected_data.extend(result.get("original_results", []))

            # 개별 검색 결과를 즉시 yield
            yield {
                "type": "search_results",
                "data": {
                    "step": result["step"],
                    "tool_name": result["tool_name"],
                    "query": result["query"],
                    "results": result["results"],
                    "message_id": state.get("message_id") if state else None
                }
            }

        # 모든 검색이 완료된 후 전체 수집된 데이터를 마지막에 yield
        yield {
            "type": "collection_complete",
            "data": {
                "total_results": len(collected_data),
                "collected_data": collected_data
            }
        }


    async def _web_search(self, query: str, **kwargs) -> List[SearchResult]:
        """웹 검색 실행 - 안정성 강화"""
        try:
            # 최적화된 쿼리 사용 (이미 _optimize_query_for_tool에서 처리됨)
            print(f"  - 웹 검색 실행 쿼리: {query}")

            # 전역 ThreadPoolExecutor 사용하여 병렬 처리
            loop = asyncio.get_event_loop()
            result_text = await loop.run_in_executor(
                _global_executor,  # 전역 executor 사용
                debug_web_search,
                query
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
                                score=0.9,  # 웹검색 결과는 높은 점수
                                timestamp=datetime.now().isoformat(),
                                document_type="web",
                                metadata={"optimized_query": query, **current_result},
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
                        score=0.9,  # 웹검색 결과는 높은 점수
                        timestamp=datetime.now().isoformat(),
                        document_type="web",
                        metadata={
                            "optimized_query": query,
                            "link": current_result.get("link"),  # 출처 링크 포함
                            **current_result
                        },
                        source_url=current_result.get("link", "웹 검색 결과")
                    )
                    search_results.append(search_result)

            print(f"  - 웹 검색 완료: {len(search_results)}개 결과")
            return search_results[:5]  # 상위 5개 결과만

        except concurrent.futures.TimeoutError:
            print(f"웹 검색 타임아웃: {query}")
            return []
        except Exception as e:
            print(f"웹 검색 오류: {e}")
            return []

    async def _vector_db_search(self, query: str, **kwargs) -> List[SearchResult]:
        """Vector DB 검색 실행 - 오류 처리 강화"""
        try:
            print(f">> Vector DB 검색 시작: {query}")
            
            # 전역 ThreadPoolExecutor 사용하여 병렬 처리
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                _global_executor,  # 전역 executor 사용
                vector_db_search,
                query
            )

            search_results = []
            for result in results:
                if isinstance(result, dict):
                    # 새로운 doc_link와 page_number 필드 사용
                    doc_link = result.get("source_url", "")
                    page_number = result.get("page_number", [])
                    # 문서 제목 추출
                    doc_title = result.get("title", "")

                    # 제목에 페이지 번호 추가
                    full_title = f"{doc_title}, ({', '.join(map(str, page_number))})".strip()
                    score = result.get("score", 5.2)

                    search_results.append(SearchResult(
                        source="_search",
                        content=result.get("content", ""),
                        search_query=query,
                        title=full_title,
                        document_type="database",
                        score=score,
                        url=doc_link,  # 새 필드 추가
                    ))


            print(f"  - Vector DB 검색 완료: {len(search_results)}개 결과")
            return search_results[:5]

        # except concurrent.futures.TimeoutError:
        #     print(f"Vector DB 검색 타임아웃: {query}")
        #     return []
        except Exception as e:
            print(f"Vector DB 검색 오류: {e}")

    async def _graph_db_search(self, query: str, **kwargs) -> List[SearchResult]:
        """Graph DB 검색 실행 - 포맷 불일치 방지 및 타임아웃 처리"""
        print(f"  - Graph DB 검색 시작: {query}")
        import concurrent.futures

        try:
            # graph_db_search가 동기 함수라면
            loop = asyncio.get_running_loop()
            raw_results = await loop.run_in_executor(None, graph_db_search, query)
            # 만약 langchain Tool일 경우:
            # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            #     raw_results = executor.submit(graph_db_search.invoke, {"query": query}).result(timeout=20)

            search_results: List[SearchResult] = []

            if isinstance(raw_results, list):
                # list of dict or list of str
                for res in raw_results[:5]:
                    if isinstance(res, dict):
                        # Graph DB 결과의 의미있는 제목 생성
                        title_candidates = [
                            res.get("title"),
                            res.get("entity"),
                            res.get("product"),
                            res.get("name"),
                            res.get("품목명"),
                            res.get("원산지"),
                            res.get("영양소명")
                        ]
                        title = next((t for t in title_candidates if t), "그래프 정보")

                        # 관계형 데이터의 의미있는 내용 생성
                        content_parts = []
                        relationship_info = []

                        # 관계 정보 추출
                        for key, value in res.items():
                            if "관계" in key or "연결" in key or key.endswith("_관계"):
                                relationship_info.append(f"🔗 {key}: {value}")
                            elif key not in ['title', 'content', 'entity', 'product'] and value:
                                content_parts.append(f"• {key}: {value}")

                        if relationship_info:
                            content_parts = relationship_info + content_parts

                        content = res.get("content") or "\n".join(content_parts[:8]) or json.dumps(res, ensure_ascii=False, indent=2)
                        score = res.get("confidence") or res.get("score") or 0.8
                    else:
                        title = f"그래프 관계 정보"
                        content = str(res)
                        score = 0.8

                    search_results.append(
                        SearchResult(
                            source="graph_db",
                            content=content,
                            search_query=query,
                            title=title,
                            score=score,
                            document_type="graph",
                            url=""  # 빈 문자열로 통일 (None 쓰면 후단에서 깨지는 경우 있음)
                        )
                    )
            elif isinstance(raw_results, dict):
                # 단일 그래프 결과의 의미있는 제목 생성
                title_candidates = [
                    raw_results.get("title"),
                    raw_results.get("entity"),
                    raw_results.get("product"),
                    raw_results.get("name"),
                    raw_results.get("품목명")
                ]
                title = next((t for t in title_candidates if t), "그래프 관계 정보")

                # 구조화된 내용 생성
                content_parts = []
                for key, value in raw_results.items():
                    if key not in ['title', 'content', 'entity', 'product'] and value:
                        if "관계" in key or "연결" in key:
                            content_parts.append(f"🔗 {key}: {value}")
                        else:
                            content_parts.append(f"• {key}: {value}")

                content = raw_results.get("content") or "\n".join(content_parts[:8]) or json.dumps(raw_results, ensure_ascii=False, indent=2)
                score = raw_results.get("confidence") or raw_results.get("score") or 0.8
                search_results.append(
                    SearchResult(
                        source="graph_db",
                        content=content,
                        search_query=query,
                        title=title,
                        score=score,
                        document_type="graph",
                        url=""
                    )
                )
            else:
                # 문자열 summary 같은 경우
                content = str(raw_results)
                title = "그래프 검색 요약"
                search_results.append(
                    SearchResult(
                        source="graph_db",
                        content=content,
                        search_query=query,
                        title=title,
                        score=0.6,
                        document_type="graph",
                        url=""
                    )
                )

            print(f"  - Graph DB 검색 완료: {len(search_results)}개 결과")
            return search_results

        except concurrent.futures.TimeoutError:
            print(f"Graph DB 검색 타임아웃: {query}")
            return []
        except Exception as e:
            print(f"Graph DB 검색 오류: {e}")
            return []

    async def _rdb_search(self, query: str, **kwargs) -> List[SearchResult]:
        """RDB 검색 실행 - 반환 표준화"""
        try:
            loop = asyncio.get_running_loop()
            result_text = await loop.run_in_executor(None, rdb_search, query)

            # rdb_search는 문자열을 반환하므로, 이를 단일 SearchResult로 변환
            if isinstance(result_text, str) and result_text.strip():
                # "PostgreSQL 검색 결과: "로 시작하는지 체크
                if "PostgreSQL 검색 결과:" in result_text and "찾을 수 없습니다" in result_text:
                    # 검색 결과가 없는 경우
                    print(f"  - RDB에서 '{query}' 관련 데이터 없음")
                    return []

                search_result = SearchResult(
                    source="rdb_search",
                    content=result_text,
                    search_query=query,
                    title="PostgreSQL 데이터베이스 검색 결과",
                    url="",
                    score=0.9,
                    document_type="database",
                    metadata={"raw_result": result_text}
                )
                print(f"  - rdb_search 래퍼 반환: 1개 (텍스트 결과)")
                return [search_result]
            else:
                print(f"  - RDB 검색 결과가 비어있음")
                return []


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
        # 보고서 최종 생성을 위한 고품질 모델 (Gemini)
        self.llm_pro = ChatGoogleGenerativeAI(model=model_pro, temperature=temperature)
        # 구조 설계, 요약 등 빠른 작업에 사용할 경량 모델 (Gemini)
        self.llm_flash = ChatGoogleGenerativeAI(model=model_flash, temperature=0.1)

        # OpenAI fallback 모델들
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.llm_openai_mini = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                api_key=self.openai_api_key
            )
            self.llm_openai_4o = ChatOpenAI(
                model="gpt-4o",
                temperature=temperature,
                api_key=self.openai_api_key
            )
            print("ProcessorAgent: OpenAI fallback 모델 초기화 완료")
        else:
            self.llm_openai_mini = None
            self.llm_openai_4o = None
            print("ProcessorAgent: 경고: OPENAI_API_KEY가 설정되지 않음")

        self.personas = PERSONA_PROMPTS

        # Orchestrator가 호출할 수 있는 작업 목록 정의
        self.processor_mapping = {
            "design_report_structure": self._design_report_structure,
            "create_chart_data": self._create_charts,
        }

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
                print(f"ProcessorAgent: Gemini API rate limit 감지, OpenAI로 fallback 시도: {e}")
                if fallback_model:
                    try:
                        result = await fallback_model.ainvoke(prompt)
                        print("ProcessorAgent: OpenAI fallback 성공")
                        return result
                    except Exception as fallback_error:
                        print(f"ProcessorAgent: OpenAI fallback도 실패: {fallback_error}")
                        raise fallback_error
                else:
                    print("ProcessorAgent: OpenAI 모델이 초기화되지 않음")
                    raise e
            else:
                raise e

    async def _astream_with_fallback(self, prompt, primary_model, fallback_model):
        """
        스트리밍을 위한 Gemini API rate limit 시 OpenAI로 fallback 처리
        """
        primary_chunks_received = 0
        primary_content_length = 0

        try:
            print(f"- Primary 모델로 스트리밍 시도 ({type(primary_model).__name__})")
            async for chunk in primary_model.astream(prompt):
                primary_chunks_received += 1
                if hasattr(chunk, 'content') and chunk.content:
                    primary_content_length += len(chunk.content)
                yield chunk

            print(f"- Primary 스트리밍 완료: {primary_chunks_received}개 청크, {primary_content_length} 문자")

            # 청크를 받았지만 내용이 비어있는 경우도 실패로 간주
            if primary_chunks_received == 0 or primary_content_length == 0:
                print(f"- Primary 모델에서 유효한 내용이 생성되지 않음, fallback 실행")
                raise Exception("No valid content generated")

        except Exception as e:
            error_str = str(e).lower()
            rate_limit_indicators = ['429', 'quota', 'rate limit', 'exceeded', 'resource_exhausted', 'no valid content', 'no generation chunks']

            if any(indicator in error_str for indicator in rate_limit_indicators) or primary_chunks_received == 0:
                print(f"ProcessorAgent: Gemini API 문제 감지 (청크:{primary_chunks_received}, 내용:{primary_content_length}), OpenAI로 fallback: {e}")
                if fallback_model:
                    try:
                        print("ProcessorAgent: OpenAI fallback으로 스트리밍 시작")
                        fallback_chunks_received = 0
                        async for chunk in fallback_model.astream(prompt):
                            fallback_chunks_received += 1
                            yield chunk
                        print(f"ProcessorAgent: OpenAI fallback 완료: {fallback_chunks_received}개 청크")
                    except Exception as fallback_error:
                        print(f"ProcessorAgent: OpenAI fallback도 실패: {fallback_error}")
                        raise fallback_error
                else:
                    print("ProcessorAgent: OpenAI 모델이 초기화되지 않음")
                    raise e
            else:
                print(f"ProcessorAgent: 복구 불가능한 오류: {e}")
                raise e

    async def process(self, processor_type: str, data: Any, param2: Any, param3: str, param4: str = "", yield_callback=None, state: Optional[Dict[str, Any]] = None):
        """Orchestrator로부터 동기식 작업을 받아 처리합니다."""
        print(f"\n>> Processor 실행: {processor_type}")

        if processor_type == "design_report_structure":
            selected_indexes = param2
            original_query = param3
            result = await self._design_report_structure(data, selected_indexes, original_query, state)
            yield {"type": "result", "data": result}

        elif processor_type == "create_chart_data":
            section_title = param2
            generated_content = param3
            async for result in self._create_charts(data, section_title, generated_content, yield_callback, state):
                yield result
        else:
            yield {"type": "error", "data": {"error": f"알 수 없는 처리 타입: {processor_type}"}}

    async def _design_report_structure(self, data: List[SearchResult], selected_indexes: List[int], query: str, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """보고서 구조 설계 + 섹션별 사용할 데이터 인덱스 선택"""

        print(f"\n>> 보고서 구조 설계 시작:")
        print(f"   전체 데이터: {len(data)}개")
        print(f"   선택된 인덱스: {selected_indexes} ({len(selected_indexes)}개)")

        # 페르소나 정보 추출
        persona_name = state.get("persona", "기본") if state else "기본"
        persona_description = self.personas.get(persona_name, {}).get("description", "일반적인 분석가")
        print(f"  - 보고서 구조 설계에 '{persona_name}' 페르소나 관점 적용")

        # 선택된 인덱스의 데이터를 인덱스와 함께 매핑하여 컨텍스트 생성
        indexed_context = ""
        for idx in selected_indexes:
            if 0 <= idx < len(data):
                res = data[idx]
                source = getattr(res, 'source', 'Unknown')
                title = getattr(res, 'title', 'No Title')
                content = getattr(res, 'content', '')  # 전체 내용 (요약 없이)

                indexed_context += f"""
    --- 데이터 인덱스 [{idx}] ---
    출처: {source}
    제목: {title}
    내용: {content}

    """

        # 컨텍스트 길이 제한 (너무 길면 잘라내기)
        limited_indexed_context = indexed_context[:20000]  # 더 많은 정보 포함

        print(f"   생성된 컨텍스트 길이: {len(indexed_context)} 문자")
        print(f"   제한된 컨텍스트 길이: {len(limited_indexed_context)} 문자")

        prompt = f"""
    당신은 '{persona_name}'({persona_description})의 관점을 가진 데이터 분석가이자 AI 에이전트 워크플로우 설계자입니다.
    주어진 **선별된 데이터**와 사용자 질문을 분석하여, **'{persona_name}'가 가장 중요하게 생각할 만한 주제**들로 **내용이 절대 중복되지 않는** 논리적인 보고서 목차를 설계하고 **각 섹션별로 사용할 데이터 인덱스**를 명확히 분배해주세요.

    **가장 중요한 목표**: 각 보고서 섹션이 '{persona_name}'의 관점에서 고유한 주제를 다루게 하여, 내용 반복을 원천적으로 방지하는 것입니다.

    **사용자 질문**: "{query}"

    **선별된 데이터 (인덱스와 전체 내용 포함)**:
    {limited_indexed_context}

    **작업 지침**:
    1. **보고서 목차 설계**
    - 주어진 데이터를 분석하여, 사용자 질문에 답할 수 있는 3~5개의 **고유하고 구체적인 섹션**으로 목차를 구성하세요.
    - **핵심 규칙**: 각 섹션의 제목은 **서로 다른 분석 관점이나 주제**를 다루어야 합니다. 모호하거나 유사한 제목은 절대 금지됩니다.

        - **절대 피해야 할 나쁜 예시 (주제가 유사하여 내용이 중복됨):**
            - "1. 만두 수출 현황 분석", "2. 주요 수출국 및 시장 동향"  (X)
            - "1. 시장 트렌드", "2. 최신 동향 분석" (X)

        - **반드시 따라야 할 좋은 예시 (주제가 명확히 분리됨):**
            - "1. **국가별 수출액 데이터**를 통한 핵심 시장 분석" (정량적, 순위)
            - "2. **소비자 선호도 및 최신 식품 트렌드** 분석" (정성적, 트렌드)
            - "3. **주요 경쟁사 전략 및 성공/실패 사례** 연구" (경쟁, 사례 분석)

    2. **각 섹션별 사용 데이터 선택**
    - **1단계에서 설계한 고유한 섹션 제목**을 기준으로, 각 섹션마다 `use_contents` 필드에 **해당 섹션에서 사용할 데이터 인덱스 번호들을 배열로** 포함하세요.
    - **데이터 중복 할당 엄격 금지**: **서로 다른 섹션은 반드시 서로 다른 `use_contents` 목록을 가져야 합니다.** 동일한 데이터를 여러 섹션에서 사용하는 것을 최소화해야 합니다.
    - **최적 할당 원칙**: 각 데이터는 그것이 가장 핵심적으로 뒷받침할 수 있는 **단 하나의 섹션에만 할당**하는 것을 원칙으로 합니다.
    - **인덱스 범위 및 개수 제한**:
        - 위에 제시된 데이터 인덱스 중에서만 선택하세요: {selected_indexes}
        - 한 섹션에 너무 많은 데이터(5개 초과)를 할당하지 마세요.

    3. **'결론' 섹션 추가 (필수)**: 보고서의 가장 마지막에는 항상 '결론' 섹션을 포함하세요.
    - `content_type`은 'synthesis'로 설정
    - `use_contents`에는 주요 섹션들의 핵심 데이터를 종합하여 포함

    4. **데이터 필요 유형 명시**:
    - **수치, 통계, 트렌드, 분석 관련 섹션**: 'full_data_for_chart' (차트 생성)
    - **일반 설명, 개요, 결론**: 'synthesis' (텍스트만)

    5. **데이터 충분성 검증**:
    - 기본적으로 `is_sufficient`는 `true`로 설정
    - 해당 섹션 주제와 관련된 데이터가 전혀 없는 경우에만 `false`로 설정
    - `feedback_for_gatherer` 필드에 추가 데이터 요청이 필요한 경우, 정보가 부족한 부분은 보강할 수 있는 구체적 쿼리를 작성하세요

    **섹션별 데이터 선택 예시**:
    - "시장 규모 분석" 섹션 → 시장, 매출, 규모 관련 데이터 인덱스만 선택
    - "소비자 트렌드" 섹션 → 소비자, 구매, 선호도 관련 데이터 인덱스만 선택
    - "결론" 섹션 → 각 섹션의 핵심 데이터들을 종합하여 선택

    **출력 포맷 (반드시 JSON 형식으로만 응답):**
    {{
        "title": "보고서의 최종 제목",
        "structure": [
            {{
                "section_title": "1. 시장 현황 분석",
                "content_type": "full_data_for_chart",
                "use_contents": [0, 3, 7],
                "is_sufficient": true,
                "feedback_for_gatherer": ""
            }},
            {{
                "section_title": "2. 소비자 트렌드 분석",
                "content_type": "full_data_for_chart",
                "use_contents": [1, 5, 9],
                "is_sufficient": true,
                "feedback_for_gatherer": ""
            }},
            {{
                "section_title": "3. 경쟁 환경 분석",
                "content_type": "synthesis",
                "use_contents": [2, 4, 8],
                "is_sufficient": false,
                "feedback_for_gatherer": {{
                    "tool": "vector_db_search",
                    "query": "오뚜기 경쟁사의 시장 점유율과 전략 분석"
                }}
            }},
            {{
                "section_title": "4. 결론",
                "content_type": "synthesis",
                "use_contents": [0, 1, 3, 5, 7, 9],
                "is_sufficient": true,
                "feedback_for_gatherer": ""
            }}
        ]
    }}

    **중요**: `use_contents` 배열에는 반드시 위에 제시된 인덱스 번호 {selected_indexes} 중에서만 선택하세요.
    """

        try:
            response = await self._invoke_with_fallback(
                prompt,
                self.llm_flash,
                self.llm_openai_mini
            )

            print(f"  - 보고서 구조 설계 응답 길이: {len(response.content)} 문자")

            # JSON 파싱
            design_result = json.loads(re.search(r'\{.*\}', response.content, re.DOTALL).group())

            # 인덱스 유효성 검증 및 디버깅
            print(f"  - 구조 설계 결과 검증:")
            for i, section in enumerate(design_result.get("structure", [])):
                section_title = section.get("section_title", f"섹션 {i+1}")
                use_contents = section.get("use_contents", [])

                # 유효한 인덱스만 필터링
                valid_use_contents = []
                for idx in use_contents:
                    if isinstance(idx, int) and idx in selected_indexes:
                        valid_use_contents.append(idx)
                    else:
                        print(f"    경고: 잘못된 인덱스 {idx} 제거됨 (허용된 인덱스: {selected_indexes})")

                section["use_contents"] = valid_use_contents

                print(f"    '{section_title}': {len(valid_use_contents)}개 데이터 사용")
                print(f"      사용 인덱스: {valid_use_contents}")

                # 사용될 데이터 미리보기
                for idx in valid_use_contents[:2]:  # 처음 2개만
                    if 0 <= idx < len(data):
                        data_item = data[idx]
                        print(f"      [{idx:2d}] {getattr(data_item, 'source', 'Unknown'):10s} | {getattr(data_item, 'title', 'No Title')[:40]}")

            print(f"  - 보고서 구조 설계 완료: '{design_result.get('title', '제목없음')}'")
            print(f"{design_result.get('structure', [])}")
            return design_result

        except Exception as e:
            print(f"  - 보고서 구조 설계 실패: {e}")
            print(f"  - 안전 모드로 기본 구조 생성")

            # 안전 모드: 모든 선택된 인덱스를 하나의 섹션에서 사용
            return {
                "title": f"{query} - 통합 분석 보고서",
                "structure": [{
                    "section_title": "종합 분석",
                    "content_type": "synthesis",
                    "use_contents": selected_indexes[:10],  # 최대 10개까지만
                    "is_sufficient": True,
                    "feedback_for_gatherer": ""
                }]
            }


    # worker_agents.py - ProcessorAgent 클래스의 수정된 함수들

    async def _synthesize_data_for_section(self, section_title: str, section_data: List[SearchResult]) -> str:
        """⭐ 수정: 섹션별 선택된 데이터만 사용하여 출처 번호 정확히 매핑"""

        # ⭐ 핵심 개선: 섹션별 선택된 데이터만 사용하여 출처 정보 생성
        context_with_sources = ""
        for i, res in enumerate(section_data):  # section_data만 사용 (all_data 대신)
            source_info = ""
            source_link = ""

            # Web search 결과인 경우
            if hasattr(res, 'source') and 'web_search' in str(res.source).lower():
                if hasattr(res, 'url') and res.url:
                    source_link = res.url
                    source_info = f"웹 출처: {res.url}"
                elif hasattr(res, 'metadata') and res.metadata and 'link' in res.metadata:
                    source_link = res.metadata['link']
                    source_info = f"웹 출처: {res.metadata['link']}"
                else:
                    source_info = "웹 검색 결과"
                    source_link = "웹 검색"

            # Vector DB 결과인 경우
            elif hasattr(res, 'source_url'):
                source_info = f"문서 출처: {res.source_url}"
                source_link = res.source_url
            elif hasattr(res, 'title'):
                source_info = f"문서: {res.title}"
                source_link = res.title
            else:
                source_name = res.source if hasattr(res, 'source') else 'Vector DB'
                source_info = f"출처: {source_name}"
                source_link = source_name

            # 핵심: 섹션 데이터 내에서의 인덱스 사용 (0, 1, 2...)
            context_with_sources += f"--- 문서 ID {i}: [{source_info}] ---\n제목: {res.title}\n내용: {res.content}\n출처_링크: {source_link}\n\n"

        prompt = f"""
    당신은 여러 데이터 소스를 종합하여 특정 주제에 대한 분석 보고서의 한 섹션을 저술하는 주제 전문가입니다.

    **작성할 섹션의 주제**: "{section_title}"

    **사용 데이터 인덱스**

    **참고할 선택된 데이터** (섹션별로 엄선된 관련 데이터):
    {context_with_sources[:8000]}

    **작성 지침**:
    1. **핵심 정보 추출**: '{section_title}' 주제와 직접적으로 관련된 핵심 사실, 수치, 통계 위주로 정보를 추출하세요.
    2. **간결한 요약**: 정보를 단순히 나열하지 말고, 1~2 문단 이내의 간결하고 논리적인 핵심 요약문으로 재구성해주세요.
    3. **중복 제거**: 여러 문서에 걸쳐 반복되는 내용은 하나로 통합하여 제거하세요.
    4. **객관성 유지**: 데이터에 기반하여 객관적인 사실만을 전달해주세요.
    5. **⭐ 출처 정보 보존**: 중요한 정보나 수치를 언급할 때 해당 정보의 출처를 [SOURCE:숫자] 형식으로 표기하세요. 반드시 숫자만 사용하세요.
    - **문서 ID 번호를 사용**
    - 예시: "시장 규모가 증가했습니다 [SOURCE:1]", "매출이 상승했습니다 [SOURCE:2]"
    - 잘못된 예시: [SOURCE:데이터 1], [SOURCE:문서 1] (이런 형식 사용 금지)
    6. **⭐ 노션 스타일 마크다운 적극 활용**:
    - **중요한 키워드나 수치**: **굵은 글씨**로 강조
    - *일반적인 강조나 트렌드*: *기울임체*로 표현
    - **핵심 포인트나 결론**: > 인용문 형태로 강조
    - **항목이 여러 개**: - 첫 번째 항목, - 두 번째 항목 형태
    - **하위 분류**:   - 세부 항목 (들여쓰기)
    - **단락 구분**: 내용 변화 시 공백 라인으로 명확히 구분

    **결과물 (핵심 요약본)**:
    """
        response = await self._invoke_with_fallback(
            prompt,
            self.llm_flash,
            self.llm_openai_mini
        )
        return response.content

    async def generate_section_streaming(self, section: Dict[str, Any], full_data_dict: Dict[int, Dict], original_query: str, use_indexes: List[int], state: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        """페르소나를 반영하여 전체 데이터 딕셔너리에서 해당 섹션 인덱스만 사용하여 스트리밍 생성"""
        section_title = section.get("section_title", "제목 없음")
        content_type = section.get("content_type", "synthesis")
        description = section.get("description", "")

        # 페르소나 정보 추출
        persona_name = state.get("persona", "기본") if state else "기본"
        persona_instruction = self.personas.get(persona_name, {}).get("prompt", "당신은 전문적인 AI 분석가입니다.")
        print(f"  - 섹션 생성에 '{persona_name}' 페르소나 스타일 적용")

        # 섹션 시작 시 H2 헤더로 출력
        section_header = f"\n\n## {section_title}\n\n"
        yield section_header

        if content_type == "synthesis":
            print(f"\n🔍 === SECTION STREAMING 디버깅 ({section_title}) ===")
            print(f"use_indexes: {use_indexes}")
            print(f"full_data_dict 키들: {list(full_data_dict.keys()) if full_data_dict else 'None'}")

            # 전체 데이터 딕셔너리에서 해당 인덱스만 선별하여 프롬프트용으로 포맷팅
            section_data_content = ""
            valid_indexes = []
            for actual_index in use_indexes:
                if actual_index in full_data_dict:
                    valid_indexes.append(actual_index)
                    data_info = full_data_dict[actual_index]
                    section_data_content += f"**데이터 {actual_index}: {data_info['source']}**\n"
                    section_data_content += f"- **제목**: {data_info['title']}\n"
                    section_data_content += f"- **내용**: {data_info['content']}\n\n"
                    print(f"  ✅ [{actual_index}] 데이터 매핑 성공: '{data_info['title'][:30]}...'")
                else:
                    print(f"  ❌ [{actual_index}] full_data_dict에서 찾을 수 없음!")

            print(f"유효한 인덱스들: {valid_indexes}")
            print(f"프롬프트에서 사용할 SOURCE 번호들: {valid_indexes}")

            prompt_template = """

    {persona_instruction}

    당신은 위의 페르소나 지침을 반드시 따라서, 주어진 데이터를 바탕으로 전문가 수준의 보고서의 한 섹션을 작성하는 AI입니다.

    **사용자의 전체 질문**: "{original_query}"
    **현재 작성할 섹션 제목**: "{section_title}"
    **섹션 목표**: "{description}"

    **참고 데이터 (실제 인덱스 번호 포함)**:
    {section_data_content}

    **작성 지침 (매우 중요)**:
    1. **페르소나 역할 유지**: 당신의 역할과 말투, 분석 관점을 반드시 유지하며 작성하세요.
    2. **간결성 유지**: 반드시 1~2 문단 이내로, 가장 핵심적인 내용만 간결하게 요약하여 작성하세요.
    3. **제목 반복 금지**: 주어진 섹션 제목을 절대 반복해서 출력하지 마세요. 바로 본문 내용으로 시작해야 합니다.
    4. **데이터 기반**: 참고 데이터에 있는 구체적인 수치, 사실, 인용구를 적극적으로 활용하여 내용을 구성하세요.
    5. **전문가적 문체**: 명확하고 간결하며 논리적인 전문가의 톤으로 글을 작성하세요.
    6. **⭐ 노션 스타일 마크다운 적극 활용 (매우 중요)**:
    - **핵심 키워드나 중요한 수치**: **굵은 글씨**로 강조
    - *일반적인 강조나 변화*: *기울임체*로 표현
    - **주요 포인트나 결론**: > 중요한 인사이트나 결론 형태로 강조
    - **목록이 필요한 경우**: - 첫 번째 항목, - 두 번째 항목 형태로 구조화
    - **하위 분류가 있는 경우**:   - 세부 항목 (들여쓰기 사용)
    - **세부 카테고리**: ### 소제목 활용
    - **단락 구분**: 내용이 바뀔 때마다 명확하게 단락을 나누어 공백 라인 삽입
    7. **⭐ 출처 표기 (실제 인덱스 번호 사용)**: 특정 정보를 참고하여 작성한 문장 바로 뒤에 [SOURCE:숫자] 형식으로 출처를 표기하세요. 반드시 숫자만 사용하고 "데이터"라는 단어는 사용하지 마세요.
    - 예시: "**매출이 증가했습니다** [SOURCE:8]", "시장 점유율이 상승했습니다 [SOURCE:12]"
    - 예시: **매출이 5% 감소**했습니다. [SOURCE:1, 4, 8]
    - 잘못된 예시: [SOURCE:데이터 1], [SOURCE:문서 1] (이런 형식 사용 금지)

    **보고서 섹션 내용**:
    """

            prompt = prompt_template.format(
                persona_instruction=persona_instruction,
                original_query=original_query,
                section_title=section_title,
                description=description,
                section_data_content=section_data_content
            )

        else:  # "full_data_for_chart"
            section_data_with_sources = ""
            for actual_index in use_indexes:
                if actual_index in full_data_dict:
                    data_info = full_data_dict[actual_index]
                    section_data_with_sources += f"**데이터 {actual_index}: {data_info['source']}**\n"
                    section_data_with_sources += f"- **제목**: {data_info['title']}\n"
                    section_data_with_sources += f"- **내용**: {data_info['content']}\n"
                    section_data_with_sources += f"- **출처_링크**: {data_info.get('url') or data_info.get('source_url', '')}\n\n"

            prompt_template = """
    당신은 데이터 분석가이자 보고서 작성가입니다. 주어진 선택된 데이터를 분석하여, 텍스트 설명과 시각적 차트를 결합한 전문가 수준의 보고서 섹션을 작성합니다.

    **사용자의 전체 질문**: "{original_query}"
    **현재 작성할 섹션 제목**: "{section_title}"
    **섹션 목표**: "{description}"

    **참고 데이터 (실제 인덱스 번호 포함)**:
    {section_data}

    **작성 지침 (매우 중요)**:
    1. **간결성 유지**: 반드시 1~2 문단 이내로, 데이터에서 가장 중요한 인사이트와 분석 내용만 간결하게 요약하여 작성하세요.
    2. **제목 반복 금지**: 주어진 섹션 제목을 절대 반복해서 출력하지 마세요. 바로 본문 내용으로 시작해야 합니다.
    3. **데이터 기반**: 설명에 구체적인 수치, 사실, 통계 자료를 적극적으로 인용하여 신뢰도를 높이세요.
    4. **⭐ 차트 마커 삽입**: 텍스트 설명의 흐름 상, 시각적 데이터가 필요한 적절한 위치에 [GENERATE_CHART] 마커를 한 줄에 단독으로 삽입하세요.
    5. **서술 계속**: 마커를 삽입한 후, 이어서 나머지 텍스트 설명을 자연스럽게 계속 작성하세요.
    6. **⭐ 노션 스타일 마크다운 적극 활용**: 굵은 글씨, 기울임체, 인용문, 목록 등을 적절히 사용하세요.
    7. **⭐ 출처 표기 (실제 인덱스 번호 사용)**: 특정 정보를 참고하여 작성한 문장 바로 뒤에 [SOURCE:숫자1, 숫자2, 숫자3] 형식으로 출처를 표기하세요. 반드시 숫자만 사용하고 "데이터", "문서" 등의 단어는 사용하지 마세요.
    - 예시: **시장 규모가 10% 증가**했습니다. [SOURCE:8]
    - 예시: **매출이 5% 감소**했습니다. [SOURCE:1, 4, 8]
    - 잘못된 예시: [SOURCE:데이터 8], [SOURCE:문서 8] (이런 형식 사용 금지)

    **보고서 섹션 본문**:
    """

            prompt = prompt_template.format(
                persona_instruction=persona_instruction,
                original_query=original_query,
                section_title=section_title,
                description=description,
                section_data=section_data_with_sources
            )

        try:
            print(f"\n>> 섹션 스트리밍 시작: {section_title} (사용 인덱스: {use_indexes})")
            total_content = ""
            chunk_count = 0
            valid_content_count = 0

            async for chunk in self._astream_with_fallback(
                prompt,
                self.llm_pro,
                self.llm_openai_4o
            ):
                chunk_count += 1
                if hasattr(chunk, 'content') and chunk.content:
                    total_content += chunk.content
                    chunk_text = chunk.content
                    valid_content_count += 1

                    print(f"- 원본 청크 {chunk_count}: {len(chunk_text)} 문자")

                    # 5자 단위로 쪼개서 전송
                    for i in range(0, len(chunk_text), 5):
                        mini_chunk = chunk_text[i:i+5]
                        yield mini_chunk

            print(f"\n>> 섹션 완료: {section_title}, 총 {chunk_count}개 원본 청크, {valid_content_count}개 유효 청크, {len(total_content)} 문자")

            if not total_content.strip() or valid_content_count == 0:
                print(f"- 섹션 스트리밍 오류 ({section_title}): No generation chunks were returned")
                raise Exception("No generation chunks were returned")

        except Exception as e:
            print(f"- 섹션 스트리밍 오류 ({section_title}): {e}")
            if "No generation chunks" in str(e) or "no valid content" in str(e).lower():
                try:
                    print(f"- OpenAI로 직접 재시도: {section_title}")
                    total_content = ""
                    chunk_count = 0

                    async for chunk in self.llm_openai_4o.astream(prompt):
                        chunk_count += 1
                        if hasattr(chunk, 'content') and chunk.content:
                            total_content += chunk.content
                            chunk_text = chunk.content
                            print(f"- OpenAI 재시도 청크 {chunk_count}: {len(chunk_text)} 문자")

                            for i in range(0, len(chunk_text), 5):
                                mini_chunk = chunk_text[i:i+5]
                                yield mini_chunk

                    print(f"- OpenAI 재시도 완료: {section_title}, {chunk_count}개 청크, {len(total_content)} 문자")

                    if not total_content.strip():
                        print(f"- OpenAI 재시도도 실패, fallback 내용 생성")
                        raise Exception("OpenAI retry also failed")

                except Exception as retry_error:
                    print(f"- OpenAI 재시도 실패: {retry_error}")
                    fallback_content = f"*'{section_title}' 섹션에 대한 상세한 분석을 생성하는 중 문제가 발생했습니다.*\n\n"
                    yield fallback_content
            else:
                error_content = f"*'{section_title}' 섹션 생성 중 오류가 발생했습니다: {str(e)}*\n\n"
                yield error_content

    async def _create_charts(self, section_data: List[SearchResult], section_title: str, generated_content: str = "", yield_callback=None, state: Dict[str, Any] = None):
        """⭐ 수정: 페르소나 관점을 반영하여 섹션별 선택된 데이터와 생성된 내용을 바탕으로 정확한 차트 생성"""
        print(f"  - 차트 데이터 생성: '{section_title}' (데이터 {len(section_data)}개)")

        # >> 데이터 보강을 위한 DataGatherer 인스턴스 생성
        from .worker_agents import DataGathererAgent
        data_gatherer = DataGathererAgent() if not hasattr(self, 'data_gatherer') else self.data_gatherer

        # 페르소나 정보 추출
        persona_name = state.get("persona", "기본") if state else "기본"
        print(f"  - 차트 생성에 '{persona_name}' 페르소나 관점 적용")

        async def _generate_chart_with_data(current_data: List[SearchResult], attempt: int = 1):
            """실제 차트 생성 로직 (재시도 가능)"""
            try:
                # 데이터 요약 생성
                data_summary = ""
                for i, item in enumerate(current_data):
                    source = getattr(item, 'source', 'Unknown')
                    title = getattr(item, 'title', 'No Title')
                    content = getattr(item, 'content', '')[:500]
                    data_summary += f"[{i}] [{source}] {title}\n내용: {content}...\n\n"

                # 직전에 생성된 보고서 내용 추가
                context_info = ""
                if generated_content:
                    content_preview = generated_content[:800] if generated_content else ""
                    context_info = f"\n**직전에 생성된 보고서 내용 (차트와 일맥상통해야 함)**:\n{content_preview}\n"

                chart_prompt = f"""
        CRITICAL: You MUST respond with ONLY valid JSON. No explanations, no markdown, no other text.

        '{persona_name}'의 관점에서 다음 섹션을 위해 **선별된 실제 데이터**를 바탕으로 **정확하고 복잡한**, **가장 흥미롭고 유용한** Chart.js 차트를 생성해주세요.
        (예: 구매 담당자는 '원가 변동' 라인 차트, 마케터는 '소비자 관심도' 버블 차트를 선호할 것입니다.)

        **섹션 제목**: "{section_title}"
        **시도 횟수**: {attempt} (최대 2회)

        **섹션별로 엄선된 실제 데이터**:
        {data_summary}
        {context_info}

        **⭐ 중요한 제약사항**:
        1. **절대 임의 수치 생성 금지** - 위 데이터에서 명시된 실제 수치만 사용
        2. **'{persona_name}'의 관점 반영** - 해당 페르소나가 가장 중요하게 생각할 데이터를 시각화하세요.
        3. **섹션 제목과 직접 관련된 차트만** 생성
        4. **복잡하고 상세한 차트** 생성 (최소 3개 이상의 데이터 포인트)
        5. **정확한 라벨과 수치** 사용

        **데이터 추출 규칙**:
        - 위 데이터에서 숫자, 퍼센트, 금액 등 수치 정보 추출
        - 연도, 월, 분기 등 시간 정보 추출
        - 카테고리, 부문, 지역 등 분류 정보 추출

        **데이터가 충분하지 않은 경우, 다음 JSON 형식으로 부족한 데이터 정보를 제공하세요**:
        {{
            "insufficient_data": true,
            "missing_info": "구체적으로 부족한 데이터 설명 (예: '글로벌 식품 시장의 연도별 매출액 데이터', '주요 국가별 시장 점유율 수치' 등)",
            "suggested_search_query": "부족한 데이터를 찾기 위한 구체적인 검색 쿼리"
        }}

        **데이터가 충분한 경우, Chart.js 형식의 완전한 JSON을 출력하세요**:
        {{
            "type": "적절한_차트_타입",
            "data": {{
                "labels": ["실제_데이터_기반_라벨1", "라벨2", "라벨3"],
                "datasets": [{{
                    "label": "실제_데이터셋_이름",
                    "data": [실제_수치1, 실제_수치2, 실제_수치3],
                    "backgroundColor": ["#4F46E5", "#7C3AED", "#EC4899"],
                    "borderColor": "#4F46E5",
                    "borderWidth": 2
                }}]
            }},
            "options": {{
                "responsive": true,
                "plugins": {{
                    "title": {{
                        "display": true,
                        "text": "{section_title} - 실제 데이터 기반 차트"
                    }},
                    "legend": {{
                        "display": true,
                        "position": "top"
                    }}
                }},
                "scales": {{
                    "y": {{
                        "beginAtZero": true,
                    }}
                }}
            }}
        }}

        OUTPUT ONLY JSON (no other text):
        """

                response = await self._invoke_with_fallback(
                    chart_prompt,
                    self.llm_flash,
                    self.llm_openai_mini
                )
                response_text = response.content.strip()

                # JSON 추출 개선
                try:
                    # 코드 블록 제거
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0].strip()

                    # JSON 파싱
                    chart_response = json.loads(response_text)

                    # >> 데이터 부족 여부 확인
                    if chart_response.get("insufficient_data", False) and attempt == 1:
                        missing_info = chart_response.get("missing_info", "")
                        search_query = chart_response.get("suggested_search_query", "")

                        print(f"  - 차트 데이터 부족 감지: {missing_info}")
                        print(f"  - 추가 검색 실행: '{search_query}'")

                        yield {
                            "type": "status",
                            "data": {"message": f"차트 데이터 부족으로 '{search_query[:50]}...' 관련 정보를 추가 수집 중입니다"}
                        }

                        # >> 웹 검색으로 추가 데이터 수집
                        try:
                            additional_data = []
                            web_results, _ = await data_gatherer.execute("web_search", {"query": search_query})
                            vector_results, _ = await data_gatherer.execute("vector_db_search", {"query": search_query})
                            additional_data.extend(web_results)
                            additional_data.extend(vector_results)

                            if additional_data:
                                print(f"  - 추가 데이터 수집 완료: {len(additional_data)}개")

                                # >> 프론트엔드에 검색 결과 스트리밍 전송
                                if additional_data:
                                    yield {
                                        "type": "status",
                                        "data": {"message": f"추가 데이터 {len(additional_data)}개 수집 완료. 차트를 재생성합니다..."}
                                    }
                                    # 검색 결과를 프론트엔드 형식으로 변환
                                    formatted_results = []
                                    for search_result in additional_data:
                                        # SearchResult 객체인지 dict인지 확인
                                        if hasattr(search_result, 'model_dump'):
                                            result_dict = search_result.model_dump()
                                        elif isinstance(search_result, dict):
                                            result_dict = search_result
                                        else:
                                            continue  # 지원하지 않는 형식은 건너뛰기

                                        formatted_result = {
                                            "title": result_dict.get("title", "제목 없음"),
                                            "content_preview": result_dict.get("content", "내용 없음")[:200] + "...",
                                            "url": result_dict.get("url", "URL 없음"),
                                            "source": result_dict.get("source", "web_search"),
                                            "score": result_dict.get("score", 0.0),
                                            "document_type": result_dict.get("document_type", "web")
                                        }
                                        formatted_results.append(formatted_result)

                                    # 검색 결과 이벤트 전송 (중간 검색 표시)
                                    yield {
                                        "type": "search_results",
                                        "data": {
                                            "step": f"chart_enhancement_{attempt}",
                                            "tool_name": "web_search",
                                            "query": search_query,
                                            "results": formatted_results,
                                            "is_intermediate_search": True,
                                            "section_context": {
                                                "section_title": section_title,
                                                "search_reason": "차트 데이터 부족으로 인한 추가 검색",
                                                "attempt": attempt
                                            },
                                            "message_id": state.get("message_id") if state else None
                                        }
                                    }

                                    # 데이터 보강 완료 상태 전송
                                    yield {
                                        "type": "status",
                                        "data": {"message": f"차트 데이터 보강 완료. 차트를 다시 생성합니다."}
                                    }
                            else:
                                print(f"  - 추가 데이터 수집 실패")
                                yield {
                                    "type": "status",
                                    "data": {"message": "추가 데이터 수집에 실패했습니다. 기본 데이터로 차트를 생성합니다..."}
                                }
                        except Exception as search_error:
                            print(f"  - 데이터 보강 검색 실패: {search_error}")
                            yield {
                                "type": "status",
                                "data": {"message": f"데이터 보강 중 오류 발생: {str(search_error)}"}
                            }

                        # 검색 실패시 fallback 차트 반환
                        yield {
                            "type": "chart",
                            "data": {
                                "type": "bar",
                            "data": {
                                "labels": ["데이터 부족"],
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
                                        "text": f"{section_title} - 데이터 수집 중"
                                    }
                                }
                            }
                            }
                        }
                        return

                    # >> 정상적인 차트 데이터인 경우
                    elif "type" in chart_response and "data" in chart_response:
                        # 필수 필드 검증
                        datasets = chart_response.get("data", {}).get("datasets", [])
                        if datasets and len(datasets) > 0:
                            data_points = datasets[0].get("data", [])
                            if len(data_points) < 2:
                                print(f"  - 경고: 차트 데이터 포인트가 부족함 ({len(data_points)}개)")

                        # 콜백 함수 제거 (프론트엔드 오류 방지)
                        def remove_callbacks(obj):
                            if isinstance(obj, dict):
                                # 콜백 함수 관련 키들 제거
                                callback_keys = ['callback', 'callbacks', 'generateLabels']
                                for key in list(obj.keys()):
                                    if key in callback_keys:
                                        del obj[key]
                                    elif isinstance(obj[key], str) and 'function' in obj[key]:
                                        del obj[key]
                                    else:
                                        remove_callbacks(obj[key])
                            elif isinstance(obj, list):
                                for item in obj:
                                    remove_callbacks(item)

                        remove_callbacks(chart_response)

                        print(f"  - 차트 생성 성공: {chart_response['type']} 타입, {len(datasets)}개 데이터셋 (시도 {attempt})")
                        yield {
                            "type": "chart",
                            "data": chart_response
                        }
                        return
                    else:
                        raise ValueError("올바르지 않은 JSON 형식")

                except (json.JSONDecodeError, ValueError) as e:
                    print(f"  - 차트 JSON 파싱 실패 (시도 {attempt}): {e}")
                    print(f"  - 원본 응답: {response_text[:200]}...")

                    # 첫 번째 시도이고 파싱 실패시에도 웹 검색 시도
                    if attempt == 1:
                        print(f"  - JSON 파싱 실패로 인한 데이터 보강 시도")

                        # 섹션 제목 기반으로 검색 쿼리 생성
                        fallback_query = f"{section_title} 시장 데이터 통계 수치"

                        try:
                            # async generator를 올바르게 처리
                            additional_data, _ = await data_gatherer.execute("web_search", {"query": fallback_query})

                            if additional_data:
                                print(f"  - Fallback 추가 데이터 수집: {len(additional_data)}개")
                                enhanced_data = current_data + additional_data
                                async for result in _generate_chart_with_data(enhanced_data, attempt=2):
                                    yield result
                                return
                        except Exception as search_error:
                            print(f"  - Fallback 검색 실패: {search_error}")

                    # 최종 fallback 차트
                    yield {
                        "type": "chart",
                        "data": {
                            "type": "bar",
                            "data": {
                                "labels": [f"{section_title} 관련 데이터"],
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
                                        "text": f"{section_title} - 데이터 분석 중"
                                    }
                                },
                                "scales": {
                                    "y": {
                                        "beginAtZero": True,
                                        "max": 2,
                                        "ticks": {
                                            "stepSize": 1
                                        }
                                    }
                                }
                            }
                        }
                    }

            except Exception as e:
                print(f"  - 차트 생성 전체 오류 (시도 {attempt}): {e}")
                yield {
                    "type": "chart",
                    "data": {
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
                            }
                        }
                    }
                }

        # >> 메인 로직 실행
        async for result in _generate_chart_with_data(section_data, attempt=1):
            yield result
