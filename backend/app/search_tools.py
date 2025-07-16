"""
에이전트가 사용할 수 있는 모든 검색 도구(Tools)를 정의하는 파일입니다.
각 도구는 @tool 데코레이터로 장식되어 있으며, 명확하고 상세한 description을 가지고 있어
LLM 에이전트가 사용자의 질문 의도에 따라 최적의 도구를 선택할 수 있도록 안내합니다.
"""

import os
import json
import requests
from typing import Dict, List, Any
from langchain_core.tools import tool

# 로컬 mock_databases 파일에서 DB 인스턴스 생성 함수를 가져옵니다.
from .mock_databases import create_mock_databases

# Neo4j 연결을 위한 import 추가
from .neo4j_query import run_cypher

mock_graph_db, mock_vector_db, mock_rdb = create_mock_databases()


# --------------------------------------------------
# Tool Definitions
# --------------------------------------------------


@tool
def debug_web_search(query: str) -> str:
    """
    내부 데이터베이스(RDB, Vector, Graph)에 없는 최신 정보나 일반적인 지식을 실제 웹(구글)에서 검색합니다.
    - 사용 시점:
      1. '오늘', '현재', '실시간' 등 내부 DB에 아직 반영되지 않았을 수 있는 최신 정보가 필요할 때 (예: '오늘자 A기업 주가', '현재 서울 날씨')
      2. 내부 DB의 주제(농업/식품)를 벗어나는 일반적인 질문일 때 (예: '대한민국의 수도는 어디야?')
      3. 특정 인물, 사건, 제품에 대한 최신 뉴스와 같이 시의성이 매우 중요한 정보를 찾을 때
    - 주의: 농산물 시세, 영양 정보, 문서 내용 분석, 데이터 관계 분석 등 내부 DB로 해결 가능한 질문에는 절대 사용하지 마세요. 최후의 수단으로 사용해야 합니다.
    """
    print(f"Web 검색 실행: {query}")
    try:
        api_key = os.environ.get("SERPER_API_KEY")
        if not api_key:
            return "SERPER_API_KEY가 설정되지 않았습니다."

        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": 3, "gl": "kr", "hl": "ko"}
        response = requests.post(url, headers=headers, json=payload, timeout=10)

        if response.status_code == 200:
            data = response.json()
            results = []
            if "answerBox" in data:
                answer = data["answerBox"].get("answer", "")
                if answer:
                    results.append({"title": "Direct Answer", "snippet": answer})
            if "organic" in data and data["organic"]:
                for result in data["organic"][:3]:
                    results.append(
                        {
                            "title": result.get("title", "No title"),
                            "snippet": result.get("snippet", "No snippet"),
                        }
                    )
            if results:
                result_text = f"웹 검색 결과 (검색어: {query}):\n\n"
                for i, result in enumerate(results):
                    result_text += (
                        f"{i+1}. {result['title']}\n   {result['snippet']}\n\n"
                    )
                return result_text
            else:
                return f"'{query}'에 대한 웹 검색 결과를 찾을 수 없습니다."
        else:
            return f"API 오류: {response.status_code}, {response.text}"
    except Exception as e:
        return f"웹 검색 중 예외 발생: {str(e)}"


@tool
def mock_rdb_search(query: str) -> str:
    """
    PostgreSQL DB에 저장된 정형 데이터를 조회하여 정확한 수치나 통계를 제공합니다.
    - 사용 시점:
      1. 구체적인 품목, 날짜, 지역 등의 조건으로 정확한 데이터를 찾을 때 (예: '최근 일주일간 제주도산 감귤의 평균 가격 알려줘')
      2. 영양 정보, 수급량 등 명확한 스펙이나 통계 수치를 물을 때 (예: '사과 100g당 칼로리와 비타민C 함량은?')
      3. 특정 기간의 데이터 순위나 추이를 알고 싶을 때 (예: '작년 한국이 가장 많이 수입한 과일은?')
    - 데이터 종류: 농산물 시세, 원산지, 영양소 정보, 수출입량 통계, 실시간 트렌드 키워드, 뉴스 메타데이터(URL, 날짜 등).
    """
    print(f"RDB 검색 실행: {query}")
    search_result = mock_rdb.search(query)
    summary = f"RDB 검색 결과 (총 {search_result['total_results']}개 레코드 발견):\n"
    data = search_result.get("data", {})
    for category, records in data.items():
        if records:
            summary += f"- {category}: {len(records)}건\n"
    summary += "\n### 주요 데이터 (JSON 형식)\n"
    summary += json.dumps(data, ensure_ascii=False, indent=2)
    return summary


@tool
def mock_vector_search(query: str) -> str:
    """
    Elasticsearch에 저장된 뉴스 기사 본문, 논문, 보고서 전문에서 '의미 기반'으로 유사한 내용을 검색합니다.
    - 사용 시점:
      1. 특정 주제에 대한 심층적인 분석이나 여러 문서에 걸친 종합적인 정보가 필요할 때 (예: '기후 변화가 농산물 가격에 미치는 영향에 대한 보고서 찾아줘')
      2. 문서의 단순 키워드 매칭이 아닌, 문맥적 의미나 논조를 파악해야 할 때 (예: 'AI 기술의 긍정적 측면을 다룬 뉴스 기사 요약해줘')
      3. 특정 보고서나 논문의 내용을 확인하고 싶을 때
    - 데이터 종류: 뉴스 기사 본문, 논문 초록/본문, KREI 보고서 등 텍스트 데이터.
    """
    print(f"Vector DB 검색 실행: {query}")
    search_results = mock_vector_db.search(query)
    if not search_results:
        return f"'{query}'에 대한 관련 문서를 찾을 수 없습니다."
    summary = f"Vector DB 검색 결과 (상위 {len(search_results)}개 문서):\n\n"
    for i, doc in enumerate(search_results):
        summary += f"{i+1}. 제목: {doc.get('title', 'N/A')}\n"
        summary += f"   - 출처: {doc.get('metadata', {}).get('source', 'N/A')}\n"
        summary += f"   - 유사도: {doc.get('similarity_score', 0):.2f}\n"
        content_preview = doc.get("content", "")[:100]
        summary += f"   - 내용 미리보기: {content_preview}...\n\n"
    return summary

@tool
def graph_db_search(query: str) -> str:
    """
    Neo4j 지식 그래프에서 농산물/수산물과 지역 정보를 검색합니다.

    검색 가능한 정보:
    - 농산물: product(품목명), category(분류) - 569개
    - 수산물: product(품목명), fishState(상태) - 369개
    - 지역: city(시/군), region(도/시) - 213개
    - 관계: isFrom(품목 → 지역)
    """
    print(f"Neo4j Graph DB 검색 실행: {query}")

    try:
        # 1. 키워드 추출 및 정제
        keywords = _extract_and_clean_keywords(query)
        print(f"추출된 키워드: {keywords}")

        all_results = []

        # 2. 각 키워드별 최적화된 검색
        for keyword in keywords[:3]:
            # 2-1. 농산물 검색
            agricultural_results = _search_agricultural_products(keyword)
            all_results.extend(agricultural_results)

            # 2-2. 수산물 검색
            marine_results = _search_marine_products(keyword)
            all_results.extend(marine_results)

            # 2-3. 지역 검색
            region_results = _search_regions(keyword)
            all_results.extend(region_results)

        # 3. 중복 제거 및 관계 정보 추가
        unique_results = _deduplicate_results(all_results)

        # 4. 관계 정보 검색 (상위 결과 기준)
        relationships = []
        if unique_results:
            relationships = _search_relationships(unique_results[:3])

        # 5. 결과 포맷팅
        if not unique_results and not relationships:
            return f"'{query}'에 대한 관련 정보를 Neo4j에서 찾을 수 없습니다."

        summary = f"Neo4j Graph DB 검색 결과: {len(unique_results)}개 항목, {len(relationships)}개 관계 발견\n\n"

        # 노드 정보
        if unique_results:
            summary += "### 검색된 항목:\n"
            for item in unique_results[:8]:
                summary += _format_search_result(item)

        # 관계 정보
        if relationships:
            summary += "\n### 연관 관계:\n"
            for rel in relationships[:5]:
                summary += f"- {rel['start_item']} → {rel['end_location']} ({rel['relationship']})\n"

        print(f"- Neo4j 검색 완료: {len(unique_results)}개 항목, {len(relationships)}개 관계")
        return summary

    except Exception as e:
        print(f"- Neo4j 검색 오류: {e}")
        return f"Neo4j 검색 중 오류가 발생했습니다: {str(e)}"


def _extract_and_clean_keywords(query: str) -> list:
    """쿼리에서 Neo4j 검색에 유용한 키워드 추출"""
    # 불용어 제거
    stop_words = [
        '의', '을', '를', '이', '가', '에', '에서', '로', '으로', '와', '과',
        '는', '은', '도', '만', '알려줘', '검색', '찾아', '정보', '데이터',
        '어디', '언제', '어떻게', '무엇', '누구', '왜'
    ]

    # 단어 분리 및 정제
    words = query.replace(',', ' ').replace('.', ' ').split()
    keywords = []

    for word in words:
        word = word.strip()
        if len(word) > 1 and word not in stop_words:
            keywords.append(word)

    return keywords


def _search_agricultural_products(keyword: str) -> list:
    """농산물 검색 최적화"""
    try:
        # 정확한 매칭 우선
        exact_query = """
        MATCH (n:농산물)
        WHERE n.product = $keyword
        RETURN 'agricultural' as type, n.product as product, n.category as category,
               labels(n) as labels, properties(n) as properties
        """
        exact_results = run_cypher(exact_query, {"keyword": keyword})

        if exact_results:
            print(f"  농산물 정확 매칭: {len(exact_results)}개")
            return exact_results

        # 부분 매칭
        partial_query = """
        MATCH (n:농산물)
        WHERE n.product CONTAINS $keyword OR n.category CONTAINS $keyword
        RETURN 'agricultural' as type, n.product as product, n.category as category,
               labels(n) as labels, properties(n) as properties
        LIMIT 5
        """
        partial_results = run_cypher(partial_query, {"keyword": keyword})
        print(f"  농산물 부분 매칭: {len(partial_results)}개")
        return partial_results

    except Exception as e:
        print(f"  농산물 검색 오류: {e}")
        return []


def _search_marine_products(keyword: str) -> list:
    """수산물 검색 최적화"""
    try:
        # 정확한 매칭 우선
        exact_query = """
        MATCH (n:수산물)
        WHERE n.product = $keyword
        RETURN 'marine' as type, n.product as product, n.fishState as fishState,
               labels(n) as labels, properties(n) as properties
        """
        exact_results = run_cypher(exact_query, {"keyword": keyword})

        if exact_results:
            print(f"  수산물 정확 매칭: {len(exact_results)}개")
            return exact_results

        # 부분 매칭
        partial_query = """
        MATCH (n:수산물)
        WHERE n.product CONTAINS $keyword OR n.fishState CONTAINS $keyword
        RETURN 'marine' as type, n.product as product, n.fishState as fishState,
               labels(n) as labels, properties(n) as properties
        LIMIT 5
        """
        partial_results = run_cypher(partial_query, {"keyword": keyword})
        print(f"  수산물 부분 매칭: {len(partial_results)}개")
        return partial_results

    except Exception as e:
        print(f"  수산물 검색 오류: {e}")
        return []


def _search_regions(keyword: str) -> list:
    """지역 검색 최적화"""
    try:
        # 정확한 매칭 우선
        exact_query = """
        MATCH (n:Origin)
        WHERE n.city = $keyword OR n.region = $keyword
        RETURN 'region' as type, n.city as city, n.region as region,
               labels(n) as labels, properties(n) as properties
        """
        exact_results = run_cypher(exact_query, {"keyword": keyword})

        if exact_results:
            print(f"  지역 정확 매칭: {len(exact_results)}개")
            return exact_results

        # 부분 매칭
        partial_query = """
        MATCH (n:Origin)
        WHERE n.city CONTAINS $keyword OR n.region CONTAINS $keyword
        RETURN 'region' as type, n.city as city, n.region as region,
               labels(n) as labels, properties(n) as properties
        LIMIT 5
        """
        partial_results = run_cypher(partial_query, {"keyword": keyword})
        print(f"  지역 부분 매칭: {len(partial_results)}개")
        return partial_results

    except Exception as e:
        print(f"  지역 검색 오류: {e}")
        return []


def _search_relationships(items: list) -> list:
    """관계 정보 검색"""
    relationships = []

    try:
        for item in items:
            if item.get('type') in ['agricultural', 'marine']:
                product_name = item.get('product')
                if product_name:
                    # 해당 품목의 생산지 찾기
                    rel_query = """
                    MATCH (product {product: $product_name})-[r:isFrom]->(location:Origin)
                    RETURN r, location.city as city, location.region as region
                    LIMIT 3
                    """
                    rel_results = run_cypher(rel_query, {"product_name": product_name})

                    for rel in rel_results:
                        relationships.append({
                            'start_item': product_name,
                            'end_location': f"{rel.get('city', '')} ({rel.get('region', '')})",
                            'relationship': 'isFrom'
                        })

        print(f"  관계 검색: {len(relationships)}개")
        return relationships

    except Exception as e:
        print(f"  관계 검색 오류: {e}")
        return []


def _deduplicate_results(results: list) -> list:
    """결과 중복 제거"""
    seen = set()
    unique_results = []

    for result in results:
        # 고유 키 생성
        if result.get('type') == 'agricultural':
            key = f"agri_{result.get('product', '')}"
        elif result.get('type') == 'marine':
            key = f"marine_{result.get('product', '')}"
        elif result.get('type') == 'region':
            key = f"region_{result.get('city', '')}_{result.get('region', '')}"
        else:
            key = str(result)

        if key not in seen:
            seen.add(key)
            unique_results.append(result)

    return unique_results


def _format_search_result(item: dict) -> str:
    """검색 결과 포맷팅"""
    item_type = item.get('type', 'unknown')

    if item_type == 'agricultural':
        product = item.get('product', '알 수 없음')
        category = item.get('category', '미분류')
        return f"- 🌾 농산물: {product} (분류: {category})\n"

    elif item_type == 'marine':
        product = item.get('product', '알 수 없음')
        fish_state = item.get('fishState', '미분류')
        return f"- 🐟 수산물: {product} (상태: {fish_state})\n"

    elif item_type == 'region':
        city = item.get('city', '알 수 없음')
        region = item.get('region', '미분류')
        return f"- 📍 지역: {city} ({region})\n"

    else:
        return f"- ❓ 기타: {str(item)}\n"
