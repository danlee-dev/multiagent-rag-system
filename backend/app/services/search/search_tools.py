# search_tools.py

import os
import requests
from langchain_core.tools import tool
import json

# 각 RAG 툴의 메인 함수를 import 합니다.
from ..database.postgres_rag_tool import postgres_rdb_search
from ..database.neo4j_rag_tool import neo4j_graph_search

from ..database.mock_databases import create_mock_vector_db

mock_vector_db = create_mock_vector_db()


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
def rdb_search(query: str) -> str:
    """
    PostgreSQL DB에 저장된 정형 데이터를 조회하여 정확한 수치나 통계를 제공합니다.
    - 사용 시점:
      1. 구체적인 품목, 날짜, 지역 등의 조건으로 정확한 데이터를 찾을 때 (예: '최근 일주일간 제주도산 감귤의 평균 가격 알려줘')
      2. 영양 정보, 수급량 등 명확한 스펙이나 통계 수치를 물을 때 (예: '사과 100g당 칼로리와 비타민C 함량은?')
      3. 특정 기간의 데이터 순위나 추이를 알고 싶을 때 (예: '작년 한국이 가장 많이 수입한 과일은?')
    """
    print(f"\n>> 실제 PostgreSQL 검색 시작: {query}")
    try:
        return postgres_rdb_search(query)
    except Exception as e:
        error_msg = f"PostgreSQL 연결 오류: {str(e)}"
        print(f">> {error_msg}")
        return error_msg


@tool
def mock_vector_search(query: str) -> str:
    """
    Elasticsearch에 저장된 뉴스 기사 본문, 논문, 보고서 전문에서 '의미 기반'으로 유사한 내용을 검색합니다.
    - 사용 시점:
      1. 특정 주제에 대한 심층적인 분석이나 여러 문서에 걸친 종합적인 정보가 필요할 때 (예: '기후 변화가 농산물 가격에 미치는 영향에 대한 보고서 찾아줘')
      2. 문서의 단순 키워드 매칭이 아닌, 문맥적 의미나 논조를 파악해야 할 때 (예: 'AI 기술의 긍정적 측면을 다룬 뉴스 기사 요약해줘')
      3. 특정 보고서나 논문의 내용을 확인하고 싶을 때
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
    Neo4j 지식 그래프에서 농산물, 수산물, 지역 등의 개체와 그들 간의 관계를 검색합니다.
    - 사용 시점: 'A의 생산지는 어디야?', 'B와 관련된 품목은 뭐야?'와 같이 개체 간의 연결 관계나 소속 정보가 필요할 때 사용합니다.
    """
    print(f"Neo4j Graph DB 검색 실행: {query}")
    return neo4j_graph_search(query)
