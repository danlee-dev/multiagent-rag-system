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
def mock_graph_db_search(query: str) -> str:
    """
    Neo4j 지식 그래프(Knowledge Graph)에서 개체(노드) 간의 복잡한 관계나 연결 구조를 탐색하고 분석합니다.
    - 사용 시점:
      1. 특정 인물, 기관, 문서 간의 직접적 또는 간접적 관계를 파악해야 할 때 (예: 'A 연구원이 참여한 모든 보고서는?', 'B 회사와 협력 관계인 모든 기관을 알려줘')
      2. 연결 경로, 영향력, 숨겨진 패턴을 찾아야 할 때 (예: 'X 기술이 어떤 논문들을 통해 Y 산업에 영향을 미쳤는지 경로를 추적해줘')
      3. 여러 개체와 조건이 얽힌 복잡한 질문에 답해야 할 때 (예: 'KREI 소속 저자가 작성하고 '기후 변화'를 다루는 논문과 관련된 모든 키워드는?')
    - 데이터 종류: 인물, 기관, 문서(뉴스/논문/보고서), 키워드 등의 개체(노드)와 이들을 연결하는 관계(엣지)로 구성된 그래프 데이터.
    - 주의: 단순 통계 조회(RDB), 최신 뉴스 검색(Web Search), 문서 본문 검색(Vector Search)에는 적합하지 않습니다.
    """
    print(f"Graph DB 검색 실행: {query}")
    search_result = mock_graph_db.search(query)
    nodes = search_result.get("nodes", [])
    relationships = search_result.get("relationships", [])
    if not nodes and not relationships:
        return f"'{query}'에 대한 관련 개체나 관계를 찾을 수 없습니다."
    summary = (
        f"Graph DB 검색 결과: {len(nodes)}개 노드, {len(relationships)}개 관계 발견\n\n"
    )
    summary += "### 주요 노드:\n"
    for node in nodes[:5]:
        props = node.get("properties", {})
        summary += (
            f"- {props.get('name', node.get('id'))} (레이블: {node.get('labels')})\n"
        )
    summary += "\n### 주요 관계:\n"
    for rel in relationships[:5]:
        start_node = rel.get("start_node")
        end_node = rel.get("end_node")
        rel_type = rel.get("type")
        summary += f"- ({start_node}) -[{rel_type}]-> ({end_node})\n"
    return summary
