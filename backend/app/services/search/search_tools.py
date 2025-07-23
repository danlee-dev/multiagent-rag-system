import os
import requests
import json
import asyncio
import concurrent.futures
import io
from pypdf import PdfReader

# 각 RAG 툴의 메인 함수를 import
from ..database.postgres_rag_tool import postgres_rdb_search
from ..database.neo4j_rag_tool import neo4j_search_sync

from ..database.mock_databases import create_mock_vector_db

from ...core.models.models import ScrapeInput



from playwright.sync_api import sync_playwright
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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
        payload = {"q": query, "num": 5, "gl": "kr", "hl": "ko"}  # 결과 수 증가
        response = requests.post(url, headers=headers, json=payload, timeout=10)

        if response.status_code == 200:
            data = response.json()
            results = []

            # Answer box 우선 처리
            if "answerBox" in data:
                answer = data["answerBox"].get("answer", "")
                if answer:
                    results.append({
                        "title": "Direct Answer",
                        "snippet": answer,
                        "link": "google_answer_box",
                        "source": "google_answer_box"
                    })

            # Organic 결과 처리
            if "organic" in data and data["organic"]:
                for result in data["organic"][:5]:  # 상위 5개
                    link = result.get("link", "")
                    title = result.get("title", "No title")
                    snippet = result.get("snippet", "No snippet")

                    # 유효한 URL인지 확인
                    if link and link.startswith(('http://', 'https://')):
                        # PDF 파일인 경우 제목 수정
                        if link.endswith('.pdf'):
                            title = f"PDF: {title}"

                        results.append({
                            "title": title,
                            "snippet": snippet,
                            "link": link,
                            "source": "web_search"
                        })
                    else:
                        print(f"- 유효하지 않은 URL 스킵: {link}")

            if results:
                # 텍스트 형태로 반환 (ReAct 에이전트용)
                result_text = f"웹 검색 결과 (검색어: {query}):\n\n"
                for i, result in enumerate(results):
                    result_text += f"{i+1}. {result['title']}\n"
                    result_text += f"   링크: {result['link']}\n"
                    result_text += f"   요약: {result['snippet']}\n\n"

                print(f"- 유효한 검색 결과: {len(results)}개")
                return result_text
            else:
                return f"'{query}'에 대한 유효한 웹 검색 결과를 찾을 수 없습니다."
        else:
            return f"API 오류: {response.status_code}, {response.text}"
    except Exception as e:
        return f"웹 검색 중 예외 발생: {str(e)}"


@tool
def scrape_and_extract_content(action_input: str) -> str:
    """
    주어진 URL의 웹페이지 또는 PDF에 접속하여 본문 내용을 추출하고, 사용자의 원래 질문과 관련된 핵심 정보를 요약합니다.
    Action Input은 반드시 '{"url": "...", "query": "..."}' 형태의 JSON(딕셔너리) 문자열이어야 합니다.
    """
    try:
        input_data = json.loads(action_input)
        url = input_data['url']
        query = input_data['query']
        print(f"Scraping 시작 (URL: {url}, Query: {query})")

    except (json.JSONDecodeError, KeyError) as e:
        return f"입력값 파싱 오류: Action Input은 '{{\"url\": \"...\", \"query\": \"...\"}}' 형태여야 합니다. 오류: {e}"

    # URL이 PDF로 끝나는지에 따라 다른 처리 함수를 호출
    if url.lower().endswith('.pdf'):
        return _scrape_pdf_content(url, query)
    else:
        return _scrape_html_content(url, query)


def _scrape_pdf_content(url: str, query: str) -> str:
    """PDF URL에서 텍스트를 추출하고 요약합니다."""
    print(f"  → PDF 처리 모드 시작: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status() # HTTP 오류가 있으면 예외 발생

        with io.BytesIO(response.content) as f:
            reader = PdfReader(f)
            text = "".join(page.extract_text() for page in reader.pages if page.extract_text())

        if not text:
            return "PDF에서 텍스트를 추출할 수 없었습니다."

        print(f"  ✓ PDF 텍스트 추출 완료 ({len(text)}자)")
        return _extract_key_info(text, query)

    except Exception as e:
        return f"PDF 처리 중 오류 발생: {e}"


def _scrape_html_content(url: str, query: str) -> str:
    """HTML 웹페이지에서 텍스트를 추출하고 요약합니다."""
    print(f"  → HTML 처리 모드 시작: {url}")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=30000) # 타임아웃 증가

            # 더 견고한 방식으로 본문 탐색 (article -> main -> body 순서)
            locators = ['article', 'main', '[role="main"]']
            content = ""
            for loc in locators:
                try:
                    # 해당 선택자의 첫 번째 요소만 선택
                    content = page.locator(loc).first.inner_text(timeout=2000)
                    if len(content) > 100:
                        print(f"  ✓ '{loc}' 선택자에서 본문 발견")
                        break
                except Exception:
                    continue

            # 위에서 못 찾으면 body 전체를 최후의 수단으로 사용
            if not content:
                content = page.locator('body').inner_text(timeout=2000)
                print("  ✓ 최후의 수단으로 'body' 선택자 사용")

            browser.close()

            if not content:
                return "웹페이지에서 내용을 추출할 수 없었습니다."

            print(f"  ✓ HTML 텍스트 추출 완료 ({len(content)}자)")
            return _extract_key_info(content, query)

    except Exception as e:
        return f"웹페이지 스크래핑 중 오류 발생: {e}"


def _extract_key_info(content: str, query: str) -> str:
    """추출된 전체 텍스트에서 LLM을 이용해 핵심 정보를 다시 추출합니다."""
    print(f"  → LLM 분석 시작...")
    try:
        extractor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = ChatPromptTemplate.from_template(
            """당신은 유능한 데이터 분석가입니다. 아래 원본 텍스트에서 사용자의 질문과 가장 관련 있는 핵심 정보, 특히 수치 데이터, 통계, 주요 사실들을 정확하게 추출하고 요약해주세요.

            사용자 질문: "{user_query}"

            원본 텍스트 (최대 15000자):
            {web_content}

            핵심 정보 요약:"""
        )
        chain = prompt | extractor_llm | StrOutputParser()

        extracted_info = chain.invoke({
            "user_query": query,
            "web_content": content[:15000]
        })

        print(f"  ✓ LLM 분석 완료")
        return extracted_info
    except Exception as e:
        return f"LLM 분석 중 오류 발생: {e}"


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
    - 사용 시점: 'A의 생산지는 어디야?'와 같이 개체 간의 연결 관계나 소속 정보(원산지 정보)가 필요할 때 사용합니다.
    """
    print(f"Neo4j Graph DB 검색 실행: {query}")
    try:
        # 이벤트 루프가 이미 실행 중인지 확인
        try:
            loop = asyncio.get_running_loop()
            # 이미 루프가 실행 중이면 thread pool에서 실행
            print(f"- 기존 이벤트 루프 감지됨, 별도 스레드에서 실행")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(neo4j_search_sync, query)
                return future.result()
        except RuntimeError:
            # 실행 중인 루프가 없으면 직접 동기 함수 호출
            print(f"- 동기 방식으로 Neo4j 검색 실행")
            return neo4j_search_sync(query)
    except Exception as e:
        print(f"- Graph DB 검색 중 오류 발생: {e}")
        return f"Graph DB 검색 중 오류: {e}"
