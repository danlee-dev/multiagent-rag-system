import os
import json
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from openai import OpenAI
import typing
from typing import Dict, Any, List
from datetime import datetime, timedelta

class SmartKeywordExtractor:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def extract_search_params(self, query: str) -> Dict[str, Any]:
        """LLM을 사용하여 검색 파라미터 추출"""

        prompt = f"""
다음 질문을 분석하여 농수산물 데이터베이스 검색에 필요한 정보를 JSON 형태로 추출해주세요.

질문: {query}

다음 형태의 JSON으로 응답해주세요:
{{
    "search_type": "price|nutrition|trade|news|general",
    "items": ["품목1", "품목2"],
    "regions": ["지역1", "지역2"],
    "time_period": "recent|today|this_week|this_month|this_year|specific_date",
    "specific_info": ["가격", "영양성분", "칼로리", "비타민" 등 구체적 정보],
    "search_intent": "사용자가 원하는 정보를 한 문장으로 요약"
}}

## 중요한 품목 추출 규칙
1. **완전한 품목명을 정확히 추출**: "귀리 영양성분" → items: ["귀리"]
2. **농수산물 품목명만 추출**: 귀리, 사과, 감자, 쌀, 보리, 옥수수, 콩, 당근, 양파, 배추, 무, 고구마, 딸기, 포도, 복숭아, 수박, 참외, 호박, 오이, 토마토, 상추, 시금치, 깻잎, 마늘, 생강, 파, 대파, 쪽파, 부추, 고추, 피망, 감귤, 귤, 바나나, 키위, 망고 등
3. **명시적으로 언급된 품목만 추출**: 질문에 없는 품목은 추가하지 말 것

## 예시
- "귀리 영양성분이 궁금해요" → items: ["귀리"], search_type: "nutrition"
- "사과 가격 알려주세요" → items: ["사과"], search_type: "price"
- "오늘 양파 시세는?" → items: ["양파"], search_type: "price", time_period: "today"
- "농산물 영양정보" → items: [], search_type: "nutrition" (구체적 품목 없음)

## 절대 금지사항
- "귀리"를 "파"로 잘못 추출하는 것
- 품목명의 일부만 추출하는 것
- 질문에 없는 품목을 임의로 추가하는 것

JSON만 응답하고 다른 설명은 하지 마세요.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            result = json.loads(response.choices[0].message.content)
            print(f">> LLM 추출 결과: {result}")
            return result

        except Exception as e:
            print(f">> LLM 키워드 추출 실패: {e}")
            return {
                "search_type": "general",
                "items": [],
                "regions": [],
                "time_period": "recent",
                "specific_info": [],
                "search_intent": query
            }

class PostgreSQLRAG:
    def __init__(self):
        """PostgreSQL 연결 풀 초기화"""
        self.pool = None
        self.keyword_extractor = SmartKeywordExtractor()
        self._init_connection_pool()

    def _init_connection_pool(self):
        """PostgreSQL 연결 풀 초기화"""
        try:
            db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': os.getenv('DB_PORT', '5432'),
                'database': os.getenv('DB_NAME', 'your_database'),
                'user': os.getenv('DB_USER', 'your_user'),
                'password': os.getenv('DB_PASSWORD', 'your_password')
            }

            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=20,
                **db_config
            )
            print(">> PostgreSQL 연결 풀 초기화 완료")

        except Exception as e:
            print(f">> PostgreSQL 연결 풀 초기화 실패: {e}")
            self.pool = None

    @contextmanager
    def get_connection(self):
        """DB 연결 컨텍스트 매니저"""
        if not self.pool:
            raise Exception("DB 연결 풀이 초기화되지 않음")

        conn = None
        try:
            conn = self.pool.getconn()
            conn.cursor_factory = RealDictCursor
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                self.pool.putconn(conn)

    def __del__(self):
        """소멸자에서 연결 풀 정리"""
        if self.pool:
            self.pool.closeall()
            print(">> PostgreSQL 연결 풀 정리 완료")

    def _extract_search_params(self, query: str) -> Dict[str, Any]:
        """스마트 키워드 추출"""
        llm_result = self.keyword_extractor.extract_search_params(query)

        params = {
            'items': llm_result.get('items', []),
            'regions': llm_result.get('regions', []),
            'date_range': self._parse_time_period(llm_result.get('time_period')),
            'price_range': None,
            'sort_by': 'date_desc',
            'limit': 100,
            'search_type': llm_result.get('search_type', 'general'),
            'specific_info': llm_result.get('specific_info', []),
            'search_intent': llm_result.get('search_intent', query)
        }

        return params

    def _parse_time_period(self, time_period: str) -> tuple:
        """시간 범위 파싱"""
        now = datetime.now()

        if time_period == "today":
            return (now.replace(hour=0, minute=0, second=0), now)
        elif time_period == "this_week":
            week_start = now - timedelta(days=now.weekday())
            return (week_start, now)
        elif time_period == "this_month":
            month_start = now.replace(day=1, hour=0, minute=0, second=0)
            return (month_start, now)
        elif time_period == "recent":
            return (now - timedelta(days=30), now)
        else:
            return None

    def smart_search(self, query: str) -> Dict[str, Any]:
        """스마트 검색 수행"""
        print(f"\n>> 스마트 RDB 검색 시작: {query}")

        params = self._extract_search_params(query)
        print(f">> 추출된 파라미터: {params}")

        results = {
            'query': query,
            'extracted_params': params,
            'price_data': [],
            'nutrition_data': [],
            'trade_data': [],
            'news_data': [],
            'total_results': 0
        }

        search_type = params.get('search_type', 'general')

        if search_type == 'nutrition' or '영양' in query or '칼로리' in query:
            print(">> 영양 정보 우선 검색")
            results['nutrition_data'] = self.search_nutrition_data_smart(params)

        if search_type == 'price' or '가격' in query or '시세' in query:
            print(">> 가격 정보 우선 검색")
            results['price_data'] = self.search_price_data_smart(params)

        if search_type == 'general':
            results['nutrition_data'] = self.search_nutrition_data_smart(params)
            results['price_data'] = self.search_price_data_smart(params)

        results['total_results'] = (
            len(results['price_data']) +
            len(results['nutrition_data']) +
            len(results['trade_data']) +
            len(results['news_data'])
        )

        print(f">> 검색 완료 - 총 {results['total_results']}건")
        return results

    def search_nutrition_data_smart(self, params: Dict[str, Any]) -> List[Dict]:
        """스마트 영양소 정보 검색"""
        items = params.get('items', [])

        # items가 비어있으면 쿼리에서 품목명 추출 시도
        if not items:
            query_text = params.get('search_intent', '')
            # 농수산물 품목명 직접 검색 (더 포괄적으로)
            food_items = ['귀리', '사과', '감자', '쌀', '보리', '옥수수', '콩', '당근', '양파', '배추', '무', '고구마',
                         '딸기', '포도', '복숭아', '수박', '참외', '호박', '오이', '토마토', '상추', '시금치',
                         '깻잎', '마늘', '생강', '파', '대파', '쪽파', '부추', '고추', '피망', '파프리카',
                         '감귤', '귤', '오렌지', '바나나', '키위', '망고', '배', '자두', '체리']

            for possible_item in food_items:
                if possible_item in query_text:
                    items.append(possible_item)
                    print(f"    → 폴백으로 '{possible_item}' 품목 추가")
                    break

        if not items:
            print(">> 검색할 품목이 없어서 영양소 검색 건너뜀")
            return []

        base_query = """
        SELECT
            식품명,
            식품군,
            출처,
            "에너지 (kcal/100g)" as 칼로리,
            "수분 (g/100g)" as 수분,
            "단백질 (g/100g)" as 단백질,
            "지방 (g/100g)" as 지방,
            "탄수화물 (g/100g)" as 탄수화물,
            "총 식이섬유 (g/100g)" as 식이섬유,
            "당류 (g/100g)" as 당류,
            "칼슘 (mg/100g)" as 칼슘,
            "인 (mg/100g)" as 인,
            "철 (mg/100g)" as 철,
            "나트륨 (mg/100g)" as 나트륨,
            "칼륨 (mg/100g)" as 칼륨,
            "마그네슘 (mg/100g)" as 마그네슘,
            "아연 (mg/100g)" as 아연,
            "비타민 A (μg/100g)" as 비타민A,
            "베타카로틴 (μg/100g)" as 베타카로틴,
            "티아민 (mg/100g)" as 비타민B1,
            "리보플라빈 (mg/100g)" as 비타민B2,
            "비타민 B6 (mg/100g)" as 비타민B6,
            "비타민 C (mg/100g)" as 비타민C,
            "비타민 E (mg/100g)" as 비타민E,
            "엽산_ 엽산당량 (μg/100g)" as 엽산,
            "총 포화 지방산 (g/100g)" as 포화지방산,
            "총 단일 불포화지방산 (g/100g)" as 단일불포화지방산,
            "총 다가 불포화지방산 (g/100g)" as 다가불포화지방산,
            "콜레스테롤 (mg/100g)" as 콜레스테롤
        FROM nutrition_facts
        WHERE 1=1
        """

        where_conditions = []
        query_params = []

        if items:
            item_conditions = []
            for item in items:
                item_conditions.append("식품명 ILIKE %s")
                query_params.append(f"%{item}%")

            where_conditions.append(f"({' OR '.join(item_conditions)})")

        if where_conditions:
            base_query += " AND " + " AND ".join(where_conditions)

        base_query += " LIMIT 50"

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    print(f">> 영양소 검색 쿼리: {base_query}")
                    print(f">> 파라미터: {query_params}")

                    cursor.execute(base_query, query_params)
                    results = cursor.fetchall()

                    print(f">> 영양소 검색 결과: {len(results)}건")
                    return [dict(row) for row in results]

        except Exception as e:
            print(f">> 영양소 데이터 검색 오류: {e}")
            return []

    def search_price_data_smart(self, params: Dict[str, Any]) -> List[Dict]:
        """스마트 가격 데이터 검색"""
        base_query = """
        SELECT
            product_cls_name,
            category_name,
            regday,
            product_cls_code,
            category_code,
            item_name,
            unit,
            day1,
            dpr1,
            day2,
            dpr2,
            day3,
            dpr3,
            day4,
            dpr4,
            value,
            id
        FROM kamis_product_price_latest
        WHERE 1=1
        """

        where_conditions = []
        query_params = []

        if params.get('items'):
            item_conditions = []
            for item in params['items']:
                item_conditions.append("product_cls_name ILIKE %s")
                query_params.append(f"%{item}%")

            if item_conditions:
                where_conditions.append(f"({' OR '.join(item_conditions)})")

        if params.get('regions'):
            region_conditions = []
            for region in params['regions']:
                region_conditions.append("category_name ILIKE %s")
                query_params.append(f"%{region}%")

            if region_conditions:
                where_conditions.append(f"({' OR '.join(region_conditions)})")

        if params.get('date_range'):
            where_conditions.append("regday >= %s AND regday <= %s")
            query_params.extend([d.date() for d in params['date_range']])

        if params.get('price_range'):
            if len(params['price_range']) == 1:
                where_conditions.append("value <= %s")
                query_params.append(params['price_range'][0])
            elif len(params['price_range']) == 2:
                where_conditions.append("value BETWEEN %s AND %s")
                query_params.extend(params['price_range'])

        if where_conditions:
            base_query += " AND " + " AND ".join(where_conditions)

        sort_by = params.get('sort_by', 'date_desc')
        if sort_by == 'price_desc':
            base_query += " ORDER BY value DESC"
        elif sort_by == 'price_asc':
            base_query += " ORDER BY value ASC"
        elif sort_by == 'date_desc':
            base_query += " ORDER BY regday DESC"
        else:
            base_query += " ORDER BY regday DESC"

        limit = params.get('limit', 100)
        base_query += f" LIMIT {limit}"

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    print(f">> 실행 쿼리: {base_query}")
                    print(f">> 파라미터: {query_params}")

                    cursor.execute(base_query, query_params)
                    results = cursor.fetchall()

                    print(f">> 검색 결과: {len(results)}건")
                    return [dict(row) for row in results]

        except Exception as e:
            print(f">> 가격 데이터 검색 오류: {e}")
            return []

    def search_nutrition_data(self, items: List[str]) -> List[Dict]:
        """영양소 정보 검색"""
        if not items:
            return []

        query = """
        SELECT
            식품군,
            식품명,
            출처
        FROM nutrition_facts
        WHERE 식품명 ILIKE ANY(%s)
        LIMIT 50
        """

        like_patterns = [f"%{item}%" for item in items]

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (like_patterns,))
                    results = cursor.fetchall()
                    return [dict(row) for row in results]

        except Exception as e:
            print(f">> 영양소 데이터 검색 오류: {e}")
            return []

    def search_trade_data(self, params: Dict[str, Any]) -> List[Dict]:
        """수출입 통계 검색"""
        query = """
        SELECT
            item_name,
            trade_type,
            country,
            quantity_kg,
            value_usd,
            DATE(trade_date) as date
        FROM trade_statistics
        WHERE 1=1
        """

        where_conditions = []
        query_params = []

        if params.get('items'):
            placeholders = ','.join(['%s'] * len(params['items']))
            where_conditions.append(f"item_name IN ({placeholders})")
            query_params.extend(params['items'])

        if params.get('date_range'):
            where_conditions.append("trade_date >= %s AND trade_date <= %s")
            query_params.extend(params['date_range'])

        if where_conditions:
            query += " AND " + " AND ".join(where_conditions)

        query += " ORDER BY trade_date DESC LIMIT 100"

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, query_params)
                    results = cursor.fetchall()
                    return [dict(row) for row in results]

        except Exception as e:
            print(f">> 무역 데이터 검색 오류: {e}")
            return []

    def search_news_metadata(self, keywords: List[str]) -> List[Dict]:
        """뉴스 메타데이터 검색"""
        if not keywords:
            return []

        query = """
        SELECT
            title,
            url,
            published_date,
            source,
            keywords,
            sentiment_score
        FROM news_metadata
        WHERE title ILIKE ANY(%s)
        ORDER BY published_date DESC
        LIMIT 50
        """

        like_patterns = [f"%{keyword}%" for keyword in keywords]

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (like_patterns,))
                    results = cursor.fetchall()
                    return [dict(row) for row in results]

        except Exception as e:
            print(f">> 뉴스 메타데이터 검색 오류: {e}")
            return []

    def comprehensive_search(self, query: str) -> Dict[str, Any]:
        """종합 검색 수행"""
        print(f"\n>> 포괄적 RDB 검색 시작: {query}")

        params = self._extract_search_params(query)
        print(f">> 추출된 파라미터: {params}")

        results = {
            'query': query,
            'extracted_params': params,
            'price_data': [],
            'nutrition_data': [],
            'trade_data': [],
            'news_data': [],
            'total_results': 0
        }

        if '가격' in query or '시세' in query or params.get('items'):
            results['price_data'] = self.search_price_data_smart(params)

        if '영양' in query or '칼로리' in query or '비타민' in query:
            results['nutrition_data'] = self.search_nutrition_data_smart(params)

        if '수출' in query or '수입' in query or '무역' in query:
            results['trade_data'] = self.search_trade_data(params)

        if '뉴스' in query or '기사' in query:
            results['news_data'] = self.search_news_metadata(params.get('items', []))

        results['total_results'] = (
            len(results['price_data']) +
            len(results['nutrition_data']) +
            len(results['trade_data']) +
            len(results['news_data'])
        )

        print(f">> 검색 완료 - 총 {results['total_results']}건")
        return results

# PostgreSQL RAG 인스턴스 생성
postgres_rag = PostgreSQLRAG()

def postgres_rdb_search(query: str) -> str:
    """
    PostgreSQL DB에 저장된 정형 데이터를 조회하여 정확한 수치나 통계를 제공합니다.
    - 사용 시점:
      1. 구체적인 품목, 날짜, 지역 등의 조건으로 정확한 데이터를 찾을 때
      2. 영양 정보, 수급량 등 명확한 스펙이나 통계 수치를 물을 때
      3. 특정 기간의 데이터 순위나 추이를 알고 싶을 때
    - 데이터 종류: 농산물 시세, 원산지, 영양소 정보, 수출입량 통계, 뉴스 메타데이터
    """
    try:
        search_results = postgres_rag.comprehensive_search(query)

        summary = f"PostgreSQL 검색 결과 (총 {search_results['total_results']}건):\n\n"

        if search_results['price_data']:
            summary += f"### 가격 데이터 ({len(search_results['price_data'])}건)\n"
            for item in search_results['price_data'][:5]:
                summary += f"- {item.get('product_cls_name', 'N/A')} ({item.get('category_name', 'N/A')}): {item.get('value', 'N/A')}원/{item.get('unit', 'kg')} [{item.get('regday', 'N/A')}]\n"
            if len(search_results['price_data']) > 5:
                summary += f"... 외 {len(search_results['price_data'])-5}건\n"
            summary += "\n"

        if search_results['nutrition_data']:
            summary += f"### 영양 정보 ({len(search_results['nutrition_data'])}건)\n"
            for item in search_results['nutrition_data']:
                summary += f"- {item.get('식품명', 'N/A')} ({item.get('식품군', 'N/A')}): 출처 - {item.get('출처', 'N/A')}\n"
            summary += "\n"

        if search_results['trade_data']:
            summary += f"### 수출입 통계 ({len(search_results['trade_data'])}건)\n"
            for item in search_results['trade_data'][:3]:
                summary += f"- {item.get('item_name', 'N/A')} {item.get('trade_type', 'N/A')} ({item.get('country', 'N/A')}): {item.get('quantity_kg', 'N/A')}kg, ${item.get('value_usd', 'N/A')}\n"
            summary += "\n"

        if search_results['news_data']:
            summary += f"### 관련 뉴스 ({len(search_results['news_data'])}건)\n"
            for item in search_results['news_data'][:3]:
                summary += f"- [{item.get('source', 'N/A')}] {item.get('title', 'N/A')} [{item.get('published_date', 'N/A')}]\n"
            summary += "\n"

        summary += "### 상세 데이터 (JSON)\n"
        summary += json.dumps(search_results, ensure_ascii=False, indent=2, default=str)

        return summary

    except Exception as e:
        error_msg = f"PostgreSQL 검색 중 오류 발생: {str(e)}"
        print(f">> {error_msg}")
        return error_msg
