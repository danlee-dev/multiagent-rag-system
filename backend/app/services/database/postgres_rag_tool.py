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
                'database': os.getenv('DB_NAME', ''),
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
                if possible_item in query_text and possible_item not in items:
                    items.append(possible_item)
                    print(f"    → 폴백으로 '{possible_item}' 품목 추가")

        if not items:
            print(">> 검색할 품목이 없어서 영양소 검색 건너뜀")
            return []

        base_query = """
        SELECT
            "식품명" as "food_name",
            "식품군" as "food_group",
            "출처" as "source",
            "에너지 (kcal/100g)" as "energy_kcal",
            "수분 (g/100g)" as "moisture_g",
            "단백질 (g/100g)" as "protein_g",
            "지방 (g/100g)" as "fat_g",
            "회분 (g/100g)" as "ash_g",
            "탄수화물 (g/100g)" as "carbohydrate_g",
            "당류 (g/100g)" as "sugars_g",
            "자당 (g/100g)" as "sucrose_g",
            "포도당 (g/100g)" as "glucose_g",
            "과당 (g/100g)" as "fructose_g",
            "유당 (g/100g)" as "lactose_g",
            "맥아당 (g/100g)" as "maltose_g",
            "갈락토오스 (g/100g)" as "galactose_g",
            "총 식이섬유 (g/100g)" as "total_dietary_fiber_g",
            "수용성 식이섬유 (g/100g)" as "soluble_dietary_fiber_g",
            "불용성 식이섬유 (g/100g)" as "insoluble_dietary_fiber_g",
            "칼슘 (mg/100g)" as "calcium_mg",
            "철 (mg/100g)" as "iron_mg",
            "마그네슘 (mg/100g)" as "magnesium_mg",
            "인 (mg/100g)" as "phosphorus_mg",
            "칼륨 (mg/100g)" as "potassium_mg",
            "나트륨 (mg/100g)" as "sodium_mg",
            "아연 (mg/100g)" as "zinc_mg",
            "구리 (mg/100g)" as "copper_mg",
            "망간 (mg/100g)" as "manganese_mg",
            "셀레늄 (μg/100g)" as "selenium_ug",
            "몰리브덴 (μg/100g)" as "molybdenum_ug",
            "요오드 (μg/100g)" as "iodine_ug",
            "비타민 A (μg/100g)" as "vitamin_a_ug_rae",
            "레티놀 (μg/100g)" as "retinol_ug",
            "베타카로틴 (μg/100g)" as "beta_carotene_ug",
            "티아민 (mg/100g)" as "thiamin_mg",
            "리보플라빈 (mg/100g)" as "riboflavin_mg",
            "니아신 (mg/100g)" as "niacin_mg",
            "니아신당량(NE) (mg/100g)" as "niacin_eq_mg_ne",
            "니코틴산 (mg/100g)" as "nicotinic_acid_mg",
            "니코틴아미드 (mg/100g)" as "nicotinamide_mg",
            "판토텐산 (mg/100g)" as "pantothenic_acid_mg",
            "비타민 B6 (mg/100g)" as "vitamin_b6_mg",
            "피리독신 (mg/100g)" as "pyridoxine_mg",
            "비오틴 (μg/100g)" as "biotin_ug",
            "엽산_ 엽산당량 (μg/100g)" as "folate_ug_dfe",
            "엽산_ 식품 엽산 (μg/100g)" as "folate_food_ug",
            "엽산_ 합성 엽산 (μg/100g)" as "folate_synthetic_ug",
            "비타민 B12 (μg/100g)" as "vitamin_b12_ug",
            "비타민 C (mg/100g)" as "vitamin_c_mg",
            "비타민 D (μg/100g)" as "vitamin_d_ug",
            "비타민 D2 (μg/100g)" as "vitamin_d2_ug",
            "비타민 D3 (μg/100g)" as "vitamin_d3_ug",
            "비타민 E (mg/100g)" as "vitamin_e_mg_ate",
            "알파 토코페롤 (mg/100g)" as "alpha_tocopherol_mg",
            "베타 토코페롤 (mg/100g)" as "beta_tocopherol_mg",
            "감마 토코페롤 (mg/100g)" as "gamma_tocopherol_mg",
            "델타 토코페롤 (mg/100g)" as "delta_tocopherol_mg",
            "알파 토코트리에놀 (mg/100g)" as "alpha_tocotrienol_mg",
            "베타 토코트리에놀 (mg/100g)" as "beta_tocotrienol_mg",
            "감마 토코트리에놀 (mg/100g)" as "gamma_tocotrienol_mg",
            "델타 토코트리에놀 (mg/100g)" as "delta_tocotrienol_mg",
            "비타민 K (μg/100g)" as "vitamin_k_ug",
            "비타민 K1 (μg/100g)" as "vitamin_k1_ug",
            "비타민 K2 (μg/100g)" as "vitamin_k2_ug",
            "총 아미노산 (mg/100g)" as "total_amino_acids_mg",
            "총 필수 아미노산 (mg/100g)" as "total_essential_amino_acids_mg",
            "이소류신 (mg/100g)" as "isoleucine_mg",
            "류신 (mg/100g)" as "leucine_mg",
            "라이신 (mg/100g)" as "lysine_mg",
            "메티오닌 (mg/100g)" as "methionine_mg",
            "페닐알라닌 (mg/100g)" as "phenylalanine_mg",
            "트레오닌 (mg/100g)" as "threonine_mg",
            "트립토판 (mg/100g)" as "tryptophan_mg",
            "발린 (mg/100g)" as "valine_mg",
            "히스티딘 (mg/100g)" as "histidine_mg",
            "아르기닌 (mg/100g)" as "arginine_mg",
            "티로신 (mg/100g)" as "tyrosine_mg",
            "시스테인 (mg/100g)" as "cysteine_mg",
            "알라닌 (mg/100g)" as "alanine_mg",
            "아스파르트산 (mg/100g)" as "aspartic_acid_mg",
            "글루탐산 (mg/100g)" as "glutamic_acid_mg",
            "글라이신 (mg/100g)" as "glycine_mg",
            "프롤린 (mg/100g)" as "proline_mg",
            "세린 (mg/100g)" as "serine_mg",
            "타우린 (mg/100g)" as "taurine_mg",
            "콜레스테롤 (mg/100g)" as "cholesterol_mg",
            "총 지방산 (g/100g)" as "total_fatty_acids_g",
            "총 필수 지방산 (g/100g)" as "total_essential_fatty_acids_g",
            "총 포화 지방산 (g/100g)" as "total_saturated_fatty_acids_g",
            "부티르산 (4:0) (mg/100g)" as "butyric_acid_4_0_mg",
            "카프로산 (6:0) (mg/100g)" as "caproic_acid_6_0_mg",
            "카프릴산 (8:0) (mg/100g)" as "caprylic_acid_8_0_mg",
            "카프르산 (10:0) (mg/100g)" as "capric_acid_10_0_mg",
            "라우르산 (12:0) (mg/100g)" as "lauric_acid_12_0_mg",
            "트라이데칸산 (13:0) (mg/100g)" as "tridecanoic_acid_13_0_mg",
            "미리스트산 (14:0) (mg/100g)" as "myristic_acid_14_0_mg",
            "펜타데칸산 (15:0) (mg/100g)" as "pentadecanoic_acid_15_0_mg",
            "팔미트산 (16:0) (mg/100g)" as "palmitic_acid_16_0_mg",
            "헵타데칸산 (17:0) (mg/100g)" as "heptadecanoic_acid_17_0_mg",
            "스테아르산 (18:0) (mg/100g)" as "stearic_acid_18_0_mg",
            "아라키드산 (20:0) (mg/100g)" as "arachidic_acid_20_0_mg",
            "헨에이코산산 (21:0) (mg/100g)" as "heneicosanoic_acid_21_0_mg",
            "베헨산 (22:0) (mg/100g)" as "behenic_acid_22_0_mg",
            "트리코산산 (23:0) (mg/100g)" as "tricosanoic_acid_23_0_mg",
            "리그노세르산 (24:0) (mg/100g)" as "lignoceric_acid_24_0_mg",
            "총 불포화 지방산 (g/100g)" as "total_unsaturated_fatty_acids_g",
            "총 단일 불포화지방산 (g/100g)" as "total_monounsaturated_fatty_acids_g",
            "미리스톨레산 (14:1) (mg/100g)" as "myristoleic_acid_14_1_mg",
            "팔미톨레산 (16:1) (mg/100g)" as "palmitoleic_acid_16_1_mg",
            "헵타데센산 (17:1) (mg/100g)" as "heptadecenoic_acid_17_1_mg",
            "올레산 (18:1(n-9)) (mg/100g)" as "oleic_acid_18_1_n9_mg",
            "박센산 (18:1(n-7)) (mg/100g)" as "vaccenic_acid_18_1_n7_mg",
            "가돌레산 (20:1) (mg/100g)" as "gadoleic_acid_20_1_mg",
            "에루크산 (22:1) (mg/100g)" as "erucic_acid_22_1_mg",
            "네르본산 (24:1) (mg/100g)" as "nervonic_acid_24_1_mg",
            "총 다가 불포화지방산 (g/100g)" as "total_polyunsaturated_fatty_acids_g",
            "리놀레산 (18:2(n-6)) (mg/100g)" as "linoleic_acid_18_2_n6_mg",
            "알파 리놀렌산 (18:3 (n-3)) (mg/100g)" as "alpha_linolenic_acid_18_3_n3_mg",
            "감마 리놀렌산 (18:3 (n-6)) (mg/100g)" as "gamma_linolenic_acid_18_3_n6_mg",
            "에이코사 디에노산 (20:2(n-6)) (mg/100g)" as "eicosadienoic_acid_20_2_n6_mg",
            "디호모 리놀렌산 (20:3(n-3)) (mg/100g)" as "dihomo_linolenic_acid_20_3_n3_mg",
            "에이코사 트리에노산 (20:3(n-6)) (mg/100g)" as "eicosatrienoic_acid_20_3_n6_mg",
            "아라키돈산 (20:4(n-6)) (mg/100g)" as "arachidonic_acid_20_4_n6_mg",
            "에이코사 펜타에노산 (20:5(n-3)) (mg/100g)" as "eicosapentaenoic_acid_20_5_n3_mg",
            "도코사 디에노산(22:2) (mg/100g)" as "docosadienoic_acid_22_2_mg",
            "도코사 펜타에노산 (22:5(n-3)) (mg/100g)" as "docosapentaenoic_acid_22_5_n3_mg",
            "도코사 헥사에노산 (22:6(n-3)) (mg/100g)" as "docosahexaenoic_acid_22_6_n3_mg",
            "오메가3 지방산 (g/100g)" as "omega_3_fatty_acids_g",
            "오메가6 지방산 (g/100g)" as "omega_6_fatty_acids_g",
            "총 트랜스 지방산 (g/100g)" as "total_trans_fatty_acids_g",
            "트랜스 올레산(18:1(n-9)t) (mg/100g)" as "trans_oleic_acid_18_1_n9t_mg",
            "트랜스 리놀레산(18:2t) (mg/100g)" as "trans_linoleic_acid_18_2t_mg",
            "트랜스 리놀렌산(18:3t) (mg/100g)" as "trans_linolenic_acid_18_3t_mg",
            "식염상당량 (g/100g)" as "salt_equivalent_g",
            "폐기율 (%)" as "waste_rate_percent"
        FROM nutrition_facts
        WHERE "식품명" ILIKE ANY(%s)
        """

        like_patterns = [f"%{item}%" for item in items]
        query_params = (like_patterns,)

        base_query += " LIMIT 50"

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    print(f">> 영양소 검색 쿼리: {cursor.mogrify(base_query, query_params).decode('utf-8')}")
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
            id,
            regday,
            product_cls_code,
            product_cls_name,
            category_code,
            category_name,
            productno,
            lastest_day,
            "productName",
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
            direction,
            value
        FROM kamis_product_price_latest
        WHERE 1=1
        """

        where_conditions = []
        query_params = []

        if params.get('items'):
            item_conditions = []
            for item in params['items']:
                item_conditions.append("item_name ILIKE %s")
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
            summary += f"### 📈 가격 데이터 ({len(search_results['price_data'])}건)\n"
            for item in search_results['price_data'][:5]:
                # --- 1. 단기(전일 대비) 가격 정보 생성 (0: 하락, 1: 상승, 2: 변동없음)---
                direction_map = {0: '▼', 1: '▲', 2: '-'}
                direction_symbol = direction_map.get(item.get('direction'), '')

                price_now = item.get('dpr1')
                price_yesterday = item.get('dpr2')

                # 날짜 객체를 'YYYY-MM-DD' 형식의 문자열로 변환 (값이 없을 경우 대비)
                lastest_day_obj = item.get('lastest_day')
                display_date = lastest_day_obj.strftime('%Y-%m-%d') if lastest_day_obj else 'N/A'

                # 어제 대비 가격 변동 정보
                daily_trend_info = ""
                if price_now is not None and price_yesterday is not None and price_yesterday > 0:
                    price_diff = price_now - price_yesterday
                    if price_diff != 0:
                        daily_trend_info = f" (어제보다 {abs(price_diff):,}원 {direction_symbol})"

                # --- 2. 장기(월/년 단위) 가격 정보 생성 ---
                historical_info_parts = []
                price_month_ago = item.get('dpr3')
                price_year_ago = item.get('dpr4')

                # 1개월 전 대비
                if price_now is not None and price_month_ago is not None and price_month_ago > 0:
                    month_change_pct = ((price_now - price_month_ago) / price_month_ago) * 100
                    month_symbol = '▲' if month_change_pct > 0 else '▼'
                    historical_info_parts.append(f"1개월 전: {month_change_pct:+.1f}% {month_symbol}")

                # 1년 전 대비
                if price_now is not None and price_year_ago is not None and price_year_ago > 0:
                    year_change_pct = ((price_now - price_year_ago) / price_year_ago) * 100
                    year_symbol = '▲' if year_change_pct > 0 else '▼'
                    historical_info_parts.append(f"1년 전: {year_change_pct:+.1f}% {year_symbol}")

                # --- 3. 최종 요약 라인 조합 ---
                # 메인 정보 (현재 가격, 전일 대비)
                summary += (
                    f"- **{item.get('item_name', 'N/A')}** ({item.get('category_name', 'N/A')}): "
                    f"**{price_now:,}원**/{item.get('unit', 'N/A')} "
                    f"[{display_date} 기준]{daily_trend_info}\n"
                )

                # 추가 정보 (장기 추세)
                if historical_info_parts:
                    summary += f"    - `추세: {' | '.join(historical_info_parts)}`\n"

        if len(search_results['price_data']) > 5:
            summary += f"... 외 {len(search_results['price_data']) - 5}건\n"
        summary += "\n"

        if search_results['nutrition_data']:
            summary += f"### 🥗 영양 정보 ({len(search_results['nutrition_data'])}건)\n"
        
            # 1. (개선) 동의어를 그룹으로 묶어 중복을 제거한 매핑 테이블
            NUTRIENT_MAP = {
                # DB 컬럼명: {표시 이름, 동의어 리스트, 단위}
                'energy_kcal': {'display': '에너지(칼로리)', 'keywords': ['에너지', '칼로리'], 'unit': 'kcal'},
                'moisture_g': {'display': '수분', 'keywords': ['수분'], 'unit': 'g'},
                'protein_g': {'display': '단백질', 'keywords': ['단백질'], 'unit': 'g'},
                'fat_g': {'display': '지방', 'keywords': ['지방'], 'unit': 'g'},
                'carbohydrate_g': {'display': '탄수화물', 'keywords': ['탄수화물'], 'unit': 'g'},
                'sugars_g': {'display': '당류', 'keywords': ['당류'], 'unit': 'g'},
                'glucose_g': {'display': '포도당', 'keywords': ['포도당'], 'unit': 'g'},
                'fructose_g': {'display': '과당', 'keywords': ['과당'], 'unit': 'g'},
                'total_dietary_fiber_g': {'display': '식이섬유', 'keywords': ['식이섬유', '총식이섬유'], 'unit': 'g'},
                'calcium_mg': {'display': '칼슘', 'keywords': ['칼슘'], 'unit': 'mg'},
                'iron_mg': {'display': '철(철분)', 'keywords': ['철', '철분'], 'unit': 'mg'},
                'magnesium_mg': {'display': '마그네슘', 'keywords': ['마그네슘'], 'unit': 'mg'},
                'potassium_mg': {'display': '칼륨', 'keywords': ['칼륨'], 'unit': 'mg'},
                'sodium_mg': {'display': '나트륨', 'keywords': ['나트륨'], 'unit': 'mg'},
                'vitamin_a_ug_rae': {'display': '비타민 A', 'keywords': ['비타민A', '비타민 A'], 'unit': 'μg'},
                'vitamin_b6_mg': {'display': '비타민 B6', 'keywords': ['비타민B6', '비타민 B6'], 'unit': 'mg'},
                'vitamin_b12_ug': {'display': '비타민 B12', 'keywords': ['비타민B12', '비타민 B12'], 'unit': 'μg'},
                'vitamin_c_mg': {'display': '비타민 C', 'keywords': ['비타민C', '비타민 C'], 'unit': 'mg'},
                'vitamin_d_ug': {'display': '비타민 D', 'keywords': ['비타민D', '비타민 D'], 'unit': 'μg'},
                'vitamin_d2_ug': {'display': '비타민 D2', 'keywords': ['비타민D2', '비타민 D2'], 'unit': 'μg'},
                'vitamin_d3_ug': {'display': '비타민 D3', 'keywords': ['비타민D3', '비타민 D3'], 'unit': 'μg'},
                'vitamin_e_mg_ate': {'display': '비타민 E', 'keywords': ['비타민E', '비타민 E'], 'unit': 'mg'},
                'vitamin_k_ug': {'display': '비타민 K', 'keywords': ['비타민K', '비타민 K'], 'unit': 'μg'},
                'vitamin_k1_ug': {'display': '비타민 K1', 'keywords': ['비타민K1', '비타민 K1'], 'unit': 'μg'},
                'vitamin_k2_ug': {'display': '비타민 K2', 'keywords': ['비타민K2', '비타민 K2'], 'unit': 'μg'},
            }

            params = search_results.get('extracted_params', {})
            specific_info = [info.replace(" ", "") for info in params.get('specific_info', [])]

            for item in search_results['nutrition_data']:
                summary += f"- **{item.get('food_name', 'N/A')}** ({item.get('food_group', 'N/A')}):\n"
            
                highlighted_summaries = []
                processed_nutrients = set() # 이미 처리된 영양성분 기록 (중복 출력 방지)
            
                # 2. (개선) 새로운 MAP 구조에 맞춰 사용자 요청 정보 확인
                if specific_info:
                    for db_key, nutrient_details in NUTRIENT_MAP.items():
                        # 이미 요약에 추가된 성분이면 건너뛰기
                        if db_key in processed_nutrients:
                            continue
                    
                        # 사용자가 요청한 키워드가 현재 영양성분의 동의어 목록에 있는지 확인
                        for keyword in nutrient_details['keywords']:
                            if keyword.replace(" ", "") in specific_info:
                                value = item.get(db_key)
                                if value is not None:
                                    display_name = nutrient_details['display']
                                    unit = nutrient_details['unit']
                                    highlighted_summaries.append(f"**{display_name}**: {value}{unit}/100g")
                                    processed_nutrients.add(db_key) # 처리된 것으로 기록
                                    break # 해당 영양성분 처리가 끝났으므로 다음 MAP 항목으로 이동

                # 3. 강조할 정보가 있으면 그것만 보여주고, 없으면 기본 정보를 보여줌
                if highlighted_summaries:
                    summary += "    - " + " | ".join(highlighted_summaries) + "\n"
                else:
                    # 기본으로 보여줄 주요 영양 정보 (에너지, 단백질, 지방, 탄수화물, 당류)
                    energy = item.get('energy_kcal', 'N/A')
                    protein = item.get('protein_g', 'N/A')
                    fat = item.get('fat_g', 'N/A')
                    carb = item.get('carbohydrate_g', 'N/A')
                    sugar = item.get('sugars_g', 'N/A')
                
                    summary += (
                        f"    - **주요성분**: "
                        f"칼로리 {energy}kcal/100g | "
                        f"단백질 {protein}g/100g | "
                        f"지방 {fat}g/100g | "
                        f"탄수화물 {carb}g/100g | "
                        f"당류 {sugar}g/100g\n"
                    )
                
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
