import os
import json
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import google.generativeai as genai
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# -------------------------------------------
# KeywordExtractor (2-Stage: route -> extract)
# -------------------------------------------
class KeywordExtractor:
    """
    1) 라우팅 LLM: 어떤 테이블을 볼지와 search_type을 결정
    2) 컬럼 인지 LLM: 선택된 테이블의 실제 컬럼명/표시명만 사용해 파라미터를 추출
    - 실패 시 휴리스틱 폴백을 사용
    """

    # 스키마 메타 (스크린샷 기준 요약)
    TABLE_SCHEMAS: Dict[str, Dict[str, Any]] = {
        "kamis_product_price_latest": {
            "key_columns": [
                "regday", "category_name", "item_name", "unit",
                "dpr1", "dpr2", "dpr3", "dpr4", "direction", "value"
            ],
            "display_to_alias": {
                "가격": "value",
                "현재가격": "dpr1",
                "어제가격": "dpr2",
                "1개월전": "dpr3",
                "1년전": "dpr4",
                "단위": "unit",
                "품목": "item_name",
                "품목명": "item_name",
                "분류": "category_name",
                "날짜": "regday"
            }
        },
        # 영양은 foods + 세부 5테이블 조합
        "foods": {
            "key_columns": ["식품군", "식품명", "출처", "폐기율_percent"],
        },
        "proximates": {
            "key_columns": [
                "에너지 (kcal/100g)","수분 (g/100g)","단백질 (g/100g)","지방 (g/100g)",
                "회분 (g/100g)","탄수화물 (g/100g)","당류 (g/100g)","자당 (g/100g)",
                "포도당 (g/100g)","과당 (g/100g)","유당 (g/100g)","맥아당 (g/100g)",
                "갈락토오스 (g/100g)","총 식이섬유 (g/100g)","수용성 식이섬유 (g/100g)",
                "불용성 식이섬유 (g/100g)","식염상당량 (g/100g)"
            ],
        },
        "minerals": {
            "key_columns": [
                "칼슘 (mg/100g)","철 (mg/100g)","마그네슘 (mg/100g)","인 (mg/100g)",
                "칼륨 (mg/100g)","나트륨 (mg/100g)","아연 (mg/100g)","구리 (mg/100g)",
                "망간 (mg/100g)","셀레늄 (μg/100g)","몰리브덴 (μg/100g)","요오드 (μg/100g)"
            ],
        },
        "vitamins": {
            "key_columns": [
                "비타민 A (μg/100g)","레티놀 (μg/100g)","베타카로틴 (μg/100g)",
                "티아민 (mg/100g)","리보플라빈 (mg/100g)","니아신 (mg/100g)",
                "니아신당량(NE) (mg/100g)","니코틴산 (mg/100g)","니코틴아미드 (mg/100g)",
                "판토텐산 (mg/100g)","비타민 B6 (mg/100g)","피리독신 (mg/100g)",
                "비오틴 (μg/100g)","엽산_ 엽산당량 (μg/100g)","엽산_ 식품 엽산 (μg/100g)",
                "엽산_ 합성 엽산 (μg/100g)","비타민 B12 (μg/100g)","비타민 C (mg/100g)",
                "비타민 D (μg/100g)","비타민 D2 (μg/100g)","비타민 D3 (μg/100g)",
                "비타민 E (mg/100g)","알파 토코페롤 (mg/100g)","베타 토코페롤 (mg/100g)",
                "감마 토코페롤 (mg/100g)","델타 토코페롤 (mg/100g)",
                "알파 토코트리에놀 (mg/100g)","베타 토코트리에놀 (mg/100g)",
                "감마 토코트리에놀 (mg/100g)","델타 토코트리에놀 (mg/100g)",
                "비타민 K (μg/100g)","비타민 K1 (μg/100g)","비타민 K2 (μg/100g)"
            ],
        },
        "amino_acids": {
            "key_columns": [
                "총 아미노산 (mg/100g)","총 필수 아미노산 (mg/100g)","이소류신 (mg/100g)",
                "류신 (mg/100g)","라이신 (mg/100g)","메티오닌 (mg/100g)","페닐알라닌 (mg/100g)",
                "트레오닌 (mg/100g)","트립토판 (mg/100g)","발린 (mg/100g)","히스티딘 (mg/100g)",
                "아르기닌 (mg/100g)","티로신 (mg/100g)","시스테인 (mg/100g)","알라닌 (mg/100g)",
                "아스파르트산 (mg/100g)","글루탐산 (mg/100g)","글라이신 (mg/100g)",
                "프롤린 (mg/100g)","세린 (mg/100g)","타우린 (mg/100g)"
            ],
        },
        "fatty_acids": {
            "key_columns": [
                "콜레스테롤 (mg/100g)","총 지방산 (g/100g)","총 필수 지방산 (g/100g)",
                "총 포화 지방산 (g/100g)","부티르산 (4:0) (mg/100g)","카프로산 (6:0) (mg/100g)",
                "카프릴산 (8:0) (mg/100g)","카프르산 (10:0) (mg/100g)","라우르산 (12:0) (mg/100g)",
                "트라이데칸산 (13:0) (mg/100g)","미리스트산 (14:0) (mg/100g)","펜타데칸산 (15:0) (mg/100g)",
                "팔미트산 (16:0) (mg/100g)","헵타데칸산 (17:0) (mg/100g)","스테아르산 (18:0) (mg/100g)",
                "아라키드산 (20:0) (mg/100g)","헨에이코산산 (21:0) (mg/100g)","베헨산 (22:0) (mg/100g)",
                "트리코산산 (23:0) (mg/100g)","리그노세르산 (24:0) (mg/100g)",
                "총 불포화 지방산 (g/100g)","총 단일 불포화지방산 (g/100g)",
                "미리스톨레산 (14:1) (mg/100g)","팔미톨레산 (16:1) (mg/100g)","헵타데센산 (17:1) (mg/100g)",
                "올레산 (18:1(n-9)) (mg/100g)","박센산 (18:1(n-7)) (mg/100g)","가돌레산 (20:1) (mg/100g)",
                "에루크산 (22:1) (mg/100g)","네르본산 (24:1) (mg/100g)","총 다가 불포화지방산 (g/100g)",
                "리놀레산 (18:2(n-6)) (mg/100g)","알파 리놀렌산 (18:3 (n-3)) (mg/100g)",
                "감마 리놀렌산 (18:3 (n-6)) (mg/100g)","에이코사 디에노산 (20:2(n-6)) (mg/100g)",
                "디호모 리놀렌산 (20:3(n-3)) (mg/100g)","에이코사 트리에노산 (20:3(n-6)) (mg/100g)",
                "아라키돈산 (20:4(n-6)) (mg/100g)","에이코사 펜타에노산 (20:5(n-3)) (mg/100g)",
                "도코사 디에노산(22:2) (mg/100g)","도코사 펜타에노산 (22:5(n-3)) (mg/100g)",
                "도코사 헥사에노산 (22:6(n-3)) (mg/100g)","오메가3 지방산 (g/100g)",
                "오메가6 지방산 (g/100g)","총 트랜스 지방산 (g/100g)",
                "트랜스 올레산(18:1(n-9)t) (mg/100g)","트랜스 리놀레산(18:2t) (mg/100g)",
                "트랜스 리놀렌산(18:3t) (mg/100g)"
            ],
        },
    }

    # 검색 타입 키워드 폴백
    _PRICE_HINTS = ("가격", "시세", "얼마", "비싸", "dpr", "kamis")
    _NUTR_HINTS  = ("영양", "칼로리", "비타민", "미네랄", "아미노산", "지방산", "nutri")

    # 자연어 동의어 → 영문 키(검색 결과 요약에서 사용하는 alias) 매핑
    SPECIFIC_SYNONYM_TO_RESULT_KEY = {
        # 주요 10여개만 우선 매핑 (필요 시 추가)
        "칼로리": "energy_kcal", "에너지": "energy_kcal",
        "단백질": "protein_g", "지방": "fat_g", "탄수화물": "carbohydrate_g", "당류": "sugars_g",
        "칼슘": "calcium_mg", "철": "iron_mg", "철분": "iron_mg",
        "나트륨": "sodium_mg", "칼륨": "potassium_mg", "마그네슘": "magnesium_mg",
        "비타민a": "vitamin_a_ug_rae", "비타민 a": "vitamin_a_ug_rae",
        "비타민c": "vitamin_c_mg", "비타민 c": "vitamin_c_mg",
        "비타민d": "vitamin_d_ug", "비타민 d": "vitamin_d_ug",
        "비타민e": "vitamin_e_mg_ate", "비타민 e": "vitamin_e_mg_ate",
        "비타민k": "vitamin_k_ug", "비타민 k": "vitamin_k_ug",
        "b6": "vitamin_b6_mg", "비타민b6": "vitamin_b6_mg",
        "b12": "vitamin_b12_ug", "비타민b12": "vitamin_b12_ug",
        "오메가3": "omega_3_fatty_acids_g", "오메가6": "omega_6_fatty_acids_g",
        "콜레스테롤": "cholesterol_mg",
        "식이섬유": "total_dietary_fiber_g",
        "류신": "leucine_mg", "라이신": "lysine_mg", "트립토판": "tryptophan_mg",
    }

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel("gemini-2.5-flash")

    # ---------- public ----------
    def extract_search_params(self, query: str) -> Dict[str, Any]:
        """LLM 2단계 + 폴백. 기존 반환 스키마 유지."""
        print(f"- LLM 키워드 추출 시작: {query}")

        # 1) 라우팅
        route = self._route_tables_with_llm(query)
        search_type = route.get("search_type", "general")
        chosen = route.get("chosen_tables", [])

        # 영양 검색이면 foods + 5테이블 강제 포함
        if search_type in ("nutrition", "general"):
            must = ["foods","proximates","minerals","vitamins","amino_acids","fatty_acids"]
            chosen = list(dict.fromkeys(chosen + must))

        # 2) 컬럼 인지 파라미터 추출
        params = self._extract_params_with_llm(query, chosen)

        # 3) 정규화/동의어 매핑/기본값
        result = self._normalize(query, search_type, params)
        print(f"- LLM 추출 성공: {result}")
        return result

    # ---------- stage 1 ----------
    def _route_tables_with_llm(self, query: str) -> Dict[str, Any]:
        prompt = f"""
너는 쿼리 라우터다. 질문을 보고 어떤 테이블을 조회할지 결정한다.

[테이블]
- 가격: kamis_product_price_latest
- 영양: foods + (proximates, minerals, vitamins, amino_acids, fatty_acids)  ← foods.id = 각 테이블.food_id 로 조인

다음 JSON만 응답:
{{
  "search_type": "price|nutrition|general",
  "chosen_tables": ["테이블명1","테이블명2", ...]
}}

규칙:
- '가격/시세' 문맥이면 search_type="price", chosen=["kamis_product_price_latest"].
- '영양/칼로리/비타민/미네랄/아미노산/지방산' 문맥이면 search_type="nutrition",
  chosen=["foods","proximates","minerals","vitamins","amino_acids","fatty_acids"].
- 애매하면 "general"로 하고 관련 테이블을 포괄적으로 포함.

질문: {query}
오직 JSON으로만 응답.
"""
        data = self._safe_json_from_llm(prompt)
        if not data:
            # 폴백 라우팅
            st = "nutrition" if any(k in query for k in self._NUTR_HINTS) else \
                 ("price" if any(k in query for k in self._PRICE_HINTS) else "general")
            chosen = (["kamis_product_price_latest"] if st == "price"
                      else ["foods","proximates","minerals","vitamins","amino_acids","fatty_acids"])
            return {"search_type": st, "chosen_tables": chosen}
        return data

    # ---------- stage 2 ----------
    def _extract_params_with_llm(self, query: str, chosen_tables: List[str]) -> Dict[str, Any]:
        picked = {t: self.TABLE_SCHEMAS.get(t, {}) for t in chosen_tables}
        table_cols = {t: v.get("key_columns", []) for t, v in picked.items()}
        price_display = self.TABLE_SCHEMAS["kamis_product_price_latest"].get("display_to_alias", {})

        prompt = f"""
너는 SQL 파라미터 추출기다. 아래 질문에서 '실제 컬럼이 존재하는 값'만 뽑아낸다.

[중요]
- 영양 검색은 반드시 foods."식품명"으로 품목을 매칭하고, foods.id = 각 테이블.food_id 로 조인한다.
- 가격 검색은 kamis_product_price_latest.item_name, category_name, regday 등을 사용한다.
- specific_info는 아래 컬럼 목록에 존재하는 항목(표시명 or 일반명)만 포함한다.
- 기간 언급 없으면 time_period="recent".

선택된 테이블과 주요 컬럼:
{json.dumps(table_cols, ensure_ascii=False, indent=2)}

가격 표시명→실제 컬럼:
{json.dumps(price_display, ensure_ascii=False, indent=2)}

반환 JSON 스키마(이 형식만):
{{
  "search_type": "price|nutrition|general",
  "items": ["foods.\"식품명\"에 들어갈 명시 품목명들"],
  "regions": ["지역명 목록 (가격일 때만 사용, 없으면 [])"],
  "time_period": "recent|today|this_week|this_month|this_year|specific_date",
  "specific_info": ["위 컬럼 목록에 있는 영양/지표 이름만", "..."],
  "search_intent": "사용자 의도를 한 문장으로 요약"
}}

질문: {query}
오직 JSON으로만 응답.
"""
        data = self._safe_json_from_llm(prompt)
        # 실패 시 폴백
        if not data:
            return self._get_fallback_result(query)
        return data

    # ---------- utils ----------
    def _safe_json_from_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        try:
            resp = self.client.generate_content(prompt)
            if not resp or not hasattr(resp, "text"):
                return None
            text = resp.text.strip()
            if not text:
                return None
            # 코드블록 제거
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif text.startswith("```"):
                text = text.split("```")[1].split("```")[0].strip()
            return json.loads(text)
        except Exception as e:
            print(f"- LLM JSON 파싱 실패: {e}")
            return None

    def _normalize(self, query: str, search_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        # 기본 골격
        items = data.get("items") or []
        regions = data.get("regions") or []
        time_period = data.get("time_period") or "recent"
        specific_info_raw = data.get("specific_info") or []
        intent = data.get("search_intent") or query

        # specific_info: 동의어 -> 결과 요약에서 쓰는 영문 alias 로 정리
        normalized_specific = []
        for s in specific_info_raw:
            key = s.strip().lower().replace(" ", "")
            mapped = self.SPECIFIC_SYNONYM_TO_RESULT_KEY.get(key)
            if mapped and mapped not in normalized_specific:
                normalized_specific.append(mapped)

        # search_type 폴백
        if not search_type or search_type not in ("price","nutrition","general"):
            if any(k in query for k in self._PRICE_HINTS):
                search_type = "price"
            elif any(k in query for k in self._NUTR_HINTS):
                search_type = "nutrition"
            else:
                search_type = "general"

        return {
            "search_type": search_type,
            "items": items,
            "regions": regions,
            "time_period": time_period,
            "specific_info": normalized_specific,  # 영문 alias로 정규화 (없으면 [])
            "search_intent": intent
        }

    def _get_fallback_result(self, query: str) -> Dict[str, Any]:
        # 아주 보수적 폴백: 타입만 유추
        st = "nutrition" if any(k in query for k in self._NUTR_HINTS) else \
             ("price" if any(k in query for k in self._PRICE_HINTS) else "general")
        return {
            "search_type": st,
            "items": [],
            "regions": [],
            "time_period": "recent",
            "specific_info": [],
            "search_intent": query
        }


# -----------------------------
# PostgreSQL RAG (동일 시그니처)
# -----------------------------
class PostgreSQLRAG:
    def __init__(self):
        self.pool = None
        self.keyword_extractor = KeywordExtractor()
        self._init_connection_pool()

    def _init_connection_pool(self):
        try:
            db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': os.getenv('DB_PORT', '5432'),
                'database': os.getenv('DB_NAME', ''),
                'user': os.getenv('DB_USER', 'your_user'),
                'password': os.getenv('DB_PASSWORD', 'your_password')
            }
            self.pool = psycopg2.pool.ThreadedConnectionPool(minconn=1, maxconn=20, **db_config)
            print(">> PostgreSQL 연결 풀 초기화 완료")
        except Exception as e:
            print(f">> PostgreSQL 연결 풀 초기화 실패: {e}")
            self.pool = None

    @contextmanager
    def get_connection(self):
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
        if self.pool:
            self.pool.closeall()
            print(">> PostgreSQL 연결 풀 정리 완료")

    # ---- helpers ----
    def _extract_search_params(self, query: str) -> Dict[str, Any]:
        llm_result = self.keyword_extractor.extract_search_params(query)
        params = {
            'items': llm_result.get('items', []),
            'regions': llm_result.get('regions', []),
            'date_range': self._parse_time_period(llm_result.get('time_period')),
            'price_range': None,
            'sort_by': 'date_desc',
            'limit': 100,
            'search_type': llm_result.get('search_type', 'general'),
            'specific_info': llm_result.get('specific_info', []),  # 이미 정규화된 alias 사용
            'search_intent': llm_result.get('search_intent', query)
        }
        return params

    def _parse_time_period(self, time_period: str) -> Optional[tuple]:
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
        return None

    # ---- public ----
    def comprehensive_search(self, query: str) -> Dict[str, Any]:
        print(f"\n>> RDB 종합 검색 시작: {query}")
        params = self._extract_search_params(query)

        results = {
            'query': query,
            'extracted_params': params,
            'price_data': [],
            'nutrition_data': [],
        }

        st = params.get('search_type', 'general')
        if st in ['price', 'general']:
            print(">> 가격 정보 검색 수행")
            results['price_data'] = self.search_price_data(params)
        if st in ['nutrition', 'general']:
            print(">> 영양 정보 검색 수행")
            results['nutrition_data'] = self.search_nutrition_data(params)

        results['total_results'] = sum(
            len(data) for key, data in results.items() if key.endswith('_data')
        )
        print(f">> 검색 완료 - 총 {results['total_results']}건")
        return results

    def search_nutrition_data(self, params: Dict[str, Any]) -> List[Dict]:
        items = params.get('items', [])

        if not items:
            # 폴백: 의도 문장에서 품목 추정 (최소한의 보수적 목록)
            query_text = params.get('search_intent', '')
            food_items = ['귀리', '사과', '감자', '쌀', '보리', '옥수수', '콩', '당근', '양파', '배추', '무',
                          '고구마','딸기','포도','복숭아','수박','참외','호박','오이','토마토','상추','시금치',
                          '깻잎','마늘','생강','파','대파','쪽파','부추','고추','피망','파프리카','감귤','귤',
                          '오렌지','바나나','키위','망고','배','자두','체리']
            for w in food_items:
                if w in query_text and w not in items:
                    items.append(w)
                    print(f"    - 폴백으로 '{w}' 품목 추가")

        if not items:
            print(">> 검색할 품목이 없어서 영양소 검색 건너뜀")
            return []

        base_query = """
        SELECT
            f."식품군" as "food_group",
            f."식품명" as "food_name",
            f."출처" as "source",
            f."폐기율_percent" as "waste_rate_percent",

            -- proximates
            p."에너지 (kcal/100g)" as "energy_kcal",
            p."수분 (g/100g)" as "moisture_g",
            p."단백질 (g/100g)" as "protein_g",
            p."지방 (g/100g)" as "fat_g",
            p."회분 (g/100g)" as "ash_g",
            p."탄수화물 (g/100g)" as "carbohydrate_g",
            p."당류 (g/100g)" as "sugars_g",
            p."자당 (g/100g)" as "sucrose_g",
            p."포도당 (g/100g)" as "glucose_g",
            p."과당 (g/100g)" as "fructose_g",
            p."유당 (g/100g)" as "lactose_g",
            p."맥아당 (g/100g)" as "maltose_g",
            p."갈락토오스 (g/100g)" as "galactose_g",
            p."총 식이섬유 (g/100g)" as "total_dietary_fiber_g",
            p."수용성 식이섬유 (g/100g)" as "soluble_dietary_fiber_g",
            p."불용성 식이섬유 (g/100g)" as "insoluble_dietary_fiber_g",
            p."식염상당량 (g/100g)" as "salt_equivalent_g",

            -- minerals
            m."칼슘 (mg/100g)" as "calcium_mg",
            m."철 (mg/100g)" as "iron_mg",
            m."마그네슘 (mg/100g)" as "magnesium_mg",
            m."인 (mg/100g)" as "phosphorus_mg",
            m."칼륨 (mg/100g)" as "potassium_mg",
            m."나트륨 (mg/100g)" as "sodium_mg",
            m."아연 (mg/100g)" as "zinc_mg",
            m."구리 (mg/100g)" as "copper_mg",
            m."망간 (mg/100g)" as "manganese_mg",
            m."셀레늄 (μg/100g)" as "selenium_ug",
            m."몰리브덴 (μg/100g)" as "molybdenum_ug",
            m."요오드 (μg/100g)" as "iodine_ug",

            -- vitamins
            v."비타민 A (μg/100g)" as "vitamin_a_ug_rae",
            v."레티놀 (μg/100g)" as "retinol_ug",
            v."베타카로틴 (μg/100g)" as "beta_carotene_ug",
            v."티아민 (mg/100g)" as "thiamin_mg",
            v."리보플라빈 (mg/100g)" as "riboflavin_mg",
            v."니아신 (mg/100g)" as "niacin_mg",
            v."니아신당량(NE) (mg/100g)" as "niacin_eq_mg_ne",
            v."니코틴산 (mg/100g)" as "nicotinic_acid_mg",
            v."니코틴아미드 (mg/100g)" as "nicotinamide_mg",
            v."판토텐산 (mg/100g)" as "pantothenic_acid_mg",
            v."비타민 B6 (mg/100g)" as "vitamin_b6_mg",
            v."피리독신 (mg/100g)" as "pyridoxine_mg",
            v."비오틴 (μg/100g)" as "biotin_ug",
            v."엽산_ 엽산당량 (μg/100g)" as "folate_ug_dfe",
            v."엽산_ 식품 엽산 (μg/100g)" as "folate_food_ug",
            v."엽산_ 합성 엽산 (μg/100g)" as "folate_synthetic_ug",
            v."비타민 B12 (μg/100g)" as "vitamin_b12_ug",
            v."비타민 C (mg/100g)" as "vitamin_c_mg",
            v."비타민 D (μg/100g)" as "vitamin_d_ug",
            v."비타민 D2 (μg/100g)" as "vitamin_d2_ug",
            v."비타민 D3 (μg/100g)" as "vitamin_d3_ug",
            v."비타민 E (mg/100g)" as "vitamin_e_mg_ate",
            v."알파 토코페롤 (mg/100g)" as "alpha_tocopherol_mg",
            v."베타 토코페롤 (mg/100g)" as "beta_tocopherol_mg",
            v."감마 토코페롤 (mg/100g)" as "gamma_tocopherol_mg",
            v."델타 토코페롤 (mg/100g)" as "delta_tocopherol_mg",
            v."알파 토코트리에놀 (mg/100g)" as "alpha_tocotrienol_mg",
            v."베타 토코트리에놀 (mg/100g)" as "beta_tocotrienol_mg",
            v."감마 토코트리에놀 (mg/100g)" as "gamma_tocotrienol_mg",
            v."델타 토코트리에놀 (mg/100g)" as "delta_tocotrienol_mg",
            v."비타민 K (μg/100g)" as "vitamin_k_ug",
            v."비타민 K1 (μg/100g)" as "vitamin_k1_ug",
            v."비타민 K2 (μg/100g)" as "vitamin_k2_ug",

            -- amino_acids
            aa."총 아미노산 (mg/100g)" as "total_amino_acids_mg",
            aa."총 필수 아미노산 (mg/100g)" as "total_essential_amino_acids_mg",
            aa."이소류신 (mg/100g)" as "isoleucine_mg",
            aa."류신 (mg/100g)" as "leucine_mg",
            aa."라이신 (mg/100g)" as "lysine_mg",
            aa."메티오닌 (mg/100g)" as "methionine_mg",
            aa."페닐알라닌 (mg/100g)" as "phenylalanine_mg",
            aa."트레오닌 (mg/100g)" as "threonine_mg",
            aa."트립토판 (mg/100g)" as "tryptophan_mg",
            aa."발린 (mg/100g)" as "valine_mg",
            aa."히스티딘 (mg/100g)" as "histidine_mg",
            aa."아르기닌 (mg/100g)" as "arginine_mg",
            aa."티로신 (mg/100g)" as "tyrosine_mg",
            aa."시스테인 (mg/100g)" as "cysteine_mg",
            aa."알라닌 (mg/100g)" as "alanine_mg",
            aa."아스파르트산 (mg/100g)" as "aspartic_acid_mg",
            aa."글루탐산 (mg/100g)" as "glutamic_acid_mg",
            aa."글라이신 (mg/100g)" as "glycine_mg",
            aa."프롤린 (mg/100g)" as "proline_mg",
            aa."세린 (mg/100g)" as "serine_mg",
            aa."타우린 (mg/100g)" as "taurine_mg",

            -- fatty_acids
            fa."콜레스테롤 (mg/100g)" as "cholesterol_mg",
            fa."총 지방산 (g/100g)" as "total_fatty_acids_g",
            fa."총 필수 지방산 (g/100g)" as "total_essential_fatty_acids_g",
            fa."총 포화 지방산 (g/100g)" as "total_saturated_fatty_acids_g",
            fa."부티르산 (4:0) (mg/100g)" as "butyric_acid_4_0_mg",
            fa."카프로산 (6:0) (mg/100g)" as "caproic_acid_6_0_mg",
            fa."카프릴산 (8:0) (mg/100g)" as "caprylic_acid_8_0_mg",
            fa."카프르산 (10:0) (mg/100g)" as "capric_acid_10_0_mg",
            fa."라우르산 (12:0) (mg/100g)" as "lauric_acid_12_0_mg",
            fa."트라이데칸산 (13:0) (mg/100g)" as "tridecanoic_acid_13_0_mg",
            fa."미리스트산 (14:0) (mg/100g)" as "myristic_acid_14_0_mg",
            fa."펜타데칸산 (15:0) (mg/100g)" as "pentadecanoic_acid_15_0_mg",
            fa."팔미트산 (16:0) (mg/100g)" as "palmitic_acid_16_0_mg",
            fa."헵타데칸산 (17:0) (mg/100g)" as "heptadecanoic_acid_17_0_mg",
            fa."스테아르산 (18:0) (mg/100g)" as "stearic_acid_18_0_mg",
            fa."아라키드산 (20:0) (mg/100g)" as "arachidic_acid_20_0_mg",
            fa."헨에이코산산 (21:0) (mg/100g)" as "heneicosanoic_acid_21_0_mg",
            fa."베헨산 (22:0) (mg/100g)" as "behenic_acid_22_0_mg",
            fa."트리코산산 (23:0) (mg/100g)" as "tricosanoic_acid_23_0_mg",
            fa."리그노세르산 (24:0) (mg/100g)" as "lignoceric_acid_24_0_mg",
            fa."총 불포화 지방산 (g/100g)" as "total_unsaturated_fatty_acids_g",
            fa."총 단일 불포화지방산 (g/100g)" as "total_monounsaturated_fatty_acids_g",
            fa."미리스톨레산 (14:1) (mg/100g)" as "myristoleic_acid_14_1_mg",
            fa."팔미톨레산 (16:1) (mg/100g)" as "palmitoleic_acid_16_1_mg",
            fa."헵타데센산 (17:1) (mg/100g)" as "heptadecenoic_acid_17_1_mg",
            fa."올레산 (18:1(n-9)) (mg/100g)" as "oleic_acid_18_1_n9_mg",
            fa."박센산 (18:1(n-7)) (mg/100g)" as "vaccenic_acid_18_1_n7_mg",
            fa."가돌레산 (20:1) (mg/100g)" as "gadoleic_acid_20_1_mg",
            fa."에루크산 (22:1) (mg/100g)" as "erucic_acid_22_1_mg",
            fa."네르본산 (24:1) (mg/100g)" as "nervonic_acid_24_1_mg",
            fa."총 다가 불포화지방산 (g/100g)" as "total_polyunsaturated_fatty_acids_g",
            fa."리놀레산 (18:2(n-6)) (mg/100g)" as "linoleic_acid_18_2_n6_mg",
            fa."알파 리놀렌산 (18:3 (n-3)) (mg/100g)" as "alpha_linolenic_acid_18_3_n3_mg",
            fa."감마 리놀렌산 (18:3 (n-6)) (mg/100g)" as "gamma_linolenic_acid_18_3_n6_mg",
            fa."에이코사 디에노산 (20:2(n-6)) (mg/100g)" as "eicosadienoic_acid_20_2_n6_mg",
            fa."디호모 리놀렌산 (20:3(n-3)) (mg/100g)" as "dihomo_linolenic_acid_20_3_n3_mg",
            fa."에이코사 트리에노산 (20:3(n-6)) (mg/100g)" as "eicosatrienoic_acid_20_3_n6_mg",
            fa."아라키돈산 (20:4(n-6)) (mg/100g)" as "arachidonic_acid_20_4_n6_mg",
            fa."에이코사 펜타에노산 (20:5(n-3)) (mg/100g)" as "eicosapentaenoic_acid_20_5_n3_mg",
            fa."도코사 디에노산(22:2) (mg/100g)" as "docosadienoic_acid_22_2_mg",
            fa."도코사 펜타에노산 (22:5(n-3)) (mg/100g)" as "docosapentaenoic_acid_22_5_n3_mg",
            fa."도코사 헥사에노산 (22:6(n-3)) (mg/100g)" as "docosahexaenoic_acid_22_6_n3_mg",
            fa."오메가3 지방산 (g/100g)" as "omega_3_fatty_acids_g",
            fa."오메가6 지방산 (g/100g)" as "omega_6_fatty_acids_g",
            fa."총 트랜스 지방산 (g/100g)" as "total_trans_fatty_acids_g",
            fa."트랜스 올레산(18:1(n-9)t) (mg/100g)" as "trans_oleic_acid_18_1_n9t_mg",
            fa."트랜스 리놀레산(18:2t) (mg/100g)" as "trans_linoleic_acid_18_2t_mg",
            fa."트랜스 리놀렌산(18:3t) (mg/100g)" as "trans_linolenic_acid_18_3t_mg"
        FROM foods f
        LEFT JOIN proximates  p  ON f.id = p.food_id
        LEFT JOIN minerals   m  ON f.id = m.food_id
        LEFT JOIN vitamins   v  ON f.id = v.food_id
        LEFT JOIN amino_acids aa ON f.id = aa.food_id
        LEFT JOIN fatty_acids fa ON f.id = fa.food_id
        WHERE 1=1
        """

        where_conditions, query_params = [], []
        if items:
            cond = " OR ".join(['f."식품명" ILIKE %s' for _ in items])
            where_conditions.append(f"({cond})")
            query_params.extend([f"%{it}%" for it in items])

        if where_conditions:
            base_query += " AND " + " AND ".join(where_conditions)
        base_query += " LIMIT 10"

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    print(">> 영양소 검색 쿼리:", cursor.mogrify(base_query, tuple(query_params)).decode("utf-8"))
                    cursor.execute(base_query, tuple(query_params))
                    results = cursor.fetchall()
                    print(f">> 영양소 검색 결과: {len(results)}건")
                    return [dict(row) for row in results]
        except Exception as e:
            print(f">> 영양소 데이터 검색 오류: {e}")
            return []

    def search_price_data(self, params: Dict[str, Any]) -> List[Dict]:
        base_query = """
        SELECT
            id, regday, product_cls_code, product_cls_name,
            category_code, category_name, productno, lastest_day,
            "productName", item_name, unit,
            day1, dpr1, day2, dpr2, day3, dpr3, day4, dpr4,
            direction, value
        FROM kamis_product_price_latest
        WHERE 1=1
        """
        where_conditions, query_params = [], []

        if params.get('items'):
            conds = []
            for item in params['items']:
                conds.append("item_name ILIKE %s")
                query_params.append(f"%{item}%")
            if conds:
                where_conditions.append(f"({' OR '.join(conds)})")

        if params.get('date_range'):
            where_conditions.append("regday >= %s AND regday <= %s")
            query_params.extend([d.date() for d in params['date_range']])

        if params.get('price_range'):
            pr = params['price_range']
            if len(pr) == 1:
                where_conditions.append("value <= %s")
                query_params.append(pr[0])
            elif len(pr) == 2:
                where_conditions.append("value BETWEEN %s AND %s")
                query_params.extend(pr)

        if where_conditions:
            base_query += " AND " + " AND ".join(where_conditions)

        sort_by = params.get('sort_by', 'date_desc')
        if sort_by == 'price_desc':
            base_query += " ORDER BY value DESC"
        elif sort_by == 'price_asc':
            base_query += " ORDER BY value ASC"
        else:
            base_query += " ORDER BY regday DESC"

        limit = params.get('limit', 100)
        base_query += f" LIMIT {limit}"

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    print(">> 실행 쿼리:", cursor.mogrify(base_query, tuple(query_params)).decode("utf-8"))
                    cursor.execute(base_query, tuple(query_params) if query_params else None)
                    results = cursor.fetchall()
                    print(f">> 검색 결과: {len(results)}건")
                    return [dict(row) for row in results]
        except Exception as e:
            print(f">> 가격 데이터 검색 오류: {e}")
            return []


# -----------------------------
# Public API (리턴: str)  ---- 기존 호출부 호환
# -----------------------------
postgres_rag = PostgreSQLRAG()

def postgres_rdb_search(query: str) -> str:
    try:
        search_results = postgres_rag.comprehensive_search(query)
        has_data = any([
            search_results['price_data'],
            search_results['nutrition_data']
        ])
        if not has_data:
            print(f">> RDB에서 '{query}' 관련 데이터 없음")
            return f"PostgreSQL 검색 결과: '{query}'와 관련된 데이터를 찾을 수 없습니다."

        # LLM이 이해하기 쉬운 구조화된 형태로 변경
        structured_data = {
            "data_type": "database_search",
            "query": query,
            "total_results": search_results['total_results'],
            "search_params": search_results.get('extracted_params', {}),
            "structured_results": {}
        }

        # 가격 데이터 구조화
        if search_results['price_data']:
            price_items = []
            for item in search_results['price_data'][:10]:  # 상위 10개
                price_now = item.get('dpr1')
                price_yesterday = item.get('dpr2')
                price_1m_ago = item.get('dpr3')
                price_1y_ago = item.get('dpr4')
                
                # 가격 변동 계산
                daily_change = None
                monthly_change = None
                yearly_change = None
                
                if price_now is not None and price_yesterday is not None and price_yesterday > 0:
                    daily_change = ((price_now - price_yesterday) / price_yesterday) * 100
                    
                if price_now is not None and price_1m_ago is not None and price_1m_ago > 0:
                    monthly_change = ((price_now - price_1m_ago) / price_1m_ago) * 100
                    
                if price_now is not None and price_1y_ago is not None and price_1y_ago > 0:
                    yearly_change = ((price_now - price_1y_ago) / price_1y_ago) * 100

                price_item = {
                    "item_name": item.get('item_name'),
                    "category": item.get('category_name'),
                    "current_price": price_now,
                    "unit": item.get('unit'),
                    "date": item.get('lastest_day').strftime('%Y-%m-%d') if item.get('lastest_day') else None,
                    "price_trends": {
                        "daily_change_percent": round(daily_change, 1) if daily_change is not None else None,
                        "monthly_change_percent": round(monthly_change, 1) if monthly_change is not None else None,
                        "yearly_change_percent": round(yearly_change, 1) if yearly_change is not None else None,
                        "yesterday_price": price_yesterday,
                        "month_ago_price": price_1m_ago,
                        "year_ago_price": price_1y_ago
                    },
                    "market_trend": "상승" if daily_change and daily_change > 0 else "하락" if daily_change and daily_change < 0 else "보합"
                }
                price_items.append(price_item)
            
            structured_data["structured_results"]["price_data"] = {
                "count": len(search_results['price_data']),
                "items": price_items
            }

        # 영양 데이터 구조화
        if search_results['nutrition_data']:
            nutrition_items = []
            for item in search_results['nutrition_data'][:10]:  # 상위 10개
                
                # 주요 영양소 정리
                nutrition_facts = {}
                
                # 기본 영양소
                basic_nutrients = {
                    "energy_kcal": {"name": "칼로리", "unit": "kcal"},
                    "protein_g": {"name": "단백질", "unit": "g"},
                    "fat_g": {"name": "지방", "unit": "g"},
                    "carbohydrate_g": {"name": "탄수화물", "unit": "g"},
                    "sugars_g": {"name": "당류", "unit": "g"},
                    "total_dietary_fiber_g": {"name": "식이섬유", "unit": "g"}
                }
                
                # 미네랄
                minerals = {
                    "calcium_mg": {"name": "칼슘", "unit": "mg"},
                    "iron_mg": {"name": "철분", "unit": "mg"},
                    "potassium_mg": {"name": "칼륨", "unit": "mg"},
                    "sodium_mg": {"name": "나트륨", "unit": "mg"},
                    "magnesium_mg": {"name": "마그네슘", "unit": "mg"}
                }
                
                # 비타민
                vitamins = {
                    "vitamin_a_ug_rae": {"name": "비타민A", "unit": "μg"},
                    "vitamin_c_mg": {"name": "비타민C", "unit": "mg"},
                    "vitamin_d_ug": {"name": "비타민D", "unit": "μg"},
                    "vitamin_e_mg_ate": {"name": "비타민E", "unit": "mg"},
                    "vitamin_k_ug": {"name": "비타민K", "unit": "μg"}
                }
                
                # 모든 영양소 정보 수집
                for key, info in {**basic_nutrients, **minerals, **vitamins}.items():
                    value = item.get(key)
                    if value is not None and value != 0:
                        nutrition_facts[info["name"]] = {
                            "value": value,
                            "unit": info["unit"],
                            "per_100g": f"{value}{info['unit']}/100g"
                        }

                nutrition_item = {
                    "food_name": item.get('food_name'),
                    "food_group": item.get('food_group'),
                    "source": item.get('source'),
                    "nutrition_facts": nutrition_facts,
                    "summary": {
                        "high_in": [],  # 높은 함량 영양소
                        "notable_nutrients": []  # 주목할 영양소
                    }
                }
                
                # 높은 함량 영양소 식별 (임계값 기반)
                high_thresholds = {
                    "칼로리": 200, "단백질": 10, "칼슘": 100, "철분": 5, 
                    "비타민C": 30, "비타민A": 300, "칼륨": 300
                }
                
                for nutrient, data in nutrition_facts.items():
                    threshold = high_thresholds.get(nutrient)
                    if threshold and data["value"] >= threshold:
                        nutrition_item["summary"]["high_in"].append(f"{nutrient} {data['per_100g']}")
                
                nutrition_items.append(nutrition_item)
            
            structured_data["structured_results"]["nutrition_data"] = {
                "count": len(search_results['nutrition_data']),
                "items": nutrition_items
            }

        # 최종 결과를 LLM이 이해하기 쉬운 형태로 포맷팅
        summary = f"""=== RDB 검색 결과 ===
검색어: {query}
총 결과: {structured_data['total_results']}건

"""

        # 가격 정보 요약
        if "price_data" in structured_data["structured_results"]:
            price_data = structured_data["structured_results"]["price_data"]
            summary += f"📊 가격 정보 ({price_data['count']}건):\n"
            for item in price_data["items"][:5]:
                trend_emoji = "📈" if item["market_trend"] == "상승" else "📉" if item["market_trend"] == "하락" else "➡️"
                daily_trend = f"({item['price_trends']['daily_change_percent']:+.1f}%)" if item['price_trends']['daily_change_percent'] else ""
                
                summary += f"• {item['item_name']} ({item['category']}): {item['current_price']:,}원/{item['unit']} {trend_emoji} {daily_trend}\n"
                
                trends = []
                if item['price_trends']['monthly_change_percent']:
                    trends.append(f"1개월: {item['price_trends']['monthly_change_percent']:+.1f}%")
                if item['price_trends']['yearly_change_percent']:
                    trends.append(f"1년: {item['price_trends']['yearly_change_percent']:+.1f}%")
                if trends:
                    summary += f"  └ 추세: {' | '.join(trends)}\n"
            summary += "\n"

        # 영양 정보 요약
        if "nutrition_data" in structured_data["structured_results"]:
            nutrition_data = structured_data["structured_results"]["nutrition_data"]
            summary += f"🥗 영양 정보 ({nutrition_data['count']}건):\n"
            for item in nutrition_data["items"][:5]:
                summary += f"• {item['food_name']} ({item['food_group']}):\n"
                
                # 주요 영양소 표시
                key_nutrients = []
                for nutrient in ["칼로리", "단백질", "지방", "탄수화물"]:
                    if nutrient in item['nutrition_facts']:
                        data = item['nutrition_facts'][nutrient]
                        key_nutrients.append(f"{nutrient} {data['per_100g']}")
                
                if key_nutrients:
                    summary += f"  └ 주요성분: {' | '.join(key_nutrients)}\n"
                
                # 높은 함량 영양소
                if item['summary']['high_in']:
                    summary += f"  └ 풍부한 영양소: {', '.join(item['summary']['high_in'])}\n"
            summary += "\n"

        # 구조화된 데이터도 포함 (분석용)
        summary += "=== 구조화된 데이터 (분석용) ===\n"
        summary += json.dumps(structured_data, ensure_ascii=False, indent=2, default=str)
        
        return summary

    except Exception as e:
        error_msg = f"PostgreSQL 검색 중 오류 발생: {str(e)}"
        print(f">> {error_msg}")
        return error_msg
