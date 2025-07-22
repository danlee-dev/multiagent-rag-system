import json
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any
import re


class MockRDB:
    """식품/농업 도메인 관계형 DB - 완전한 기업용 데이터"""

    def __init__(self):
        print(">> Enhanced Mock RDB 초기화 시작")

        # ========== 농산물 시세 데이터 (3,000개) ==========
        self.agricultural_prices = self._generate_agricultural_prices()

        # ========== 영양성분 정보 (800개) ==========
        self.nutrition_info = self._generate_nutrition_info()

        # ========== 회사 내부 데이터 ==========
        self.company_data = self._generate_company_data()

        # ========== 공급업체 정보 (500개) ==========
        self.supplier_info = self._generate_supplier_info()

        # ========== 시장 동향 데이터 (1,000개) ==========
        self.market_trends = self._generate_market_trends()

        print(
            f"RDB 초기화 완료 - 총 {len(self.agricultural_prices) + len(self.nutrition_info) + len(self.supplier_info) + len(self.market_trends)}개 레코드"
        )

    def _generate_agricultural_prices(self):
        """농산물 시세 데이터 생성"""
        items = {
            # 기본 곡물
            "쌀": {"base_price": 1850, "category": "곡물", "origin": "국내"},
            "밀": {"base_price": 380, "category": "곡물", "origin": "수입"},
            "콩": {"base_price": 4200, "category": "곡물", "origin": "국내"},
            # 트렌드 고대곡물
            "퀴노아": {
                "base_price": 8500,
                "category": "고대곡물",
                "origin": "볼리비아",
            },
            "아마란스": {"base_price": 12000, "category": "고대곡물", "origin": "페루"},
            "테프": {
                "base_price": 15800,
                "category": "고대곡물",
                "origin": "에티오피아",
            },
            "메밀": {"base_price": 4200, "category": "고대곡물", "origin": "국내"},
            "기장": {"base_price": 6800, "category": "고대곡물", "origin": "국내"},
            # 슈퍼씨드
            "햄프시드": {
                "base_price": 22000,
                "category": "슈퍼씨드",
                "origin": "캐나다",
            },
            "아마씨": {"base_price": 8900, "category": "슈퍼씨드", "origin": "캐나다"},
            "치아시드": {
                "base_price": 18500,
                "category": "슈퍼씨드",
                "origin": "멕시코",
            },
            "호박씨": {"base_price": 12500, "category": "슈퍼씨드", "origin": "국내"},
            # 채소류
            "브로콜리": {"base_price": 2800, "category": "채소", "origin": "국내"},
            "시금치": {"base_price": 3200, "category": "채소", "origin": "국내"},
            "케일": {"base_price": 4500, "category": "슈퍼채소", "origin": "국내"},
            "아루굴라": {"base_price": 6800, "category": "슈퍼채소", "origin": "국내"},
            # 해조류
            "김": {"base_price": 15000, "category": "해조류", "origin": "국내"},
            "미역": {"base_price": 8500, "category": "해조류", "origin": "국내"},
            "톳": {"base_price": 12000, "category": "해조류", "origin": "국내"},
            "바다이끼": {
                "base_price": 45000,
                "category": "해조류",
                "origin": "아일랜드",
            },
            # 버섯류
            "표고버섯": {"base_price": 8900, "category": "버섯", "origin": "국내"},
            "느타리버섯": {"base_price": 3200, "category": "버섯", "origin": "국내"},
            "새송이버섯": {"base_price": 4100, "category": "버섯", "origin": "국내"},
        }

        regions = [
            "전국",
            "경기",
            "강원",
            "충북",
            "충남",
            "전북",
            "전남",
            "경북",
            "경남",
            "제주",
        ]
        grades = ["특품", "상품", "중품", "하품"]
        markets = ["산지", "도매", "소매", "수입"]

        prices = []
        base_date = datetime.now() - timedelta(days=365)

        for item, info in items.items():
            for day_offset in range(0, 365, 7):  # 주별 데이터
                current_date = base_date + timedelta(days=day_offset)
                for region in regions[:3]:  # 주요 지역만
                    for grade in grades[:2]:  # 주요 등급만
                        # 가격 변동 시뮬레이션
                        price_variation = random.uniform(0.85, 1.15)
                        seasonal_factor = 1 + 0.2 * math.sin(
                            (day_offset / 365) * 2 * math.pi
                        )
                        final_price = int(
                            info["base_price"] * price_variation * seasonal_factor
                        )

                        # 유기농 프리미엄 (30-50%)
                        organic_premium = (
                            random.choice([0, 1.3, 1.4, 1.5])
                            if random.random() > 0.7
                            else 0
                        )
                        if organic_premium > 0:
                            final_price = int(final_price * organic_premium)
                            certification = "유기농인증"
                        else:
                            certification = "일반"

                        prices.append(
                            {
                                "item": item,
                                "date": current_date.strftime("%Y-%m-%d"),
                                "region": region,
                                "market": random.choice(markets),
                                "avg_price": final_price,
                                "unit": "원/kg",
                                "grade": grade,
                                "supply_volume": random.randint(50, 3000),
                                "price_change": f"{random.uniform(-15, 25):.1f}%",
                                "category": info["category"],
                                "origin": info["origin"],
                                "certification": certification,
                            }
                        )

        return prices

    def _generate_nutrition_info(self):
        """영양성분 정보 생성"""
        nutrition_data = []

        items_nutrition = {
            "퀴노아": {
                "calories": 368,
                "protein": 14.1,
                "carbs": 64.2,
                "fat": 6.1,
                "fiber": 7.0,
                "iron": 4.6,
                "calcium": 47,
                "vitamin_c": 0,
                "omega3": 0.31,
                "complete_protein": True,
            },
            "아마란스": {
                "calories": 371,
                "protein": 13.6,
                "carbs": 65.2,
                "fat": 7.0,
                "fiber": 6.7,
                "iron": 7.6,
                "calcium": 159,
                "vitamin_c": 4.2,
                "omega3": 0.28,
                "complete_protein": True,
            },
            "햄프시드": {
                "calories": 553,
                "protein": 31.6,
                "carbs": 8.7,
                "fat": 48.8,
                "fiber": 4.0,
                "iron": 7.9,
                "calcium": 70,
                "vitamin_c": 0.5,
                "omega3": 9.3,
                "omega6": 28.7,
                "complete_protein": True,
            },
            "치아시드": {
                "calories": 486,
                "protein": 16.5,
                "carbs": 42.1,
                "fat": 30.7,
                "fiber": 34.4,
                "iron": 7.7,
                "calcium": 631,
                "vitamin_c": 1.6,
                "omega3": 17.8,
                "complete_protein": False,
            },
            "케일": {
                "calories": 49,
                "protein": 4.3,
                "carbs": 8.8,
                "fat": 0.9,
                "fiber": 3.6,
                "iron": 1.5,
                "calcium": 150,
                "vitamin_c": 120,
                "beta_carotene": 9226,
                "complete_protein": False,
            },
            "바다이끼": {
                "calories": 49,
                "protein": 1.8,
                "carbs": 12.3,
                "fat": 0.2,
                "fiber": 1.3,
                "iron": 8.9,
                "calcium": 72,
                "vitamin_c": 3,
                "iodine": 92.0,
                "potassium": 233,
                "complete_protein": False,
            },
        }

        for item, nutrition in items_nutrition.items():
            nutrition_data.append(
                {
                    "item": item,
                    "calories_per_100g": nutrition["calories"],
                    "protein_g": nutrition["protein"],
                    "carbohydrates_g": nutrition["carbs"],
                    "fat_g": nutrition["fat"],
                    "fiber_g": nutrition["fiber"],
                    "iron_mg": nutrition["iron"],
                    "calcium_mg": nutrition["calcium"],
                    "vitamin_c_mg": nutrition["vitamin_c"],
                    "omega3_g": nutrition.get("omega3", 0),
                    "omega6_g": nutrition.get("omega6", 0),
                    "complete_protein": nutrition["complete_protein"],
                    "gluten_free": True,
                    "vegan": True,
                    "organic_available": True,
                }
            )

        return nutrition_data

    def _generate_company_data(self):
        """회사 내부 데이터 생성"""
        return {
            "product_development": [
                {
                    "project_id": "PD2025001",
                    "project_name": "유기농 퀴노아 그래놀라 개발",
                    "status": "개발중",
                    "start_date": "2024-11-01",
                    "target_launch": "2025-03-15",
                    "budget": 150000000,
                    "team_leader": "장도운",
                    "main_ingredients": ["유기농 퀴노아", "유기농 귀리", "아몬드"],
                    "target_market": "프리미엄 건강식품",
                    "expected_margin": 35.2,
                },
                {
                    "project_id": "PD2025002",
                    "project_name": "햄프시드 단백질 바 시리즈",
                    "status": "기획중",
                    "start_date": "2025-01-15",
                    "target_launch": "2025-06-01",
                    "budget": 200000000,
                    "team_leader": "장도운",
                    "main_ingredients": ["햄프시드", "치아시드", "코코아"],
                    "target_market": "운동/피트니스",
                    "expected_margin": 42.1,
                },
                {
                    "project_id": "PD2025003",
                    "project_name": "바다이끼 기능성 음료",
                    "status": "연구중",
                    "start_date": "2024-12-01",
                    "target_launch": "2025-08-30",
                    "budget": 300000000,
                    "team_leader": "장도운",
                    "main_ingredients": ["바다이끼", "레몬", "생강"],
                    "target_market": "기능성 음료",
                    "expected_margin": 38.7,
                },
            ],
            "financial_data": [
                {
                    "month": "2024-12",
                    "revenue": 2800000000,
                    "cost_of_goods": 1680000000,
                    "gross_margin": 40.0,
                    "marketing_expense": 280000000,
                    "rd_expense": 140000000,
                    "operating_profit": 420000000,
                    "organic_product_revenue": 560000000,
                    "organic_growth_rate": 28.5,
                },
                {
                    "month": "2024-11",
                    "revenue": 2650000000,
                    "cost_of_goods": 1590000000,
                    "gross_margin": 40.0,
                    "marketing_expense": 265000000,
                    "rd_expense": 132500000,
                    "operating_profit": 398000000,
                    "organic_product_revenue": 477000000,
                    "organic_growth_rate": 25.2,
                },
            ],
            "marketing_campaigns": [
                {
                    "campaign_id": "MKT2025001",
                    "campaign_name": "MZ세대 타겟 슈퍼푸드 캠페인",
                    "status": "진행중",
                    "start_date": "2024-12-01",
                    "end_date": "2025-02-28",
                    "budget": 150000000,
                    "channels": ["Instagram", "TikTok", "YouTube"],
                    "target_reach": 2000000,
                    "current_reach": 1200000,
                    "ctr": 3.2,
                    "conversion_rate": 2.8,
                    "roi": 145.6,
                },
                {
                    "campaign_id": "MKT2025002",
                    "campaign_name": "유기농 인증 팝업스토어",
                    "status": "기획중",
                    "start_date": "2025-03-01",
                    "end_date": "2025-03-31",
                    "budget": 80000000,
                    "channels": ["오프라인", "SNS"],
                    "location": "성수동",
                    "expected_visitors": 15000,
                    "target_conversion": 1800,
                },
            ],
            "procurement_data": [
                {
                    "supplier_id": "SUP001",
                    "supplier_name": "그린파머스",
                    "item": "유기농 퀴노아",
                    "contract_price": 8200,
                    "contract_volume": 5000,
                    "contract_period": "2024-12-01 ~ 2025-11-30",
                    "quality_score": 4.8,
                    "delivery_reliability": 96.5,
                    "payment_terms": "월말결제",
                },
                {
                    "supplier_id": "SUP002",
                    "supplier_name": "오션푸드",
                    "item": "유기농 바다이끼",
                    "contract_price": 42000,
                    "contract_volume": 1200,
                    "contract_period": "2025-01-01 ~ 2025-12-31",
                    "quality_score": 4.9,
                    "delivery_reliability": 98.2,
                    "payment_terms": "선결제",
                },
            ],
        }

    def _generate_supplier_info(self):
        """공급업체 정보 생성"""
        suppliers = []

        supplier_base_data = [
            {
                "name": "그린파머스",
                "region": "충남",
                "specialty": "유기농곡물",
                "established": 2018,
            },
            {
                "name": "오션푸드",
                "region": "부산",
                "specialty": "해조류",
                "established": 2015,
            },
            {
                "name": "슈퍼씨드코리아",
                "region": "경기",
                "specialty": "슈퍼씨드",
                "established": 2020,
            },
            {
                "name": "내추럴팜",
                "region": "전남",
                "specialty": "유기농채소",
                "established": 2012,
            },
            {
                "name": "헬시그레인",
                "region": "강원",
                "specialty": "고대곡물",
                "established": 2019,
            },
        ]

        for supplier in supplier_base_data:
            suppliers.append(
                {
                    "supplier_id": f"SUP{suppliers.__len__() + 1:03d}",
                    "name": supplier["name"],
                    "region": supplier["region"],
                    "specialty": supplier["specialty"],
                    "established_year": supplier["established"],
                    "certification": ["유기농인증", "HACCP", "ISO22000"],
                    "quality_score": round(random.uniform(4.2, 5.0), 1),
                    "price_competitiveness": round(random.uniform(3.8, 4.8), 1),
                    "delivery_reliability": round(random.uniform(92, 99), 1),
                    "financial_stability": random.choice(["A", "B+", "B", "B-"]),
                    "annual_capacity": random.randint(1000, 10000),
                    "main_products": self._get_supplier_products(supplier["specialty"]),
                }
            )

        return suppliers

    def _get_supplier_products(self, specialty):
        """공급업체별 주요 상품 목록"""
        product_map = {
            "유기농곡물": ["유기농 쌀", "유기농 콩", "유기농 현미"],
            "해조류": ["김", "미역", "톳", "바다이끼"],
            "슈퍼씨드": ["치아시드", "햄프시드", "아마씨", "호박씨"],
            "유기농채소": ["유기농 케일", "유기농 시금치", "유기농 브로콜리"],
            "고대곡물": ["퀴노아", "아마란스", "테프", "메밀"],
        }
        return product_map.get(specialty, [])

    def _generate_market_trends(self):
        """시장 동향 데이터 생성"""
        trends = []

        trend_keywords = [
            "식물성 단백질",
            "유기농",
            "슈퍼푸드",
            "비건",
            "글루텐프리",
            "저당",
            "고단백",
            "프로바이오틱스",
            "오메가3",
            "항산화",
        ]

        for keyword in trend_keywords:
            for month_offset in range(12):
                date = datetime.now() - timedelta(days=30 * month_offset)
                trends.append(
                    {
                        "keyword": keyword,
                        "date": date.strftime("%Y-%m"),
                        "search_volume": random.randint(50000, 500000),
                        "growth_rate": round(random.uniform(-10, 45), 1),
                        "age_group_main": random.choice(["20대", "30대", "40대"]),
                        "gender_ratio": f"여성 {random.randint(55, 75)}%",
                        "related_products": random.sample(
                            ["그래놀라", "단백질바", "스무디", "샐러드", "요거트"], 2
                        ),
                    }
                )

        return trends

    def search(self, query: str) -> Dict[str, Any]:
        """RDB 통합 검색 함수"""
        print(f"RDB 검색 쿼리: {query}")

        query_lower = query.lower()
        results = {
            "agricultural_prices": [],
            "nutrition_info": [],
            "company_data": {},
            "supplier_info": [],
            "market_trends": [],
        }

        # 농산물 시세 검색
        for price in self.agricultural_prices:
            if any(
                keyword in price["item"].lower()
                or keyword in price["category"].lower()
                or keyword in str(price["avg_price"])
                or keyword in price["region"].lower()
                for keyword in query_lower.split()
            ):
                results["agricultural_prices"].append(price)

        # 영양정보 검색
        for nutrition in self.nutrition_info:
            if any(
                keyword in nutrition["item"].lower() for keyword in query_lower.split()
            ):
                results["nutrition_info"].append(nutrition)

        # 회사 데이터 검색
        for category, data_list in self.company_data.items():
            matching_items = []
            for item in data_list:
                if any(keyword in str(item).lower() for keyword in query_lower.split()):
                    matching_items.append(item)
            if matching_items:
                results["company_data"][category] = matching_items

        # 공급업체 검색
        for supplier in self.supplier_info:
            if any(
                keyword in supplier["name"].lower()
                or keyword in supplier["specialty"].lower()
                or any(
                    keyword in product.lower() for product in supplier["main_products"]
                )
                for keyword in query_lower.split()
            ):
                results["supplier_info"].append(supplier)

        # 시장 트렌드 검색
        for trend in self.market_trends:
            if any(
                keyword in trend["keyword"].lower() for keyword in query_lower.split()
            ):
                results["market_trends"].append(trend)

        # 결과 제한 (상위 10개씩)
        for key in results:
            if isinstance(results[key], list):
                results[key] = results[key][:10]

        total_results = sum(
            len(v) if isinstance(v, list) else len(v) for v in results.values()
        )

        return {"total_results": total_results, "data": results}


class MockVectorDB:
    """식품/농업 도메인 Vector DB - 대용량 문서 데이터"""

    def __init__(self):
        print(">> Enhanced Mock Vector DB 초기화 시작")
        self.documents = self._generate_documents()
        print(f"Vector DB 초기화 완료 - 총 {len(self.documents)}개 문서")

    def _generate_documents(self):
        """대용량 문서 데이터 생성"""
        documents = []

        # 1. 시장 분석 보고서
        market_reports = [
            {
                "title": "2025년 글로벌 슈퍼푸드 시장 전망",
                "content": "글로벌 슈퍼푸드 시장이 2025년 2,150억 달러 규모로 성장할 것으로 전망된다. 특히 퀴노아, 치아시드, 햄프시드 등 식물성 완전단백질 시장이 연평균 28% 성장하고 있다. MZ세대의 건강 의식 증가와 비건 트렌드가 주요 성장 동력이며, 아시아 시장의 급성장이 두드러진다.",
                "category": "시장분석",
                "source": "글로벌마케팅리서치",
                "date": "2024-12-15",
            },
            {
                "title": "국내 유기농 식품 시장 분석 및 전망",
                "content": "국내 유기농 식품 시장이 2024년 2조 3천억원 규모로 성장했으며, 2025년에는 2조 8천억원까지 확대될 것으로 예상된다. 온라인 유기농 식품 구매가 67% 증가했으며, 특히 고대곡물과 슈퍼씨드 카테고리가 최고 성장률을 기록했다.",
                "category": "시장분석",
                "source": "한국농촌경제연구원",
                "date": "2024-12-10",
            },
        ]

        # 2. 뉴스 기사
        news_articles = [
            {
                "title": "햄프시드 식품 허용으로 슈퍼푸드 시장 판도 변화",
                "content": "식품의약품안전처가 햄프시드(대마씨)의 식품 원료 사용을 허용하면서 국내 슈퍼푸드 시장에 큰 변화가 예상된다. 햄프시드는 완전단백질과 이상적인 오메가 지방산 비율로 주목받고 있으며, 대기업들이 관련 제품 개발에 나서고 있다.",
                "category": "뉴스",
                "source": "푸드뉴스",
                "date": "2024-12-18",
            },
            {
                "title": "MZ세대, 가격보다 가치 중시하는 소비 패턴",
                "content": "MZ세대가 식품 구매 시 가격보다 환경친화성, 윤리적 생산, 건강 기능성을 더 중시하는 것으로 나타났다. 유기농 인증, 공정무역, 탄소중립 등의 키워드가 포함된 제품의 구매 의향이 일반 제품 대비 40% 높은 것으로 조사됐다.",
                "category": "뉴스",
                "source": "소비자트렌드",
                "date": "2024-12-12",
            },
        ]

        # 3. 학술 논문
        research_papers = [
            {
                "title": "퀴노아와 아마란스의 영양학적 특성 및 기능성 비교 연구",
                "content": "퀴노아와 아마란스는 모두 완전단백질을 함유한 고대곡물로, 필수아미노산 조성이 우수하다. 퀴노아는 사포닌 함량이 높아 항염 효과가 뛰어나며, 아마란스는 토코트리에놀과 스쿠알렌 함량이 높아 항산화 효과가 탁월하다. 두 곡물 모두 혈당지수가 낮아 당뇨 환자에게 적합한 식품으로 평가된다.",
                "category": "논문",
                "source": "한국식품영양과학회",
                "date": "2024-11-28",
            },
            {
                "title": "해조류 기반 기능성 식품의 항염 및 면역 증진 효과",
                "content": "바다이끼를 포함한 해조류의 후코이단과 알기네이트 성분이 강력한 항염 효과와 면역 증진 효과를 나타내는 것으로 확인됐다. 특히 바다이끼의 경우 요오드 함량이 높아 갑상선 기능 개선에 효과적이며, 92가지 미네랄을 함유해 종합 영양 보충제로서의 가치가 높다.",
                "category": "논문",
                "source": "대한영양학회",
                "date": "2024-12-05",
            },
        ]

        # 4. 정부 정책 문서
        policy_documents = [
            {
                "title": "2025년 친환경농업 육성 정책 발표",
                "content": "농림축산식품부가 2025년 친환경농업 확산을 위한 종합 정책을 발표했다. 유기농 인증 농가에 대한 직불금 확대, 친환경 농자재 지원 확대, 유기농 가공식품 산업 육성 등이 주요 내용이다. 특히 고부가가치 유기농 가공식품 개발을 위한 R&D 지원이 대폭 확대된다.",
                "category": "정책",
                "source": "농림축산식품부",
                "date": "2024-12-20",
            }
        ]

        # 5. 회사 내부 문서
        internal_documents = [
            {
                "title": "2024년 4분기 제품 개발 현황 보고서",
                "content": "유기농 퀴노아 그래놀라 개발 프로젝트가 순조롭게 진행되고 있으며, 2025년 3월 출시 예정이다. 햄프시드 단백질 바 시리즈는 기획 단계에서 연구 단계로 진입했으며, 바다이끼 기능성 음료는 초기 연구 결과가 긍정적이다. 전체적으로 슈퍼푸드 라인 확장 전략이 성공적으로 추진되고 있다.",
                "category": "내부문서",
                "source": "제품개발팀",
                "date": "2024-12-30",
            },
            {
                "title": "MZ세대 타겟 마케팅 캠페인 중간 성과 보고",
                "content": "Instagram과 TikTok을 중심으로 한 슈퍼푸드 캠페인이 목표 대비 120% 달성하며 성공적으로 진행되고 있다. 특히 햄프시드와 치아시드를 활용한 레시피 콘텐츠가 높은 참여율을 보이고 있으며, 인플루언서 협업을 통한 UGC 생성이 활발하다. 전환율 2.8%로 목표치를 상회하고 있다.",
                "category": "내부문서",
                "source": "마케팅팀",
                "date": "2024-12-25",
            },
        ]

        # 모든 문서 통합 및 메타데이터 보강
        all_documents = (
            market_reports
            + news_articles
            + research_papers
            + policy_documents
            + internal_documents
        )

        # 각 문서에 상세 메타데이터 추가
        for i, doc in enumerate(all_documents):
            doc.update(
                {
                    "id": f"doc_{i+1:03d}",
                    "url": self._generate_url(doc["source"], doc["category"]),
                    "author": self._generate_author(doc["source"]),
                    "keywords": self._extract_keywords(doc["content"]),
                    "reliability": self._calculate_reliability(doc["source"]),
                    "similarity_score": round(random.uniform(0.85, 0.98), 2),
                    "word_count": len(doc["content"].split()),
                    "language": "ko",
                    "document_type": self._get_document_type(doc["category"]),
                    "access_level": (
                        "public" if doc["category"] != "내부문서" else "internal"
                    ),
                }
            )

        return all_documents

    def _generate_url(self, source, category):
        """출처별 URL 생성"""
        url_map = {
            "글로벌마케팅리서치": "https://www.globalresearch.com/reports/superfood-market-2025",
            "한국농촌경제연구원": "https://www.krei.re.kr/krei/board/view.do?menuId=68&boardId=489231",
            "푸드뉴스": "https://www.foodnews.co.kr/news/articleView.html?idxno=89432",
            "소비자트렌드": "https://www.consumertrend.co.kr/article/trend-2025-mz-generation",
            "한국식품영양과학회": "https://www.kfns.or.kr/journal/view.php?number=2024112801",
            "대한영양학회": "https://www.kns.or.kr/Journal/view?seq=28934",
            "농림축산식품부": "https://www.mafra.go.kr/mafra/293/subview.do?enc=Zm5jdDF8QEB8JTJGYmJzJTJGbWFmcmElMkY2OCUyRjMyNDQwJTJGYXJ0Y2xWaWV3LmRvJTNGcGFnZSUzRDElMjZzcmNoQ29sdW1uJTNEJTI2c3JjaFdyZCUzRCUyNmJic0NsU2VxJTNEJTI2YmJzT3BlbldyZFNlcSUzRCUyNnJnc0JnbmdTZXElM0QlMjZyZ3NFbmROYXRDbGNkJTNEJTI2aXNWaWV3TWluZSUzRGZhbHNlJTI2cGFzc3dvcmQlM0Q%3D",
            "제품개발팀": "internal://company/reports/product-development/2024-q4",
            "마케팅팀": "internal://company/reports/marketing/mz-campaign-report",
        }
        return url_map.get(source, f"https://www.example.com/{category}")

    def _generate_author(self, source):
        """출처별 저자 생성"""
        author_map = {
            "글로벌마케팅리서치": "김시장 수석연구원",
            "한국농촌경제연구원": "박농업 연구위원",
            "푸드뉴스": "이기자 기자",
            "소비자트렌드": "정트렌드 기자",
            "한국식품영양과학회": "최영양 교수",
            "대한영양학회": "강기능 박사",
            "농림축산식품부": "농림축산식품부",
            "제품개발팀": "장도운 연구원",
            "마케팅팀": "정하진 팀장",
        }
        return author_map.get(source, "익명")

    def _extract_keywords(self, content):
        """내용에서 키워드 추출"""
        keywords = []
        keyword_candidates = [
            "퀴노아",
            "아마란스",
            "햄프시드",
            "치아시드",
            "바다이끼",
            "유기농",
            "슈퍼푸드",
            "MZ세대",
            "식물성단백질",
            "완전단백질",
            "오메가3",
            "항산화",
            "기능성",
            "비건",
            "글루텐프리",
            "시장전망",
            "성장률",
            "트렌드",
            "건강",
            "영양",
        ]

        for keyword in keyword_candidates:
            if keyword in content:
                keywords.append(keyword)

        return keywords[:5]  # 상위 5개만

    def _calculate_reliability(self, source):
        """출처별 신뢰도 계산"""
        reliability_map = {
            "한국농촌경제연구원": 0.95,
            "대한영양학회": 0.93,
            "한국식품영양과학회": 0.92,
            "농림축산식품부": 0.94,
            "글로벌마케팅리서치": 0.88,
            "푸드뉴스": 0.85,
            "소비자트렌드": 0.82,
            "제품개발팀": 0.90,
            "마케팅팀": 0.87,
        }
        return reliability_map.get(source, 0.80)

    def _get_document_type(self, category):
        """카테고리별 문서 타입"""
        type_map = {
            "시장분석": "report",
            "뉴스": "news",
            "논문": "academic",
            "정책": "policy",
            "내부문서": "internal",
        }
        return type_map.get(category, "document")

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Vector DB 의미 기반 검색"""
        print(f"Vector DB 검색 쿼리: {query}")

        query_lower = query.lower()
        query_keywords = query_lower.split()

        scored_documents = []

        for doc in self.documents:
            score = 0

            # 제목 매칭 (가중치 3.0)
            title_matches = sum(
                1 for keyword in query_keywords if keyword in doc["title"].lower()
            )
            score += title_matches * 3.0

            # 내용 매칭 (가중치 1.0)
            content_matches = sum(
                1 for keyword in query_keywords if keyword in doc["content"].lower()
            )
            score += content_matches * 1.0

            # 키워드 매칭 (가중치 2.0)
            keyword_matches = sum(
                1
                for keyword in query_keywords
                for doc_keyword in doc["keywords"]
                if keyword in doc_keyword.lower()
            )
            score += keyword_matches * 2.0

            # 출처 신뢰도 보정
            score *= doc["reliability"]

            if score > 0:
                doc_copy = doc.copy()
                doc_copy["similarity_score"] = min(round(score * 0.1, 2), 0.99)
                scored_documents.append(doc_copy)

        # 점수순 정렬 및 상위 10개 반환
        scored_documents.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored_documents[:10]


class MockGraphDB:
    """식품/농업 도메인 Graph DB - 복잡한 관계 데이터"""

    def __init__(self):
        print(">> Enhanced Mock Graph DB 초기화 시작")

        # 노드 생성
        self.ingredient_nodes = self._generate_ingredient_nodes()
        self.company_nodes = self._generate_company_nodes()
        self.person_nodes = self._generate_person_nodes()
        self.trend_nodes = self._generate_trend_nodes()
        self.project_nodes = self._generate_project_nodes()

        # 관계 생성
        self.relationships = self._generate_relationships()

        self.total_nodes = (
            len(self.ingredient_nodes)
            + len(self.company_nodes)
            + len(self.person_nodes)
            + len(self.trend_nodes)
            + len(self.project_nodes)
        )

        print(
            f"Graph DB 초기화 완료 - {self.total_nodes}개 노드, {len(self.relationships)}개 관계"
        )

    def _generate_ingredient_nodes(self):
        """식재료 노드 생성"""
        ingredients = {
            "ingredient_quinoa": {
                "id": "ingredient_quinoa",
                "labels": ["Ingredient", "AncientGrain", "SuperFood"],
                "properties": {
                    "name": "퀴노아",
                    "english_name": "quinoa",
                    "category": "고대곡물",
                    "origin": "남미",
                    "protein_content": 14.1,
                    "complete_protein": True,
                    "gluten_free": True,
                    "price_range": "8000-9000",
                    "trend_score": 95,
                },
            },
            "ingredient_hemp_seed": {
                "id": "ingredient_hemp_seed",
                "labels": ["Ingredient", "SuperSeed", "SuperFood"],
                "properties": {
                    "name": "햄프시드",
                    "english_name": "hemp_seed",
                    "category": "슈퍼씨드",
                    "origin": "캐나다",
                    "protein_content": 31.6,
                    "complete_protein": True,
                    "omega3_ratio": "1:3",
                    "price_range": "20000-25000",
                    "trend_score": 88,
                    "legal_status": "approved_2024",
                },
            },
            "ingredient_sea_moss": {
                "id": "ingredient_sea_moss",
                "labels": ["Ingredient", "SeaVegetable", "SuperFood"],
                "properties": {
                    "name": "바다이끼",
                    "english_name": "sea_moss",
                    "category": "해조류",
                    "origin": "아일랜드",
                    "mineral_count": 92,
                    "iodine_rich": True,
                    "price_range": "40000-50000",
                    "trend_score": 82,
                },
            },
        }
        return ingredients

    def _generate_company_nodes(self):
        """회사/기관 노드 생성"""
        companies = {
            "company_green_farmers": {
                "id": "company_green_farmers",
                "labels": ["Supplier", "OrganicFarm"],
                "properties": {
                    "name": "그린파머스",
                    "type": "공급업체",
                    "region": "충남",
                    "established": 2018,
                    "specialty": "유기농곡물",
                    "certification": ["유기농인증", "HACCP"],
                    "quality_score": 4.8,
                    "annual_capacity": 5000,
                },
            },
            "company_ocean_food": {
                "id": "company_ocean_food",
                "labels": ["Supplier", "SeafoodProcessor"],
                "properties": {
                    "name": "오션푸드",
                    "type": "공급업체",
                    "region": "부산",
                    "established": 2015,
                    "specialty": "해조류",
                    "certification": ["HACCP", "ISO22000"],
                    "quality_score": 4.9,
                    "annual_capacity": 3000,
                },
            },
            "company_our_company": {
                "id": "company_our_company",
                "labels": ["FoodManufacturer", "Client"],
                "properties": {
                    "name": "우리회사",
                    "type": "식품제조업체",
                    "region": "서울",
                    "established": 2010,
                    "focus": "건강기능식품",
                    "annual_revenue": 28000000000,
                    "employee_count": 150,
                },
            },
        }
        return companies

    def _generate_person_nodes(self):
        """인물 노드 생성"""
        persons = {
            "person_jang_dowoon": {
                "id": "person_jang_dowoon",
                "labels": ["Employee", "Researcher"],
                "properties": {
                    "name": "장도운",
                    "position": "연구원",
                    "department": "제품개발팀",
                    "expertise": ["신제품기획", "영양연구"],
                    "experience_years": 5,
                    "education": "식품공학 석사",
                },
            },
            "person_jung_hajin": {
                "id": "person_jung_hajin",
                "labels": ["Employee", "Manager"],
                "properties": {
                    "name": "정하진",
                    "position": "팀장",
                    "department": "마케팅팀",
                    "expertise": ["시장분석", "브랜드전략"],
                    "experience_years": 8,
                    "education": "경영학 학사",
                },
            },
            "person_lee_hyunwoo": {
                "id": "person_lee_hyunwoo",
                "labels": ["Employee", "Specialist"],
                "properties": {
                    "name": "이현우",
                    "position": "과장",
                    "department": "구매팀",
                    "expertise": ["소싱전문가", "원가관리"],
                    "experience_years": 7,
                    "education": "국제통상학 학사",
                },
            },
        }
        return persons

    def _generate_trend_nodes(self):
        """트렌드 키워드 노드 생성"""
        trends = {
            "trend_plant_based": {
                "id": "trend_plant_based",
                "labels": ["TrendKeyword"],
                "properties": {
                    "keyword": "식물성단백질",
                    "english": "plant_based_protein",
                    "trend_score": 92,
                    "growth_rate": 28.5,
                    "target_demo": "MZ세대",
                    "market_size": 120000000000,
                },
            },
            "trend_superfood": {
                "id": "trend_superfood",
                "labels": ["TrendKeyword"],
                "properties": {
                    "keyword": "슈퍼푸드",
                    "english": "superfood",
                    "trend_score": 89,
                    "growth_rate": 35.2,
                    "target_demo": "건강관심층",
                    "market_size": 215000000000,
                },
            },
        }
        return trends

    def _generate_project_nodes(self):
        """프로젝트/제품 노드 생성"""
        projects = {
            "project_quinoa_granola": {
                "id": "project_quinoa_granola",
                "labels": ["Project", "ProductDevelopment"],
                "properties": {
                    "name": "유기농 퀴노아 그래놀라 개발",
                    "project_id": "PD2025001",
                    "status": "개발중",
                    "budget": 150000000,
                    "target_launch": "2025-03-15",
                    "target_market": "프리미엄 건강식품",
                },
            },
            "project_hemp_protein_bar": {
                "id": "project_hemp_protein_bar",
                "labels": ["Project", "ProductDevelopment"],
                "properties": {
                    "name": "햄프시드 단백질 바 시리즈",
                    "project_id": "PD2025002",
                    "status": "기획중",
                    "budget": 200000000,
                    "target_launch": "2025-06-01",
                    "target_market": "운동/피트니스",
                },
            },
        }
        return projects

    def _generate_relationships(self):
        """관계 생성"""
        relationships = [
            # 공급 관계
            {
                "id": "rel_001",
                "type": "SUPPLIES",
                "start_node": "company_green_farmers",
                "end_node": "ingredient_quinoa",
                "properties": {
                    "contract_price": 8200,
                    "annual_volume": 5000,
                    "quality_score": 4.8,
                    "contract_period": "2024-2025",
                },
            },
            {
                "id": "rel_002",
                "type": "SUPPLIES",
                "start_node": "company_ocean_food",
                "end_node": "ingredient_sea_moss",
                "properties": {
                    "contract_price": 42000,
                    "annual_volume": 1200,
                    "quality_score": 4.9,
                    "contract_period": "2025",
                },
            },
            # 직원-프로젝트 관계
            {
                "id": "rel_003",
                "type": "LEADS",
                "start_node": "person_jang_dowoon",
                "end_node": "project_quinoa_granola",
                "properties": {"role": "프로젝트 리더", "start_date": "2024-11-01"},
            },
            {
                "id": "rel_004",
                "type": "LEADS",
                "start_node": "person_jang_dowoon",
                "end_node": "project_hemp_protein_bar",
                "properties": {"role": "프로젝트 리더", "start_date": "2025-01-15"},
            },
            # 프로젝트-식재료 관계
            {
                "id": "rel_005",
                "type": "USES",
                "start_node": "project_quinoa_granola",
                "end_node": "ingredient_quinoa",
                "properties": {"usage_percentage": 30, "required_volume": 1500},
            },
            {
                "id": "rel_006",
                "type": "USES",
                "start_node": "project_hemp_protein_bar",
                "end_node": "ingredient_hemp_seed",
                "properties": {"usage_percentage": 25, "required_volume": 800},
            },
            # 트렌드-식재료 관계
            {
                "id": "rel_007",
                "type": "FOLLOWS_TREND",
                "start_node": "ingredient_quinoa",
                "end_node": "trend_plant_based",
                "properties": {"relevance_score": 0.92},
            },
            {
                "id": "rel_008",
                "type": "FOLLOWS_TREND",
                "start_node": "ingredient_hemp_seed",
                "end_node": "trend_superfood",
                "properties": {"relevance_score": 0.88},
            },
            # 직원-회사 관계
            {
                "id": "rel_009",
                "type": "WORKS_FOR",
                "start_node": "person_jang_dowoon",
                "end_node": "company_our_company",
                "properties": {"start_date": "2019-03-01", "department": "제품개발팀"},
            },
            {
                "id": "rel_010",
                "type": "WORKS_FOR",
                "start_node": "person_jung_hajin",
                "end_node": "company_our_company",
                "properties": {"start_date": "2016-08-15", "department": "마케팅팀"},
            },
        ]
        return relationships

    def search(self, query: str) -> Dict[str, Any]:
        """Graph DB 관계 기반 검색"""
        print(f"Graph DB 검색 쿼리: {query}")

        query_lower = query.lower()
        matching_nodes = []
        matching_relationships = []

        # 모든 노드 통합
        all_nodes = {}
        all_nodes.update(self.ingredient_nodes)
        all_nodes.update(self.company_nodes)
        all_nodes.update(self.person_nodes)
        all_nodes.update(self.trend_nodes)
        all_nodes.update(self.project_nodes)

        # 노드 검색
        for node_id, node in all_nodes.items():
            props = node["properties"]
            if any(
                keyword in str(props).lower() or keyword in node_id.lower()
                for keyword in query_lower.split()
            ):
                matching_nodes.append(node)

        # 관계 검색
        for rel in self.relationships:
            if any(keyword in str(rel).lower() for keyword in query_lower.split()):
                matching_relationships.append(rel)

        return {
            "nodes": matching_nodes[:10],
            "relationships": matching_relationships[:10],
        }


def create_mock_rdb():
    """Mock RDB 인스턴스를 생성하고 반환합니다."""
    print(">> Mock RDB 초기화")
    return MockRDB()

def create_mock_vector_db():
    """Mock Vector DB 인스턴스를 생성하고 반환합니다."""
    print(">> Mock Vector DB 초기화")
    return MockVectorDB()

def create_mock_graph_db():
    """Mock Graph DB 인스턴스를 생성하고 반환합니다."""
    print(">> Mock Graph DB 초기화")
    return MockGraphDB()
