import json
import random
import re
from typing import Dict, List, Any
from datetime import datetime, timedelta


class MockGraphDB:
    """식품/농업 도메인 Graph DB - 대폭 강화"""

    def __init__(self):
        print(">> Mock Graph DB 초기화 시작")
        # 재료 노드들 (대폭 확장)
        self.ingredient_nodes = {
            # 곡물류
            "ingredient_rice": {
                "id": "ingredient_rice",
                "labels": ["Ingredient", "Grain"],
                "properties": {
                    "name": "쌀",
                    "english_name": "rice",
                    "category": "grain",
                    "protein": 2.7,
                    "carbs": 28.1,
                    "origin": "아시아",
                    "harvest_season": "가을",
                    "storage_period": 12,
                },
            },
            "ingredient_wheat": {
                "id": "ingredient_wheat",
                "labels": ["Ingredient", "Grain"],
                "properties": {
                    "name": "밀",
                    "english_name": "wheat",
                    "category": "grain",
                    "protein": 10.7,
                    "carbs": 71.2,
                    "origin": "중동",
                    "harvest_season": "여름",
                    "storage_period": 24,
                },
            },
            "ingredient_corn": {
                "id": "ingredient_corn",
                "labels": ["Ingredient", "Grain"],
                "properties": {
                    "name": "옥수수",
                    "english_name": "corn",
                    "category": "grain",
                    "protein": 3.3,
                    "carbs": 22.8,
                    "origin": "아메리카",
                    "harvest_season": "가을",
                    "storage_period": 18,
                },
            },
            "ingredient_barley": {
                "id": "ingredient_barley",
                "labels": ["Ingredient", "Grain"],
                "properties": {
                    "name": "보리",
                    "english_name": "barley",
                    "category": "grain",
                    "protein": 12.5,
                    "carbs": 73.5,
                    "origin": "중동",
                    "harvest_season": "여름",
                    "storage_period": 24,
                },
            },
            # 채소류
            "ingredient_cabbage": {
                "id": "ingredient_cabbage",
                "labels": ["Ingredient", "Vegetable"],
                "properties": {
                    "name": "배추",
                    "english_name": "cabbage",
                    "category": "vegetable",
                    "vitamin_c": 36,
                    "fiber": 2.5,
                    "harvest_season": "가을,겨울",
                    "storage_period": 3,
                },
            },
            "ingredient_spinach": {
                "id": "ingredient_spinach",
                "labels": ["Ingredient", "Vegetable"],
                "properties": {
                    "name": "시금치",
                    "english_name": "spinach",
                    "category": "vegetable",
                    "iron": 2.7,
                    "vitamin_a": 469,
                    "harvest_season": "봄,가을",
                    "storage_period": 1,
                },
            },
            "ingredient_onion": {
                "id": "ingredient_onion",
                "labels": ["Ingredient", "Vegetable"],
                "properties": {
                    "name": "양파",
                    "english_name": "onion",
                    "category": "vegetable",
                    "quercetin": "high",
                    "sulfur_compounds": "high",
                    "harvest_season": "여름",
                    "storage_period": 6,
                },
            },
            # 과일류
            "ingredient_apple": {
                "id": "ingredient_apple",
                "labels": ["Ingredient", "Fruit"],
                "properties": {
                    "name": "사과",
                    "english_name": "apple",
                    "category": "fruit",
                    "vitamin_c": 4.6,
                    "fiber": 2.4,
                    "antioxidant": "high",
                    "harvest_season": "가을",
                    "storage_period": 4,
                },
            },
            "ingredient_strawberry": {
                "id": "ingredient_strawberry",
                "labels": ["Ingredient", "Fruit"],
                "properties": {
                    "name": "딸기",
                    "english_name": "strawberry",
                    "category": "fruit",
                    "vitamin_c": 58.8,
                    "anthocyanin": "high",
                    "harvest_season": "봄",
                    "storage_period": 0.3,
                },
            },
            # 축산물
            "ingredient_beef": {
                "id": "ingredient_beef",
                "labels": ["Ingredient", "Meat"],
                "properties": {
                    "name": "소고기",
                    "english_name": "beef",
                    "category": "meat",
                    "protein": 26.1,
                    "iron": 2.9,
                    "vitamin_b12": 2.6,
                    "storage_period": 3,
                },
            },
            "ingredient_pork": {
                "id": "ingredient_pork",
                "labels": ["Ingredient", "Meat"],
                "properties": {
                    "name": "돼지고기",
                    "english_name": "pork",
                    "category": "meat",
                    "protein": 25.7,
                    "thiamine": 0.7,
                    "storage_period": 3,
                },
            },
            # 수산물
            "ingredient_salmon": {
                "id": "ingredient_salmon",
                "labels": ["Ingredient", "Seafood"],
                "properties": {
                    "name": "연어",
                    "english_name": "salmon",
                    "category": "seafood",
                    "protein": 25.4,
                    "omega3": "very_high",
                    "vitamin_d": 11,
                    "storage_period": 2,
                },
            },
            "ingredient_mackerel": {
                "id": "ingredient_mackerel",
                "labels": ["Ingredient", "Seafood"],
                "properties": {
                    "name": "고등어",
                    "english_name": "mackerel",
                    "category": "seafood",
                    "protein": 23.9,
                    "omega3": "high",
                    "selenium": 44.1,
                    "storage_period": 1,
                },
            },
            # 슈퍼푸드
            "ingredient_quinoa": {
                "id": "ingredient_quinoa",
                "labels": ["Ingredient", "Superfood"],
                "properties": {
                    "name": "퀴노아",
                    "english_name": "quinoa",
                    "category": "pseudocereal",
                    "protein": 14.1,
                    "complete_protein": True,
                    "gluten_free": True,
                    "fiber": 7.0,
                    "magnesium": 197,
                },
            },
            "ingredient_chia": {
                "id": "ingredient_chia",
                "labels": ["Ingredient", "Superfood"],
                "properties": {
                    "name": "치아시드",
                    "english_name": "chia_seed",
                    "category": "seed",
                    "protein": 16.5,
                    "omega3": "very_high",
                    "fiber": 34.4,
                    "calcium": 631,
                },
            },
            "ingredient_avocado": {
                "id": "ingredient_avocado",
                "labels": ["Ingredient", "Superfood"],
                "properties": {
                    "name": "아보카도",
                    "english_name": "avocado",
                    "category": "fruit",
                    "healthy_fats": "very_high",
                    "potassium": 485,
                    "folate": 81,
                    "storage_period": 1,
                },
            },
        }

        # 트렌드 키워드 (확장)
        self.trend_nodes = {
            "trend_plant_based": {
                "id": "trend_plant_based",
                "labels": ["TrendKeyword"],
                "properties": {
                    "keyword": "식물성",
                    "english": "plant_based",
                    "trend_score": 92,
                    "growth_rate": "30%",
                    "peak_season": "전년도",
                    "target_demo": "MZ세대",
                },
            },
            "trend_keto": {
                "id": "trend_keto",
                "labels": ["TrendKeyword"],
                "properties": {
                    "keyword": "케토",
                    "english": "keto",
                    "trend_score": 85,
                    "growth_rate": "25%",
                    "peak_season": "1분기",
                    "target_demo": "30-40대",
                },
            },
            "trend_gluten_free": {
                "id": "trend_gluten_free",
                "labels": ["TrendKeyword"],
                "properties": {
                    "keyword": "글루텐프리",
                    "english": "gluten_free",
                    "trend_score": 78,
                    "growth_rate": "18%",
                    "peak_season": "연중",
                    "target_demo": "건강관심층",
                },
            },
            "trend_fermented": {
                "id": "trend_fermented",
                "labels": ["TrendKeyword"],
                "properties": {
                    "keyword": "발효식품",
                    "english": "fermented",
                    "trend_score": 88,
                    "growth_rate": "35%",
                    "peak_season": "겨울",
                    "target_demo": "전연령",
                },
            },
            "trend_protein": {
                "id": "trend_protein",
                "labels": ["TrendKeyword"],
                "properties": {
                    "keyword": "고단백",
                    "english": "high_protein",
                    "trend_score": 90,
                    "growth_rate": "28%",
                    "peak_season": "여름",
                    "target_demo": "운동인구",
                },
            },
        }

        # 뉴스/기사 노드들 (대폭 확장)
        self.news_nodes = {
            "news_001": {
                "id": "news_001",
                "labels": ["NewsArticle"],
                "properties": {
                    "title": "2025년 식물성 대체육 시장 1조원 돌파 전망",
                    "source": "식품저널",
                    "published_date": "2024-12-15",
                    "url": "https://www.foodnews.co.kr/news/123456",
                    "category": "market_trend",
                    "views": 15420,
                },
            },
            "news_002": {
                "id": "news_002",
                "labels": ["NewsArticle"],
                "properties": {
                    "title": "김치 수출량 역대 최고치 경신, K푸드 열풍 지속",
                    "source": "농민신문",
                    "published_date": "2024-12-18",
                    "url": "https://www.nongmin.com/news/234567",
                    "category": "export",
                    "views": 8930,
                },
            },
            "news_003": {
                "id": "news_003",
                "labels": ["NewsArticle"],
                "properties": {
                    "title": "폭우로 인한 채소 가격 급등, 배추 1포기 5천원 시대",
                    "source": "한국농어민신문",
                    "published_date": "2024-12-20",
                    "url": "https://www.agrinet.co.kr/news/345678",
                    "category": "price_alert",
                    "views": 23450,
                },
            },
            "news_004": {
                "id": "news_004",
                "labels": ["NewsArticle"],
                "properties": {
                    "title": "MZ세대 건강식품 소비 패턴 변화, 슈퍼푸드 인기 급상승",
                    "source": "식품산업신문",
                    "published_date": "2024-12-12",
                    "url": "https://www.foodindustry.co.kr/news/456789",
                    "category": "consumer_trend",
                    "views": 12340,
                },
            },
            "news_005": {
                "id": "news_005",
                "labels": ["NewsArticle"],
                "properties": {
                    "title": "AI 기반 농업 기술 도입으로 생산성 30% 향상",
                    "source": "농업기술신문",
                    "published_date": "2024-12-10",
                    "url": "https://www.agritech.co.kr/news/567890",
                    "category": "technology",
                    "views": 7890,
                },
            },
        }

        # 가격 정보 노드들 (확장)
        self.price_nodes = {
            "price_rice_2024": {
                "id": "price_rice_2024",
                "labels": ["Price"],
                "properties": {
                    "item": "쌀",
                    "date": "2024-12-20",
                    "avg_price": 1850,
                    "unit": "원/kg",
                    "market": "전국평균",
                    "grade": "상품",
                    "change_rate": "+2.1%",
                },
            },
            "price_cabbage_2024": {
                "id": "price_cabbage_2024",
                "labels": ["Price"],
                "properties": {
                    "item": "배추",
                    "date": "2024-12-20",
                    "avg_price": 4200,
                    "unit": "원/포기",
                    "market": "가락시장",
                    "grade": "상품",
                    "change_rate": "+45.2%",
                },
            },
            "price_beef_2024": {
                "id": "price_beef_2024",
                "labels": ["Price"],
                "properties": {
                    "item": "한우",
                    "date": "2024-12-20",
                    "avg_price": 28500,
                    "unit": "원/kg",
                    "market": "전국평균",
                    "grade": "1등급",
                    "change_rate": "-1.2%",
                },
            },
            "price_salmon_2024": {
                "id": "price_salmon_2024",
                "labels": ["Price"],
                "properties": {
                    "item": "연어",
                    "date": "2024-12-20",
                    "avg_price": 12800,
                    "unit": "원/kg",
                    "market": "수입가격",
                    "grade": "특급",
                    "change_rate": "+8.3%",
                },
            },
        }

        # 기업/브랜드 노드들 (추가)
        self.company_nodes = {
            "company_cj": {
                "id": "company_cj",
                "labels": ["Company"],
                "properties": {
                    "name": "CJ제일제당",
                    "english_name": "CJ CheilJedang",
                    "sector": "식품제조",
                    "market_cap": "8.5조원",
                    "main_products": ["햇반", "비비고", "백설"],
                },
            },
            "company_orion": {
                "id": "company_orion",
                "labels": ["Company"],
                "properties": {
                    "name": "오리온",
                    "english_name": "Orion",
                    "sector": "제과",
                    "market_cap": "1.2조원",
                    "main_products": ["초코파이", "오감자", "꼬북칩"],
                },
            },
        }

        # 모든 노드 통합
        self.nodes = {
            **self.ingredient_nodes,
            **self.trend_nodes,
            **self.news_nodes,
            **self.price_nodes,
            **self.company_nodes,
        }

        # 관계 데이터 (대폭 확장)
        self.relationships = [
            # 가격 관계
            {
                "id": "rel_001",
                "type": "HAS_PRICE",
                "start_node": "ingredient_rice",
                "end_node": "price_rice_2024",
                "properties": {"stability": "stable"},
            },
            {
                "id": "rel_002",
                "type": "HAS_PRICE",
                "start_node": "ingredient_cabbage",
                "end_node": "price_cabbage_2024",
                "properties": {"stability": "volatile"},
            },
            {
                "id": "rel_003",
                "type": "HAS_PRICE",
                "start_node": "ingredient_beef",
                "end_node": "price_beef_2024",
                "properties": {"stability": "stable"},
            },
            {
                "id": "rel_004",
                "type": "HAS_PRICE",
                "start_node": "ingredient_salmon",
                "end_node": "price_salmon_2024",
                "properties": {"stability": "moderate"},
            },
            # 트렌드 관계
            {
                "id": "rel_005",
                "type": "TRENDING_WITH",
                "start_node": "ingredient_quinoa",
                "end_node": "trend_plant_based",
                "properties": {"correlation": 0.92},
            },
            {
                "id": "rel_006",
                "type": "TRENDING_WITH",
                "start_node": "ingredient_avocado",
                "end_node": "trend_keto",
                "properties": {"correlation": 0.85},
            },
            {
                "id": "rel_007",
                "type": "TRENDING_WITH",
                "start_node": "ingredient_quinoa",
                "end_node": "trend_gluten_free",
                "properties": {"correlation": 0.88},
            },
            {
                "id": "rel_008",
                "type": "TRENDING_WITH",
                "start_node": "ingredient_beef",
                "end_node": "trend_protein",
                "properties": {"correlation": 0.78},
            },
            # 뉴스 관계
            {
                "id": "rel_009",
                "type": "MENTIONED_IN",
                "start_node": "trend_plant_based",
                "end_node": "news_001",
                "properties": {"sentiment": "positive", "mentions": 12},
            },
            {
                "id": "rel_010",
                "type": "MENTIONED_IN",
                "start_node": "ingredient_cabbage",
                "end_node": "news_003",
                "properties": {"sentiment": "negative", "mentions": 8},
            },
            {
                "id": "rel_011",
                "type": "MENTIONED_IN",
                "start_node": "trend_fermented",
                "end_node": "news_002",
                "properties": {"sentiment": "positive", "mentions": 15},
            },
            # 회사 관계
            {
                "id": "rel_012",
                "type": "PRODUCES",
                "start_node": "company_cj",
                "end_node": "ingredient_rice",
                "properties": {"product_type": "가공식품"},
            },
            {
                "id": "rel_013",
                "type": "PRODUCES",
                "start_node": "company_orion",
                "end_node": "ingredient_wheat",
                "properties": {"product_type": "제과제빵"},
            },
            # 영양소 관계
            {
                "id": "rel_014",
                "type": "RICH_IN",
                "start_node": "ingredient_salmon",
                "end_node": "ingredient_salmon",
                "properties": {"nutrient": "omega3", "level": "very_high"},
            },
            {
                "id": "rel_015",
                "type": "RICH_IN",
                "start_node": "ingredient_spinach",
                "end_node": "ingredient_spinach",
                "properties": {"nutrient": "iron", "level": "high"},
            },
            # 계절성 관계
            {
                "id": "rel_016",
                "type": "SEASONAL",
                "start_node": "ingredient_strawberry",
                "end_node": "ingredient_strawberry",
                "properties": {"season": "spring", "peak_month": "4"},
            },
            {
                "id": "rel_017",
                "type": "SEASONAL",
                "start_node": "ingredient_apple",
                "end_node": "ingredient_apple",
                "properties": {"season": "autumn", "peak_month": "10"},
            },
        ]

        print("- Mock Graph DB 초기화 완료")

    def search(self, query: str) -> Dict[str, Any]:
        """Graph DB 검색 - 대폭 강화된 버전"""
        print(f">> Graph DB 검색 실행: {query}")

        query_lower = query.lower()
        matched_nodes = []
        matched_relationships = []

        # 키워드 매핑 (대폭 확장)
        keyword_mappings = {
            # 곡물류
            ("쌀", "rice", "밥"): ["ingredient_rice", "price_rice_2024"],
            ("밀", "wheat", "밀가루"): ["ingredient_wheat"],
            ("옥수수", "corn"): ["ingredient_corn"],
            ("보리", "barley"): ["ingredient_barley"],
            # 채소류
            ("배추", "cabbage", "김치"): [
                "ingredient_cabbage",
                "price_cabbage_2024",
                "news_002",
                "news_003",
            ],
            ("시금치", "spinach"): ["ingredient_spinach"],
            ("양파", "onion"): ["ingredient_onion"],
            # 과일류
            ("사과", "apple"): ["ingredient_apple"],
            ("딸기", "strawberry"): ["ingredient_strawberry"],
            ("아보카도", "avocado"): ["ingredient_avocado"],
            # 축산물
            ("소고기", "beef", "한우"): ["ingredient_beef", "price_beef_2024"],
            ("돼지고기", "pork"): ["ingredient_pork"],
            # 수산물
            ("연어", "salmon"): ["ingredient_salmon", "price_salmon_2024"],
            ("고등어", "mackerel"): ["ingredient_mackerel"],
            # 슈퍼푸드
            ("퀴노아", "quinoa"): ["ingredient_quinoa"],
            ("치아시드", "chia"): ["ingredient_chia"],
            # 트렌드
            ("식물성", "plant", "비건", "vegan"): ["trend_plant_based", "news_001"],
            ("케토", "keto", "저탄수화물"): ["trend_keto"],
            ("글루텐프리", "gluten"): ["trend_gluten_free"],
            ("발효", "fermented", "프로바이오틱스"): ["trend_fermented", "news_002"],
            ("단백질", "protein", "고단백"): ["trend_protein"],
            # 가격/시장
            ("가격", "price", "시세", "시장"): [
                "price_rice_2024",
                "price_cabbage_2024",
                "price_beef_2024",
                "price_salmon_2024",
            ],
            ("급등", "상승", "폭등"): ["news_003", "price_cabbage_2024"],
            # 뉴스/트렌드
            ("뉴스", "news", "기사"): [
                "news_001",
                "news_002",
                "news_003",
                "news_004",
                "news_005",
            ],
            ("MZ세대", "건강", "소비자"): ["news_004"],
            ("AI", "기술", "농업"): ["news_005"],
            ("수출", "K푸드", "한류"): ["news_002"],
            # 회사
            ("CJ", "제일제당"): ["company_cj"],
            ("오리온", "orion"): ["company_orion"],
        }

        # 매칭된 노드 찾기
        for keywords, node_ids in keyword_mappings.items():
            if any(keyword in query_lower for keyword in keywords):
                for node_id in node_ids:
                    if node_id in self.nodes and node_id not in [
                        n["id"] for n in matched_nodes
                    ]:
                        matched_nodes.append(self.nodes[node_id])

        # 관계 추가
        matched_node_ids = [node["id"] for node in matched_nodes]
        for rel in self.relationships:
            if (
                rel["start_node"] in matched_node_ids
                or rel["end_node"] in matched_node_ids
            ):
                matched_relationships.append(rel)

        # 기본 결과 보장
        if not matched_nodes:
            matched_nodes = [
                self.nodes["ingredient_rice"],
                self.nodes["trend_plant_based"],
            ]
            matched_relationships = [self.relationships[0]]

        print(
            f"- Graph DB 검색 결과: {len(matched_nodes)}개 노드, {len(matched_relationships)}개 관계"
        )

        return {
            "query": query,
            "total_nodes": len(matched_nodes),
            "total_relationships": len(matched_relationships),
            "nodes": matched_nodes,
            "relationships": matched_relationships,
            "execution_time": f"{random.uniform(0.1, 0.8):.2f}s",
            "database": "FoodAgriGraphDB_Enhanced",
        }


class MockVectorDB:
    """식품/농업 도메인 Vector DB - 대폭 강화"""

    def __init__(self):
        print(">> Mock Vector DB 초기화 시작")

        self.documents = [
            {
                "id": "doc_001",
                "title": "2025년 글로벌 식물성 단백질 시장 전망 보고서",
                "content": "글로벌 식물성 단백질 시장이 2025년 1,200억 달러 규모로 성장할 것으로 전망된다. 특히 완두콩, 대두, 퀴노아 기반 제품의 수요가 급증하고 있으며, 국내 시장도 연평균 25% 성장률을 보이고 있다. MZ세대의 환경 의식과 건강 관심이 주요 성장 동력으로 작용하고 있다.",
                "metadata": {
                    "source": "소비자원",
                    "category": "소비자조사",
                    "reliability": 0.89,
                    "published_date": "2024-12-05",
                    "keywords": ["소비패턴", "건강식품", "슈퍼푸드", "온라인구매"],
                },
                "similarity_score": 0.87,
            },
            {
                "id": "doc_005",
                "title": "국내 양식업 현황 및 수산물 가격 동향",
                "content": "국내 양식업이 기술 발전과 함께 성장하고 있다. 연어, 광어, 전복 등의 양식 생산량이 증가하면서 가격 안정화에 기여하고 있다. 특히 연어의 경우 국내 양식 기술 발전으로 수입 의존도를 줄이고 있으며, 노르웨이산 대비 30% 저렴한 가격으로 공급되고 있다.",
                "metadata": {
                    "source": "수산청",
                    "category": "양식업",
                    "reliability": 0.91,
                    "published_date": "2024-12-12",
                    "keywords": ["양식업", "연어", "수산물가격", "국내생산"],
                },
                "similarity_score": 0.84,
            },
            {
                "id": "doc_006",
                "title": "기능성 식품 시장 성장 전망",
                "content": "국내 기능성 식품 시장이 연평균 8.5% 성장하며 2025년 5조원 규모에 달할 것으로 예상된다. 프로바이오틱스, 오메가3, 비타민D 등이 주요 성장 동력이며, 고령화 사회 진입과 건강 관심 증가가 시장 확대를 이끌고 있다.",
                "metadata": {
                    "source": "한국건강기능식품협회",
                    "category": "기능성식품",
                    "reliability": 0.93,
                    "published_date": "2024-12-08",
                    "keywords": ["기능성식품", "프로바이오틱스", "오메가3", "고령화"],
                },
                "similarity_score": 0.89,
            },
            {
                "id": "doc_007",
                "title": "AI 기반 농업 기술 도입 사례",
                "content": "국내 농가에서 AI 기술 도입이 활발해지고 있다. 스마트팜 시설에서 AI를 활용한 생육 관리, 병해충 예측, 수확량 최적화 등이 이뤄지고 있으며, 이를 통해 생산성이 평균 30% 향상되었다. 정부는 2030년까지 스마트팜 확산을 위해 3조원을 투자할 예정이다.",
                "metadata": {
                    "source": "농림축산식품부",
                    "category": "농업기술",
                    "reliability": 0.96,
                    "published_date": "2024-12-01",
                    "keywords": ["AI농업", "스마트팜", "생산성", "정부투자"],
                },
                "similarity_score": 0.86,
            },
            {
                "id": "doc_008",
                "title": "대체육 시장 동향 및 소비자 수용도",
                "content": "국내 대체육 시장이 급성장하고 있으며, 2024년 시장 규모가 500억원을 넘어섰다. 식물성 대체육의 소비자 수용도는 60%에 달하며, 특히 20-30대에서 높은 관심을 보이고 있다. 맛과 식감 개선이 지속적으로 이뤄지면서 시장 확대가 가속화되고 있다.",
                "metadata": {
                    "source": "식품산업협회",
                    "category": "대체육",
                    "reliability": 0.88,
                    "published_date": "2024-11-28",
                    "keywords": ["대체육", "식물성", "소비자수용도", "2030대"],
                },
                "similarity_score": 0.92,
            },
            {
                "id": "doc_009",
                "title": "친환경 농업 확산 정책 및 효과",
                "content": "정부의 친환경 농업 정책이 성과를 보이고 있다. 무농약, 유기농 인증 농가가 전년 대비 15% 증가했으며, 친환경 농산물 생산량도 크게 늘었다. 소비자들의 안전한 먹거리에 대한 관심이 높아지면서 프리미엄 시장도 확대되고 있다.",
                "metadata": {
                    "source": "국립농산물품질관리원",
                    "category": "친환경농업",
                    "reliability": 0.94,
                    "published_date": "2024-11-25",
                    "keywords": ["친환경농업", "유기농", "무농약", "프리미엄시장"],
                },
                "similarity_score": 0.85,
            },
            {
                "id": "doc_010",
                "title": "글로벌 곡물 수급 전망 및 국내 영향",
                "content": "우크라이나 전쟁과 기후변화로 인한 글로벌 곡물 수급 불안이 지속되고 있다. 밀, 옥수수 가격이 연초 대비 20% 이상 상승했으며, 국내 사료값과 밀가루 가격에도 영향을 미치고 있다. 정부는 곡물 비축량 확대와 수입선 다변화를 추진하고 있다.",
                "metadata": {
                    "source": "한국농촌경제연구원",
                    "category": "시장분석",
                    "reliability": 0.95,
                    "published_date": "2024-12-15",
                    "keywords": ["식물성단백질", "시장전망", "퀴노아", "MZ세대"],
                },
                "similarity_score": 0.95,
            },
            {
                "id": "doc_002",
                "title": "기후변화가 국내 농업에 미치는 영향 연구",
                "content": "기후변화로 인한 극한 날씨 현상이 국내 농업 생산성에 심각한 영향을 미치고 있다. 2024년 폭우로 인해 배추, 무, 당근 등 주요 채소류 가격이 평년 대비 40% 이상 상승했다. 정부는 스마트팜 기술 도입과 품종 개량을 통한 대응 방안을 모색하고 있다.",
                "metadata": {
                    "source": "농촌진흥청",
                    "category": "기후영향",
                    "reliability": 0.92,
                    "published_date": "2024-12-18",
                    "keywords": ["기후변화", "채소가격", "배추", "스마트팜"],
                },
                "similarity_score": 0.88,
            },
            {
                "id": "doc_003",
                "title": "K-푸드 해외 진출 성과 및 전략 분석",
                "content": "K-푸드의 해외 진출이 가속화되고 있다. 2024년 김치 수출액이 1억 5천만 달러를 돌파하며 역대 최고치를 기록했다. 라면, 김, 고추장 등도 꾸준한 성장세를 보이고 있으며, 특히 동남아시아와 북미 지역에서의 인기가 높다.",
                "metadata": {
                    "source": "한국농수산식품유통공사",
                    "category": "수출동향",
                    "reliability": 0.94,
                    "published_date": "2024-12-10",
                    "keywords": ["K푸드", "김치", "수출", "한류"],
                },
                "similarity_score": 0.91,
            },
            {
                "id": "doc_004",
                "title": "소비자 식품 구매 패턴 변화 조사",
                "content": "코로나19 이후 소비자들의 식품 구매 패턴이 크게 변화했다. 건강기능식품에 대한 관심이 높아졌으며, 특히 면역력 강화, 항산화 효과가 있는 슈퍼푸드의 인기가 급상승했다. 온라인 구매 비중도 30% 이상 증가했다.",
                "metadata": {
                    "source": "한국농촌경제연구원",
                    "category": "곡물수급",
                    "reliability": 0.97,
                    "published_date": "2024-11-30",
                    "keywords": ["곡물수급", "밀", "옥수수", "우크라이나전쟁"],
                },
                "similarity_score": 0.90,
            },
        ]

        print("- Mock Vector DB 초기화 완료")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """벡터 검색 시뮬레이션 - 대폭 강화"""
        print(f">> Vector DB 검색 실행: {query} (top_k={top_k})")

        query_lower = query.lower()
        results = []

        for doc in self.documents:
            content_text = f"{doc.get('title', '')} {doc.get('content', '')}"
            keywords = doc.get("metadata", {}).get("keywords", [])
            content_lower = content_text.lower()
            keywords_lower = " ".join(keywords).lower()

            # 키워드와 내용 매칭 점수 계산
            query_words = set(re.findall(r"\w+", query_lower))
            content_words = set(re.findall(r"\w+", content_lower))
            keyword_words = set(re.findall(r"\w+", keywords_lower))

            # 다중 매칭 점수 계산
            if query_words and (content_words or keyword_words):
                # 제목/내용 매칭
                content_intersection = len(query_words.intersection(content_words))
                content_union = len(query_words.union(content_words))
                content_score = (
                    content_intersection / content_union if content_union > 0 else 0
                )

                # 키워드 매칭 (가중치 적용)
                keyword_intersection = len(query_words.intersection(keyword_words))
                keyword_union = len(query_words.union(keyword_words))
                keyword_score = (
                    keyword_intersection / keyword_union if keyword_union > 0 else 0
                )

                # 최종 점수 (키워드에 더 높은 가중치)
                final_score = (content_score * 0.4) + (keyword_score * 0.6)

                if final_score > 0.05:  # 임계값
                    doc_copy = doc.copy()
                    doc_copy["similarity_score"] = round(final_score, 3)
                    results.append(doc_copy)

        # 기본 결과 보장
        if not results:
            results = [doc.copy() for doc in self.documents[:3]]
            for i, result in enumerate(results):
                result["similarity_score"] = 0.7 - (i * 0.1)

        # 점수 순으로 정렬
        results = sorted(
            results, key=lambda x: x.get("similarity_score", 0), reverse=True
        )[:top_k]

        print(f"- Vector DB 검색 결과: {len(results)}개 문서")
        return results


class MockRDB:
    """식품/농업 도메인 관계형 DB - 대폭 강화"""

    def __init__(self):
        print(">> Mock RDB 초기화 시작")

        # 농산물 시세 데이터 (대폭 확장)
        self.agricultural_prices = [
            # 곡물류
            {
                "item": "쌀",
                "date": "2024-12-20",
                "region": "전국",
                "market": "산지",
                "avg_price": 1850,
                "unit": "원/kg",
                "grade": "상품",
                "supply_volume": 2500,
                "price_change": "+2.1%",
                "category": "곡물",
            },
            {
                "item": "밀",
                "date": "2024-12-20",
                "region": "전국",
                "market": "수입",
                "avg_price": 420,
                "unit": "원/kg",
                "grade": "1급",
                "supply_volume": 15000,
                "price_change": "+15.2%",
                "category": "곡물",
            },
            {
                "item": "옥수수",
                "date": "2024-12-20",
                "region": "전국",
                "market": "수입",
                "avg_price": 380,
                "unit": "원/kg",
                "grade": "사료용",
                "supply_volume": 25000,
                "price_change": "+12.8%",
                "category": "곡물",
            },
            {
                "item": "보리",
                "date": "2024-12-20",
                "region": "전남",
                "market": "산지",
                "avg_price": 1200,
                "unit": "원/kg",
                "grade": "상품",
                "supply_volume": 800,
                "price_change": "+3.5%",
                "category": "곡물",
            },
            # 채소류
            {
                "item": "배추",
                "date": "2024-12-20",
                "region": "서울",
                "market": "가락시장",
                "avg_price": 4200,
                "unit": "원/포기",
                "grade": "상품",
                "supply_volume": 450,
                "price_change": "+45.2%",
                "category": "채소",
            },
            {
                "item": "무",
                "date": "2024-12-20",
                "region": "서울",
                "market": "가락시장",
                "avg_price": 1800,
                "unit": "원/개",
                "grade": "상품",
                "supply_volume": 380,
                "price_change": "+35.1%",
                "category": "채소",
            },
            {
                "item": "당근",
                "date": "2024-12-20",
                "region": "서울",
                "market": "가락시장",
                "avg_price": 2500,
                "unit": "원/kg",
                "grade": "상품",
                "supply_volume": 220,
                "price_change": "+28.9%",
                "category": "채소",
            },
            {
                "item": "양파",
                "date": "2024-12-20",
                "region": "서울",
                "market": "가락시장",
                "avg_price": 1200,
                "unit": "원/kg",
                "grade": "상품",
                "supply_volume": 890,
                "price_change": "-5.2%",
                "category": "채소",
            },
            {
                "item": "시금치",
                "date": "2024-12-20",
                "region": "서울",
                "market": "가락시장",
                "avg_price": 3800,
                "unit": "원/kg",
                "grade": "상품",
                "supply_volume": 150,
                "price_change": "+18.3%",
                "category": "채소",
            },
            # 과일류
            {
                "item": "사과",
                "date": "2024-12-20",
                "region": "경북",
                "market": "산지",
                "avg_price": 3200,
                "unit": "원/kg",
                "grade": "특품",
                "supply_volume": 1200,
                "price_change": "-2.1%",
                "category": "과일",
            },
            {
                "item": "배",
                "date": "2024-12-20",
                "region": "전남",
                "market": "산지",
                "avg_price": 4500,
                "unit": "원/kg",
                "grade": "특품",
                "supply_volume": 800,
                "price_change": "+1.2%",
                "category": "과일",
            },
            {
                "item": "감귤",
                "date": "2024-12-20",
                "region": "제주",
                "market": "산지",
                "avg_price": 2800,
                "unit": "원/kg",
                "grade": "상품",
                "supply_volume": 2200,
                "price_change": "+8.5%",
                "category": "과일",
            },
            {
                "item": "딸기",
                "date": "2024-12-20",
                "region": "경남",
                "market": "산지",
                "avg_price": 12000,
                "unit": "원/kg",
                "grade": "특품",
                "supply_volume": 450,
                "price_change": "+5.3%",
                "category": "과일",
            },
            # 축산물
            {
                "item": "한우",
                "date": "2024-12-20",
                "region": "전국",
                "market": "도매",
                "avg_price": 28500,
                "unit": "원/kg",
                "grade": "1등급",
                "supply_volume": 120,
                "price_change": "-1.2%",
                "category": "축산",
            },
            {
                "item": "돼지고기",
                "date": "2024-12-20",
                "region": "전국",
                "market": "도매",
                "avg_price": 5800,
                "unit": "원/kg",
                "grade": "상품",
                "supply_volume": 380,
                "price_change": "+2.8%",
                "category": "축산",
            },
            {
                "item": "닭고기",
                "date": "2024-12-20",
                "region": "전국",
                "market": "도매",
                "avg_price": 3200,
                "unit": "원/kg",
                "grade": "상품",
                "supply_volume": 450,
                "price_change": "+1.5%",
                "category": "축산",
            },
            {
                "item": "계란",
                "date": "2024-12-20",
                "region": "전국",
                "market": "도매",
                "avg_price": 2800,
                "unit": "원/30개",
                "grade": "특란",
                "supply_volume": 2200,
                "price_change": "+12.3%",
                "category": "축산",
            },
            # 수산물
            {
                "item": "연어",
                "date": "2024-12-20",
                "region": "전국",
                "market": "수입",
                "avg_price": 12800,
                "unit": "원/kg",
                "grade": "특급",
                "supply_volume": 85,
                "price_change": "+8.3%",
                "category": "수산",
            },
            {
                "item": "고등어",
                "date": "2024-12-20",
                "region": "부산",
                "market": "수협",
                "avg_price": 4500,
                "unit": "원/kg",
                "grade": "상품",
                "supply_volume": 220,
                "price_change": "-3.2%",
                "category": "수산",
            },
            {
                "item": "명태",
                "date": "2024-12-20",
                "region": "전국",
                "market": "수입",
                "avg_price": 8900,
                "unit": "원/kg",
                "grade": "냉동",
                "supply_volume": 180,
                "price_change": "+6.7%",
                "category": "수산",
            },
            {
                "item": "조기",
                "date": "2024-12-20",
                "region": "서해",
                "market": "수협",
                "avg_price": 15000,
                "unit": "원/kg",
                "grade": "상품",
                "supply_volume": 95,
                "price_change": "+15.8%",
                "category": "수산",
            },
        ]

        # 영양 정보 데이터 (대폭 확장)
        self.nutrition_data = [
            {
                "item": "쌀",
                "serving_size": "100g",
                "calories": 130,
                "protein": 2.7,
                "fat": 0.3,
                "carbohydrate": 28.1,
                "fiber": 0.4,
                "sodium": 5,
                "vitamin_b1": 0.02,
            },
            {
                "item": "밀",
                "serving_size": "100g",
                "calories": 339,
                "protein": 10.7,
                "fat": 1.5,
                "carbohydrate": 71.2,
                "fiber": 2.7,
                "iron": 3.2,
                "folate": 26,
            },
            {
                "item": "옥수수",
                "serving_size": "100g",
                "calories": 96,
                "protein": 3.3,
                "fat": 1.4,
                "carbohydrate": 22.8,
                "fiber": 2.4,
                "vitamin_c": 6.8,
                "magnesium": 37,
            },
            {
                "item": "배추",
                "serving_size": "100g",
                "calories": 15,
                "protein": 1.2,
                "fat": 0.1,
                "carbohydrate": 2.8,
                "fiber": 1.2,
                "vitamin_c": 25,
                "calcium": 45,
            },
            {
                "item": "시금치",
                "serving_size": "100g",
                "calories": 23,
                "protein": 2.9,
                "fat": 0.4,
                "carbohydrate": 3.6,
                "fiber": 2.2,
                "iron": 2.7,
                "vitamin_a": 469,
            },
            {
                "item": "당근",
                "serving_size": "100g",
                "calories": 41,
                "protein": 0.9,
                "fat": 0.2,
                "carbohydrate": 9.6,
                "fiber": 2.8,
                "beta_carotene": 8285,
                "potassium": 320,
            },
            {
                "item": "사과",
                "serving_size": "100g",
                "calories": 52,
                "protein": 0.3,
                "fat": 0.2,
                "carbohydrate": 13.8,
                "fiber": 2.4,
                "vitamin_c": 4.6,
                "potassium": 107,
            },
            {
                "item": "딸기",
                "serving_size": "100g",
                "calories": 32,
                "protein": 0.7,
                "fat": 0.3,
                "carbohydrate": 7.7,
                "fiber": 2.0,
                "vitamin_c": 58.8,
                "folate": 24,
            },
            {
                "item": "한우",
                "serving_size": "100g",
                "calories": 250,
                "protein": 26.1,
                "fat": 15.0,
                "carbohydrate": 0,
                "iron": 2.9,
                "zinc": 4.8,
                "vitamin_b12": 2.6,
            },
            {
                "item": "돼지고기",
                "serving_size": "100g",
                "calories": 242,
                "protein": 25.7,
                "fat": 14.6,
                "carbohydrate": 0,
                "thiamine": 0.7,
                "niacin": 4.8,
                "phosphorus": 200,
            },
            {
                "item": "닭고기",
                "serving_size": "100g",
                "calories": 165,
                "protein": 31.0,
                "fat": 3.6,
                "carbohydrate": 0,
                "niacin": 10.9,
                "vitamin_b6": 0.5,
                "selenium": 22.0,
            },
            {
                "item": "연어",
                "serving_size": "100g",
                "calories": 208,
                "protein": 25.4,
                "fat": 12.4,
                "carbohydrate": 0,
                "omega3": 1.8,
                "vitamin_d": 11,
                "selenium": 36.5,
            },
            {
                "item": "고등어",
                "serving_size": "100g",
                "calories": 205,
                "protein": 23.9,
                "fat": 11.9,
                "carbohydrate": 0,
                "omega3": 2.3,
                "vitamin_b12": 19,
                "selenium": 44.1,
            },
            {
                "item": "퀴노아",
                "serving_size": "100g",
                "calories": 368,
                "protein": 14.1,
                "fat": 6.1,
                "carbohydrate": 64.2,
                "fiber": 7.0,
                "magnesium": 197,
                "iron": 4.6,
            },
            {
                "item": "아보카도",
                "serving_size": "100g",
                "calories": 160,
                "protein": 2.0,
                "fat": 14.7,
                "carbohydrate": 8.5,
                "fiber": 6.7,
                "potassium": 485,
                "folate": 81,
            },
        ]

        # 시장 데이터 (대폭 확장)
        self.market_data = [
            {
                "category": "식물성 단백질",
                "year": 2024,
                "market_size_billion_won": 850,
                "growth_rate": "18.5%",
                "forecast_2025_billion_won": 1008,
                "key_players": ["CJ제일제당", "대상", "삼양사"],
                "export_ratio": "15%",
            },
            {
                "category": "기능성 곡물",
                "year": 2024,
                "market_size_billion_won": 1200,
                "growth_rate": "12.3%",
                "forecast_2025_billion_won": 1348,
                "key_players": ["롯데웰푸드", "오뚜기", "농심"],
                "export_ratio": "8%",
            },
            {
                "category": "대체육",
                "year": 2024,
                "market_size_billion_won": 500,
                "growth_rate": "45.2%",
                "forecast_2025_billion_won": 726,
                "key_players": ["지구인컴퍼니", "더플랜잇", "언리미트"],
                "export_ratio": "5%",
            },
            {
                "category": "프로바이오틱스",
                "year": 2024,
                "market_size_billion_won": 780,
                "growth_rate": "22.1%",
                "forecast_2025_billion_won": 952,
                "key_players": ["일동제약", "종근당", "한국야쿠르트"],
                "export_ratio": "12%",
            },
            {
                "category": "유기농 식품",
                "year": 2024,
                "market_size_billion_won": 2100,
                "growth_rate": "8.9%",
                "forecast_2025_billion_won": 2287,
                "key_players": ["풀무원", "초록마을", "오가니아"],
                "export_ratio": "3%",
            },
            {
                "category": "수산양식",
                "year": 2024,
                "market_size_billion_won": 3200,
                "growth_rate": "6.5%",
                "forecast_2025_billion_won": 3408,
                "key_players": ["한국수산", "동원F&B", "사조산업"],
                "export_ratio": "25%",
            },
            {
                "category": "스마트팜",
                "year": 2024,
                "market_size_billion_won": 4500,
                "growth_rate": "15.8%",
                "forecast_2025_billion_won": 5211,
                "key_players": ["LG CNS", "KT", "네이버클라우드"],
                "export_ratio": "0%",
            },
        ]

        # 지역별 생산량 데이터 (추가)
        self.regional_production = [
            {
                "region": "경기",
                "item": "쌀",
                "production_tons": 284000,
                "farmland_hectares": 58000,
                "avg_yield": "4.9t/ha",
                "main_varieties": ["추청", "고시히카리"],
            },
            {
                "region": "전남",
                "item": "쌀",
                "production_tons": 512000,
                "farmland_hectares": 102000,
                "avg_yield": "5.0t/ha",
                "main_varieties": ["신동진", "일미"],
            },
            {
                "region": "제주",
                "item": "감귤",
                "production_tons": 580000,
                "farmland_hectares": 21000,
                "avg_yield": "27.6t/ha",
                "main_varieties": ["온주밀감", "한라봉"],
            },
            {
                "region": "경북",
                "item": "사과",
                "production_tons": 285000,
                "farmland_hectares": 23000,
                "avg_yield": "12.4t/ha",
                "main_varieties": ["후지", "홍로"],
            },
            {
                "region": "충남",
                "item": "배",
                "production_tons": 156000,
                "farmland_hectares": 13500,
                "avg_yield": "11.6t/ha",
                "main_varieties": ["신고", "원황"],
            },
            {
                "region": "강원",
                "item": "감자",
                "production_tons": 245000,
                "farmland_hectares": 18500,
                "avg_yield": "13.2t/ha",
                "main_varieties": ["수미", "대지"],
            },
        ]

        # 소비자 트렌드 데이터 (추가)
        self.consumer_trends = [
            {
                "trend": "식물성 대체식품",
                "interest_score": 92,
                "age_group": "20-30대",
                "growth_period": "2년",
                "main_drivers": ["환경의식", "건강관심", "동물복지"],
            },
            {
                "trend": "홈쿡",
                "interest_score": 88,
                "age_group": "30-40대",
                "growth_period": "3년",
                "main_drivers": ["코로나19", "가족시간", "건강식"],
            },
            {
                "trend": "간편식",
                "interest_score": 85,
                "age_group": "20-30대",
                "growth_period": "5년",
                "main_drivers": ["1인가구", "시간절약", "다양성"],
            },
            {
                "trend": "기능성 식품",
                "interest_score": 79,
                "age_group": "40-50대",
                "growth_period": "4년",
                "main_drivers": ["건강관심", "고령화", "질병예방"],
            },
            {
                "trend": "프리미엄 식품",
                "interest_score": 76,
                "age_group": "30-50대",
                "growth_period": "3년",
                "main_drivers": ["소득증가", "품질중시", "브랜드선호"],
            },
        ]

        print("- Mock RDB 초기화 완료")

    def search(self, query: str) -> Dict[str, Any]:
        """RDB 통합 검색 - 대폭 강화"""
        print(f">> RDB 검색 실행: {query}")

        query_lower = query.lower()
        all_results = {
            "prices": [],
            "nutrition": [],
            "market_data": [],
            "regional_production": [],
            "consumer_trends": [],
        }

        # 가격 정보 검색 (강화)
        if any(
            keyword in query_lower
            for keyword in ["가격", "시세", "price", "급등", "급락", "상승", "하락"]
        ):
            # 특정 품목 검색
            for price in self.agricultural_prices:
                if any(
                    item in query_lower for item in [price["item"], price["category"]]
                ):
                    all_results["prices"].append(price)

            # 카테고리별 검색
            if "곡물" in query_lower:
                all_results["prices"].extend(
                    [p for p in self.agricultural_prices if p["category"] == "곡물"]
                )
            if "채소" in query_lower:
                all_results["prices"].extend(
                    [p for p in self.agricultural_prices if p["category"] == "채소"]
                )
            if "과일" in query_lower:
                all_results["prices"].extend(
                    [p for p in self.agricultural_prices if p["category"] == "과일"]
                )
            if "축산" in query_lower:
                all_results["prices"].extend(
                    [p for p in self.agricultural_prices if p["category"] == "축산"]
                )
            if "수산" in query_lower:
                all_results["prices"].extend(
                    [p for p in self.agricultural_prices if p["category"] == "수산"]
                )

        # 영양 정보 검색 (강화)
        if any(
            keyword in query_lower
            for keyword in [
                "영양",
                "성분",
                "nutrition",
                "단백질",
                "비타민",
                "칼슘",
                "철분",
            ]
        ):
            for nutrition in self.nutrition_data:
                if nutrition["item"] in query_lower:
                    all_results["nutrition"].append(nutrition)

        # 시장 데이터 검색 (강화)
        if any(
            keyword in query_lower
            for keyword in ["시장", "market", "규모", "성장", "전망"]
        ):
            for market in self.market_data:
                if any(word in query_lower for word in market["category"].split()):
                    all_results["market_data"].append(market)

        # 지역별 생산량 검색 (추가)
        if any(
            keyword in query_lower
            for keyword in [
                "지역",
                "생산",
                "production",
                "경기",
                "전남",
                "제주",
                "경북",
                "충남",
                "강원",
            ]
        ):
            for production in self.regional_production:
                if (
                    production["region"] in query_lower
                    or production["item"] in query_lower
                ):
                    all_results["regional_production"].append(production)

        # 소비자 트렌드 검색 (추가)
        if any(
            keyword in query_lower
            for keyword in ["트렌드", "trend", "소비자", "consumer", "인기", "유행"]
        ):
            for trend in self.consumer_trends:
                if any(word in query_lower for word in trend["trend"].split()):
                    all_results["consumer_trends"].append(trend)

        # 기본 결과 보장 (카테고리별)
        if not any(all_results.values()):
            all_results["prices"] = self.agricultural_prices[:5]
            all_results["nutrition"] = self.nutrition_data[:3]
            all_results["market_data"] = self.market_data[:2]
            all_results["regional_production"] = self.regional_production[:2]
            all_results["consumer_trends"] = self.consumer_trends[:2]

        # 중복 제거
        for key in all_results:
            seen = set()
            unique_results = []
            for item in all_results[key]:
                item_id = str(item)
                if item_id not in seen:
                    seen.add(item_id)
                    unique_results.append(item)
            all_results[key] = unique_results

        total_results = sum(len(results) for results in all_results.values())

        print(f"- RDB 검색 결과: {total_results}개 레코드")

        return {
            "query": query,
            "total_results": total_results,
            "data": all_results,
            "database": "Enhanced_RelationalDB_FoodAgri",
        }


class MockWebSearch:
    """웹 검색 시뮬레이션 - 대폭 강화"""

    def __init__(self):
        print(">> Mock Web Search 초기화 시작")

        # 검색 결과 데이터 (대폭 확장)
        self.search_results = {
            "식물성 단백질": [
                {
                    "title": "2025년 글로벌 식물성 단백질 시장 1,200억 달러 돌파 전망",
                    "url": "https://www.foodbiz.co.kr/news/article2025123",
                    "snippet": "글로벌 식물성 단백질 시장이 연평균 15% 성장하며 2025년 1,200억 달러 규모에 달할 것으로 예상된다. 완두콩, 대두, 퀴노아 기반 제품이 주도하고 있다.",
                    "published_date": "2024-12-22",
                    "relevance": 0.95,
                    "source_type": "industry_news",
                },
                {
                    "title": "국내 식물성 대체육 시장 급성장, MZ세대 견인",
                    "url": "https://www.agrinet.co.kr/plant-meat-trend2024",
                    "snippet": "국내 식물성 대체육 시장이 전년 대비 45% 성장했다. 특히 20-30대 소비자층에서 높은 관심을 보이며 시장 확대를 견인하고 있다.",
                    "published_date": "2024-12-18",
                    "relevance": 0.92,
                    "source_type": "market_analysis",
                },
            ],
            "완두콩": [
                {
                    "title": "완두콩 기반 단백질 제품 출시 러시, 글로벌 트렌드 반영",
                    "url": "https://www.foodnews.co.kr/pea-protein-boom",
                    "snippet": "국내 식품업계에서 완두콩 기반 단백질 제품 출시가 잇따르고 있다. CJ제일제당, 대상 등 주요 업체들이 시장 선점에 나섰다.",
                    "published_date": "2024-12-20",
                    "relevance": 0.88,
                    "source_type": "industry_news",
                },
                {
                    "title": "완두콩 수입가격 상승세, 공급량 부족 영향",
                    "url": "https://www.agrinews.co.kr/pea-price-surge",
                    "snippet": "완두콩 수입가격이 전월 대비 8% 상승했다. 주요 수출국인 캐나다의 생산량 감소가 원인으로 분석된다.",
                    "published_date": "2024-12-19",
                    "relevance": 0.85,
                    "source_type": "price_alert",
                },
            ],
            "퀴노아": [
                {
                    "title": "퀴노아 열풍 지속, 슈퍼푸드 시장 견인",
                    "url": "https://www.healthfood.co.kr/quinoa-trend2024",
                    "snippet": "퀴노아가 K-푸드 트렌드와 함께 국내 건강식품 시장에서 급부상하고 있다. 글루텐프리 특성과 완전단백질 함유로 주목받고 있다.",
                    "published_date": "2024-12-17",
                    "relevance": 0.92,
                    "source_type": "health_trend",
                },
                {
                    "title": "퀴노아 재배 기술 개발, 국내 생산 가능성 높아져",
                    "url": "https://www.ruralnews.co.kr/quinoa-cultivation",
                    "snippet": "농촌진흥청이 개발한 퀴노아 재배 기술로 국내 생산 가능성이 높아졌다. 수입 의존도를 줄이고 농가 소득 증대에 기여할 것으로 기대된다.",
                    "published_date": "2024-12-15",
                    "relevance": 0.87,
                    "source_type": "agricultural_tech",
                },
            ],
            "배추": [
                {
                    "title": "폭우 피해로 배추값 급등, 1포기 5천원 시대",
                    "url": "https://www.farmernews.co.kr/cabbage-price-spike",
                    "snippet": "연이은 폭우로 인한 배추 작황 부진으로 가격이 급등했다. 평년 대비 40% 이상 오르며 소비자 부담이 가중되고 있다.",
                    "published_date": "2024-12-21",
                    "relevance": 0.94,
                    "source_type": "price_alert",
                },
                {
                    "title": "배추 비축량 확대, 가격 안정화 방안 모색",
                    "url": "https://www.agripolicy.co.kr/cabbage-reserve",
                    "snippet": "정부가 배추 가격 안정화를 위해 비축량 확대와 수급 조절 방안을 검토하고 있다. 내년 봄까지 가격 상승세가 지속될 것으로 전망된다.",
                    "published_date": "2024-12-20",
                    "relevance": 0.89,
                    "source_type": "policy_news",
                },
            ],
            "연어": [
                {
                    "title": "국내 연어 양식 기술 발전, 수입 의존도 감소",
                    "url": "https://www.fishery.co.kr/salmon-aquaculture",
                    "snippet": "국내 연어 양식 기술이 크게 발전하면서 수입 의존도가 줄어들고 있다. 노르웨이산 대비 30% 저렴한 가격으로 공급 가능해졌다.",
                    "published_date": "2024-12-16",
                    "relevance": 0.91,
                    "source_type": "aquaculture_news",
                },
                {
                    "title": "연어 소비량 급증, 오메가3 효능 주목받아",
                    "url": "https://www.healthnews.co.kr/salmon-omega3",
                    "snippet": "연어 소비량이 전년 대비 25% 증가했다. 오메가3 지방산의 건강 효능이 알려지면서 건강 관심층을 중심으로 인기가 높아지고 있다.",
                    "published_date": "2024-12-14",
                    "relevance": 0.88,
                    "source_type": "health_news",
                },
            ],
            "기능성 식품": [
                {
                    "title": "국내 기능성 식품 시장 5조원 돌파, 고령화 영향",
                    "url": "https://www.functional-food.co.kr/market-5trillion",
                    "snippet": "국내 기능성 식품 시장이 5조원을 돌파했다. 고령화 사회 진입과 건강에 대한 관심 증가가 시장 성장을 이끌고 있다.",
                    "published_date": "2024-12-13",
                    "relevance": 0.93,
                    "source_type": "market_report",
                },
                {
                    "title": "프로바이오틱스 제품 다양화, 면역력 강화 효과 주목",
                    "url": "https://www.probiotics.co.kr/immunity-boost",
                    "snippet": "프로바이오틱스 제품이 다양화되고 있다. 면역력 강화 효과가 입증되면서 코로나19 이후 수요가 급증하고 있다.",
                    "published_date": "2024-12-11",
                    "relevance": 0.89,
                    "source_type": "health_product",
                },
            ],
            "AI 농업": [
                {
                    "title": "AI 기반 스마트팜 확산, 생산성 30% 향상",
                    "url": "https://www.smartfarm.co.kr/ai-productivity",
                    "snippet": "AI 기술을 도입한 스마트팜에서 생산성이 평균 30% 향상된 것으로 나타났다. 정부는 2030년까지 3조원을 투자해 스마트팜을 확산할 계획이다.",
                    "published_date": "2024-12-10",
                    "relevance": 0.96,
                    "source_type": "agtech_news",
                },
                {
                    "title": "농업용 드론 활용 급증, 정밀농업 시대 개막",
                    "url": "https://www.agridrone.co.kr/precision-farming",
                    "snippet": "농업용 드론 활용이 급증하고 있다. 작물 모니터링, 방제 작업 등에 활용되면서 정밀농업 시대가 본격 개막되고 있다.",
                    "published_date": "2024-12-08",
                    "relevance": 0.91,
                    "source_type": "technology",
                },
            ],
            "K푸드": [
                {
                    "title": "김치 수출 1억5천만 달러 돌파, 역대 최고치",
                    "url": "https://www.kfood-export.co.kr/kimchi-record",
                    "snippet": "김치 수출액이 1억 5천만 달러를 돌파하며 역대 최고치를 기록했다. 한류 열풍과 함께 K-푸드의 세계화가 가속화되고 있다.",
                    "published_date": "2024-12-12",
                    "relevance": 0.94,
                    "source_type": "export_news",
                },
                {
                    "title": "라면·김 등 K푸드 동남아 진출 확대",
                    "url": "https://www.asiafood.co.kr/kfood-sea-expansion",
                    "snippet": "라면, 김, 고추장 등 K-푸드의 동남아 진출이 확대되고 있다. 현지 입맛에 맞춘 제품 개발로 시장 점유율을 높이고 있다.",
                    "published_date": "2024-12-09",
                    "relevance": 0.88,
                    "source_type": "market_expansion",
                },
            ],
            "친환경 농업": [
                {
                    "title": "유기농 인증 농가 15% 증가, 친환경 농업 확산",
                    "url": "https://www.organic-farm.co.kr/certification-increase",
                    "snippet": "유기농 인증 농가가 전년 대비 15% 증가했다. 소비자들의 안전한 먹거리에 대한 관심이 높아지면서 친환경 농업이 확산되고 있다.",
                    "published_date": "2024-12-07",
                    "relevance": 0.92,
                    "source_type": "sustainable_agriculture",
                },
                {
                    "title": "무농약 채소 프리미엄 30% 상승, 시장 확대",
                    "url": "https://www.pesticide-free.co.kr/premium-increase",
                    "snippet": "무농약 채소의 프리미엄이 30% 상승했다. 건강과 환경을 중시하는 소비 트렌드가 확산되면서 친환경 농산물 시장이 확대되고 있다.",
                    "published_date": "2024-12-05",
                    "relevance": 0.87,
                    "source_type": "market_trend",
                },
            ],
        }

        # 시장 동향 데이터 (확장)
        self.market_trends = [
            {
                "trend": "식물성 대체육",
                "growth_rate": "45%",
                "market_size": "500억원",
                "key_factors": ["환경의식", "건강관심", "동물복지", "MZ세대"],
                "forecast": "2025년 726억원 전망",
            },
            {
                "trend": "슈퍼푸드",
                "growth_rate": "22%",
                "market_size": "2.1조원",
                "key_factors": ["프리미엄화", "건강기능성", "다양성", "고령화"],
                "forecast": "연평균 15% 성장 지속",
            },
            {
                "trend": "스마트팜",
                "growth_rate": "16%",
                "market_size": "4.5조원",
                "key_factors": ["AI기술", "생산성향상", "인력절약", "정부지원"],
                "forecast": "2030년 10조원 규모",
            },
            {
                "trend": "기능성 식품",
                "growth_rate": "8.5%",
                "market_size": "5조원",
                "key_factors": ["고령화", "건강관심", "면역력", "개인맞춤"],
                "forecast": "지속적 성장 전망",
            },
            {
                "trend": "친환경 농업",
                "growth_rate": "12%",
                "market_size": "1.8조원",
                "key_factors": ["환경보호", "안전먹거리", "지속가능성", "정책지원"],
                "forecast": "유기농 확산 가속화",
            },
        ]

        print("- Mock Web Search 초기화 완료")

    def search(self, query: str) -> Dict[str, Any]:
        """웹 검색 시뮬레이션 - 대폭 강화"""
        print(f">> Web Search 검색 실행: {query}")

        query_lower = query.lower()
        results = []

        # 키워드별 검색 결과 매칭 (확장)
        for keyword, articles in self.search_results.items():
            keyword_parts = keyword.split()
            if (
                any(part.lower() in query_lower for part in keyword_parts)
                or keyword.lower() in query_lower
            ):
                results.extend(articles)

        # 추가 키워드 매칭 로직
        additional_keywords = {
            ("시장", "전망", "성장", "규모"): ["식물성 단백질", "기능성 식품"],
            ("가격", "급등", "상승", "시세"): ["완두콩", "배추"],
            ("건강", "영양", "오메가3"): ["연어", "기능성 식품"],
            ("수출", "한류", "김치"): ["K푸드"],
            ("농업", "기술", "드론", "스마트"): ["AI 농업"],
            ("환경", "유기농", "무농약"): ["친환경 농업"],
            ("양식", "수산", "어업"): ["연어"],
            ("날씨", "폭우", "기후"): ["배추"],
        }

        for keywords, categories in additional_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                for category in categories:
                    if category in self.search_results:
                        results.extend(self.search_results[category])

        # 중복 제거
        seen_urls = set()
        unique_results = []
        for result in results:
            if result["url"] not in seen_urls:
                unique_results.append(result)
                seen_urls.add(result["url"])

        # 기본 결과 보장
        if not unique_results:
            # 인기 검색 결과 제공
            default_results = [
                {
                    "title": "2024년 식품산업 종합 동향 리포트",
                    "url": "https://www.foodindustry.co.kr/2024-report",
                    "snippet": "2024년 식품산업의 주요 동향과 시장 전망을 종합 분석한 리포트입니다. 식물성 식품, 기능성 식품 등 주요 트렌드를 다루고 있습니다.",
                    "published_date": "2024-12-01",
                    "relevance": 0.75,
                    "source_type": "industry_report",
                },
                {
                    "title": "농식품 시장 변화와 소비자 트렌드",
                    "url": "https://www.agritrend.co.kr/consumer-trend",
                    "snippet": "농식품 시장의 변화와 소비자 트렌드를 분석한 자료입니다. MZ세대의 소비 패턴과 건강식품 시장 동향을 포함하고 있습니다.",
                    "published_date": "2024-11-28",
                    "relevance": 0.72,
                    "source_type": "market_analysis",
                },
            ]
            unique_results = default_results

        # 관련도 순으로 정렬
        unique_results = sorted(
            unique_results, key=lambda x: x.get("relevance", 0), reverse=True
        )

        print(f"- Web Search 검색 결과: {len(unique_results)}개 기사")

        return {
            "total_results": len(unique_results),
            "query": query,
            "results": unique_results,
            "trends": self.market_trends,
            "database": "Enhanced_WebSearch_FoodAgri",
        }


def create_mock_databases():
    """모든 Enhanced Mock Database 인스턴스를 생성하고 반환"""
    print(">> Enhanced Mock Databases 초기화 시작")

    graph_db = MockGraphDB()
    vector_db = MockVectorDB()
    rdb = MockRDB()
    web_search = MockWebSearch()

    print("======= Enhanced Mock Databases 초기화 완료 =======")
    print(
        f"Graph DB: {len(graph_db.nodes)}개 노드, {len(graph_db.relationships)}개 관계"
    )
    print(f"Vector DB: {len(vector_db.documents)}개 문서")
    print(f"RDB: 5개 테이블 (가격, 영양, 시장, 지역생산, 소비자트렌드)")
    print(
        f"Web Search: {len(web_search.search_results)}개 카테고리, {len(web_search.market_trends)}개 트렌드"
    )

    return graph_db, vector_db, rdb, web_search


def test_enhanced_databases():
    """Enhanced Mock Database 기능 테스트"""
    print("======= Enhanced Mock Database 기능 테스트 =======")

    graph_db, vector_db, rdb, web_search = create_mock_databases()

    # 다양한 쿼리로 테스트
    test_queries = [
        "완두콩 가격 트렌드",
        "식물성 단백질 시장 전망",
        "배추 급등 원인",
        "연어 양식 기술",
        "K푸드 수출 현황",
        "기능성 식품 시장 규모",
        "AI 농업 기술",
        "친환경 농업 동향",
    ]

    for query in test_queries:
        print(f"\n=== 테스트 쿼리: '{query}' ===")

        # Graph DB 테스트
        graph_result = graph_db.search(query)
        print(f"Graph DB: {graph_result['total_nodes']}개 노드")

        # Vector DB 테스트
        vector_result = vector_db.search(query, top_k=3)
        print(f"Vector DB: {len(vector_result)}개 문서")

        # RDB 테스트
        rdb_result = rdb.search(query)
        print(f"RDB: {rdb_result['total_results']}개 레코드")

        # Web Search 테스트
        web_result = web_search.search(query)
        print(f"Web Search: {web_result['total_results']}개 기사")

    print("\n모든 Enhanced Database 정상 작동")


if __name__ == "__main__":
    # Enhanced 테스트 실행
    test_enhanced_databases()
