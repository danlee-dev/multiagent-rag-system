import json
import random
import re
from typing import Dict, List, Any
from datetime import datetime, timedelta


class MockGraphDB:
    """식품/농업 도메인 Graph DB - 친환경 트렌드 식재료 대폭 추가"""

    def __init__(self):
        print(">> Mock Graph DB 초기화 시작")
        # 기존 재료 노드들
        self.ingredient_nodes = {
            # 기존 곡물류
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
            # ========== 새로운 고대곡물/슈퍼그레인 추가 ==========
            "ingredient_amaranth": {
                "id": "ingredient_amaranth",
                "labels": ["Ingredient", "Superfood", "AncientGrain"],
                "properties": {
                    "name": "아마란스",
                    "english_name": "amaranth",
                    "category": "ancient_grain",
                    "protein": 13.6,
                    "carbs": 65.2,
                    "fiber": 6.7,
                    "lysine": "complete",
                    "gluten_free": True,
                    "origin": "남미",
                    "trend_score": 88,
                    "eco_friendly": True,
                    "drought_resistant": True,
                },
            },
            "ingredient_buckwheat": {
                "id": "ingredient_buckwheat",
                "labels": ["Ingredient", "Superfood", "AncientGrain"],
                "properties": {
                    "name": "메밀",
                    "english_name": "buckwheat",
                    "category": "ancient_grain",
                    "protein": 13.2,
                    "carbs": 71.5,
                    "rutin": "very_high",
                    "gluten_free": True,
                    "origin": "중앙아시아",
                    "trend_score": 82,
                    "eco_friendly": True,
                    "cold_resistant": True,
                },
            },
            "ingredient_teff": {
                "id": "ingredient_teff",
                "labels": ["Ingredient", "Superfood", "AncientGrain"],
                "properties": {
                    "name": "테프",
                    "english_name": "teff",
                    "category": "ancient_grain",
                    "protein": 13.3,
                    "carbs": 73.1,
                    "iron": 7.6,
                    "calcium": 180,
                    "gluten_free": True,
                    "origin": "에티오피아",
                    "trend_score": 91,
                    "eco_friendly": True,
                    "climate_resilient": True,
                },
            },
            "ingredient_millet": {
                "id": "ingredient_millet",
                "labels": ["Ingredient", "Superfood", "AncientGrain"],
                "properties": {
                    "name": "기장",
                    "english_name": "millet",
                    "category": "ancient_grain",
                    "protein": 11.0,
                    "carbs": 72.8,
                    "magnesium": 114,
                    "gluten_free": True,
                    "origin": "아프리카",
                    "trend_score": 85,
                    "eco_friendly": True,
                    "water_efficient": True,
                },
            },
            "ingredient_farro": {
                "id": "ingredient_farro",
                "labels": ["Ingredient", "Superfood", "AncientGrain"],
                "properties": {
                    "name": "파로",
                    "english_name": "farro",
                    "category": "ancient_grain",
                    "protein": 15.1,
                    "carbs": 67.1,
                    "fiber": 7.8,
                    "origin": "이탈리아",
                    "trend_score": 79,
                    "eco_friendly": True,
                    "sustainable_farming": True,
                },
            },
            # ========== 새로운 슈퍼씨드 추가 ==========
            "ingredient_hemp_seed": {
                "id": "ingredient_hemp_seed",
                "labels": ["Ingredient", "Superfood", "Seed"],
                "properties": {
                    "name": "햄프시드",
                    "english_name": "hemp_seed",
                    "category": "super_seed",
                    "protein": 31.6,
                    "omega3": "high",
                    "omega6": "high",
                    "complete_protein": True,
                    "trend_score": 92,
                    "eco_friendly": True,
                    "carbon_negative": True,
                },
            },
            "ingredient_flax_seed": {
                "id": "ingredient_flax_seed",
                "labels": ["Ingredient", "Superfood", "Seed"],
                "properties": {
                    "name": "아마씨",
                    "english_name": "flax_seed",
                    "category": "super_seed",
                    "protein": 18.3,
                    "omega3": "very_high",
                    "lignans": "highest",
                    "fiber": 27.3,
                    "trend_score": 87,
                    "eco_friendly": True,
                },
            },
            "ingredient_pumpkin_seed": {
                "id": "ingredient_pumpkin_seed",
                "labels": ["Ingredient", "Superfood", "Seed"],
                "properties": {
                    "name": "호박씨",
                    "english_name": "pumpkin_seed",
                    "category": "super_seed",
                    "protein": 30.2,
                    "zinc": 7.8,
                    "magnesium": 592,
                    "trend_score": 76,
                    "eco_friendly": True,
                    "zero_waste": True,
                },
            },
            "ingredient_sunflower_seed": {
                "id": "ingredient_sunflower_seed",
                "labels": ["Ingredient", "Superfood", "Seed"],
                "properties": {
                    "name": "해바라기씨",
                    "english_name": "sunflower_seed",
                    "category": "super_seed",
                    "protein": 20.8,
                    "vitamin_e": 35.2,
                    "selenium": 53,
                    "trend_score": 73,
                    "eco_friendly": True,
                    "pollinator_friendly": True,
                },
            },
            # ========== 새로운 슈퍼푸드 채소/과일 추가 ==========
            "ingredient_moringa": {
                "id": "ingredient_moringa",
                "labels": ["Ingredient", "Superfood", "Leaf"],
                "properties": {
                    "name": "모링가",
                    "english_name": "moringa",
                    "category": "super_leaf",
                    "protein": 27.1,
                    "vitamin_c": 220,
                    "calcium": 2003,
                    "iron": 28.2,
                    "trend_score": 94,
                    "eco_friendly": True,
                    "drought_resistant": True,
                },
            },
            "ingredient_spirulina": {
                "id": "ingredient_spirulina",
                "labels": ["Ingredient", "Superfood", "Algae"],
                "properties": {
                    "name": "스피룰리나",
                    "english_name": "spirulina",
                    "category": "super_algae",
                    "protein": 57.5,
                    "chlorophyll": "very_high",
                    "phycocyanin": "highest",
                    "trend_score": 89,
                    "eco_friendly": True,
                    "carbon_capture": True,
                },
            },
            "ingredient_chlorella": {
                "id": "ingredient_chlorella",
                "labels": ["Ingredient", "Superfood", "Algae"],
                "properties": {
                    "name": "클로렐라",
                    "english_name": "chlorella",
                    "category": "super_algae",
                    "protein": 45.0,
                    "chlorophyll": "extremely_high",
                    "cgf": "unique",
                    "trend_score": 83,
                    "eco_friendly": True,
                    "minimal_resources": True,
                },
            },
            "ingredient_acai": {
                "id": "ingredient_acai",
                "labels": ["Ingredient", "Superfood", "Berry"],
                "properties": {
                    "name": "아사이베리",
                    "english_name": "acai",
                    "category": "super_berry",
                    "antioxidant": "extremely_high",
                    "anthocyanins": "highest",
                    "origin": "아마존",
                    "trend_score": 91,
                    "eco_friendly": True,
                    "rainforest_preservation": True,
                },
            },
            "ingredient_goji": {
                "id": "ingredient_goji",
                "labels": ["Ingredient", "Superfood", "Berry"],
                "properties": {
                    "name": "고지베리",
                    "english_name": "goji_berry",
                    "category": "super_berry",
                    "vitamin_c": 48.4,
                    "zeaxanthin": "highest",
                    "polysaccharides": "unique",
                    "trend_score": 84,
                    "eco_friendly": True,
                    "traditional_medicine": True,
                },
            },
            "ingredient_cacao": {
                "id": "ingredient_cacao",
                "labels": ["Ingredient", "Superfood", "Bean"],
                "properties": {
                    "name": "카카오닙스",
                    "english_name": "cacao_nibs",
                    "category": "super_bean",
                    "flavonoids": "very_high",
                    "theobromine": "natural",
                    "magnesium": 272,
                    "trend_score": 86,
                    "eco_friendly": True,
                    "fair_trade": True,
                },
            },
            # ========== 새로운 발효식품 추가 ==========
            "ingredient_tempeh": {
                "id": "ingredient_tempeh",
                "labels": ["Ingredient", "Fermented", "Protein"],
                "properties": {
                    "name": "템페",
                    "english_name": "tempeh",
                    "category": "fermented_protein",
                    "protein": 19.0,
                    "probiotics": "high",
                    "vitamin_b12": "natural",
                    "trend_score": 88,
                    "eco_friendly": True,
                    "plant_based": True,
                },
            },
            "ingredient_miso": {
                "id": "ingredient_miso",
                "labels": ["Ingredient", "Fermented", "Seasoning"],
                "properties": {
                    "name": "미소",
                    "english_name": "miso",
                    "category": "fermented_seasoning",
                    "probiotics": "very_high",
                    "umami": "intense",
                    "isoflavones": "high",
                    "trend_score": 82,
                    "eco_friendly": True,
                    "traditional": True,
                },
            },
            "ingredient_kombucha_scoby": {
                "id": "ingredient_kombucha_scoby",
                "labels": ["Ingredient", "Fermented", "Probiotic"],
                "properties": {
                    "name": "콤부차스코비",
                    "english_name": "kombucha_scoby",
                    "category": "fermented_culture",
                    "probiotics": "diverse",
                    "antioxidants": "high",
                    "trend_score": 79,
                    "eco_friendly": True,
                    "zero_waste": True,
                },
            },
            # ========== 새로운 대체단백질 추가 ==========
            "ingredient_pea_protein": {
                "id": "ingredient_pea_protein",
                "labels": ["Ingredient", "PlantProtein", "Alternative"],
                "properties": {
                    "name": "완두콩단백질",
                    "english_name": "pea_protein",
                    "category": "plant_protein",
                    "protein": 80.0,
                    "bcaa": "complete",
                    "allergen_free": True,
                    "trend_score": 95,
                    "eco_friendly": True,
                    "nitrogen_fixing": True,
                },
            },
            "ingredient_cricket_protein": {
                "id": "ingredient_cricket_protein",
                "labels": ["Ingredient", "InsectProtein", "Alternative"],
                "properties": {
                    "name": "귀뚜라미단백질",
                    "english_name": "cricket_protein",
                    "category": "insect_protein",
                    "protein": 65.0,
                    "vitamin_b12": "high",
                    "water_efficiency": "extremely_high",
                    "trend_score": 76,
                    "eco_friendly": True,
                    "future_food": True,
                },
            },
            # ========== 기존 재료들 ==========
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

        # ========== 대폭 확장된 트렌드 키워드 ==========
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
            "trend_ancient_grains": {
                "id": "trend_ancient_grains",
                "labels": ["TrendKeyword"],
                "properties": {
                    "keyword": "고대곡물",
                    "english": "ancient_grains",
                    "trend_score": 89,
                    "growth_rate": "42%",
                    "peak_season": "연중",
                    "target_demo": "건강관심층",
                },
            },
            "trend_superfood": {
                "id": "trend_superfood",
                "labels": ["TrendKeyword"],
                "properties": {
                    "keyword": "슈퍼푸드",
                    "english": "superfood",
                    "trend_score": 95,
                    "growth_rate": "38%",
                    "peak_season": "연중",
                    "target_demo": "전연령",
                },
            },
            "trend_sustainable": {
                "id": "trend_sustainable",
                "labels": ["TrendKeyword"],
                "properties": {
                    "keyword": "지속가능",
                    "english": "sustainable",
                    "trend_score": 93,
                    "growth_rate": "45%",
                    "peak_season": "연중",
                    "target_demo": "친환경의식층",
                },
            },
            "trend_zero_waste": {
                "id": "trend_zero_waste",
                "labels": ["TrendKeyword"],
                "properties": {
                    "keyword": "제로웨이스트",
                    "english": "zero_waste",
                    "trend_score": 87,
                    "growth_rate": "52%",
                    "peak_season": "연중",
                    "target_demo": "밀레니얼",
                },
            },
            "trend_functional": {
                "id": "trend_functional",
                "labels": ["TrendKeyword"],
                "properties": {
                    "keyword": "기능성",
                    "english": "functional",
                    "trend_score": 91,
                    "growth_rate": "35%",
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
            "trend_adaptogenic": {
                "id": "trend_adaptogenic",
                "labels": ["TrendKeyword"],
                "properties": {
                    "keyword": "어댑토겐",
                    "english": "adaptogenic",
                    "trend_score": 84,
                    "growth_rate": "68%",
                    "peak_season": "연중",
                    "target_demo": "스트레스관리층",
                },
            },
            "trend_nootropic": {
                "id": "trend_nootropic",
                "labels": ["TrendKeyword"],
                "properties": {
                    "keyword": "뇌건강",
                    "english": "nootropic",
                    "trend_score": 82,
                    "growth_rate": "55%",
                    "peak_season": "연중",
                    "target_demo": "직장인",
                },
            },
        }

        # ========== 대폭 확장된 뉴스/기사 노드들 ==========
        self.news_nodes = {
            # 기존 뉴스들
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
            # 새로운 뉴스들 추가
            "news_eco_001": {
                "id": "news_eco_001",
                "labels": ["NewsArticle"],
                "properties": {
                    "title": "고대곡물 열풍, 테프와 아마란스 수입량 300% 급증",
                    "source": "친환경식품뉴스",
                    "published_date": "2024-12-22",
                    "url": "https://www.ecofoodnews.co.kr/ancient-grains-boom",
                    "category": "eco_trend",
                    "views": 23450,
                },
            },
            "news_eco_002": {
                "id": "news_eco_002",
                "labels": ["NewsArticle"],
                "properties": {
                    "title": "햄프시드 합법화 이후 국내 생산 첫 시작, 탄소중립 농업의 새 희망",
                    "source": "지속가능농업신문",
                    "published_date": "2024-12-20",
                    "url": "https://www.sustainableagri.co.kr/hemp-seed-legal",
                    "category": "sustainable_agriculture",
                    "views": 18930,
                },
            },
            "news_eco_003": {
                "id": "news_eco_003",
                "labels": ["NewsArticle"],
                "properties": {
                    "title": "스피룰리나 국내 양식장 확대, 미래 단백질원으로 주목",
                    "source": "바이오식품타임즈",
                    "published_date": "2024-12-18",
                    "url": "https://www.biofoodtimes.co.kr/spirulina-farms",
                    "category": "alternative_protein",
                    "views": 16720,
                },
            },
            "news_eco_004": {
                "id": "news_eco_004",
                "labels": ["NewsArticle"],
                "properties": {
                    "title": "모링가 파우더 수요 급증, 아프리카 공정무역으로 윈윈 효과",
                    "source": "공정무역매거진",
                    "published_date": "2024-12-16",
                    "url": "https://www.fairtrademagazine.co.kr/moringa-demand",
                    "category": "fair_trade",
                    "views": 12340,
                },
            },
            "news_eco_005": {
                "id": "news_eco_005",
                "labels": ["NewsArticle"],
                "properties": {
                    "title": "발효식품 템페 인기 급상승, 국내 제조업체 러브콜",
                    "source": "발효식품전문지",
                    "published_date": "2024-12-14",
                    "url": "https://www.fermentedfoods.co.kr/tempeh-popularity",
                    "category": "fermented_trend",
                    "views": 9850,
                },
            },
            "news_eco_006": {
                "id": "news_eco_006",
                "labels": ["NewsArticle"],
                "properties": {
                    "title": "곤충단백질 법제화 완료, 귀뚜라미 제품 상용화 본격 시작",
                    "source": "미래식품리포트",
                    "published_date": "2024-12-12",
                    "url": "https://www.futurefood.co.kr/insect-protein-legal",
                    "category": "future_food",
                    "views": 14230,
                },
            },
            "news_eco_007": {
                "id": "news_eco_007",
                "labels": ["NewsArticle"],
                "properties": {
                    "title": "아사이베리 직수입 늘어, 레인포레스트 보존 캠페인과 함께",
                    "source": "열대과일뉴스",
                    "published_date": "2024-12-10",
                    "url": "https://www.tropicalfruits.co.kr/acai-import",
                    "category": "rainforest_conservation",
                    "views": 8760,
                },
            },
            "news_eco_008": {
                "id": "news_eco_008",
                "labels": ["NewsArticle"],
                "properties": {
                    "title": "제로웨이스트 식품포장재 혁신, 식용 필름으로 주목",
                    "source": "패키징혁신지",
                    "published_date": "2024-12-08",
                    "url": "https://www.packaging-innovation.co.kr/edible-packaging",
                    "category": "zero_waste_packaging",
                    "views": 11450,
                },
            },
        }

        # ========== 대폭 확장된 가격 정보 노드들 ==========
        self.price_nodes = {
            # 기존 가격들
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
            # 새로운 친환경 트렌드 식재료 가격들
            "price_quinoa_2024": {
                "id": "price_quinoa_2024",
                "labels": ["Price"],
                "properties": {
                    "item": "퀴노아",
                    "date": "2024-12-20",
                    "avg_price": 8500,
                    "unit": "원/kg",
                    "market": "수입가격",
                    "grade": "유기농",
                    "change_rate": "+12.3%",
                },
            },
            "price_amaranth_2024": {
                "id": "price_amaranth_2024",
                "labels": ["Price"],
                "properties": {
                    "item": "아마란스",
                    "date": "2024-12-20",
                    "avg_price": 12000,
                    "unit": "원/kg",
                    "market": "수입가격",
                    "grade": "유기농",
                    "change_rate": "+25.4%",
                },
            },
            "price_teff_2024": {
                "id": "price_teff_2024",
                "labels": ["Price"],
                "properties": {
                    "item": "테프",
                    "date": "2024-12-20",
                    "avg_price": 15800,
                    "unit": "원/kg",
                    "market": "수입가격",
                    "grade": "유기농",
                    "change_rate": "+18.7%",
                },
            },
            "price_hemp_seed_2024": {
                "id": "price_hemp_seed_2024",
                "labels": ["Price"],
                "properties": {
                    "item": "햄프시드",
                    "date": "2024-12-20",
                    "avg_price": 22000,
                    "unit": "원/kg",
                    "market": "국내외혼합",
                    "grade": "유기농",
                    "change_rate": "+35.2%",
                },
            },
            "price_moringa_2024": {
                "id": "price_moringa_2024",
                "labels": ["Price"],
                "properties": {
                    "item": "모링가파우더",
                    "date": "2024-12-20",
                    "avg_price": 45000,
                    "unit": "원/kg",
                    "market": "공정무역",
                    "grade": "유기농",
                    "change_rate": "+8.9%",
                },
            },
            "price_spirulina_2024": {
                "id": "price_spirulina_2024",
                "labels": ["Price"],
                "properties": {
                    "item": "스피룰리나",
                    "date": "2024-12-20",
                    "avg_price": 38000,
                    "unit": "원/kg",
                    "market": "국내양식",
                    "grade": "프리미엄",
                    "change_rate": "-5.2%",
                },
            },
            "price_acai_2024": {
                "id": "price_acai_2024",
                "labels": ["Price"],
                "properties": {
                    "item": "아사이파우더",
                    "date": "2024-12-20",
                    "avg_price": 52000,
                    "unit": "원/kg",
                    "market": "직수입",
                    "grade": "프리즈드라이",
                    "change_rate": "+15.6%",
                },
            },
            "price_cricket_protein_2024": {
                "id": "price_cricket_protein_2024",
                "labels": ["Price"],
                "properties": {
                    "item": "귀뚜라미단백질",
                    "date": "2024-12-20",
                    "avg_price": 28000,
                    "unit": "원/kg",
                    "market": "국내생산",
                    "grade": "식품등급",
                    "change_rate": "-12.8%",
                },
            },
        }

        # ========== 새로운 친환경 인증기관 노드들 ==========
        self.certification_nodes = {
            "cert_organic": {
                "id": "cert_organic",
                "labels": ["Certification"],
                "properties": {
                    "name": "유기농인증",
                    "english": "organic_certification",
                    "authority": "국립농산물품질관리원",
                    "validity_period": 1,
                    "requirements": ["3년무농약", "화학비료금지", "GMO금지"],
                },
            },
            "cert_fair_trade": {
                "id": "cert_fair_trade",
                "labels": ["Certification"],
                "properties": {
                    "name": "공정무역인증",
                    "english": "fair_trade",
                    "authority": "국제공정무역기구",
                    "social_impact": "high",
                    "premium_rate": "10-15%",
                },
            },
            "cert_rainforest": {
                "id": "cert_rainforest",
                "labels": ["Certification"],
                "properties": {
                    "name": "레인포레스트얼라이언스",
                    "english": "rainforest_alliance",
                    "authority": "Rainforest Alliance",
                    "focus": "환경보호",
                    "biodiversity": "high",
                },
            },
            "cert_carbon_neutral": {
                "id": "cert_carbon_neutral",
                "labels": ["Certification"],
                "properties": {
                    "name": "탄소중립인증",
                    "english": "carbon_neutral",
                    "authority": "한국환경공단",
                    "scope": "전과정평가",
                    "trend_score": 94,
                },
            },
        }

        # 모든 노드 통합
        self.nodes = {
            **self.ingredient_nodes,
            **self.trend_nodes,
            **self.news_nodes,
            **self.price_nodes,
            **self.certification_nodes,
        }

        # ========== 대폭 확장된 관계 데이터 ==========
        self.relationships = [
            # 기존 관계들 (일부)
            {
                "id": "rel_001",
                "type": "HAS_PRICE",
                "start_node": "ingredient_quinoa",
                "end_node": "price_quinoa_2024",
                "properties": {"stability": "rising"},
            },
            # 새로운 슈퍼푸드 가격 관계
            {
                "id": "rel_eco_001",
                "type": "HAS_PRICE",
                "start_node": "ingredient_amaranth",
                "end_node": "price_amaranth_2024",
                "properties": {"stability": "volatile", "trend": "rising"},
            },
            {
                "id": "rel_eco_002",
                "type": "HAS_PRICE",
                "start_node": "ingredient_teff",
                "end_node": "price_teff_2024",
                "properties": {"stability": "rising", "supply": "limited"},
            },
            {
                "id": "rel_eco_003",
                "type": "HAS_PRICE",
                "start_node": "ingredient_hemp_seed",
                "end_node": "price_hemp_seed_2024",
                "properties": {"stability": "emerging", "regulation": "new"},
            },
            {
                "id": "rel_eco_004",
                "type": "HAS_PRICE",
                "start_node": "ingredient_moringa",
                "end_node": "price_moringa_2024",
                "properties": {"stability": "stable", "fair_trade": True},
            },
            {
                "id": "rel_eco_005",
                "type": "HAS_PRICE",
                "start_node": "ingredient_spirulina",
                "end_node": "price_spirulina_2024",
                "properties": {"stability": "stable", "domestic_production": True},
            },
            {
                "id": "rel_eco_006",
                "type": "HAS_PRICE",
                "start_node": "ingredient_acai",
                "end_node": "price_acai_2024",
                "properties": {"stability": "volatile", "import_dependent": True},
            },
            {
                "id": "rel_eco_007",
                "type": "HAS_PRICE",
                "start_node": "ingredient_cricket_protein",
                "end_node": "price_cricket_protein_2024",
                "properties": {"stability": "decreasing", "scaling_effect": True},
            },
            # 트렌드 관계들
            {
                "id": "rel_trend_001",
                "type": "TRENDING_WITH",
                "start_node": "ingredient_amaranth",
                "end_node": "trend_ancient_grains",
                "properties": {"correlation": 0.94, "strength": "very_high"},
            },
            {
                "id": "rel_trend_002",
                "type": "TRENDING_WITH",
                "start_node": "ingredient_teff",
                "end_node": "trend_ancient_grains",
                "properties": {"correlation": 0.91, "strength": "very_high"},
            },
            {
                "id": "rel_trend_003",
                "type": "TRENDING_WITH",
                "start_node": "ingredient_hemp_seed",
                "end_node": "trend_sustainable",
                "properties": {"correlation": 0.96, "strength": "extremely_high"},
            },
            {
                "id": "rel_trend_004",
                "type": "TRENDING_WITH",
                "start_node": "ingredient_moringa",
                "end_node": "trend_superfood",
                "properties": {"correlation": 0.93, "strength": "very_high"},
            },
            {
                "id": "rel_trend_005",
                "type": "TRENDING_WITH",
                "start_node": "ingredient_spirulina",
                "end_node": "trend_superfood",
                "properties": {"correlation": 0.89, "strength": "high"},
            },
            {
                "id": "rel_trend_006",
                "type": "TRENDING_WITH",
                "start_node": "ingredient_tempeh",
                "end_node": "trend_fermented",
                "properties": {"correlation": 0.87, "strength": "high"},
            },
            {
                "id": "rel_trend_007",
                "type": "TRENDING_WITH",
                "start_node": "ingredient_cricket_protein",
                "end_node": "trend_sustainable",
                "properties": {"correlation": 0.85, "strength": "high"},
            },
            {
                "id": "rel_trend_008",
                "type": "TRENDING_WITH",
                "start_node": "ingredient_acai",
                "end_node": "trend_zero_waste",
                "properties": {"correlation": 0.78, "strength": "moderate"},
            },
            # 뉴스 관계들
            {
                "id": "rel_news_001",
                "type": "MENTIONED_IN",
                "start_node": "trend_ancient_grains",
                "end_node": "news_eco_001",
                "properties": {"sentiment": "very_positive", "mentions": 15},
            },
            {
                "id": "rel_news_002",
                "type": "MENTIONED_IN",
                "start_node": "ingredient_hemp_seed",
                "end_node": "news_eco_002",
                "properties": {"sentiment": "positive", "mentions": 22},
            },
            {
                "id": "rel_news_003",
                "type": "MENTIONED_IN",
                "start_node": "ingredient_spirulina",
                "end_node": "news_eco_003",
                "properties": {"sentiment": "positive", "mentions": 18},
            },
            {
                "id": "rel_news_004",
                "type": "MENTIONED_IN",
                "start_node": "ingredient_moringa",
                "end_node": "news_eco_004",
                "properties": {"sentiment": "positive", "mentions": 12},
            },
            {
                "id": "rel_news_005",
                "type": "MENTIONED_IN",
                "start_node": "ingredient_tempeh",
                "end_node": "news_eco_005",
                "properties": {"sentiment": "positive", "mentions": 8},
            },
            {
                "id": "rel_news_006",
                "type": "MENTIONED_IN",
                "start_node": "ingredient_cricket_protein",
                "end_node": "news_eco_006",
                "properties": {"sentiment": "mixed", "mentions": 25},
            },
            # 인증 관계들
            {
                "id": "rel_cert_001",
                "type": "HAS_CERTIFICATION",
                "start_node": "ingredient_quinoa",
                "end_node": "cert_organic",
                "properties": {"compliance_rate": "95%"},
            },
            {
                "id": "rel_cert_002",
                "type": "HAS_CERTIFICATION",
                "start_node": "ingredient_moringa",
                "end_node": "cert_fair_trade",
                "properties": {"compliance_rate": "88%"},
            },
            {
                "id": "rel_cert_003",
                "type": "HAS_CERTIFICATION",
                "start_node": "ingredient_acai",
                "end_node": "cert_rainforest",
                "properties": {"compliance_rate": "92%"},
            },
            {
                "id": "rel_cert_004",
                "type": "HAS_CERTIFICATION",
                "start_node": "ingredient_hemp_seed",
                "end_node": "cert_carbon_neutral",
                "properties": {"compliance_rate": "97%"},
            },
            # 영양소/기능성 관계들
            {
                "id": "rel_nutrition_001",
                "type": "RICH_IN",
                "start_node": "ingredient_amaranth",
                "end_node": "ingredient_amaranth",
                "properties": {"nutrient": "lysine", "level": "complete_protein"},
            },
            {
                "id": "rel_nutrition_002",
                "type": "RICH_IN",
                "start_node": "ingredient_moringa",
                "end_node": "ingredient_moringa",
                "properties": {"nutrient": "vitamin_c", "level": "extremely_high"},
            },
            {
                "id": "rel_nutrition_003",
                "type": "RICH_IN",
                "start_node": "ingredient_spirulina",
                "end_node": "ingredient_spirulina",
                "properties": {"nutrient": "protein", "level": "complete_protein"},
            },
            {
                "id": "rel_nutrition_004",
                "type": "RICH_IN",
                "start_node": "ingredient_hemp_seed",
                "end_node": "ingredient_hemp_seed",
                "properties": {"nutrient": "omega_ratio", "level": "optimal_3_6"},
            },
            # 지속가능성 관계들
            {
                "id": "rel_sustain_001",
                "type": "SUPPORTS",
                "start_node": "ingredient_hemp_seed",
                "end_node": "trend_sustainable",
                "properties": {"impact": "carbon_sequestration", "score": 9.5},
            },
            {
                "id": "rel_sustain_002",
                "type": "SUPPORTS",
                "start_node": "ingredient_cricket_protein",
                "end_node": "trend_sustainable",
                "properties": {"impact": "water_efficiency", "score": 9.8},
            },
            {
                "id": "rel_sustain_003",
                "type": "SUPPORTS",
                "start_node": "ingredient_acai",
                "end_node": "trend_zero_waste",
                "properties": {"impact": "rainforest_preservation", "score": 8.7},
            },
        ]

        print("- Enhanced Mock Graph DB 초기화 완료")

    def search(self, query: str) -> Dict[str, Any]:
        """Graph DB 검색 - 친환경 트렌드 대응 강화"""
        print(f">> Enhanced Graph DB 검색 실행: {query}")

        query_lower = query.lower()
        matched_nodes = []
        matched_relationships = []

        # 대폭 확장된 키워드 매핑
        keyword_mappings = {
            # 친환경 트렌드 키워드
            ("친환경", "유기농", "organic", "eco", "sustainable"): [
                "trend_sustainable",
                "cert_organic",
                "cert_fair_trade",
                "cert_carbon_neutral",
                "news_eco_001",
                "news_eco_002",
            ],
            ("지속가능", "탄소중립", "제로웨이스트"): [
                "trend_sustainable",
                "trend_zero_waste",
                "cert_carbon_neutral",
                "ingredient_hemp_seed",
                "ingredient_cricket_protein",
            ],
            # 고대곡물/슈퍼그레인
            ("고대곡물", "ancient", "그레인", "아마란스", "amaranth"): [
                "ingredient_amaranth",
                "trend_ancient_grains",
                "price_amaranth_2024",
                "news_eco_001",
            ],
            ("테프", "teff"): [
                "ingredient_teff",
                "price_teff_2024",
                "trend_ancient_grains",
            ],
            ("메밀", "buckwheat"): ["ingredient_buckwheat", "trend_ancient_grains"],
            ("기장", "millet"): ["ingredient_millet", "trend_ancient_grains"],
            ("파로", "farro"): ["ingredient_farro", "trend_ancient_grains"],
            # 슈퍼씨드
            ("햄프", "hemp", "햄프시드"): [
                "ingredient_hemp_seed",
                "price_hemp_seed_2024",
                "trend_sustainable",
                "news_eco_002",
            ],
            ("아마씨", "flax"): ["ingredient_flax_seed", "trend_superfood"],
            ("호박씨", "pumpkin"): ["ingredient_pumpkin_seed", "trend_zero_waste"],
            ("해바라기", "sunflower"): [
                "ingredient_sunflower_seed",
                "trend_sustainable",
            ],
            # 슈퍼푸드
            ("모링가", "moringa"): [
                "ingredient_moringa",
                "price_moringa_2024",
                "cert_fair_trade",
                "news_eco_004",
            ],
            ("스피룰리나", "spirulina"): [
                "ingredient_spirulina",
                "price_spirulina_2024",
                "news_eco_003",
                "trend_superfood",
            ],
            ("클로렐라", "chlorella"): ["ingredient_chlorella", "trend_superfood"],
            ("아사이", "acai"): [
                "ingredient_acai",
                "price_acai_2024",
                "cert_rainforest",
                "news_eco_007",
            ],
            ("고지베리", "goji"): ["ingredient_goji", "trend_superfood"],
            ("카카오", "cacao"): ["ingredient_cacao", "cert_fair_trade"],
            # 발효식품
            ("템페", "tempeh"): [
                "ingredient_tempeh",
                "trend_fermented",
                "news_eco_005",
            ],
            ("미소", "miso"): ["ingredient_miso", "trend_fermented"],
            ("콤부차", "kombucha"): ["ingredient_kombucha_scoby", "trend_fermented"],
            # 대체단백질
            ("완두콩단백질", "pea protein"): [
                "ingredient_pea_protein",
                "trend_plant_based",
            ],
            ("곤충단백질", "cricket", "귀뚜라미"): [
                "ingredient_cricket_protein",
                "price_cricket_protein_2024",
                "news_eco_006",
                "trend_sustainable",
            ],
            # 기능성/트렌드
            ("슈퍼푸드", "superfood"): [
                "trend_superfood",
                "ingredient_moringa",
                "ingredient_spirulina",
                "ingredient_acai",
                "ingredient_quinoa",
                "ingredient_chia",
            ],
            ("기능성", "functional"): [
                "trend_functional",
                "trend_adaptogenic",
                "trend_nootropic",
            ],
            ("발효", "fermented", "프로바이오틱스"): [
                "trend_fermented",
                "ingredient_tempeh",
                "ingredient_miso",
            ],
            # 신제품 개발 관련
            ("신제품", "개발", "트렌드", "새로운"): [
                "trend_superfood",
                "trend_ancient_grains",
                "trend_sustainable",
                "ingredient_amaranth",
                "ingredient_teff",
                "ingredient_moringa",
            ],
            ("추천", "recommend", "인기"): [
                "ingredient_hemp_seed",
                "ingredient_spirulina",
                "ingredient_tempeh",
                "trend_ancient_grains",
                "trend_sustainable",
            ],
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
            # 친환경 트렌드 기본 결과
            default_nodes = [
                "ingredient_amaranth",
                "ingredient_hemp_seed",
                "ingredient_moringa",
                "trend_superfood",
                "trend_sustainable",
                "trend_ancient_grains",
            ]
            matched_nodes = [
                self.nodes[node_id]
                for node_id in default_nodes
                if node_id in self.nodes
            ]
            matched_relationships = self.relationships[:10]

        print(
            f"- Enhanced Graph DB 검색 결과: {len(matched_nodes)}개 노드, {len(matched_relationships)}개 관계"
        )

        return {
            "query": query,
            "total_nodes": len(matched_nodes),
            "total_relationships": len(matched_relationships),
            "nodes": matched_nodes,
            "relationships": matched_relationships,
            "execution_time": f"{random.uniform(0.1, 0.8):.2f}s",
            "database": "Enhanced_FoodAgriGraphDB_EcoTrend",
        }


# ========== Enhanced Vector DB ==========
class MockVectorDB:
    """식품/농업 도메인 Vector DB - 친환경 트렌드 문서 대폭 추가"""

    def __init__(self):
        print(">> Enhanced Mock Vector DB 초기화 시작")

        self.documents = [
            # 기존 문서들 (일부)
            {
                "id": "doc_001",
                "title": "2025년 글로벌 식물성 단백질 시장 전망 보고서",
                "content": "글로벌 식물성 단백질 시장이 2025년 1,200억 달러 규모로 성장할 것으로 전망된다. 특히 완두콩, 대두, 퀴노아 기반 제품의 수요가 급증하고 있으며, 국내 시장도 연평균 25% 성장률을 보이고 있다. MZ세대의 환경 의식과 건강 관심이 주요 성장 동력으로 작용하고 있다.",
                "metadata": {
                    "source": "소비자원",
                    "category": "시장전망",
                    "reliability": 0.89,
                    "published_date": "2024-12-05",
                    "keywords": ["식물성단백질", "시장전망", "퀴노아", "MZ세대"],
                },
                "similarity_score": 0.87,
            },
            # ========== 새로운 친환경 트렌드 문서들 ==========
            {
                "id": "doc_eco_001",
                "title": "고대곡물 열풍, 아마란스와 테프의 영양학적 가치 재조명",
                "content": "아마란스와 테프 등 고대곡물이 현대 식품업계의 새로운 트렌드로 떠오르고 있다. 아마란스는 완전단백질을 함유하고 있어 비건 식단의 핵심 재료로 주목받고 있으며, 테프는 철분과 칼슘이 풍부해 영양 결핍을 예방하는 슈퍼푸드로 인정받고 있다. 또한 이들 곡물은 기후변화에 강한 내성을 가져 지속가능한 농업의 대안으로도 각광받고 있다.",
                "metadata": {
                    "source": "한국영양학회",
                    "category": "영양연구",
                    "reliability": 0.95,
                    "published_date": "2024-12-18",
                    "keywords": [
                        "고대곡물",
                        "아마란스",
                        "테프",
                        "완전단백질",
                        "지속가능",
                    ],
                },
                "similarity_score": 0.94,
            },
            {
                "id": "doc_eco_002",
                "title": "햄프시드 합법화 이후 국내 슈퍼푸드 시장 변화",
                "content": "햄프시드(대마씨)의 식품 사용이 합법화되면서 국내 슈퍼푸드 시장에 큰 변화가 일고 있다. 햄프시드는 오메가3와 오메가6의 이상적인 비율(1:3)을 가지고 있어 항염 효과가 뛰어나며, 완전단백질 공급원으로서의 가치가 높다. 특히 탄소 포집 능력이 뛰어나 재배 과정에서 환경에 긍정적 영향을 미치는 것으로 평가받고 있다.",
                "metadata": {
                    "source": "친환경식품연구소",
                    "category": "신소재연구",
                    "reliability": 0.92,
                    "published_date": "2024-12-15",
                    "keywords": [
                        "햄프시드",
                        "합법화",
                        "오메가지방산",
                        "탄소포집",
                        "환경친화",
                    ],
                },
                "similarity_score": 0.92,
            },
            {
                "id": "doc_eco_003",
                "title": "모링가, 아프리카 기적의 나무가 선사하는 영양 혁신",
                "content": "모링가는 '기적의 나무'라 불리는 만큼 뛰어난 영양학적 가치를 지니고 있다. 비타민C는 오렌지의 7배, 칼슘은 우유의 4배, 단백질은 요거트의 2배나 함유하고 있어 완전식품으로 평가받고 있다. 특히 건조한 지역에서도 잘 자라는 특성으로 기후변화 시대의 대안 작물로 주목받고 있으며, 공정무역을 통한 아프리카 농가 지원 효과도 크다.",
                "metadata": {
                    "source": "국제개발협력기구",
                    "category": "공정무역",
                    "reliability": 0.91,
                    "published_date": "2024-12-12",
                    "keywords": [
                        "모링가",
                        "기적의나무",
                        "공정무역",
                        "아프리카",
                        "완전식품",
                    ],
                },
                "similarity_score": 0.89,
            },
            {
                "id": "doc_eco_004",
                "title": "스피룰리나 국내 양식 기술 혁신, 미래 단백질의 주역으로",
                "content": "스피룰리나가 미래 단백질원으로 급부상하고 있다. 단백질 함량이 57%에 달해 소고기보다 높으며, 필수아미노산을 모두 포함하고 있다. 국내에서는 첨단 바이오리액터 기술을 활용한 대량 생산 시설이 확충되고 있으며, 기존 축산업 대비 99% 적은 토지와 물을 사용하면서도 더 많은 단백질을 생산할 수 있어 지속가능한 식품 시스템의 핵심으로 평가받고 있다.",
                "metadata": {
                    "source": "한국바이오산업협회",
                    "category": "바이오기술",
                    "reliability": 0.94,
                    "published_date": "2024-12-10",
                    "keywords": [
                        "스피룰리나",
                        "바이오리액터",
                        "미래단백질",
                        "지속가능",
                        "양식기술",
                    ],
                },
                "similarity_score": 0.91,
            },
            {
                "id": "doc_eco_005",
                "title": "발효식품 템페의 재발견, 인도네시아 전통이 만든 현대 슈퍼푸드",
                "content": "인도네시아 전통 발효식품인 템페가 서구 건강식품 시장에서 큰 주목을 받고 있다. 콩을 리조푸스 곰팡이로 발효시킨 템페는 소화하기 쉬운 단백질과 비타민B12를 자연적으로 함유하고 있어 비건 식단의 필수 요소로 자리잡고 있다. 또한 발효 과정에서 생성되는 프로바이오틱스는 장 건강 개선에 탁월한 효과를 보이며, 가공 과정이 간단해 친환경적인 단백질 공급원으로 평가받고 있다.",
                "metadata": {
                    "source": "발효식품연구원",
                    "category": "발효기술",
                    "reliability": 0.88,
                    "published_date": "2024-12-08",
                    "keywords": [
                        "템페",
                        "발효식품",
                        "프로바이오틱스",
                        "비타민B12",
                        "인도네시아",
                    ],
                },
                "similarity_score": 0.86,
            },
            {
                "id": "doc_eco_006",
                "title": "곤충단백질 상용화 원년, 귀뚜라미에서 찾은 지속가능한 미래",
                "content": "곤충단백질 식품 허용 법안 통과로 귀뚜라미 단백질이 본격적인 상용화 단계에 접어들었다. 귀뚜라미는 소고기 대비 2천배 적은 물과 12배 적은 사료로 같은 양의 단백질을 생산할 수 있어 환경 효율성이 뛰어나다. 또한 비타민B12와 철분이 풍부하며, 키틴질로 인한 면역력 강화 효과도 기대되고 있다. 현재 파우더, 바, 스낵 형태로 다양한 제품이 출시되고 있다.",
                "metadata": {
                    "source": "미래식품기술원",
                    "category": "대체단백질",
                    "reliability": 0.87,
                    "published_date": "2024-12-05",
                    "keywords": [
                        "곤충단백질",
                        "귀뚜라미",
                        "환경효율성",
                        "비타민B12",
                        "키틴질",
                    ],
                },
                "similarity_score": 0.83,
            },
            {
                "id": "doc_eco_007",
                "title": "아사이베리 직거래 확산, 아마존 보호와 공정무역의 동반 성장",
                "content": "아사이베리의 국내 수요 증가와 함께 아마존 원주민과의 직거래가 확산되고 있다. 아사이베리는 ORAC(항산화 지수) 값이 블루베리의 10배에 달하는 강력한 항산화 효과를 지니고 있어 안티에이징 식품으로 인기가 높다. 직거래를 통해 아마존 원주민들의 경제적 자립을 돕고 열대우림 보전에도 기여하고 있어, 윤리적 소비와 환경 보호를 동시에 실현하는 모범 사례로 평가받고 있다.",
                "metadata": {
                    "source": "공정무역위원회",
                    "category": "윤리적소비",
                    "reliability": 0.90,
                    "published_date": "2024-12-03",
                    "keywords": [
                        "아사이베리",
                        "직거래",
                        "아마존보호",
                        "항산화",
                        "윤리적소비",
                    ],
                },
                "similarity_score": 0.88,
            },
            {
                "id": "doc_eco_008",
                "title": "제로웨이스트 식품 포장의 혁신, 식용 필름이 열어가는 새로운 길",
                "content": "식품 포장재의 환경 문제를 해결하기 위한 식용 필름 기술이 급속도로 발전하고 있다. 해조류, 키토산, 밀 글루텐 등을 원료로 한 생분해성 포장재는 기존 플라스틱을 대체하면서도 식품의 신선도를 유지하는 데 효과적이다. 특히 호박씨, 해바라기씨 껍질 등 농업 부산물을 활용한 포장재 개발이 활발해 순환경제 실현에도 기여하고 있다.",
                "metadata": {
                    "source": "친환경포장재연구소",
                    "category": "제로웨이스트",
                    "reliability": 0.89,
                    "published_date": "2024-12-01",
                    "keywords": [
                        "제로웨이스트",
                        "식용필름",
                        "생분해성",
                        "순환경제",
                        "농업부산물",
                    ],
                },
                "similarity_score": 0.85,
            },
            {
                "id": "doc_eco_009",
                "title": "기능성 버섯류의 부상, 라이온스 메인과 코디셉스의 뇌건강 효과",
                "content": "기능성 버섯류가 뇌건강 식품 시장의 새로운 트렌드로 떠오르고 있다. 라이온스 메인(노루궁뎅이버섯)은 신경성장인자(NGF) 생성을 촉진하여 인지기능 개선에 도움을 주며, 코디셉스는 산소 이용 효율을 높여 뇌 기능 향상에 기여한다. 이들 버섯은 커피, 차, 보충제 형태로 가공되어 일상 속에서 쉽게 섭취할 수 있어 바쁜 현대인들의 뇌건강 관리 솔루션으로 주목받고 있다.",
                "metadata": {
                    "source": "기능성식품연구원",
                    "category": "뇌건강",
                    "reliability": 0.92,
                    "published_date": "2024-11-28",
                    "keywords": [
                        "기능성버섯",
                        "라이온스메인",
                        "코디셉스",
                        "뇌건강",
                        "인지기능",
                    ],
                },
                "similarity_score": 0.87,
            },
            {
                "id": "doc_eco_010",
                "title": "해양 슈퍼푸드 클로렐라의 재조명, 미세조류가 만드는 영양 혁명",
                "content": "클로렐라가 해양 슈퍼푸드로 재조명받고 있다. 클로렐라에만 존재하는 고유 성분인 CGF(클로렐라 성장인자)는 세포 재생과 면역력 강화에 탁월한 효과를 보인다. 또한 클로로필 함량이 식물 중 최고 수준으로 해독 작용이 뛰어나며, 완전단백질과 비타민B12를 자연적으로 함유하고 있어 비건 식단의 필수 보충제로 자리잡고 있다. 최근에는 태블릿뿐만 아니라 스무디, 에너지바 등 다양한 형태로 제품화되고 있다.",
                "metadata": {
                    "source": "해양바이오연구원",
                    "category": "해양식품",
                    "reliability": 0.93,
                    "published_date": "2024-11-25",
                    "keywords": [
                        "클로렐라",
                        "CGF",
                        "클로로필",
                        "해독작용",
                        "해양슈퍼푸드",
                    ],
                },
                "similarity_score": 0.90,
            },
            {
                "id": "doc_eco_011",
                "title": "완두콩 단백질의 기술 혁신, 맛과 기능성을 동시에 잡다",
                "content": "완두콩 단백질이 식물성 단백질 시장의 게임 체인저로 떠오르고 있다. 기존 대두 단백질의 알레르기 문제를 해결하면서도 필수아미노산 프로필이 우수해 운동선수들 사이에서 인기가 높다. 최신 효소 처리 기술을 통해 기존의 거친 식감과 콩비린내를 대폭 개선했으며, 80% 이상의 고순도 단백질 추출이 가능해졌다. 또한 완두콩 재배는 질소 고정 능력으로 토양을 개선하여 지속가능한 농업에도 기여한다.",
                "metadata": {
                    "source": "식물성단백질협회",
                    "category": "단백질기술",
                    "reliability": 0.95,
                    "published_date": "2024-11-22",
                    "keywords": [
                        "완두콩단백질",
                        "효소처리",
                        "알레르기프리",
                        "질소고정",
                        "지속가능농업",
                    ],
                },
                "similarity_score": 0.93,
            },
            {
                "id": "doc_eco_012",
                "title": "전통 발효 기술의 현대적 응용, 미소와 장류의 글로벌 진출",
                "content": "한국과 일본의 전통 장류 기술이 현대 건강식품 시장에서 새롭게 주목받고 있다. 미소에 함유된 이소플라본과 사포닌은 항암 효과와 콜레스테롤 저하 효과가 있으며, 장기간 발효 과정에서 생성되는 다양한 프로바이오틱스는 장 건강 개선에 탁월하다. 최근에는 저염 기술과 기능성 강화 기술을 접목하여 서구인의 입맛에 맞는 제품들이 개발되고 있으며, 글로벌 발효식품 시장에서 K-푸드의 위상을 높이고 있다.",
                "metadata": {
                    "source": "전통발효식품연구소",
                    "category": "전통발효",
                    "reliability": 0.91,
                    "published_date": "2024-11-20",
                    "keywords": [
                        "미소",
                        "이소플라본",
                        "프로바이오틱스",
                        "저염기술",
                        "K푸드",
                    ],
                },
                "similarity_score": 0.88,
            },
        ]

        print("- Enhanced Mock Vector DB 초기화 완료")

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """벡터 검색 시뮬레이션 - 친환경 트렌드 강화"""
        print(f">> Enhanced Vector DB 검색 실행: {query} (top_k={top_k})")

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

            # 친환경 트렌드 키워드 가중치 부여
            eco_keywords = {
                "친환경",
                "유기농",
                "지속가능",
                "sustainable",
                "organic",
                "eco",
                "슈퍼푸드",
                "superfood",
                "고대곡물",
                "ancient",
                "발효",
                "fermented",
                "신제품",
                "트렌드",
                "새로운",
                "추천",
                "모링가",
                "아마란스",
                "햄프",
                "스피룰리나",
                "템페",
                "아사이",
                "곤충단백질",
                "제로웨이스트",
            }

            eco_boost = 1.0
            if any(keyword in query_lower for keyword in eco_keywords):
                eco_boost = 1.5

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

                # 최종 점수 (키워드에 더 높은 가중치 + 친환경 보너스)
                final_score = (
                    (content_score * 0.4) + (keyword_score * 0.6)
                ) * eco_boost

                if final_score > 0.05:  # 임계값
                    doc_copy = doc.copy()
                    doc_copy["similarity_score"] = round(min(final_score, 1.0), 3)
                    results.append(doc_copy)

        # 기본 결과 보장
        if not results:
            # 친환경 트렌드 기본 문서들
            eco_defaults = ["doc_eco_001", "doc_eco_002", "doc_eco_003", "doc_eco_004"]
            results = [
                doc.copy() for doc in self.documents if doc["id"] in eco_defaults
            ]
            for i, result in enumerate(results):
                result["similarity_score"] = 0.9 - (i * 0.1)

        # 점수 순으로 정렬
        results = sorted(
            results, key=lambda x: x.get("similarity_score", 0), reverse=True
        )[:top_k]

        print(f"- Enhanced Vector DB 검색 결과: {len(results)}개 문서")
        return results


# ========== Enhanced RDB ==========
class MockRDB:
    """식품/농업 도메인 관계형 DB - 친환경 트렌드 데이터 대폭 추가"""

    def __init__(self):
        print(">> Enhanced Mock RDB 초기화 시작")

        # ========== 대폭 확장된 농산물 시세 데이터 ==========
        self.agricultural_prices = [
            # 기존 데이터들 (일부)
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
            # ========== 새로운 친환경 트렌드 식재료 가격 ==========
            # 고대곡물
            {
                "item": "퀴노아",
                "date": "2024-12-20",
                "region": "전국",
                "market": "수입",
                "avg_price": 8500,
                "unit": "원/kg",
                "grade": "유기농",
                "supply_volume": 85,
                "price_change": "+12.3%",
                "category": "고대곡물",
                "origin": "볼리비아",
                "certification": "유기농인증",
            },
            {
                "item": "아마란스",
                "date": "2024-12-20",
                "region": "전국",
                "market": "수입",
                "avg_price": 12000,
                "unit": "원/kg",
                "grade": "유기농",
                "supply_volume": 45,
                "price_change": "+25.4%",
                "category": "고대곡물",
                "origin": "페루",
                "certification": "유기농인증",
            },
            {
                "item": "테프",
                "date": "2024-12-20",
                "region": "전국",
                "market": "수입",
                "avg_price": 15800,
                "unit": "원/kg",
                "grade": "유기농",
                "supply_volume": 28,
                "price_change": "+18.7%",
                "category": "고대곡물",
                "origin": "에티오피아",
                "certification": "공정무역",
            },
            {
                "item": "메밀",
                "date": "2024-12-20",
                "region": "강원",
                "market": "산지",
                "avg_price": 4200,
                "unit": "원/kg",
                "grade": "유기농",
                "supply_volume": 120,
                "price_change": "+8.9%",
                "category": "고대곡물",
                "origin": "국내",
                "certification": "유기농인증",
            },
            {
                "item": "기장",
                "date": "2024-12-20",
                "region": "전남",
                "market": "산지",
                "avg_price": 6800,
                "unit": "원/kg",
                "grade": "유기농",
                "supply_volume": 75,
                "price_change": "+15.2%",
                "category": "고대곡물",
                "origin": "국내",
                "certification": "유기농인증",
            },
            # 슈퍼씨드
            {
                "item": "햄프시드",
                "date": "2024-12-20",
                "region": "전국",
                "market": "국내외혼합",
                "avg_price": 22000,
                "unit": "원/kg",
                "grade": "유기농",
                "supply_volume": 35,
                "price_change": "+35.2%",
                "category": "슈퍼씨드",
                "origin": "캐나다/국내",
                "certification": "유기농인증",
            },
            {
                "item": "아마씨",
                "date": "2024-12-20",
                "region": "전국",
                "market": "수입",
                "avg_price": 8900,
                "unit": "원/kg",
                "grade": "유기농",
                "supply_volume": 95,
                "price_change": "+6.7%",
                "category": "슈퍼씨드",
                "origin": "캐나다",
                "certification": "유기농인증",
            },
            {
                "item": "치아시드",
                "date": "2024-12-20",
                "region": "전국",
                "market": "수입",
                "avg_price": 18500,
                "unit": "원/kg",
                "grade": "유기농",
                "supply_volume": 55,
                "price_change": "+11.8%",
                "category": "슈퍼씨드",
                "origin": "멕시코",
                "certification": "유기농인증",
            },
            {
                "item": "호박씨",
                "date": "2024-12-20",
                "region": "전국",
                "market": "국내외혼합",
                "avg_price": 12500,
                "unit": "원/kg",
                "grade": "유기농",
                "supply_volume": 68,
                "price_change": "+4.3%",
                "category": "슈퍼씨드",
                "origin": "국내/중국",
                "certification": "무농약인증",
            },
            # 슈퍼푸드
            {
                "item": "모링가파우더",
                "date": "2024-12-20",
                "region": "전국",
                "market": "공정무역",
                "avg_price": 45000,
                "unit": "원/kg",
                "grade": "유기농",
                "supply_volume": 25,
                "price_change": "+8.9%",
                "category": "슈퍼푸드",
                "origin": "인도",
                "certification": "공정무역+유기농",
            },
            {
                "item": "스피룰리나",
                "date": "2024-12-20",
                "region": "전국",
                "market": "국내양식",
                "avg_price": 38000,
                "unit": "원/kg",
                "grade": "프리미엄",
                "supply_volume": 42,
                "price_change": "-5.2%",
                "category": "슈퍼푸드",
                "origin": "국내",
                "certification": "HACCP",
            },
            {
                "item": "클로렐라",
                "date": "2024-12-20",
                "region": "전국",
                "market": "국내양식",
                "avg_price": 35000,
                "unit": "원/kg",
                "grade": "프리미엄",
                "supply_volume": 38,
                "price_change": "-2.8%",
                "category": "슈퍼푸드",
                "origin": "국내",
                "certification": "HACCP",
            },
            {
                "item": "아사이파우더",
                "date": "2024-12-20",
                "region": "전국",
                "market": "직수입",
                "avg_price": 52000,
                "unit": "원/kg",
                "grade": "프리즈드라이",
                "supply_volume": 18,
                "price_change": "+15.6%",
                "category": "슈퍼푸드",
                "origin": "브라질",
                "certification": "레인포레스트얼라이언스",
            },
            {
                "item": "고지베리",
                "date": "2024-12-20",
                "region": "전국",
                "market": "수입",
                "avg_price": 28500,
                "unit": "원/kg",
                "grade": "유기농",
                "supply_volume": 32,
                "price_change": "+7.2%",
                "category": "슈퍼푸드",
                "origin": "중국",
                "certification": "유기농인증",
            },
            # 대체단백질
            {
                "item": "완두콩단백질",
                "date": "2024-12-20",
                "region": "전국",
                "market": "수입",
                "avg_price": 15800,
                "unit": "원/kg",
                "grade": "80%순도",
                "supply_volume": 85,
                "price_change": "-8.5%",
                "category": "대체단백질",
                "origin": "캐나다",
                "certification": "NON-GMO",
            },
            {
                "item": "귀뚜라미단백질",
                "date": "2024-12-20",
                "region": "전국",
                "market": "국내생산",
                "avg_price": 28000,
                "unit": "원/kg",
                "grade": "식품등급",
                "supply_volume": 15,
                "price_change": "-12.8%",
                "category": "대체단백질",
                "origin": "국내",
                "certification": "HACCP",
            },
            # 발효식품
            {
                "item": "템페",
                "date": "2024-12-20",
                "region": "전국",
                "market": "수입/국내",
                "avg_price": 8500,
                "unit": "원/kg",
                "grade": "유기농",
                "supply_volume": 45,
                "price_change": "+18.3%",
                "category": "발효식품",
                "origin": "인도네시아/국내",
                "certification": "유기농인증",
            },
            {
                "item": "미소",
                "date": "2024-12-20",
                "region": "전국",
                "market": "국내외혼합",
                "avg_price": 12000,
                "unit": "원/kg",
                "grade": "3년숙성",
                "supply_volume": 65,
                "price_change": "+5.8%",
                "category": "발효식품",
                "origin": "일본/국내",
                "certification": "전통식품인증",
            },
        ]

        # ========== 대폭 확장된 영양 정보 데이터 ==========
        self.nutrition_data = [
            # 기존 데이터들 (일부)
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
            # ========== 새로운 친환경 트렌드 식재료 영양정보 ==========
            # 고대곡물
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
                "lysine": 0.77,
                "complete_protein": True,
                "gluten_free": True,
                "gi_index": 53,
            },
            {
                "item": "아마란스",
                "serving_size": "100g",
                "calories": 371,
                "protein": 13.6,
                "fat": 7.0,
                "carbohydrate": 65.2,
                "fiber": 6.7,
                "calcium": 159,
                "iron": 7.6,
                "lysine": 0.75,
                "complete_protein": True,
                "gluten_free": True,
                "antioxidants": "high",
            },
            {
                "item": "테프",
                "serving_size": "100g",
                "calories": 367,
                "protein": 13.3,
                "fat": 2.4,
                "carbohydrate": 73.1,
                "fiber": 8.0,
                "calcium": 180,
                "iron": 7.6,
                "zinc": 3.6,
                "gluten_free": True,
                "resistant_starch": "high",
            },
            {
                "item": "메밀",
                "serving_size": "100g",
                "calories": 343,
                "protein": 13.2,
                "fat": 3.4,
                "carbohydrate": 71.5,
                "fiber": 10.0,
                "rutin": 36.4,
                "magnesium": 231,
                "gluten_free": True,
                "antioxidants": "very_high",
            },
            {
                "item": "기장",
                "serving_size": "100g",
                "calories": 378,
                "protein": 11.0,
                "fat": 4.2,
                "carbohydrate": 72.8,
                "fiber": 8.5,
                "magnesium": 114,
                "phosphorus": 285,
                "gluten_free": True,
                "alkaline_forming": True,
            },
            # 슈퍼씨드
            {
                "item": "햄프시드",
                "serving_size": "100g",
                "calories": 553,
                "protein": 31.6,
                "fat": 48.8,
                "carbohydrate": 8.7,
                "fiber": 4.0,
                "omega3": 9.3,
                "omega6": 28.7,
                "gla": 1.7,
                "complete_protein": True,
                "omega_ratio": "1:3",
            },
            {
                "item": "아마씨",
                "serving_size": "100g",
                "calories": 534,
                "protein": 18.3,
                "fat": 42.2,
                "carbohydrate": 28.9,
                "fiber": 27.3,
                "omega3": 22.8,
                "lignans": 294,
                "alpha_linolenic_acid": "highest",
                "estrogen_balancing": True,
            },
            {
                "item": "치아시드",
                "serving_size": "100g",
                "calories": 486,
                "protein": 16.5,
                "fat": 30.7,
                "carbohydrate": 42.1,
                "fiber": 34.4,
                "calcium": 631,
                "omega3": 17.8,
                "antioxidants": "high",
                "water_absorption": "12x",
            },
            {
                "item": "호박씨",
                "serving_size": "100g",
                "calories": 559,
                "protein": 30.2,
                "fat": 49.1,
                "carbohydrate": 10.7,
                "fiber": 6.0,
                "zinc": 7.8,
                "magnesium": 592,
                "tryptophan": 0.58,
                "prostate_health": True,
            },
            # 슈퍼푸드
            {
                "item": "모링가",
                "serving_size": "100g",
                "calories": 64,
                "protein": 27.1,
                "fat": 2.3,
                "carbohydrate": 38.2,
                "fiber": 19.2,
                "vitamin_c": 220,
                "calcium": 2003,
                "iron": 28.2,
                "vitamin_a": 6780,
                "potassium": 1324,
                "complete_nutrition": True,
            },
            {
                "item": "스피룰리나",
                "serving_size": "100g",
                "calories": 290,
                "protein": 57.5,
                "fat": 7.7,
                "carbohydrate": 23.9,
                "fiber": 3.6,
                "chlorophyll": 1200,
                "phycocyanin": 14.0,
                "vitamin_b12": 118,
                "iron": 28.5,
                "complete_protein": True,
                "bioavailability": "very_high",
            },
            {
                "item": "클로렐라",
                "serving_size": "100g",
                "calories": 410,
                "protein": 45.0,
                "fat": 20.0,
                "carbohydrate": 23.0,
                "fiber": 18.0,
                "chlorophyll": 2800,
                "cgf": "unique",
                "vitamin_b12": 84,
                "nucleic_acids": "high",
                "detoxification": "excellent",
            },
            {
                "item": "아사이베리",
                "serving_size": "100g",
                "calories": 70,
                "protein": 1.4,
                "fat": 5.0,
                "carbohydrate": 7.3,
                "fiber": 3.0,
                "anthocyanins": 320,
                "orac": 15405,
                "vitamin_c": 17.5,
                "antioxidants": "extremely_high",
                "anti_aging": True,
            },
            {
                "item": "고지베리",
                "serving_size": "100g",
                "calories": 349,
                "protein": 14.3,
                "fat": 0.4,
                "carbohydrate": 77.1,
                "fiber": 13.0,
                "vitamin_c": 48.4,
                "zeaxanthin": 2.4,
                "polysaccharides": "unique",
                "immune_support": "excellent",
            },
            # 대체단백질
            {
                "item": "완두콩단백질",
                "serving_size": "100g",
                "calories": 375,
                "protein": 80.0,
                "fat": 3.5,
                "carbohydrate": 5.0,
                "fiber": 2.0,
                "bcaa": 18.5,
                "leucine": 8.0,
                "lysine": 7.2,
                "digestibility": "high",
                "allergen_free": True,
            },
            {
                "item": "귀뚜라미단백질",
                "serving_size": "100g",
                "calories": 444,
                "protein": 65.0,
                "fat": 15.0,
                "carbohydrate": 5.0,
                "fiber": 6.0,
                "vitamin_b12": 24,
                "iron": 5.1,
                "chitin": 2.7,
                "complete_protein": True,
                "sustainability_score": 9.8,
            },
            # 발효식품
            {
                "item": "템페",
                "serving_size": "100g",
                "calories": 193,
                "protein": 19.0,
                "fat": 11.0,
                "carbohydrate": 9.4,
                "fiber": 9.0,
                "vitamin_b12": 0.7,
                "probiotics": "lactobacillus",
                "isoflavones": 60,
                "digestibility": "enhanced",
                "fermentation_benefits": True,
            },
            {
                "item": "미소",
                "serving_size": "100g",
                "calories": 199,
                "protein": 13.0,
                "fat": 6.0,
                "carbohydrate": 26.0,
                "fiber": 5.4,
                "sodium": 3728,
                "isoflavones": 42,
                "probiotics": "diverse",
                "umami": "high",
                "aged_benefits": True,
            },
        ]

        # ========== 새로운 시장 데이터 (친환경 트렌드) ==========
        self.market_data = [
            # 기존 시장 데이터
            {
                "category": "식물성 단백질",
                "year": 2024,
                "market_size_billion_won": 850,
                "growth_rate": "18.5%",
                "forecast_2025_billion_won": 1008,
                "key_players": ["CJ제일제당", "대상", "삼양사"],
                "export_ratio": "15%",
            },
            # 새로운 친환경 트렌드 시장들
            {
                "category": "고대곡물",
                "year": 2024,
                "market_size_billion_won": 420,
                "growth_rate": "42.3%",
                "forecast_2025_billion_won": 598,
                "key_players": ["오가닉스토리", "자연드림", "쿱자연드림"],
                "export_ratio": "3%",
                "trend_keywords": ["글루텐프리", "완전단백질", "지속가능"],
            },
            {
                "category": "슈퍼씨드",
                "year": 2024,
                "market_size_billion_won": 650,
                "growth_rate": "35.8%",
                "forecast_2025_billion_won": 883,
                "key_players": ["네이처팜", "바른생활", "건강원"],
                "export_ratio": "8%",
                "trend_keywords": ["오메가3", "항산화", "식물성지방"],
            },
            {
                "category": "해조류/미세조류",
                "year": 2024,
                "market_size_billion_won": 380,
                "growth_rate": "28.7%",
                "forecast_2025_billion_won": 489,
                "key_players": ["마린바이오", "해조류연구소", "클린스피룰리나"],
                "export_ratio": "25%",
                "trend_keywords": ["클로로필", "해독", "미래단백질"],
            },
            {
                "category": "곤충단백질",
                "year": 2024,
                "market_size_billion_won": 125,
                "growth_rate": "156.3%",
                "forecast_2025_billion_won": 320,
                "key_players": ["인섹트푸드", "크리켓프로틴", "미래식품"],
                "export_ratio": "2%",
                "trend_keywords": ["지속가능", "환경친화", "효율성"],
            },
            {
                "category": "발효식품",
                "year": 2024,
                "market_size_billion_won": 980,
                "growth_rate": "22.1%",
                "forecast_2025_billion_won": 1196,
                "key_players": ["CJ제일제당", "대상", "샘표"],
                "export_ratio": "18%",
                "trend_keywords": ["프로바이오틱스", "장건강", "면역력"],
            },
            {
                "category": "공정무역식품",
                "year": 2024,
                "market_size_billion_won": 290,
                "growth_rate": "31.4%",
                "forecast_2025_billion_won": 381,
                "key_players": ["페어트레이드코리아", "이쿱생협", "공정무역위원회"],
                "export_ratio": "5%",
                "trend_keywords": ["윤리적소비", "공정무역", "지속가능"],
            },
            {
                "category": "제로웨이스트포장",
                "year": 2024,
                "market_size_billion_won": 180,
                "growth_rate": "67.8%",
                "forecast_2025_billion_won": 302,
                "key_players": ["에코패키징", "그린랩", "제로웨이스트컴퍼니"],
                "export_ratio": "12%",
                "trend_keywords": ["생분해성", "식용포장", "순환경제"],
            },
        ]

        # ========== 새로운 소비자 트렌드 데이터 ==========
        self.consumer_trends = [
            # 기존 트렌드들
            {
                "trend": "식물성 대체식품",
                "interest_score": 92,
                "age_group": "20-30대",
                "growth_period": "2년",
                "main_drivers": ["환경의식", "건강관심", "동물복지"],
            },
            # 새로운 친환경 소비자 트렌드들
            {
                "trend": "고대곡물 섭취",
                "interest_score": 89,
                "age_group": "30-40대",
                "growth_period": "1.5년",
                "main_drivers": ["글루텐프리", "완전영양", "지속가능"],
                "seasonal_peak": "가을",
                "purchase_channel": "온라인60% 오프라인40%",
            },
            {
                "trend": "슈퍼씨드 활용",
                "interest_score": 86,
                "age_group": "25-35대",
                "growth_period": "2년",
                "main_drivers": ["오메가3", "항산화", "편의성"],
                "seasonal_peak": "연중",
                "purchase_channel": "온라인70% 오프라인30%",
            },
            {
                "trend": "미세조류 섭취",
                "interest_score": 78,
                "age_group": "20-50대",
                "growth_period": "3년",
                "main_drivers": ["해독효과", "미래식품", "환경보호"],
                "seasonal_peak": "봄여름",
                "purchase_channel": "온라인80% 오프라인20%",
            },
            {
                "trend": "발효식품 다양화",
                "interest_score": 91,
                "age_group": "전연령",
                "growth_period": "4년",
                "main_drivers": ["장건강", "면역력", "전통회귀"],
                "seasonal_peak": "겨울",
                "purchase_channel": "온라인45% 오프라인55%",
            },
            {
                "trend": "공정무역 선호",
                "interest_score": 74,
                "age_group": "30-40대",
                "growth_period": "3년",
                "main_drivers": ["윤리적소비", "품질신뢰", "사회공헌"],
                "seasonal_peak": "연중",
                "purchase_channel": "온라인50% 오프라인50%",
            },
            {
                "trend": "제로웨이스트 포장",
                "interest_score": 83,
                "age_group": "20-35대",
                "growth_period": "2년",
                "main_drivers": ["환경보호", "플라스틱줄이기", "순환경제"],
                "seasonal_peak": "연중",
                "purchase_channel": "온라인65% 오프라인35%",
            },
            {
                "trend": "곤충단백질 수용",
                "interest_score": 45,
                "age_group": "20-30대",
                "growth_period": "1년",
                "main_drivers": ["호기심", "환경의식", "영양효율"],
                "seasonal_peak": "여름",
                "purchase_channel": "온라인90% 오프라인10%",
            },
        ]

        # ========== 새로운 인증/품질 데이터 ==========
        self.certification_data = [
            {
                "certification": "유기농인증",
                "authority": "국립농산물품질관리원",
                "validity_period": 1,
                "cost_range": "50-500만원",
                "requirements": ["3년무농약", "화학비료금지", "GMO금지"],
                "market_premium": "30-50%",
                "consumer_trust": 92,
            },
            {
                "certification": "공정무역인증",
                "authority": "국제공정무역기구",
                "validity_period": 3,
                "cost_range": "100-800만원",
                "requirements": ["공정가격", "사회적프리미엄", "환경기준"],
                "market_premium": "15-25%",
                "consumer_trust": 87,
            },
            {
                "certification": "레인포레스트얼라이언스",
                "authority": "Rainforest Alliance",
                "validity_period": 3,
                "cost_range": "200-1000만원",
                "requirements": ["생물다양성", "지속가능농법", "근로자복지"],
                "market_premium": "10-20%",
                "consumer_trust": 84,
            },
            {
                "certification": "탄소중립인증",
                "authority": "한국환경공단",
                "validity_period": 2,
                "cost_range": "300-1500만원",
                "requirements": ["탄소발자국측정", "감축계획", "상쇄활동"],
                "market_premium": "20-35%",
                "consumer_trust": 89,
            },
            {
                "certification": "NON-GMO인증",
                "authority": "한국식품연구원",
                "validity_period": 1,
                "cost_range": "50-300만원",
                "requirements": ["GMO부검출", "분리보관", "이력추적"],
                "market_premium": "10-15%",
                "consumer_trust": 81,
            },
        ]

        # ========== 새로운 신제품 개발 트렌드 데이터 ==========
        self.product_development_trends = [
            {
                "trend": "고대곡물 융복합",
                "description": "퀴노아+아마란스 혼합 곡물바",
                "target_demo": "건강관심 직장인",
                "development_cost": "중간",
                "market_potential": "높음",
                "key_benefits": ["완전단백질", "글루텐프리", "포만감"],
                "recommended_ingredients": ["퀴노아", "아마란스", "치아시드"],
            },
            {
                "trend": "슈퍼씨드 스무디믹스",
                "description": "햄프시드+아마씨 기반 즉석 스무디",
                "target_demo": "운동하는 2030세대",
                "development_cost": "중간",
                "market_potential": "매우높음",
                "key_benefits": ["오메가3", "식물성단백질", "편의성"],
                "recommended_ingredients": ["햄프시드", "아마씨", "스피룰리나"],
            },
            {
                "trend": "발효 슈퍼푸드",
                "description": "템페+모링가 발효 건강식품",
                "target_demo": "중장년 건강관심층",
                "development_cost": "높음",
                "market_potential": "높음",
                "key_benefits": ["프로바이오틱스", "완전영양", "소화흡수"],
                "recommended_ingredients": ["템페", "모링가", "미소"],
            },
            {
                "trend": "제로웨이스트 패키징",
                "description": "해조류 기반 식용 포장재",
                "target_demo": "환경의식 소비자",
                "development_cost": "높음",
                "market_potential": "중간",
                "key_benefits": ["환경친화", "혁신성", "차별화"],
                "recommended_ingredients": ["다시마", "미역", "김"],
            },
            {
                "trend": "어댑토겐 블렌드",
                "description": "모링가+고지베리 스트레스 완화 제품",
                "target_demo": "스트레스 관리층",
                "development_cost": "중간",
                "market_potential": "높음",
                "key_benefits": ["스트레스완화", "항산화", "면역력"],
                "recommended_ingredients": ["모링가", "고지베리", "카카오"],
            },
        ]

        print("- Enhanced Mock RDB 초기화 완료")

    def search(self, query: str) -> Dict[str, Any]:
        """RDB 통합 검색 - 친환경 트렌드 대응 강화"""
        print(f">> Enhanced RDB 검색 실행: {query}")

        query_lower = query.lower()
        all_results = {
            "prices": [],
            "nutrition": [],
            "market_data": [],
            "consumer_trends": [],
            "certification_data": [],
            "product_development_trends": [],
        }

        # 가격 정보 검색 (친환경 트렌드 강화)
        if any(
            keyword in query_lower
            for keyword in [
                "가격",
                "시세",
                "price",
                "급등",
                "급락",
                "상승",
                "하락",
                "비용",
            ]
        ):
            for price in self.agricultural_prices:
                if any(
                    item in query_lower
                    for item in [
                        price["item"],
                        price["category"],
                        price.get("origin", ""),
                    ]
                ):
                    all_results["prices"].append(price)

        # 영양 정보 검색 (슈퍼푸드 중심)
        if any(
            keyword in query_lower
            for keyword in [
                "영양",
                "성분",
                "nutrition",
                "단백질",
                "비타민",
                "오메가",
                "항산화",
            ]
        ):
            for nutrition in self.nutrition_data:
                if nutrition["item"] in query_lower:
                    all_results["nutrition"].append(nutrition)

        # 시장 데이터 검색 (친환경 트렌드 포함)
        if any(
            keyword in query_lower
            for keyword in ["시장", "market", "규모", "성장", "전망", "트렌드"]
        ):
            for market in self.market_data:
                if any(word in query_lower for word in market["category"].split()):
                    all_results["market_data"].append(market)

        # 소비자 트렌드 검색
        if any(
            keyword in query_lower
            for keyword in ["소비자", "consumer", "트렌드", "trend", "인기", "선호"]
        ):
            for trend in self.consumer_trends:
                if any(word in query_lower for word in trend["trend"].split()):
                    all_results["consumer_trends"].append(trend)

        # 인증 데이터 검색
        if any(
            keyword in query_lower
            for keyword in ["인증", "certification", "유기농", "공정무역", "친환경"]
        ):
            for cert in self.certification_data:
                if any(word in query_lower for word in cert["certification"].split()):
                    all_results["certification_data"].append(cert)

        # 신제품 개발 트렌드 검색
        if any(
            keyword in query_lower
            for keyword in ["신제품", "개발", "추천", "새로운", "혁신", "융복합"]
        ):
            for dev_trend in self.product_development_trends:
                if any(word in query_lower for word in dev_trend["trend"].split()):
                    all_results["product_development_trends"].append(dev_trend)

        # 카테고리별 키워드 검색
        category_keywords = {
            "고대곡물": ["퀴노아", "아마란스", "테프", "메밀", "기장"],
            "슈퍼씨드": ["햄프시드", "아마씨", "치아시드", "호박씨"],
            "슈퍼푸드": ["모링가", "스피룰리나", "클로렐라", "아사이", "고지베리"],
            "발효식품": ["템페", "미소", "콤부차"],
            "대체단백질": ["완두콩단백질", "곤충단백질"],
        }

        for category, keywords in category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                # 해당 카테고리의 모든 데이터 추가
                for price in self.agricultural_prices:
                    if price.get("category") == category or price["item"] in keywords:
                        if price not in all_results["prices"]:
                            all_results["prices"].append(price)

                for nutrition in self.nutrition_data:
                    if nutrition["item"] in keywords:
                        if nutrition not in all_results["nutrition"]:
                            all_results["nutrition"].append(nutrition)

        # 기본 결과 보장 (친환경 트렌드 중심)
        if not any(all_results.values()):
            # 친환경 트렌드 기본 결과
            all_results["prices"] = [
                p
                for p in self.agricultural_prices
                if p.get("category") in ["고대곡물", "슈퍼씨드", "슈퍼푸드"]
            ][:8]
            all_results["nutrition"] = [
                n
                for n in self.nutrition_data
                if n["item"] in ["퀴노아", "아마란스", "햄프시드", "모링가"]
            ][:5]
            all_results["market_data"] = self.market_data[:4]
            all_results["consumer_trends"] = self.consumer_trends[:3]
            all_results["certification_data"] = self.certification_data[:3]
            all_results["product_development_trends"] = self.product_development_trends[
                :3
            ]

        total_results = sum(len(results) for results in all_results.values())

        print(f"- Enhanced RDB 검색 결과: {total_results}개 레코드")

        return {
            "query": query,
            "total_results": total_results,
            "data": all_results,
            "database": "Enhanced_RelationalDB_EcoTrend_FoodAgri",
        }


# ========== Enhanced Web Search ==========
class MockWebSearch:
    """웹 검색 시뮬레이션 - 친환경 트렌드 대폭 강화"""

    def __init__(self):
        print(">> Enhanced Mock Web Search 초기화 시작")

        # ========== 대폭 확장된 검색 결과 데이터 ==========
        self.search_results = {
            # 기존 검색 결과들
            "식물성 단백질": [
                {
                    "title": "2025년 글로벌 식물성 단백질 시장 1,200억 달러 돌파 전망",
                    "url": "https://www.foodbiz.co.kr/news/article2025123",
                    "snippet": "글로벌 식물성 단백질 시장이 연평균 15% 성장하며 2025년 1,200억 달러 규모에 달할 것으로 예상된다. 완두콩, 대두, 퀴노아 기반 제품이 주도하고 있다.",
                    "published_date": "2024-12-22",
                    "relevance": 0.95,
                    "source_type": "industry_news",
                },
            ],
            # ========== 새로운 친환경 트렌드 검색 결과들 ==========
            "고대곡물": [
                {
                    "title": "아마란스·테프 열풍, 국내 고대곡물 시장 300% 성장",
                    "url": "https://www.ecofoodnews.co.kr/ancient-grains-boom-2024",
                    "snippet": "아마란스와 테프 등 고대곡물이 글루텐프리와 완전단백질 트렌드를 이끌며 국내 시장이 300% 성장했다. MZ세대의 건강 관심과 지속가능 소비가 주요 동력이다.",
                    "published_date": "2024-12-21",
                    "relevance": 0.94,
                    "source_type": "market_analysis",
                },
                {
                    "title": "퀴노아 넘어선 아마란스, '슈퍼그레인'의 새로운 강자",
                    "url": "https://www.healthfoodtimes.co.kr/amaranth-superfood",
                    "snippet": "아마란스가 퀴노아를 뛰어넘는 영양가치로 주목받고 있다. 완전단백질과 고함량 라이신으로 비건 식단의 핵심 재료로 떠오르고 있다.",
                    "published_date": "2024-12-19",
                    "relevance": 0.91,
                    "source_type": "health_news",
                },
                {
                    "title": "에티오피아 전통곡물 테프, 철분왕 슈퍼푸드로 재조명",
                    "url": "https://www.superfoodreview.co.kr/teff-iron-king",
                    "snippet": "테프가 철분 함량 1위 곡물로 재조명받고 있다. 글루텐프리에 기후변화 저항성까지 갖춘 미래 식량자원으로 평가받는다.",
                    "published_date": "2024-12-17",
                    "relevance": 0.89,
                    "source_type": "superfood_analysis",
                },
            ],
            "햄프시드": [
                {
                    "title": "햄프시드 합법화 1년, 국내 슈퍼씨드 시장 판도 변화",
                    "url": "https://www.sustainablefood.co.kr/hemp-seed-legal-one-year",
                    "snippet": "햄프시드 식품 허용 1년 만에 국내 슈퍼씨드 시장이 재편되고 있다. 완벽한 오메가 비율과 탄소포집 효과로 지속가능 식품의 대표주자로 부상했다.",
                    "published_date": "2024-12-20",
                    "relevance": 0.96,
                    "source_type": "sustainability_news",
                },
                {
                    "title": "햄프시드 국내 재배 첫 성공, '탄소농업'의 새 전기",
                    "url": "https://www.carbonfarm.co.kr/hemp-cultivation-success",
                    "snippet": "국내 첫 햄프시드 재배가 성공하며 탄소중립 농업의 새 전기를 마련했다. 헥타르당 15톤의 이산화탄소를 흡수하는 친환경 작물로 주목받는다.",
                    "published_date": "2024-12-18",
                    "relevance": 0.93,
                    "source_type": "agriculture_tech",
                },
            ],
            "모링가": [
                {
                    "title": "모링가 직거래 확산, 아프리카 농가와 윈윈 모델 구축",
                    "url": "https://www.fairtradenetwork.co.kr/moringa-direct-trade",
                    "snippet": "모링가 직거래가 확산되면서 아프리카 농가와의 지속가능한 파트너십이 구축되고 있다. 공정무역 모델로 품질과 윤리를 동시에 잡았다.",
                    "published_date": "2024-12-16",
                    "relevance": 0.92,
                    "source_type": "fair_trade",
                },
                {
                    "title": "'기적의 나무' 모링가, 완전식품으로 인정받아",
                    "url": "https://www.completefood.co.kr/moringa-miracle-tree",
                    "snippet": "모링가가 WHO에서 완전식품으로 공식 인정받았다. 비타민C 오렌지의 7배, 칼슘 우유의 4배로 영양학적 가치가 입증됐다.",
                    "published_date": "2024-12-14",
                    "relevance": 0.89,
                    "source_type": "nutrition_research",
                },
            ],
            "스피룰리나": [
                {
                    "title": "스피룰리나 국내 생산 급증, 미래 단백질 자급자족 기반 마련",
                    "url": "https://www.algaetech.co.kr/spirulina-domestic-production",
                    "snippet": "국내 스피룰리나 생산이 전년 대비 250% 증가하며 미래 단백질 자급자족 기반을 마련했다. 바이오리액터 기술로 연중 안정적 생산이 가능해졌다.",
                    "published_date": "2024-12-15",
                    "relevance": 0.94,
                    "source_type": "biotech_news",
                },
                {
                    "title": "스피룰리나 VS 소고기, 단백질 효율성 비교 분석",
                    "url": "https://www.proteinanalysis.co.kr/spirulina-vs-beef",
                    "snippet": "스피룰리나가 소고기 대비 99% 적은 자원으로 더 많은 단백질을 생산할 수 있는 것으로 분석됐다. 지속가능한 단백질원으로서의 가치가 재확인됐다.",
                    "published_date": "2024-12-13",
                    "relevance": 0.91,
                    "source_type": "sustainability_analysis",
                },
            ],
            "템페": [
                {
                    "title": "템페 열풍, 인도네시아 전통 발효식품이 K-푸드로",
                    "url": "https://www.fermentedfoods.co.kr/tempeh-korean-adaptation",
                    "snippet": "인도네시아 전통 발효식품 템페가 한국식으로 재해석되며 K-푸드 트렌드를 이끌고 있다. 비건 단백질원으로서의 가치와 발효 건강효과가 주목받는다.",
                    "published_date": "2024-12-12",
                    "relevance": 0.88,
                    "source_type": "food_culture",
                },
                {
                    "title": "템페 제조 기술 국산화 성공, 대량생산 체계 구축",
                    "url": "https://www.biotechkorea.co.kr/tempeh-mass-production",
                    "snippet": "템페 제조 기술 국산화에 성공하며 대량생산 체계가 구축됐다. 리조푸스 균주 개량으로 한국인 입맛에 맞는 제품 개발이 가능해졌다.",
                    "published_date": "2024-12-10",
                    "relevance": 0.85,
                    "source_type": "biotech_development",
                },
            ],
            "곤충단백질": [
                {
                    "title": "곤충단백질 상용화 원년, 귀뚜라미 제품 대거 출시",
                    "url": "https://www.insectfood.co.kr/cricket-protein-commercialization",
                    "snippet": "곤충단백질 식품 허용 법안 통과로 귀뚜라미 기반 제품들이 대거 출시되고 있다. 바, 파우더, 스낵 등 다양한 형태로 소비자 접근성을 높였다.",
                    "published_date": "2024-12-11",
                    "relevance": 0.87,
                    "source_type": "food_regulation",
                },
                {
                    "title": "곤충농장 자동화 기술 도입, 생산효율 5배 향상",
                    "url": "https://www.smartinsectfarm.co.kr/automation-efficiency",
                    "snippet": "곤충농장에 자동화 기술이 도입되면서 생산효율이 5배 향상됐다. AI 기반 환경제어와 로봇 수확으로 안정적 대량생산이 가능해졌다.",
                    "published_date": "2024-12-09",
                    "relevance": 0.84,
                    "source_type": "agtech_innovation",
                },
            ],
            "아사이베리": [
                {
                    "title": "아사이베리 직수입 늘어, 아마존 보호와 상생 모델",
                    "url": "https://www.rainforestpartnership.co.kr/acai-amazon-protection",
                    "snippet": "아사이베리 직수입이 늘어나면서 아마존 열대우림 보호와 원주민 경제 지원의 상생 모델이 주목받고 있다. 윤리적 소비의 대표 사례로 평가받는다.",
                    "published_date": "2024-12-08",
                    "relevance": 0.90,
                    "source_type": "ethical_consumption",
                },
                {
                    "title": "아사이베리 항산화 지수 세계 1위 공식 인정",
                    "url": "https://www.antioxidantresearch.co.kr/acai-orac-champion",
                    "snippet": "아사이베리가 ORAC(항산화 지수) 세계 1위로 공식 인정받았다. 블루베리의 10배에 달하는 항산화 효과로 안티에이징 식품의 왕좌에 올랐다.",
                    "published_date": "2024-12-06",
                    "relevance": 0.88,
                    "source_type": "nutrition_research",
                },
            ],
            "제로웨이스트": [
                {
                    "title": "식용 포장재 혁신, 해조류 필름으로 플라스틱 대체",
                    "url": "https://www.ediblepackaging.co.kr/seaweed-film-innovation",
                    "snippet": "해조류 기반 식용 포장재가 플라스틱 포장재를 대체하는 혁신 기술로 주목받고 있다. 100% 생분해되면서도 방수 기능을 갖춘 친환경 포장재다.",
                    "published_date": "2024-12-07",
                    "relevance": 0.93,
                    "source_type": "packaging_innovation",
                },
                {
                    "title": "농업부산물 활용 포장재, 순환경제 모델 구축",
                    "url": "https://www.circulareconomy.co.kr/agricultural-waste-packaging",
                    "snippet": "호박씨껍질, 해바라기씨껍질 등 농업부산물을 활용한 포장재가 순환경제 모델을 구축하고 있다. 폐기물 제로화와 부가가치 창출을 동시에 실현했다.",
                    "published_date": "2024-12-05",
                    "relevance": 0.89,
                    "source_type": "circular_economy",
                },
            ],
            "친환경 인증": [
                {
                    "title": "탄소중립 인증 식품 급증, 소비자 선택 기준 변화",
                    "url": "https://www.carbonneutralcert.co.kr/consumer-preference-shift",
                    "snippet": "탄소중립 인증을 받은 식품이 급증하면서 소비자 선택 기준이 변화하고 있다. 가격보다 환경영향을 우선시하는 소비 패턴이 확산되고 있다.",
                    "published_date": "2024-12-04",
                    "relevance": 0.86,
                    "source_type": "consumer_behavior",
                },
                {
                    "title": "공정무역 인증 식품 매출 50% 증가, 윤리적 소비 확산",
                    "url": "https://www.fairtradekorea.co.kr/sales-increase-50percent",
                    "snippet": "공정무역 인증 식품 매출이 전년 대비 50% 증가하며 윤리적 소비가 확산되고 있다. MZ세대를 중심으로 사회적 가치를 중시하는 소비 트렌드가 자리잡았다.",
                    "published_date": "2024-12-03",
                    "relevance": 0.84,
                    "source_type": "ethical_market",
                },
            ],
            "신제품 개발": [
                {
                    "title": "2025 주목할 신제품 트렌드, 고대곡물 융복합 제품 대세",
                    "url": "https://www.newproducttrend.co.kr/ancient-grain-fusion-2025",
                    "snippet": "2025년 신제품 트렌드로 고대곡물 융복합 제품이 대세를 이룰 것으로 예측된다. 퀴노아+아마란스 조합이 가장 유망한 것으로 분석됐다.",
                    "published_date": "2024-12-02",
                    "relevance": 0.95,
                    "source_type": "trend_forecast",
                },
                {
                    "title": "슈퍼씨드 스무디믹스 열풍, 간편식 시장 새 바람",
                    "url": "https://www.convenientfood.co.kr/superseed-smoothie-trend",
                    "snippet": "햄프시드와 아마씨 기반 즉석 스무디믹스가 간편식 시장에 새 바람을 일으키고 있다. 운동하는 2030세대를 타겟으로 한 제품들이 인기를 끌고 있다.",
                    "published_date": "2024-12-01",
                    "relevance": 0.92,
                    "source_type": "product_innovation",
                },
                {
                    "title": "발효 슈퍼푸드 조합, 프리미엄 건강식품 시장 공략",
                    "url": "https://www.premiumhealth.co.kr/fermented-superfood-combo",
                    "snippet": "템페와 모링가를 결합한 발효 슈퍼푸드 제품이 프리미엄 건강식품 시장을 공략하고 있다. 프로바이오틱스와 완전영양의 시너지 효과가 주목받고 있다.",
                    "published_date": "2024-11-29",
                    "relevance": 0.89,
                    "source_type": "premium_health",
                },
            ],
        }

        # ========== 확장된 시장 동향 데이터 ==========
        self.market_trends = [
            # 기존 트렌드들
            {
                "trend": "식물성 대체육",
                "growth_rate": "45%",
                "market_size": "500억원",
                "key_factors": ["환경의식", "건강관심", "동물복지", "MZ세대"],
                "forecast": "2025년 726억원 전망",
            },
            # 새로운 친환경 트렌드들
            {
                "trend": "고대곡물",
                "growth_rate": "42%",
                "market_size": "420억원",
                "key_factors": ["글루텐프리", "완전단백질", "지속가능", "기후변화대응"],
                "forecast": "2025년 598억원 전망",
                "leading_products": ["퀴노아", "아마란스", "테프"],
            },
            {
                "trend": "슈퍼씨드",
                "growth_rate": "36%",
                "market_size": "650억원",
                "key_factors": ["오메가3", "식물성지방", "편의성", "영양밀도"],
                "forecast": "2025년 883억원 전망",
                "leading_products": ["햄프시드", "치아시드", "아마씨"],
            },
            {
                "trend": "해조류/미세조류",
                "growth_rate": "29%",
                "market_size": "380억원",
                "key_factors": ["미래단백질", "해독효과", "바이오기술", "지속가능"],
                "forecast": "2025년 489억원 전망",
                "leading_products": ["스피룰리나", "클로렐라"],
            },
            {
                "trend": "발효슈퍼푸드",
                "growth_rate": "33%",
                "market_size": "290억원",
                "key_factors": ["장건강", "프로바이오틱스", "전통회귀", "소화흡수"],
                "forecast": "2025년 386억원 전망",
                "leading_products": ["템페", "미소", "콤부차"],
            },
            {
                "trend": "곤충단백질",
                "growth_rate": "156%",
                "market_size": "125억원",
                "key_factors": ["환경효율", "미래식량", "규제완화", "기술발전"],
                "forecast": "2025년 320억원 전망",
                "leading_products": ["귀뚜라미파우더", "곤충바"],
            },
            {
                "trend": "공정무역식품",
                "growth_rate": "31%",
                "market_size": "290억원",
                "key_factors": ["윤리적소비", "사회적가치", "MZ세대", "투명성"],
                "forecast": "2025년 381억원 전망",
                "leading_products": ["모링가", "아사이", "카카오"],
            },
            {
                "trend": "제로웨이스트포장",
                "growth_rate": "68%",
                "market_size": "180억원",
                "key_factors": ["환경보호", "플라스틱프리", "순환경제", "기술혁신"],
                "forecast": "2025년 302억원 전망",
                "leading_products": ["식용필름", "생분해포장재"],
            },
        ]

        print("- Enhanced Mock Web Search 초기화 완료")

    def search(self, query: str) -> Dict[str, Any]:
        """웹 검색 시뮬레이션 - 친환경 트렌드 대응 강화"""
        print(f">> Enhanced Web Search 검색 실행: {query}")

        query_lower = query.lower()
        results = []

        # 키워드별 검색 결과 매칭 (대폭 확장)
        for keyword, articles in self.search_results.items():
            keyword_parts = keyword.split()
            if (
                any(part.lower() in query_lower for part in keyword_parts)
                or keyword.lower() in query_lower
            ):
                results.extend(articles)

        # 친환경 트렌드 키워드 매칭 로직 (확장)
        eco_keywords = {
            ("친환경", "유기농", "지속가능", "sustainable", "organic"): [
                "친환경 인증",
                "제로웨이스트",
            ],
            ("고대곡물", "ancient", "퀴노아", "아마란스", "테프"): ["고대곡물"],
            ("슈퍼씨드", "햄프시드", "아마씨", "치아시드"): ["햄프시드"],
            ("슈퍼푸드", "모링가", "스피룰리나", "아사이"): [
                "모링가",
                "스피룰리나",
                "아사이베리",
            ],
            ("발효", "템페", "미소", "프로바이오틱스"): ["템페"],
            ("곤충단백질", "귀뚜라미", "미래식품"): ["곤충단백질"],
            ("신제품", "개발", "트렌드", "추천"): ["신제품 개발"],
            ("제로웨이스트", "포장", "플라스틱프리"): ["제로웨이스트"],
        }

        for keywords, categories in eco_keywords.items():
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

        # 기본 결과 보장 (친환경 트렌드 중심)
        if not unique_results:
            default_categories = ["고대곡물", "햄프시드", "모링가", "신제품 개발"]
            for category in default_categories:
                if category in self.search_results:
                    unique_results.extend(self.search_results[category][:2])

        # 관련도 순으로 정렬
        unique_results = sorted(
            unique_results, key=lambda x: x.get("relevance", 0), reverse=True
        )

        print(f"- Enhanced Web Search 검색 결과: {len(unique_results)}개 기사")

        return {
            "total_results": len(unique_results),
            "query": query,
            "results": unique_results[:12],  # 더 많은 결과 반환
            "trends": self.market_trends,
            "database": "Enhanced_WebSearch_EcoTrend_FoodAgri",
            "recommendation": {
                "trending_ingredients": [
                    "아마란스",
                    "햅프시드",
                    "모링가",
                    "스피룰리나",
                    "템페",
                ],
                "development_focus": [
                    "고대곡물 융복합",
                    "슈퍼씨드 스무디",
                    "발효 슈퍼푸드",
                ],
                "market_opportunity": "고대곡물과 슈퍼씨드 조합 제품이 가장 유망",
            },
        }


# ========== 통합 생성 및 테스트 함수 ==========
def create_enhanced_eco_databases():
    """모든 Enhanced Eco-Trend Database 인스턴스를 생성하고 반환"""
    print(">> Enhanced Eco-Trend Mock Databases 초기화 시작")

    graph_db = MockGraphDB()
    vector_db = MockVectorDB()
    rdb = MockRDB()
    web_search = MockWebSearch()

    print("======= Enhanced Eco-Trend Mock Databases 초기화 완료 =======")
    print(f"Graph DB: {len(graph_db.nodes)}개 노드 (친환경 트렌드 식재료 대폭 추가)")
    print(f"Vector DB: {len(vector_db.documents)}개 문서 (친환경 연구자료 풍부)")
    print(f"RDB: 6개 테이블 (가격, 영양, 시장, 소비자트렌드, 인증, 신제품개발)")
    print(
        f"Web Search: {len(web_search.search_results)}개 카테고리 (친환경 뉴스 대폭 확장)"
    )

    return graph_db, vector_db, rdb, web_search


# 기존 함수명과의 호환성을 위한 별칭
def create_mock_databases():
    """기존 함수명 호환성을 위한 별칭"""
    return create_enhanced_eco_databases()


def test_eco_trend_query():
    """친환경 트렌드 쿼리 테스트"""
    print("======= 친환경 트렌드 쿼리 테스트 =======")

    graph_db, vector_db, rdb, web_search = create_enhanced_eco_databases()

    # 테스트 쿼리
    test_query = "요즘 트렌드에 맞는 친환경 유기농 식자재를 이용한 신제품을 개발하고 있어. 요즘 새롭게 뜨는 식재료를 추천해줘."

    print(f"\n=== 테스트 쿼리: '{test_query}' ===")

    # 각 DB 테스트
    print("\n--- Graph DB 결과 ---")
    graph_result = graph_db.search(test_query)
    print(f"노드 수: {graph_result['total_nodes']}")
    print(f"관계 수: {graph_result['total_relationships']}")
    if graph_result["nodes"]:
        print("주요 노드들:")
        for node in graph_result["nodes"][:5]:
            print(f"- {node['properties'].get('name', node['id'])}")

    print("\n--- Vector DB 결과 ---")
    vector_result = vector_db.search(test_query, top_k=5)
    print(f"문서 수: {len(vector_result)}")
    if vector_result:
        print("주요 문서들:")
        for doc in vector_result[:3]:
            print(f"- {doc['title']} (유사도: {doc['similarity_score']})")

    print("\n--- RDB 결과 ---")
    rdb_result = rdb.search(test_query)
    print(f"전체 레코드: {rdb_result['total_results']}")
    for category, data in rdb_result["data"].items():
        if data:
            print(f"- {category}: {len(data)}개")

    print("\n--- Web Search 결과 ---")
    web_result = web_search.search(test_query)
    print(f"기사 수: {web_result['total_results']}")
    if "recommendation" in web_result:
        print(
            "추천 식재료:",
            ", ".join(web_result["recommendation"]["trending_ingredients"]),
        )

    print("\n모든 Enhanced Database가 친환경 트렌드 쿼리에 풍부한 데이터 제공")


# 기존과의 호환성을 위한 테스트 함수도 추가
def test_enhanced_databases():
    """Enhanced Mock Database 기능 테스트 (기존 함수명 호환)"""
    return test_eco_trend_query()


if __name__ == "__main__":
    # 친환경 트렌드 테스트 실행
    test_eco_trend_query()
