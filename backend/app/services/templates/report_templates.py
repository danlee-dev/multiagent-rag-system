from typing import Dict, Optional
from ...core.config.report_config import (
    TeamType, ReportType, Language,
    SectionConfig, ReportTemplate
)

class ReportTemplateManager:
    """모든 팀과 보고서 타입에 대한 설정 포함"""

    def __init__(self):
        self.templates = self._load_templates()
        self.translations = self._load_translations()

    def _load_templates(self) -> Dict[str, Dict[str, ReportTemplate]]:
        """모든 팀과 보고서 타입에 대한 템플릿 설정 로드"""
        return {
            # --- 마케팅 팀 템플릿 ---
            TeamType.MARKETING.value: {
                ReportType.COMPREHENSIVE.value: ReportTemplate(
                    role_description="bain_principal_marketing",
                    sections=[
                        SectionConfig(key="marketing_insights_summary", words="450-500", details=["core_trends_5", "immediate_opportunities_3", "competitive_advantage"], chart_requirements=["trend_analysis", "opportunity_assessment"]),
                        SectionConfig(key="consumer_behavior_analysis", words="500", details=["target_segment_profiles", "customer_journey_mapping", "brand_perception_analysis"], chart_requirements=["segmentation_pie", "journey_funnel"])
                    ],
                    total_words="2000-3000", charts="6-8"
                ),
                ReportType.DETAILED.value: ReportTemplate(
                    role_description="strategic_marketing_analyst",
                    sections=[
                        SectionConfig(key="market_consumer_analysis", words="400", details=["market_size_growth", "consumer_segmentation"], chart_requirements=["market_growth", "segmentation"]),
                        SectionConfig(key="competitive_positioning", words="400", details=["competitor_analysis", "brand_positioning"], chart_requirements=["positioning_map"])
                    ],
                    total_words="1500-2000", charts="4-5"
                ),
                ReportType.STANDARD.value: ReportTemplate(
                    role_description="marketing_strategist",
                    sections=[
                        SectionConfig(key="market_consumer_insights", words="350", details=["market_size_analysis", "target_segments", "competitive_environment"], chart_requirements=["market_overview"])
                    ],
                    total_words="1000-1500", charts="3"
                ),
                ReportType.BRIEF.value: ReportTemplate(
                    role_description="marketing_consultant",
                    sections=[
                        SectionConfig(key="market_situation_opportunities", words="250", details=["key_trends", "target_analysis"], chart_requirements=["trend_snapshot"])
                    ],
                    total_words="500-800", charts="1-2"
                )
            },
            # --- 구매 팀 템플릿 ---
            TeamType.PURCHASING.value: {
                ReportType.COMPREHENSIVE.value: ReportTemplate(
                    role_description="procurement_expert",
                    sections=[
                        SectionConfig(key="price_trend_analysis", words="500", details=["commodity_price_trends", "price_forecast"], chart_requirements=["price_trend_line_chart"]),
                        SectionConfig(key="supplier_evaluation", words="500", details=["supplier_scorecard", "risk_assessment"], chart_requirements=["supplier_comparison_bar_chart"])
                    ],
                    total_words="2000-3000", charts="5-7"
                ),
                ReportType.DETAILED.value: ReportTemplate(
                    role_description="procurement_specialist",
                    sections=[
                        SectionConfig(key="price_analysis", words="400", details=["historical_price_data", "current_market_price"], chart_requirements=["price_history_chart"]),
                        SectionConfig(key="cost_reduction_strategy", words="400", details=["negotiation_points", "alternative_sourcing"], chart_requirements=[])
                    ],
                    total_words="1500-2000", charts="3-4"
                ),
                ReportType.STANDARD.value: ReportTemplate(
                    role_description="purchasing_manager",
                    sections=[
                        SectionConfig(key="market_price_summary", words="350", details=["current_price_summary", "price_change_reasons"], chart_requirements=["price_summary_table"])
                    ],
                    total_words="1000-1500", charts="2"
                ),
                ReportType.BRIEF.value: ReportTemplate(
                    role_description="buyer",
                    sections=[
                        SectionConfig(key="price_quote", words="250", details=["item_price_list", "validity_period"], chart_requirements=[])
                    ],
                    total_words="500-800", charts="1"
                )
            },
            # --- 제품개발 팀 템플릿 ---
            TeamType.DEVELOPMENT.value: {
                ReportType.COMPREHENSIVE.value: ReportTemplate(
                    role_description="organic_product_development_expert",
                    sections=[
                        SectionConfig(
                            key="eco_organic_market_analysis",
                            words="600",
                            details=[
                                "organic_ingredient_market_trends_2022_2025",
                                "quarterly_production_consumption_data",
                                "domestic_organic_ingredient_types_analysis",
                                "consumer_spending_patterns_organic_foods"
                            ],
                            chart_requirements=[
                                "organic_market_growth_trend_chart",
                                "quarterly_consumption_volume_chart",
                                "ingredient_type_market_share_pie"
                            ]
                        ),
                        SectionConfig(
                            key="sustainable_ingredient_sourcing_strategy",
                            words="500",
                            details=[
                                "certified_organic_supplier_network",
                                "eco_friendly_supply_chain_optimization",
                                "seasonal_availability_price_analysis",
                                "carbon_footprint_assessment"
                            ],
                            chart_requirements=[
                                "supplier_sustainability_scorecard",
                                "seasonal_price_fluctuation_chart"
                            ]
                        ),
                        SectionConfig(
                            key="product_development_roadmap",
                            words="500",
                            details=[
                                "eco_product_concept_validation",
                                "organic_certification_requirements",
                                "target_consumer_segment_analysis",
                                "competitive_organic_products_benchmarking"
                            ],
                            chart_requirements=[
                                "development_timeline_gantt",
                                "competitive_positioning_matrix"
                            ]
                        )
                    ],
                    total_words="2500-3500",
                    charts="8-10"
                ),
                ReportType.DETAILED.value: ReportTemplate(
                    role_description="sustainable_food_scientist",
                    sections=[
                        SectionConfig(
                            key="organic_ingredient_market_overview",
                            words="450",
                            details=[
                                "2022_2025_market_size_trends",
                                "major_organic_ingredient_categories",
                                "domestic_vs_imported_ingredient_analysis"
                            ],
                            chart_requirements=[
                                "market_growth_timeline",
                                "ingredient_category_breakdown"
                            ]
                        ),
                        SectionConfig(
                            key="eco_product_feasibility_study",
                            words="400",
                            details=[
                                "organic_certification_pathway",
                                "sustainable_packaging_options",
                                "cost_benefit_analysis_organic_vs_conventional"
                            ],
                            chart_requirements=[
                                "certification_process_flowchart",
                                "cost_comparison_analysis"
                            ]
                        )
                    ],
                    total_words="1800-2200",
                    charts="5-7"
                ),
                ReportType.STANDARD.value: ReportTemplate(
                    role_description="eco_product_developer",
                    sections=[
                        SectionConfig(
                            key="organic_ingredient_profile_analysis",
                            words="400",
                            details=[
                                "key_organic_ingredients_list",
                                "nutritional_sustainability_benefits",
                                "market_availability_pricing"
                            ],
                            chart_requirements=[
                                "ingredient_nutrition_comparison",
                                "price_trend_summary"
                            ]
                        )
                    ],
                    total_words="1000-1400",
                    charts="3-4"
                ),
                ReportType.BRIEF.value: ReportTemplate(
                    role_description="organic_research_assistant",
                    sections=[
                        SectionConfig(
                            key="eco_ingredient_summary",
                            words="300",
                            details=[
                                "trending_organic_ingredients_2025",
                                "basic_market_data_overview"
                            ],
                            chart_requirements=[
                                "trending_ingredients_snapshot"
                            ]
                        )
                    ],
                    total_words="500-700",
                    charts="1-2"
                )
            },

            # --- 총무팀 템플릿 ---
            TeamType.GENERAL_AFFAIRS.value: {
                ReportType.DETAILED.value: ReportTemplate(
                    role_description="operations_manager",
                    sections=[
                        SectionConfig(key="cafeteria_menu_analysis", words="500", details=["weekly_menu_review", "nutritional_balance_check"], chart_requirements=["nutrient_balance_radar_chart"]),
                        SectionConfig(key="budget_efficiency_report", words="400", details=["cost_per_meal_analysis", "waste_reduction_plan"], chart_requirements=["cost_breakdown_pie_chart"])
                    ],
                    total_words="1500-2000", charts="4-5"
                ),
            },
            # --- 일반팀 템플릿 (폴백용) ---
            TeamType.GENERAL.value: {
                ReportType.COMPREHENSIVE.value: ReportTemplate(
                    role_description="business_analyst",
                    sections=[
                        SectionConfig(key="executive_summary", words="400", details=["key_findings", "strategic_implications"], chart_requirements=[]),
                        SectionConfig(key="detailed_analysis", words="1000", details=["data_driven_insights", "supporting_evidence"], chart_requirements=["core_data_visualization"])
                    ],
                    total_words="2000-3000", charts="5-7"
                ),
            }
        }

    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """모든 템플릿 키에 대한 번역 데이터 로드"""
        return {
            Language.KOREAN.value: {
                # Roles
                "bain_principal_marketing": "당신은 베인앤컴퍼니의 프린시플로서 100개 이상의 브랜드를 성공으로 이끈 마케팅 전략가입니다.",
                "strategic_marketing_analyst": "당신은 전략적 마케팅 분석을 전문으로 하는 시니어 애널리스트입니다.",
                "marketing_strategist": "당신은 마케팅 전략 수립을 전문으로 하는 컨설턴트입니다.",
                "marketing_consultant": "당신은 마케팅 인사이트를 제공하는 전문 컨설턴트입니다.",
                "procurement_expert": "당신은 글로벌 컨설팅 펌의 구매/조달 전문 컨설턴트입니다.",
                "procurement_specialist": "당신은 원가 절감 및 공급망 관리를 전문으로 하는 구매 전문가입니다.",
                "purchasing_manager": "당신은 구매팀을 총괄하는 매니저입니다.",
                "buyer": "당신은 실무를 담당하는 바이어입니다.",
                "product_development_lead": "당신은 신제품 개발을 총괄하는 R&D 리더입니다.",
                "food_scientist": "당신은 식품의 영양과 기능을 연구하는 식품 과학자입니다.",
                "product_developer": "당신은 신제품을 개발하는 연구원입니다.",
                "junior_researcher": "당신은 데이터 수집 및 요약을 담당하는 연구원입니다.",
                "operations_manager": "당신은 회사 운영 및 시설 관리를 책임지는 총무팀장입니다.",
                "business_analyst": "당신은 데이터를 기반으로 비즈니스 인사이트를 도출하는 전문 분석가입니다.",

                # Section Keys
                "marketing_insights_summary": "마케팅 인사이트 종합 요약",
                "consumer_behavior_analysis": "심층 소비자 행동 분석",
                "market_consumer_analysis": "시장 및 소비자 분석",
                "competitive_positioning": "경쟁 환경 및 포지셔닝",
                "market_consumer_insights": "시장 소비자 인사이트",
                "market_situation_opportunities": "시장 현황 및 기회",
                "price_trend_analysis": "가격 동향 심층 분석",
                "supplier_evaluation": "공급업체 종합 평가",
                "price_analysis": "상세 가격 분석",
                "cost_reduction_strategy": "원가 절감 전략",
                "market_price_summary": "시장 시세 요약",
                "price_quote": "품목별 견적",
                "nutritional_analysis_deep_dive": "영양 성분 심층 분석",
                "ingredient_sourcing_strategy": "원료 소싱 전략",
                "nutritional_information": "상세 영양 정보",
                "functional_benefits": "제품의 기능적 이점",
                "ingredient_profile": "핵심 원료 프로파일",
                "nutrient_summary": "주요 영양소 요약",
                "cafeteria_menu_analysis": "구내식당 식단 분석",
                "budget_efficiency_report": "예산 효율성 보고",
                "executive_summary": "경영진 요약",
                "detailed_analysis": "상세 분석",
                "organic_product_development_expert": "당신은 친환경 유기농 제품 개발을 전문으로 하는 R&D 리더입니다. 2022년부터 2025년까지의 국내 친환경 유기농 시장 데이터와 트렌드를 깊이 분석하고, 지속가능한 신제품 개발 전략을 수립합니다.",
                "sustainable_food_scientist": "당신은 지속가능한 식품 과학을 연구하는 전문가로, 친환경 유기농 식자재의 과학적 특성과 시장 동향을 분석합니다.",
                "eco_product_developer": "당신은 친환경 제품 개발을 담당하는 연구원으로, 유기농 식자재를 활용한 제품 기획과 개발을 수행합니다.",
                "organic_research_assistant": "당신은 유기농 식품 트렌드와 기초 시장 데이터를 조사하고 정리하는 연구 보조원입니다.",

                # 새로운 Section 번역
                "eco_organic_market_analysis": "친환경 유기농 시장 종합 분석",
                "sustainable_ingredient_sourcing_strategy": "지속가능한 원료 소싱 전략",
                "product_development_roadmap": "친환경 제품 개발 로드맵",
                "organic_ingredient_market_overview": "유기농 식자재 시장 현황",
                "eco_product_feasibility_study": "친환경 제품 타당성 연구",
                "organic_ingredient_profile_analysis": "유기농 식자재 프로파일 분석",
                "eco_ingredient_summary": "친환경 식자재 요약"
            }
        }


    def get_template(self, team_type: TeamType, report_type: ReportType) -> Optional[ReportTemplate]:
        """템플릿 가져오기 - 폴백 로직 포함"""
        template = self.templates.get(team_type.value, {}).get(report_type.value)
        if not template:
            # 폴백: general 팀의 comprehensive 템플릿
            template = self.templates.get(TeamType.GENERAL.value, {}).get(ReportType.COMPREHENSIVE.value)
        return template

    def translate(self, key: str, language: Language) -> str:
        """번역 가져오기"""
        return self.translations.get(language.value, {}).get(key, key)
