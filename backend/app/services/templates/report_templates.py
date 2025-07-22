from typing import Dict, Optional
from ...core.config.report_config import (
    TeamType, ReportType, Language,
    SectionConfig, ReportTemplate
)

class ReportTemplateManager:
    """보고서 템플릿 관리자 - 설정 파일 기반으로 변경 가능"""

    def __init__(self):
        self.templates = self._load_templates()
        self.translations = self._load_translations()

    def _load_templates(self) -> Dict[str, Dict[str, ReportTemplate]]:
        """템플릿 설정 로드 - 향후 JSON/YAML 파일로 분리 가능"""
        return {
            TeamType.MARKETING.value: {
                ReportType.COMPREHENSIVE.value: ReportTemplate(
                    role_description="bain_principal_marketing",
                    sections=[
                        SectionConfig(
                            key="marketing_insights_summary",
                            words="450-500",
                            details=["core_trends_5", "immediate_opportunities_3", "competitive_advantage"],
                            chart_requirements=["trend_analysis", "opportunity_assessment"]
                        ),
                        SectionConfig(
                            key="consumer_behavior_analysis",
                            words="500",
                            details=["target_segment_profiles", "customer_journey_mapping", "brand_perception_analysis"],
                            chart_requirements=["segmentation_pie", "journey_funnel"]
                        )
                    ],
                    total_words="2000-3000",
                    charts="6-8"
                ),
                ReportType.STANDARD.value: ReportTemplate(
                    role_description="marketing_strategist",
                    sections=[
                        SectionConfig(
                            key="market_consumer_insights",
                            words="350",
                            details=["market_size_analysis", "target_segments", "competitive_environment"],
                            chart_requirements=["market_overview"]
                        )
                    ],
                    total_words="1000-1500",
                    charts="3"
                )
            }
        }

    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """번역 데이터 로드 - 향후 i18n 라이브러리로 대체 가능"""
        return {
            Language.KOREAN.value: {
                "bain_principal_marketing": "당신은 베인앤컴퍼니의 프린시플로서 100개 이상의 브랜드를 성공으로 이끈 마케팅 전략가입니다.",
                "marketing_strategist": "당신은 마케팅 전략 수립을 전문으로 하는 컨설턴트입니다.",
                "marketing_insights_summary": "마케팅 인사이트 종합 요약",
                "consumer_behavior_analysis": "심층 소비자 행동 분석",
                "core_trends_5": "핵심 트렌드 5가지: 각 트렌드별 정량적 임팩트 분석",
                "trend_analysis": "트렌드 분석 차트"
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
