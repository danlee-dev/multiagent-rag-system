from ...core.config.report_config import TeamType, ReportType, Language
from langchain_openai import ChatOpenAI
import json
import re

class QueryAnalyzer:
    """질문 분석기 - 팀 타입, 언어, 복잡도 분석"""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    @classmethod
    def detect_team_type(cls, query: str) -> TeamType:
        """팀 타입 감지"""
        if not query:
            return TeamType.GENERAL

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

        prompt = f"""다음 질문이 어느 팀의 업무와 가장 관련이 있는지 분석해줘:

질문: "{query}"

팀 분류:
- MARKETING: 마케팅, 브랜드, 광고, 캠페인, 소비자 트렌드 관련
- PURCHASING: 가격, 시세, 구매 관련
- DEVELOPMENT: 제품 개발, 영양, 성분, 기능성 식품 관련
- GENERAL_AFFAIRS: 급식, 직원 식당, 사내 복리후생 관련
- GENERAL: 위 분류에 해당하지 않는 일반적인 질문

응답은 오직 다음 중 하나만: MARKETING, PURCHASING, DEVELOPMENT, GENERAL_AFFAIRS, GENERAL"""

        try:
            response = llm.invoke(prompt)
            result = response.content.strip().upper()

            if result == "MARKETING":
                return TeamType.MARKETING
            elif result == "PURCHASING":
                return TeamType.PURCHASING
            elif result == "DEVELOPMENT":
                return TeamType.DEVELOPMENT
            elif result == "GENERAL_AFFAIRS":
                return TeamType.GENERAL_AFFAIRS
            else:
                return TeamType.GENERAL
        except:
            return TeamType.GENERAL

    @classmethod
    def detect_language(cls, query: str) -> Language:
        """언어 감지"""
        if not query:
            return Language.KOREAN

        korean_chars = sum(1 for char in query if "\uac00" <= char <= "\ud7af")
        total_chars = len([char for char in query if char.isalpha()])

        if total_chars > 0 and korean_chars / total_chars > 0.5:
            return Language.KOREAN
        return Language.ENGLISH

    @classmethod
    def analyze_complexity(cls, query: str, user_context=None) -> dict:
        """복잡도 분석"""
        if not query:
            return {
                "complexity_score": 0,
                "report_type": ReportType.STANDARD,
                "recommended_length": "1000-1500단어, 4개 섹션, 3개 차트"
            }

        from ...services.templates.report_templates import ReportTemplateManager

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

        prompt = f"""다음 질문의 복잡도를 분석해줘:

질문: "{query}"

분석 기준:
- 보고서나 상세 분석 요청시 복잡도 높음
- 간단한 질문이나 요약 요청시 복잡도 낮음
- 질문 길이와 요구사항의 구체성 고려

응답 형식 (JSON만 출력):
{{
    "complexity_score": 숫자 (0-5점),
    "report_type": "BRIEF/STANDARD/DETAILED/COMPREHENSIVE",
    "reasoning": "판단 근거"
}}"""

        try:
            response = llm.invoke(prompt)
            result_text = response.content.strip()

            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(result_text)

            complexity_score = float(result.get("complexity_score", 2))
            report_type_str = result.get("report_type", "STANDARD")

            if report_type_str == "COMPREHENSIVE":
                report_type = ReportType.COMPREHENSIVE
            elif report_type_str == "DETAILED":
                report_type = ReportType.DETAILED
            elif report_type_str == "BRIEF":
                report_type = ReportType.BRIEF
            else:
                report_type = ReportType.STANDARD

            team_type = cls.detect_team_type(query)
            template_manager = ReportTemplateManager()
            template = template_manager.get_template(team_type, report_type)

            if template:
                section_count = len(template.sections)
                recommended_length = f"{template.total_words}, {section_count}개 섹션, {template.charts}"
            else:
                recommended_length = "1000-1500단어, 4개 섹션, 3개 차트"

            return {
                "complexity_score": complexity_score,
                "report_type": report_type,
                "recommended_length": recommended_length
            }

        except Exception as e:
            print(f"- LLM 복잡도 분석 실패: {str(e)}")
            return {
                "complexity_score": 2,
                "report_type": ReportType.STANDARD,
                "recommended_length": "1000-1500단어, 4개 섹션, 3개 차트"
            }
