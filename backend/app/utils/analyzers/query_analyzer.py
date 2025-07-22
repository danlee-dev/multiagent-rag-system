from ...core.config.report_config import TeamType, ReportType, Language

class QueryAnalyzer:
    """질문 분석기 - 팀 타입, 언어, 복잡도 분석"""

    TEAM_KEYWORDS = {
        TeamType.MARKETING: ["마케팅", "브랜드", "광고", "캠페인", "소비자", "marketing", "brand", "campaign"],
        TeamType.PURCHASING: ["가격", "시세", "공급업체", "조달", "구매", "supplier", "procurement", "sourcing"],
        TeamType.DEVELOPMENT: ["개발", "제품", "영양", "성분", "기능성", "development", "nutrition", "ingredient"],
        TeamType.GENERAL_AFFAIRS: ["급식", "직원", "사내", "구내식당", "cafeteria", "employee", "facility"]
    }

    COMPLEXITY_KEYWORDS = {
        "complex": ["보고서", "report", "전략", "strategy", "분석", "analysis", "상세", "detailed", "comprehensive"],
        "simple": ["간단히", "briefly", "짧게", "요약", "summary", "개요", "overview", "빠르게", "quick"]
    }

    @classmethod
    def detect_team_type(cls, query: str) -> TeamType:
        """팀 타입 감지"""
        if not query:
            return TeamType.GENERAL

        query_lower = query.lower()
        scores = {}

        for team_type, keywords in cls.TEAM_KEYWORDS.items():
            scores[team_type] = sum(1 for keyword in keywords if keyword in query_lower)

        max_score = max(scores.values()) if scores.values() else 0
        return max(scores, key=scores.get) if max_score > 0 else TeamType.GENERAL

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

        query_lower = query.lower()
        complexity_score = 0

        # 키워드 기반 점수 계산
        for keyword in cls.COMPLEXITY_KEYWORDS["complex"]:
            if keyword in query_lower:
                complexity_score += 1.5

        for keyword in cls.COMPLEXITY_KEYWORDS["simple"]:
            if keyword in query_lower:
                complexity_score -= 1.5

        # 길이 기반 점수
        if len(query) > 100:
            complexity_score += 1.5
        elif len(query) > 50:
            complexity_score += 0.5

        # 보고서 타입 결정
        if complexity_score >= 3:
            report_type = ReportType.COMPREHENSIVE
            recommended_length = "2000-3000단어, 6-8개 섹션, 6-8개 차트"
        elif complexity_score >= 1.5:
            report_type = ReportType.DETAILED
            recommended_length = "1500-2000단어, 5개 섹션, 4-5개 차트"
        elif complexity_score <= -1.5:
            report_type = ReportType.BRIEF
            recommended_length = "500-800단어, 3개 섹션, 1-2개 차트"
        else:
            report_type = ReportType.STANDARD
            recommended_length = "1000-1500단어, 4개 섹션, 3개 차트"

        return {
            "complexity_score": complexity_score,
            "report_type": report_type,
            "recommended_length": recommended_length
        }
