from typing import Dict, List, Optional
from ...core.config.report_config import TeamType, ReportType, Language
from ...services.templates.report_templates import ReportTemplateManager


CHART_GENERATION_INSTRUCTIONS = """
## 차트 데이터 생성 지침
시작 토큰 : {{CHART_START}}
종료 토큰 : {{CHART_END}}

**절대 금지사항: 임의의 수치 생성 금지! 반드시 검색된 실제 데이터만 사용하세요.**

### 실제 데이터가 있을 때만 차트 생성:
{{CHART_START}}
{"title": "실제 농산물 가격 동향", "type": "line", "data": {"labels": ["1월", "2월", "3월"], "datasets": [{"label": "가격(원)", "data": [실제검색된값]}]}, "source": "KAMIS 농산물유통정보", "data_type": "real"}
{{CHART_END}}

**차트 생성 규칙:**
1. 반드시 검색된 실제 데이터가 있을 때만 차트 생성
2. 검색 결과에 구체적인 수치가 없으면 차트 생성하지 말 것
3. "추정", "예상", "대략" 등의 가짜 데이터 절대 금지
4. 출처는 반드시 실제 검색 소스 명시
"""

class PromptBuilder:
    """'보고서 설계도'를 기반으로 최종 보고서 생성 프롬프트를 구성합니다."""

    def __init__(self, template_manager: ReportTemplateManager):
        self.template_manager = template_manager

    def build_prompt(
        self,
        query: str,
        context: str,  # ContextIntegrator가 생성한 '보고서 설계도'
        team_type: TeamType,
        report_type: ReportType,
        language: Language,
        extracted_data: Optional[Dict] = None,
        real_charts: Optional[List[Dict]] = None,
        source_data: Optional[Dict] = None
    ) -> str:
        """'보고서 설계도(context)'를 확장하여 최종 보고서를 작성하는 프롬프트를 생성합니다."""

        print(f"\n>> 프롬프트 생성 시작 (설계도 확장 방식)")

        role_desc = self.template_manager.translate(
            self.template_manager.get_template(team_type, report_type).role_description,
            language
        )
        source_instructions = self._build_source_instructions(source_data)

        # 고정 템플릿을 완전히 제거하고, 설계도를 확장하라는 새로운 지시사항을 만듭니다.
        final_prompt = f"""
당신은 {role_desc}.
당신의 임무는 아래 '**[보고서 설계도]**'를 바탕으로, 이를 매우 상세하고 전문적인 최종 보고서로 완성하는 것입니다.

**[최종 보고서 작성 지침]**

1.  **설계도 기반 집필**: '**[보고서 설계도]**'에 명시된 섹션 구조와 '핵심 메시지'를 그대로 따릅니다.
2.  **내용 상세화**: 각 섹션의 '**활용할 데이터 포인트**'에 있는 모든 글머리 기호 항목들을 **완전하고 상세한 문장과 문단으로 풀어내어 내용을 풍부하게 작성**하세요. 각 데이터 포인트가 의미하는 바를 깊이 있게 분석하고 설명해야 합니다.
3.  **전문적인 형식**: 서론, 본론, 결론의 흐름이 자연스럽도록 문장을 연결하고, 전체적으로 전문적인 보고서의 톤앤매너를 갖추세요.
4.  **차트 생성**: 설계도의 '**주요 수치 데이터 종합**' 섹션의 데이터를 활용하여, 보고서의 내용을 뒷받침하는 차트를 1~2개 생성하세요. 생성된 모든 차트 바로 다음 줄에는 반드시 `> `로 시작하는 상세 분석 및 인사이트를 추가해야 합니다.
5.  **질문 충족**: 완성된 보고서는 아래 '**[사용자의 질문]**'에 대한 완벽하고 상세한 답변이어야 합니다.

---

**[보고서 설계도]**
{context}

---

**[사용자의 질문]**
"{query}"

---

**[기타 지침]**
- {CHART_GENERATION_INSTRUCTIONS}
- **언어**: 모든 내용은 반드시 **{language.value}**로 작성하세요.
- **형식**: 전체 보고서는 마크다운 형식을 사용하세요.
{source_instructions}
"""
        print(f"- 프롬프트 생성 완료 (총 {len(final_prompt)}자)")
        return final_prompt

    def _build_basic_prompt(self, query: str, context: str, language: Language) -> str:
        """기본 프롬프트 (템플릿이 없을 때)"""
        return f"""
당신은 전문 분석가입니다. 주어진 정보를 바탕으로 질문에 대한 보고서를 작성해주세요.

{CHART_GENERATION_INSTRUCTIONS}

**[주어진 정보]**
{context}

**[질문]**
{query}

답변은 {language.value}로 작성하고, 차트가 필요하면 {{CHART_START}} JSON {{CHART_END}} 형식을 사용해주세요.
"""

    def _build_data_instructions(self, extracted_data: Optional[Dict], context: str) -> str:
        """데이터 우선순위 지침"""

        # 실제 데이터 확인
        has_real_data = False
        if context and ("PostgreSQL 검색 결과" in context or "영양 정보" in context or "가격 데이터" in context):
            has_real_data = True

        if not has_real_data:
            return """
    **중요: 실제 검색 데이터가 없습니다.**
    - 가짜 수치나 추정 데이터 생성 금지
    - 일반적인 정성적 분석만 제공
    - 구체적인 숫자나 차트 생성하지 말 것
    """

        instructions = """
    **실제 데이터 기반 분석 수행:**
    1. **PostgreSQL**: 검색된 실제 가격/영양 데이터 우선 활용
    2. **Vector DB**: 연구 논문의 실제 수치 활용
    3. **절대 금지**: 임의의 수치 생성, 추정 데이터 사용
    """
        return instructions

    def _build_real_chart_section(self, real_charts: Optional[List[Dict]]) -> str:
        """실제 차트 데이터 섹션"""
        if not real_charts:
            return ""

        section = "\n**사용 가능한 실제 데이터 차트:**\n"
        section += "아래 차트 데이터를 보고서에 포함하세요:\n\n"

        for i, chart in enumerate(real_charts, 1):
            import json
            chart_json = json.dumps(chart, ensure_ascii=False)
            section += f"""
{i}번 차트:
{{CHART_START}}
{chart_json}
{{CHART_END}}

"""

        section += "**중요**: 위 실제 데이터 차트들을 적절한 섹션에 포함하고, 각각에 대한 인사이트 설명을 추가하세요.\n"
        return section

    def _build_source_instructions(self, source_data: Optional[Dict]) -> str:
        """출처 정보 지침"""
        if not source_data:
            return ""

        total_sources = source_data.get("total_count", 0)
        credibility = source_data.get("credibility_summary", {})

        instructions = f"""
**활용된 출처 정보:**
- 총 출처: {total_sources}개
- 평균 신뢰도: {int(credibility.get('average_reliability', 0) * 100)}%
- 고신뢰도 출처: {credibility.get('high_reliability_count', 0)}개
"""
        return instructions

