from typing import Dict, List, Optional
from ...core.config.report_config import TeamType, ReportType, Language
from ...services.templates.report_templates import ReportTemplateManager


CHART_GENERATION_INSTRUCTIONS = """
## 차트 데이터 생성 지침

**중요: 모든 차트에는 데이터 출처를 명시해야 합니다.**

보고서에 차트가 필요한 부분에서는 아래 형식을 정확히 따라 완전한 JSON 데이터를 생성해야 합니다.

### 실제 데이터 기반 차트 형식:
{{CHART_START}}
{"title": "지역별 물류 경로 효율성 (실제 데이터)", "type": "line", "data": {"labels": ["서울", "부산", "대전", "광주"], "datasets": [{"label": "배송 시간 (시간)", "data": [4, 3, 5, 4]}]}, "source": "실제 추출 데이터", "data_type": "real"}
{{CHART_END}}

### 추정 데이터 기반 차트 형식:
{{CHART_START}}
{"title": "타겟 세그먼트별 관심사 분포 (추정 데이터)", "type": "pie", "data": {"labels": ["환경친화성", "가성비", "브랜드 신뢰도"], "datasets": [{"label": "관심도 (%)", "data": [35, 25, 20]}]}, "source": "시장조사 기반 추정", "data_type": "estimated"}
{{CHART_END}}

**차트 생성 규칙:**
1. 반드시 {{CHART_START}}와 {{CHART_END}} 태그 사이에 완전한 JSON 형식으로 작성
2. JSON은 한 줄로 작성 (줄바꿈 없이)
3. 실제 데이터가 있으면 "data_type": "real", 없으면 "data_type": "estimated"
4. 제목에 반드시 "(실제 데이터)" 또는 "(추정 데이터)" 표시
5. 차트 하단에 출처와 신뢰도 설명 추가

**차트 설명 형식:**
차트 생성 후 반드시 다음과 같이 설명을 추가하세요:
> 위 차트는 [데이터 설명]을 보여줍니다. [핵심 인사이트 1-2문장]
"""

class PromptBuilder:
    """프롬프트 생성기 - 템플릿과 데이터를 조합해서 프롬프트 생성"""

    def __init__(self, template_manager: ReportTemplateManager):
        self.template_manager = template_manager

    def build_prompt(
        self,
        query: str,
        context: str,
        team_type: TeamType,
        report_type: ReportType,
        language: Language,
        extracted_data: Optional[Dict] = None,
        real_charts: Optional[List[Dict]] = None,
        source_data: Optional[Dict] = None
    ) -> str:
        """통합 프롬프트 생성"""

        print(f"\n>> 프롬프트 생성 시작")
        print(f"- 팀 타입: {team_type.value}")
        print(f"- 보고서 타입: {report_type.value}")
        print(f"- 언어: {language.value}")

        # 템플릿 가져오기
        template = self.template_manager.get_template(team_type, report_type)
        if not template:
            print("- 기본 템플릿 사용")
            return self._build_basic_prompt(query, context, language)

        # 역할 설정
        role_desc = self.template_manager.translate(template.role_description, language)

        # 기본 프롬프트
        base_prompt = f"""
당신은 {role_desc}

주어진 정보를 바탕으로 사용자의 질문에 대한 전문적인 보고서를 작성해주세요.

**보고서 구조:**
- 총 단어 수: {template.total_words}
- 총 차트 수: {template.charts}개
- 보고서 레벨: {report_type.value.upper()}

**섹션별 가이드라인:**
"""

        # 섹션별 가이드라인 추가
        for i, section in enumerate(template.sections, 1):
            section_title = self.template_manager.translate(section.key, language)
            base_prompt += f"\n### {i}. {section_title} ({section.words}단어)\n"

            for detail in section.details:
                detail_text = self.template_manager.translate(detail, language)
                base_prompt += f"- **{detail_text}**\n"

            if section.chart_requirements:
                base_prompt += f"\n**필수 차트 ({len(section.chart_requirements)}개):**\n"
                for chart_req in section.chart_requirements:
                    chart_desc = self.template_manager.translate(chart_req, language)
                    base_prompt += f"- {chart_desc}\n"

        # 데이터 우선순위 지침 추가
        data_instructions = self._build_data_instructions(extracted_data, context)

        # 실제 차트 데이터가 있으면 포함
        chart_data_section = self._build_real_chart_section(real_charts)

        # 출처 정보 추가
        source_instructions = self._build_source_instructions(source_data)

        final_prompt = f"""{base_prompt}

{data_instructions}
{chart_data_section}
{source_instructions}

{CHART_GENERATION_INSTRUCTIONS}

**[주어진 핵심 정보]**
{context}

**[사용자의 질문]**
"{query}"

**중요 지침:**
1. 모든 답변은 반드시 {language.value}로 작성
2. 마크다운 형식 사용
3. 차트는 반드시 {{CHART_START}} JSON {{CHART_END}} 형식으로 생성
4. JSON은 한 줄로 작성하고 올바른 형식 준수
5. 각 차트 후에는 반드시 "> " 로 시작하는 설명 추가
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
        if not extracted_data and not context:
            return "\n**검색 데이터 없음**: 일반적인 정보로 보고서 작성\n"

        instructions = """
**데이터 활용 우선순위:**
1. **PostgreSQL (RDB)**: 농진청 등 공식 데이터 최우선 활용
2. **Vector DB**: 연구 논문, 기술 자료 활용
3. **Graph DB**: 관계형 분석, 네트워크 정보 활용
4. **Web Search**: 최신 트렌드, 뉴스 정보 보완적 사용
5. **일반 지식**: 위 모든 소스에 없을 때만 사용 (추정 데이터 표기)
"""

        if extracted_data and extracted_data.get('extracted_numbers'):
            instructions += "\n**추출된 실제 수치:**\n"
            for num in extracted_data['extracted_numbers'][:3]:
                value = num.get('value', 'N/A')
                unit = num.get('unit', '')
                context_info = num.get('context', '')[:30]
                instructions += f"- {value}{unit}: {context_info}\n"

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

**출처 표기 규칙:**
- PostgreSQL: (출처: 농진청 '21, RDB)
- Vector DB: (출처: Vector DB - 연구자료)
- Graph DB: (출처: Graph DB - 관계분석)
- Web 검색: (출처: Web 검색 - 최신정보)
"""
        return instructions
