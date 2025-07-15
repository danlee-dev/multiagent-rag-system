import json
import random
from typing import Dict, List, Any

class ChartDataGenerator:
    """차트 데이터를 실제로 생성해주는 헬퍼 클래스"""

    @staticmethod
    def generate_sample_data(chart_type: str, labels: List[str] = None, title: str = "차트") -> Dict[str, Any]:
        """
        차트 타입에 따른 샘플 데이터 생성

        Args:
            chart_type: 차트 종류 (line, bar, pie, etc.)
            labels: 라벨 리스트
            title: 차트 제목
        """
        print("\n>> 차트 타입별 데이터 생성 시작")

        if not labels:
            labels = ["1월", "2월", "3월", "4월", "5월"]

        # 랜덤 데이터 생성 (실제로는 RAG에서 가져온 데이터 활용)
        data_values = [random.randint(10, 100) for _ in labels]

        if chart_type.lower() in ["line", "area", "flow"]:
            return {
                "title": title,
                "type": chart_type,
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": "데이터",
                        "data": data_values,
                        "borderColor": "#4F46E5",
                        "backgroundColor": "#4F46E520"
                    }]
                }
            }

        elif chart_type.lower() in ["bar", "column"]:
            return {
                "title": title,
                "type": chart_type,
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": "수량",
                        "data": data_values,
                        "backgroundColor": ["#4F46E5", "#7C3AED", "#EC4899", "#EF4444", "#F59E0B"]
                    }]
                }
            }

        elif chart_type.lower() in ["pie", "doughnut"]:
            return {
                "title": title,
                "type": chart_type,
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": "비율",
                        "data": data_values,
                        "backgroundColor": ["#4F46E5", "#7C3AED", "#EC4899", "#EF4444", "#F59E0B"]
                    }]
                }
            }

        else:
            # 기본값
            return {
                "title": title,
                "type": "bar",
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": "데이터",
                        "data": data_values,
                        "backgroundColor": ["#4F46E5", "#7C3AED", "#EC4899", "#EF4444", "#F59E0B"]
                    }]
                }
            }

    @staticmethod
    def parse_chart_from_text(text: str) -> List[Dict[str, Any]]:
        """
        LLM 응답에서 차트 정보 추출

        Args:
            text: LLM이 생성한 텍스트

        Returns:
            차트 데이터 리스트
        """
        print("\n>> 텍스트에서 차트 정보 파싱 시작")

        charts = []

        # {{CHART_START}} ~ {{CHART_END}} 패턴 찾기
        import re

        chart_pattern = r'\{\{CHART_START\}\}(.*?)\{\{CHART_END\}\}'
        matches = re.findall(chart_pattern, text, re.DOTALL)

        for match in matches:
            try:
                # JSON 파싱 시도
                chart_data = json.loads(match.strip())

                # 데이터 유효성 검사
                if ChartDataGenerator.validate_chart_data(chart_data):
                    charts.append(chart_data)
                else:
                    print(f"- 차트 데이터 유효성 검사 실패: {chart_data}")

            except json.JSONDecodeError as e:
                print(f"- JSON 파싱 실패: {e}")
                continue

        # 차트 패턴이 없으면 키워드로 추출 시도
        if not charts:
            charts = ChartDataGenerator.extract_chart_keywords(text)

        return charts

    @staticmethod
    def validate_chart_data(chart_data: Dict[str, Any]) -> bool:
        """차트 데이터 유효성 검사"""
        required_keys = ["type", "data"]

        for key in required_keys:
            if key not in chart_data:
                return False

        # data 내부 구조 검사
        data = chart_data.get("data", {})
        if not isinstance(data, dict):
            return False

        # labels와 datasets 확인
        if "labels" not in data or "datasets" not in data:
            return False

        if not isinstance(data["datasets"], list) or len(data["datasets"]) == 0:
            return False

        return True

    @staticmethod
    def extract_chart_keywords(text: str) -> List[Dict[str, Any]]:
        """
        텍스트에서 차트 관련 키워드 추출해서 차트 데이터 생성
        """
        print("\n>> 키워드 기반 차트 생성 시작")

        charts = []

        # 차트 관련 키워드 패턴
        chart_keywords = {
            "시세": {"type": "line", "labels": ["1월", "2월", "3월", "4월", "5월"]},
            "가격": {"type": "line", "labels": ["1월", "2월", "3월", "4월", "5월"]},
            "비교": {"type": "bar", "labels": ["항목1", "항목2", "항목3", "항목4"]},
            "분포": {"type": "pie", "labels": ["그룹A", "그룹B", "그룹C", "그룹D"]},
            "트렌드": {"type": "line", "labels": ["1월", "2월", "3월", "4월", "5월"]},
            "순위": {"type": "bar", "labels": ["1위", "2위", "3위", "4위", "5위"]},
        }

        for keyword, config in chart_keywords.items():
            if keyword in text:
                chart_data = ChartDataGenerator.generate_sample_data(
                    chart_type=config["type"],
                    labels=config["labels"],
                    title=f"{keyword} 차트"
                )
                charts.append(chart_data)
                break  # 첫 번째 매칭만 사용

        return charts

# ReportGeneratorAgent에서 사용할 차트 통합 함수
def integrate_charts_into_response(response_text: str, query_type: str = "complex") -> tuple[str, List[Dict[str, Any]]]:
    print("\n>> 응답에 차트 통합 시작")
    print("query_type")

    if query_type == "SIMPLE":
        print("- simple 쿼리이므로 차트 생성하지 않음")
        return response_text, []

    # 텍스트에서 차트 추출
    charts = ChartDataGenerator.parse_chart_from_text(response_text)

    # 키워드 기반 추출
    if not charts:
        charts = ChartDataGenerator.extract_chart_keywords(response_text)

    # fallback 차트 생성은 complex한 경우에만
    if not charts:
        charts = [ChartDataGenerator.generate_sample_data("bar", title="데이터 분석")]

    print(f"- 총 {len(charts)}개 차트 생성됨")

    return response_text, charts
