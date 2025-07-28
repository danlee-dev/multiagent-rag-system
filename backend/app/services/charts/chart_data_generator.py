import json
import re
from typing import Dict, List, Any, Optional, Tuple

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
        print(f"- 차트 타입: {chart_type}")
        print(f"- 라벨: {labels}")

        if not labels:
            labels = ["1월", "2월", "3월", "4월", "5월"]

        # 랜덤 데이터 생성 (실제로는 RAG에서 가져온 데이터 활용)
        import random
        data_values = [random.randint(10, 100) for _ in labels]

        base_chart = {
            "title": title,
            "type": chart_type,
            "source": "RAG 검색 결과",
            "data_type": "sample"
        }

        if chart_type.lower() in ["line", "area", "flow"]:
            base_chart["data"] = {
                "labels": labels,
                "datasets": [{
                    "label": "데이터",
                    "data": data_values,
                    "borderColor": "#4F46E5",
                    "backgroundColor": "#4F46E520"
                }]
            }

        elif chart_type.lower() in ["bar", "column"]:
            colors = ["#4F46E5", "#7C3AED", "#EC4899", "#EF4444", "#F59E0B"]
            base_chart["data"] = {
                "labels": labels,
                "datasets": [{
                    "label": "수량",
                    "data": data_values,
                    "backgroundColor": colors[:len(labels)]
                }]
            }

        elif chart_type.lower() in ["pie", "doughnut"]:
            colors = ["#4F46E5", "#7C3AED", "#EC4899", "#EF4444", "#F59E0B"]
            base_chart["data"] = {
                "labels": labels,
                "datasets": [{
                    "label": "비율",
                    "data": data_values,
                    "backgroundColor": colors[:len(labels)]
                }]
            }

        else:
            # 기본값 - bar 차트
            base_chart["type"] = "bar"
            base_chart["data"] = {
                "labels": labels,
                "datasets": [{
                    "label": "데이터",
                    "data": data_values,
                    "backgroundColor": ["#4F46E5", "#7C3AED", "#EC4899", "#EF4444", "#F59E0B"][:len(labels)]
                }]
            }

        print(f"- 생성된 차트 데이터: {json.dumps(base_chart, indent=2, ensure_ascii=False)}")
        return base_chart

    @staticmethod
    def clean_json_text(text: str) -> str:
        """
        JSON 텍스트에서 markdown 코드 블록 제거 및 정리
        """
        print("\n>> JSON 텍스트 정리 시작")

        # markdown 코드 블록 제거
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)

        # 앞뒤 공백 제거
        text = text.strip()

        print(f"- 정리된 텍스트 길이: {len(text)}자")
        return text

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
        chart_pattern = r'\{\{CHART_START\}\}(.*?)\{\{CHART_END\}\}'
        matches = re.findall(chart_pattern, text, re.DOTALL)

        print(f"- 발견된 차트 패턴 수: {len(matches)}")

        for i, match in enumerate(matches):
            try:
                print(f"- 차트 {i+1} 처리 중...")

                # JSON 텍스트 정리
                clean_text = ChartDataGenerator.clean_json_text(match.strip())

                # JSON 파싱 시도
                chart_data = json.loads(clean_text)

                # 데이터 유효성 검사
                if ChartDataGenerator.validate_chart_data(chart_data):
                    charts.append(chart_data)
                    print(f"- 차트 {i+1} 파싱 성공")
                else:
                    print(f"- 차트 {i+1} 유효성 검사 실패")

            except json.JSONDecodeError as e:
                print(f"- 차트 {i+1} JSON 파싱 실패: {e}")
                print(f"- 문제가 된 텍스트: {match.strip()[:200]}...")
                continue
            except Exception as e:
                print(f"- 차트 {i+1} 처리 중 오류: {e}")
                continue

        # 차트 패턴이 없으면 키워드로 추출 시도
        if not charts:
            print("- 패턴 기반 차트 없음, 키워드 추출 시도")
            charts = ChartDataGenerator.extract_chart_keywords(text)

        print(f"- 최종 파싱된 차트 수: {len(charts)}")
        return charts

    @staticmethod
    def validate_chart_data(chart_data: Dict[str, Any]) -> bool:
        """차트 데이터 유효성 검사"""
        print("\n>> 차트 데이터 유효성 검사")

        required_keys = ["type", "data"]

        for key in required_keys:
            if key not in chart_data:
                print(f"- 필수 키 누락: {key}")
                return False

        # data 내부 구조 검사
        data = chart_data.get("data", {})
        if not isinstance(data, dict):
            print("- data가 dict 타입이 아님")
            return False

        # labels와 datasets 확인
        if "labels" not in data:
            print("- labels 누락")
            return False

        if "datasets" not in data:
            print("- datasets 누락")
            return False

        if not isinstance(data["datasets"], list) or len(data["datasets"]) == 0:
            print("- datasets가 비어있거나 리스트가 아님")
            return False

        print("- 유효성 검사 통과")
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
            "시세": {"type": "line", "labels": ["1월", "2월", "3월", "4월", "5월"], "title": "시세 동향"},
            "가격": {"type": "line", "labels": ["1월", "2월", "3월", "4월", "5월"], "title": "가격 추이"},
            "비교": {"type": "bar", "labels": ["항목1", "항목2", "항목3", "항목4"], "title": "비교 분석"},
            "분포": {"type": "pie", "labels": ["그룹A", "그룹B", "그룹C", "그룹D"], "title": "분포 현황"},
            "트렌드": {"type": "line", "labels": ["1월", "2월", "3월", "4월", "5월"], "title": "트렌드 분석"},
            "순위": {"type": "bar", "labels": ["1위", "2위", "3위", "4위", "5위"], "title": "순위 차트"},
            "매출": {"type": "bar", "labels": ["Q1", "Q2", "Q3", "Q4"], "title": "매출 현황"},
            "점유율": {"type": "pie", "labels": ["A사", "B사", "C사", "기타"], "title": "시장 점유율"},
        }

        text_lower = text.lower()

        for keyword, config in chart_keywords.items():
            if keyword in text_lower:
                print(f"- 키워드 발견: {keyword}")
                chart_data = ChartDataGenerator.generate_sample_data(
                    chart_type=config["type"],
                    labels=config["labels"],
                    title=config["title"]
                )
                charts.append(chart_data)
                break  # 첫 번째 매칭만 사용

        if not charts:
            print("- 매칭되는 키워드 없음, 기본 차트 생성")

        return charts


def integrate_charts_into_response(response_text: str, query_type: str = "complex") -> Tuple[str, List[Dict[str, Any]]]:
    """
    응답에 차트 통합하는 메인 함수

    Args:
        response_text: LLM 응답 텍스트
        query_type: 쿼리 타입 (simple/complex)

    Returns:
        (응답_텍스트, 차트_리스트)
    """
    print("\n>> 응답에 차트 통합 시작")
    print(f"- 쿼리 타입: {query_type}")
    print(f"- 응답 텍스트 길이: {len(response_text)}자")

    if query_type.upper() == "SIMPLE":
        print("- SIMPLE 쿼리이므로 차트 생성하지 않음")
        return response_text, []

    # 1. 텍스트에서 차트 추출 시도
    charts = ChartDataGenerator.parse_chart_from_text(response_text)

    # 2. 키워드 기반 추출 (패턴 기반 추출에 실패한 경우)
    if not charts:
        print("- 패턴 기반 추출 실패, 키워드 기반 시도")
        charts = ChartDataGenerator.extract_chart_keywords(response_text)

    # 3. fallback 차트 생성 (complex한 경우에만)
    if not charts and query_type.upper() == "COMPLEX":
        print("- 키워드 기반 추출도 실패, fallback 차트 생성")
        fallback_chart = ChartDataGenerator.generate_sample_data(
            chart_type="bar",
            labels=["데이터1", "데이터2", "데이터3"],
            title="데이터 분석 결과"
        )
        charts = [fallback_chart]

    print(f"- 최종 생성된 차트 수: {len(charts)}")

    # 차트별 요약 출력
    for i, chart in enumerate(charts):
        print(f"- 차트 {i+1}: {chart.get('title', '제목없음')} ({chart.get('type', '타입없음')})")

    return response_text, charts
