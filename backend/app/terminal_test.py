"""
터미널에서 RAG 시스템을 테스트하기 위한 스크립트
기존 main.py의 터미널 대화형 기능을 분리
"""

import asyncio
import uuid
from datetime import datetime
from pathlib import Path

# 기존 main.py에서 사용하던 것들 임포트
from main import RAGWorkflow, parse_report_and_extract_charts


def save_result_as_markdown(
    query: str, final_answer: str, output_dir: str = "test-report"
):
    """
    >> 결과를 마크다운 파일로 저장
    - query: 사용자가 입력한 질문
    - final_answer: RAG 시스템이 생성한 최종 답변
    - output_dir: 저장할 디렉토리명
    """

    # 디렉토리 생성 (없으면 만들기)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 파일명 생성 (현재 시간 기준)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.md"
    file_path = output_path / filename

    # 차트 데이터 추출
    cleaned_answer, chart_data = parse_report_and_extract_charts(final_answer)

    # 마크다운 내용 구성
    markdown_content = f"""# RAG 시스템 분석 보고서

## 생성 정보
- 사용자 쿼리: "{query}"
- 생성 시간: {datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분 %S초")}
- 파일명: {filename}

---

## 최종 답변

{cleaned_answer}

---

## 추출된 차트 데이터

총 {len(chart_data)}개의 차트가 발견되었습니다.

"""

    # 차트 데이터를 마크다운에 포함
    for i, chart in enumerate(chart_data):
        markdown_content += f"""
### 차트 {i+1}: {chart.get('title', 'Unknown')}

```json
{chart}
```

"""

    markdown_content += """
---
*본 보고서는 RAG 시스템에 의해 자동 생성되었습니다.*
"""

    # 파일 저장
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print(f"\n>> 마크다운 파일 저장 완료: {file_path}")
        print(f">> 차트 데이터 {len(chart_data)}개 포함")
        return str(file_path)

    except Exception as e:
        print(f"\n>> 파일 저장 실패: {e}")
        return None


async def main():
    """대화형 RAG 시스템을 실행하는 메인 비동기 함수"""
    print(">> RAG 시스템 시작 (터미널 테스트 모드)")
    rag = RAGWorkflow()

    # 대화 세션을 구분하기 위한 고유 ID 생성
    conversation_id = f"terminal-session-{uuid.uuid4()}"
    print(f">> 대화 ID: {conversation_id}")
    print(">> 대화를 시작하세요. 종료하려면 'exit'을 입력하세요.")

    while True:
        try:
            query = input("\n질문을 입력하세요: ").strip()

            if query.lower() == "exit":
                print(">> 대화를 종료합니다.")
                break
            if not query:
                continue

            # ---- 시간 측정 시작 ----
            start_time = datetime.now()
            print(">> 답변을 생성 중입니다...")

            # RAG 워크플로우 실행 (API 버전 사용)
            result = await rag.run_api(query, conversation_id=conversation_id)

            # ---- 시간 측정 종료 ----
            end_time = datetime.now()
            processing_time = result.get("processing_time", 0.0)

            # 결과 출력
            if result["success"]:
                print(f"\n\n=== 답변 ===")
                print(result["final_answer"])

                # 차트 정보 출력
                if result["chart_data"]:
                    print(f"\n=== 차트 정보 ===")
                    print(f"총 {len(result['chart_data'])}개의 차트가 생성되었습니다:")
                    for i, chart in enumerate(result["chart_data"]):
                        print(
                            f"  {i+1}. {chart.get('title', 'Unknown')} ({chart.get('type', 'unknown')} 타입)"
                        )

                # 원본 답변으로 파일 저장 (차트 포함)
                saved_file = save_result_as_markdown(query, result["final_answer"])
                if saved_file:
                    print(f"\n>> 저장된 파일: {saved_file}")
            else:
                print(f"\n>> 오류 발생: {result['final_answer']}")

            # ---- 소요 시간 출력 ----
            print(f"\n>> 소요 시간: {processing_time:.2f}초")

        except Exception as e:
            print(f"\n>> 실행 중 오류 발생: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n>> 사용자에 의해 중단되었습니다.")
    print("\n>> 실행 완료")
