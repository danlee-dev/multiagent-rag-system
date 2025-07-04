# Multi-Agent RAG (로컬 실행)

## 개요
- Colab에서 작성된 Multi-Agent RAG 시스템을 로컬 Python 프로젝트로 리팩토링한 버전입니다.
- 식품개발팀 페르소나 기반, Mock DB 포함, LangChain/LangGraph 기반.

## 폴더 구조
```
.
├── main.py              # 메인 실행 파일
├── agents.py            # 에이전트 클래스
├── models.py            # 데이터 모델 (Pydantic)
├── mock_databases.py    # Mock DB
├── utils.py             # 헬퍼 함수
├── requirements.txt     # 의존성 명시
├── test_multi-agent-rag.ipynb # Colab test file
└── README.md            # 실행법 안내
```

## 실행 방법
1. Python 3.9 이상 설치
2. 의존성 설치
   ```bash
   pip install -r requirements.txt
   ```
3. OpenAI API 키 준비
   - .env 파일 생성 후 아래와 같이 작성
     ```
     OPENAI_API_KEY=sk-...
     ```
   - 또는 환경 변수로 직접 설정
4. 실행
   ```bash
   python main.py
   ```

## 주요 파일 설명
- `main.py` : 전체 파이프라인 진입점, 샘플 쿼리 실행
- `models.py` : 에이전트/메시지/DB 등 데이터 모델 정의
- `agents.py` : PlanningAgent 등 주요 에이전트 클래스
- `mock_databases.py` : 테스트용 Mock DB
- `utils.py` : 헬퍼 함수 및 샘플 데이터 생성

## 참고
- 실제 DB/외부 API 연동이 아닌 Mock DB 기반 예제입니다.
- LangChain, LangGraph, OpenAI API 사용
