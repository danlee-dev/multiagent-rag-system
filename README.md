# Multi-Agent RAG System v3.0 (Custom AFLOW Architecture)

## Overview
크라우드웍스의 산학 프로젝트로 개발된 차세대 B2B AI Agent 시스템입니다. **LangGraph 기반의 동적 워크플로우**와 **실시간 스트리밍 UI**를 통해 복잡한 비즈니스 쿼리를 처리하는 전문 AI 플랫폼입니다.

### 주요 특징
- **Custom AFLOW (Adaptive Flow) 아키텍처**: 동적 계획 수립 및 실행
- **실시간 스트리밍**: Claude 스타일 검색 결과 및 보고서 생성
- **ReAct Agent 지원**: 복잡한 추론과 도구 사용을 위한 선택적 활성화
- **자동 Triage**: 간단한 대화 vs 복잡한 작업 자동 분류
- **실시간 차트 생성**: 동적 데이터 시각화

---

## System Architecture

### Project Structure (v3.0)
```
multiagent-rag-system/
├── backend/
│   ├── app/
│   │   ├── core/                      # 핵심 비즈니스 로직
│   │   │   ├── agents/                # AI 에이전트들
│   │   │   │   ├── orchestrator.py    # TriageAgent, OrchestratorAgent
│   │   │   │   ├── worker_agents.py   # DataGathererAgent, ProcessorAgent
│   │   │   │   └── conversational_agent.py  # SimpleAnswererAgent
│   │   │   ├── models/                # 데이터 모델들
│   │   │   │   └── models.py          # StreamingAgentState, SearchResult
│   │   │   └── config/                # 설정 관리
│   │   │       ├── env_checker.py     # 환경 변수 체크
│   │   │       ├── rag_config.py      # RAG 설정
│   │   │       └── report_config.py   # 보고서 템플릿 설정
│   │   │
│   │   ├── services/                  # 외부 서비스 연동
│   │   │   ├── database/              # 데이터베이스 서비스
│   │   │   │   ├── postgres_rag_tool.py    # PostgreSQL RAG
│   │   │   │   ├── neo4j_rag_tool.py       # Neo4j RAG
│   │   │   │   └── mock_databases.py       # 목 데이터베이스
│   │   │   ├── search/                # 검색 서비스
│   │   │   │   └── search_tools.py    # 웹 검색, Vector DB 검색
│   │   │   ├── charts/                # 차트 생성 서비스
│   │   │   ├── templates/             # 템플릿 관리
│   │   │   └── builders/              # 프롬프트 빌더
│   │   │
│   │   ├── utils/                     # 유틸리티 함수들
│   │   │   ├── memory/                # 계층적 메모리 시스템
│   │   │   ├── analyzers/             # 분석 도구
│   │   │   └── testing/               # 테스트 유틸리티
│   │   │
│   │   ├── tools/                     # 개발 도구들
│   │   ├── tests/                     # 테스트 파일들
│   │   └── main.py                    # FastAPI 메인 서버
│   │
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/                          # Next.js 프론트엔드
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.js                # 메인 채팅 인터페이스
│   │   │   └── globals.css            # 스타일링 (Claude 스타일)
│   │   └── components/
│   │       ├── ChartComponent.js      # 동적 차트 렌더링
│   │       └── SourcesPanel.js        # 출처 정보 패널
│   │
│   ├── package.json
│   └── Dockerfile
│
├── docker-compose.yml                 # 전체 시스템 오케스트레이션
└── README.md
```

### Core Agent Architecture (Custom AFLOW)

#### 1. **TriageAgent** (Entry Point)
```python
# 자동 분류: 간단한 대화 vs 복잡한 작업
flow_type = "chat" | "task"
```

#### 2. **OrchestratorAgent** (Planning)
```python
# 동적 계획 수립
plan = {
    "steps": [
        {"agent": "DataGathererAgent", "tool": "web_search", "inputs": {...}},
        {"agent": "ProcessorAgent", "processor_type": "integrate_context", ...},
        {"agent": "ProcessorAgent", "processor_type": "generate_report", ...}
    ]
}
```

#### 3. **Worker Agents** (Execution)
- **DataGathererAgent**: 다중 소스 데이터 수집
  - Web Search (최신 정보 우선)
  - Vector DB Search
  - Graph DB Search (Neo4j)
  - RDB Search (PostgreSQL)
  - Web Scraping

- **ProcessorAgent**: 데이터 처리 및 보고서 생성
  - Context Integration
  - Result Criticism
  - Report Generation (ReAct 지원)
  - Chart Generation

#### 4. **SimpleAnswererAgent** (Direct Response)
```python
# 간단한 질문에 대한 직접 응답
query = "안녕하세요" → SimpleAnswererAgent
query = "복잡한 시장 분석 요청" → OrchestratorAgent
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- API Keys: OpenAI, Google (Gemini), Serper

### Installation

#### 1. Docker 방식 (권장)
```bash
# 1. 저장소 클론
git clone https://github.com/danlee-dev/multiagent-rag-system.git
cd multiagent-rag-system

# 2. 환경 변수 설정
cp backend/.env.example backend/.env
# .env 파일에서 API 키들 설정

# 3. 전체 시스템 실행
docker-compose up --build

# 4. 접속 확인
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000/health
```

#### 2. 로컬 개발 방식
```bash
# Backend 실행
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
python -m app.main

# Frontend 실행 (별도 터미널)
cd frontend
npm install
npm run dev
```

---

## Configuration

### 환경 변수 (.env)
```bash
# LLM API Keys
OPENAI_API_KEY=sk-your-openai-key
GOOGLE_API_KEY=your-google-gemini-key

# Search API Keys
SERPER_API_KEY=your-serper-key
LANGSMITH_API_KEY=your-langsmith-key

# ReAct Agent 설정
USE_REACT_AGENT=false  # true로 설정하면 복잡한 추론 활성화

# Database (선택사항)
POSTGRES_URL=postgresql://user:pass@localhost:5432/db
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

### 시스템 설정
```python
# ProcessorAgent ReAct 모드 설정
processor_agent = ProcessorAgent(use_react=True)  # 복잡한 추론 활성화

# Triage 민감도 조정 (간단한 질문 vs 복잡한 작업)
triage_threshold = 0.7  # 높을수록 더 많은 질문을 복잡한 작업으로 분류
```

---

## Features

### 핵심 기능
- **동적 워크플로우**: LangGraph 기반 조건부 실행 및 루프 처리
- **실시간 스트리밍**: Server-Sent Events로 검색 과정 실시간 표시
- **Claude 스타일 UI**: 검색 결과 박스, 컴팩트한 디자인, 다크 테마
- **다중 소스 검색**: Web + Vector DB + Graph DB + RDB 통합
- **자동 차트 생성**: Chart.js 기반 동적 데이터 시각화
- **메모리 관리**: 대화 히스토리 및 컨텍스트 유지

### 워크플로우 예시
```mermaid
graph TD
    A[사용자 질문] --> B[TriageAgent]
    B -->|간단한 질문| C[SimpleAnswererAgent]
    B -->|복잡한 작업| D[OrchestratorAgent]
    D --> E[계획 수립]
    E --> F[DataGathererAgent]
    F --> G[ProcessorAgent]
    G -->|품질 부족| D
    G -->|완료| H[최종 보고서]
```

### 실시간 스트리밍 UI
```javascript
// 검색 과정 실시간 표시
{
  "type": "status",
  "message": "웹에서 정보 수집 중..."
}

// 검색 결과 박스 표시
{
  "type": "search_results",
  "tool_name": "웹 검색",
  "results": [...]
}

// 보고서 내용 스트리밍
{
  "type": "content",
  "chunk": "## 시장 분석 결과\n\n..."
}
```

---

## Development

### 아키텍처 확장

#### 새로운 데이터 소스 추가
```python
# services/search/search_tools.py에 새 도구 추가
@tool
def new_data_source_search(query: str) -> List[Dict]:
    """새로운 데이터 소스 검색"""
    return search_results

# worker_agents.py의 DataGathererAgent에 매핑 추가
self.tool_mapping = {
    "new_source": self._new_source_search,
    # ... 기존 도구들
}
```

#### Agent 동작 커스터마이징
```python
# ProcessorAgent의 ReAct 모드 커스터마이징
class ProcessorAgent:
    def __init__(self, use_react: bool = False):
        self.use_react = use_react  # 환경변수 USE_REACT_AGENT로도 제어 가능

    async def _generate_report_streaming(self, data, query):
        if self.use_react:
            # 복잡한 추론과 도구 사용
            async for chunk in self._react_agent_streaming_chunks(...):
                yield chunk
        else:
            # 빠른 직접 생성
            async for chunk in self._fallback_report_generation_streaming_chunks(...):
                yield chunk
```

#### 프론트엔드 UI 커스터마이징
```javascript
// 검색 결과 박스 스타일 수정
.claude-search-results {
  max-width: 600px;  /* 컴팩트한 너비 */
  margin: 16px auto; /* 중앙 정렬 */
}

// 새로운 메시지 타입 처리
case "custom_result":
  // 커스텀 결과 렌더링 로직
  break;
```

### 테스트 실행
```bash
# Backend 테스트
cd backend
python -m pytest tests/
python tests/test_connection.py

# Frontend 테스트
cd frontend
npm test

# 전체 시스템 테스트
docker-compose -f docker-compose.test.yml up
```

---

## Docker Commands

```bash
# 전체 시스템 시작
docker-compose up --build

# 백엔드만 재시작
docker-compose restart backend

# 프론트엔드만 재시작
docker-compose restart frontend

# 로그 확인
docker-compose logs -f backend
docker-compose logs -f frontend

# 컨테이너 접속
docker-compose exec backend bash
docker-compose exec frontend sh

# 완전 정리 (데이터 포함)
docker-compose down -v --rmi all
docker system prune -af

# 캐시 무시 완전 재빌드
docker-compose build --no-cache
```

---

## API Usage Examples

### 실시간 스트리밍 요청
```javascript
// Frontend에서 Server-Sent Events 사용
const eventSource = new EventSource('/query/stream', {
  method: 'POST',
  body: JSON.stringify({
    query: "건강기능식품 시장 동향 분석해줘",
    session_id: "unique-session-id"
  })
});

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch(data.type) {
    case 'status':
      updateStatus(data.message);
      break;
    case 'search_results':
      addSearchResultBox(data.tool_name, data.results);
      break;
    case 'content':
      appendToMessage(data.chunk);
      break;
  }
};
```

### Backend API 직접 호출
```python
import requests

# 스트리밍 요청
response = requests.post('http://localhost:8000/query/stream',
    json={
        "query": "귀리 시장 분석 보고서 작성해줘",
        "session_id": "test-session"
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        data = json.loads(line.decode('utf-8').replace('data: ', ''))
        print(f"Type: {data['type']}, Content: {data.get('message', data.get('chunk', ''))}")
```

### 다양한 질문 유형 예시
```python
# 간단한 질문 → SimpleAnswererAgent
"안녕하세요"
"오늘 날씨 어때요?"

# 복잡한 분석 요청 → OrchestratorAgent + DataGathererAgent + ProcessorAgent
"건강기능식품 시장의 최신 동향을 분석하고 향후 전망을 제시해줘"
"귀리 가격 변동 요인을 분석하고 차트로 시각화해줘"
"국내외 프로바이오틱스 시장 비교 분석 보고서 작성해줘"
```

---

## Performance & Scalability

### 시스템 성능 최적화
- **TTFT (Time To First Token)**: 500ms 이하 첫 응답
- **스트리밍 처리**: 실시간 청크 단위 전송으로 사용자 경험 향상
- **ReAct 선택적 활성화**: 간단한 질문은 빠른 처리, 복잡한 질문은 정확성 우선
- **에러 핸들링**: API 할당량 초과시 자동 폴백 처리

### 확장성 고려사항
```python
# 환경 변수를 통한 동적 설정
USE_REACT_AGENT=false  # 성능 우선시
USE_REACT_AGENT=true   # 정확성 우선시

# Agent별 독립적 확장 가능
DataGathererAgent  # 새로운 데이터 소스 추가 용이
ProcessorAgent     # 처리 로직 모듈화
```

---

## Troubleshooting

### 일반적인 문제 해결

#### 1. API 키 관련 오류
```bash
# 환경 변수 확인
python backend/app/core/config/env_checker.py

# 출력 예시:
# >> OPENAI_API_KEY: 설정 완료 (sk-proj1...)
# >> GOOGLE_API_KEY: 설정되지 않음
```

#### 2. Docker 컨테이너 문제
```bash
# 컨테이너 상태 확인
docker-compose ps

# 로그 확인
docker-compose logs backend
docker-compose logs frontend

# 완전 재시작
docker-compose down && docker-compose up --build
```

#### 3. 프론트엔드 연결 문제
```bash
# Backend API 응답 확인
curl http://localhost:8000/health

# 예상 응답:
# {"status": "healthy", "timestamp": "2025-08-04T...", "version": "3.0"}
```

#### 4. 스트리밍 중단 문제
- **원인**: API 할당량 초과, 네트워크 문제
- **해결**: 자동 폴백 시스템이 작동하여 기본 요약 제공
- **확인**: 브라우저 개발자 도구 → Network 탭에서 EventSource 연결 상태 확인

---

## Project Goals & Impact

이 시스템은 크라우드웍스의 산학 협력 프로젝트의 일환으로, **차세대 B2B AI Agent 플랫폼**을 목표로 합니다.

### 프로젝트 목표
- **Custom AFLOW 아키텍처 검증**: 동적 워크플로우의 실용성 입증
- **실시간 UX 혁신**: Claude 스타일 스트리밍 인터페이스로 사용자 경험 개선
- **Multi-Modal AI 통합**: 텍스트 + 차트 + 검색 결과의 통합적 제시
- **Production-Ready 시스템**: 실제 비즈니스 환경에서 활용 가능한 안정성

### 기대 효과
- **응답 속도 3배 향상**: 실시간 스트리밍으로 체감 속도 개선
- **정보 신뢰도 강화**: 다중 소스 검증 및 실시간 출처 표시
- **의사결정 지원 강화**: 검색 과정 투명화로 AI 결과 신뢰성 증대
- **확장성 확보**: 모듈화된 Agent 시스템으로 새로운 기능 추가 용이

### 기술적 혁신점
1. **Dynamic Planning**: 사용자 질문에 따른 실시간 실행 계획 수립
2. **Conditional Execution**: LangGraph 기반 조건부 분기 및 루프 처리
3. **Real-time Streaming**: SSE를 통한 검색 과정 실시간 시각화
4. **Adaptive UI**: 검색 결과에 따른 동적 UI 컴포넌트 생성

---

## Contributing

### 개발 참여 방법
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'feat: add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### 코드 컨벤션
```bash
# Commit 메시지 형식
feat: 새로운 기능 추가
fix: 버그 수정
docs: 문서 수정
style: 코드 포맷팅
refactor: 코드 리팩토링
test: 테스트 코드
chore: 빌드 프로세스 수정
```

### 개발 환경 설정
```bash
# 1. Repository 클론
git clone https://github.com/danlee-dev/multiagent-rag-system.git
cd multiagent-rag-system

# 2. 개발용 브랜치 생성
git checkout -b feature/your-feature-name

# 3. 로컬 개발 환경 구성
docker-compose -f docker-compose.dev.yml up

# 4. 테스트 실행
docker-compose exec backend python -m pytest
```

---

## Contact & Support

### 연락처
- **Company**: 크라우드웍스 (Crowdworks)
- **GitHub**: [@danlee-dev](https://github.com/danlee-dev)

### 산학 협력
- **대학**: 고려대학교 컴퓨터학과
- **기업**: 크라우드웍스
- **분야**: B2B AI Agent 플랫폼 개발
- **기간**: 2025.06 ~ 2025.11

### 이슈 및 지원
- **버그 리포트**: [GitHub Issues](https://github.com/danlee-dev/multiagent-rag-system/issues)
- **기능 요청**: [GitHub Discussions](https://github.com/danlee-dev/multiagent-rag-system/discussions)
- **문서**: [Wiki](https://github.com/danlee-dev/multiagent-rag-system/wiki)

---

## License & Acknowledgments

### 라이선스
이 프로젝트는 크라우드웍스의 산학 협력 프로젝트의 일환으로 개발되었습니다.
상용 사용시 크라우드웍스와 사전 협의가 필요합니다.

### 감사의 말
- **LangChain/LangGraph**: 동적 워크플로우 프레임워크
- **OpenAI/Google**: LLM API 제공
- **React/Next.js**: 프론트엔드 프레임워크
- **FastAPI**: 백엔드 API 프레임워크

---

## Version History

### Recent Updates
- **v3.0.0** (2025.08 1주차): Custom AFLOW 아키텍처 도입, 실시간 스트리밍 UI
- **v2.0.0** (2025.07 4주차): 모듈화 리팩토링 및 팀별 템플릿 시스템
- **v1.5.0** (2025.07 3주차): Docker 컨테이너화
- **v1.0.0** (2025.07 2주차): 초기 Multi-Agent RAG 시스템 구현

### 로드맵
- **v3.1.0**: 멀티모달 입력 지원 (이미지, 문서)
