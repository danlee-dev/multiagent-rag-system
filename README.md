# Multi-Agent RAG System (Production-Ready Version)

## Overview
크라우드웍스의 산학 프로젝트로 개발된 B2B AI Agent 시스템입니다. 식품회사의 구매팀, 총무팀, 마케팅팀, 제품개발팀을 위한 전문 보고서 생성 시스템으로, 다중 데이터베이스 검색과 팀별 맞춤형 템플릿을 제공합니다.

---

## System Architecture

### Project Structure (Refactored)
```
test-report/
├── core/                          # 핵심 비즈니스 로직
│   ├── agents/                     # AI 에이전트들
│   │   └── agents.py              # 모든 에이전트 로직 (ReportGeneratorAgent 포함)
│   ├── models/                     # 데이터 모델들
│   │   └── models.py              # Pydantic 모델들
│   └── config/                    # 설정 관리
│       ├── report_config.py       # 보고서 템플릿 설정
│       └── env_checker.py         # 환경 변수 체크
│
├── services/                      # 외부 서비스 연동
│   ├── database/                  # 데이터베이스 서비스
│   │   ├── postgres_rag_tool.py   # PostgreSQL RAG
│   │   ├── neo4j_rag_tool.py     # Neo4j RAG
│   │   ├── mock_databases.py      # 목 데이터베이스
│   │   └── neo4j_structure.json   # Neo4j 스키마
│   ├── search/                    # 검색 서비스
│   │   └── search_tools.py        # 웹 검색 도구들
│   ├── charts/                    # 차트 생성 서비스
│   │   └── chart_data_generator.py # 차트 생성기
│   ├── templates/                 # 템플릿 관리
│   │   └── report_templates.py    # 보고서 템플릿 매니저
│   └── builders/                  # 프롬프트 빌더
│       └── prompt_builder.py      # 프롬프트 생성기
│
├── utils/                         # 유틸리티 함수들
│   ├── utils.py                   # 공통 유틸리티
│   ├── memory/                    # 메모리 관리
│   │   └── hierarchical_memory.py # 계층적 메모리
│   ├── testing/                   # 테스트 유틸리티
│   └── analyzers/                 # 분석 도구
│       └── query_analyzer.py      # 쿼리 분석기
│
├── tools/                         # 개발 도구들
│   └── query/                     # 쿼리 도구들
│       └── neo4j_query.py         # Neo4j 쿼리
│
├── tests/                         # 테스트 파일들
│   └── test_connection.py         # 연결 테스트
│
├── main.py                            # 메인 실행 파일
├── docker-compose.yml                 # Docker 컨테이너 설정
├── Dockerfile                         # Docker 이미지 설정
└── requirements.txt               # Python 의존성
```

### Core Agents
- **PlanningAgent**: 사용자 질의 분석 및 검색 계획 수립
- **RetrieverAgentX**: Graph DB 전문 검색 (Neo4j)
- **RetrieverAgentY**: 다중소스 검색 (Vector DB, PostgreSQL, Web)
- **CriticAgent**: 검색 결과 품질 평가 및 개선 제안
- **ContextIntegratorAgent**: 검색 결과 통합 및 구조화
- **ReportGeneratorAgent**: 팀별 맞춤형 보고서 생성 (NEW)

### Team-Specific Templates
- **구매팀**: 시세 분석, 공급업체 평가, 조달 전략
- **총무팀**: 급식 운영, 예산 관리, 대체 식재료
- **마케팅팀**: 트렌드 분석, 소비자 행동, 캠페인 전략
- **제품개발팀**: 신소재 연구, 기능성 평가, R&D 인사이트

---

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- OpenAI API Key

### Installation

#### 1. Docker 방식 (권장)
```bash
# 1. 저장소 클론
git clone <repository-url>
cd multiagent-rag-system/backend

# 2. 환경 변수 설정
cp .env.example .env
# .env 파일에서 OPENAI_API_KEY 설정

# 3. Docker 빌드 및 실행
docker-compose up --build

# 4. 접속 확인
curl http://localhost:8000/health
```

#### 2. 로컬 개발 방식
```bash
# 1. 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 환경 변수 설정
export OPENAI_API_KEY="your-api-key"

# 4. 실행
python main.py
```

---

## Configuration

### 환경 변수 (.env)
```bash
# OpenAI
OPENAI_API_KEY=sk-your-key-here

# Database
POSTGRES_URL=postgresql://user:pass@localhost:5432/db
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# API Keys
GOOGLE_API_KEY=your-google-key
SERPAPI_KEY=your-serpapi-key
```

### 보고서 템플릿 설정
```python
# core/config/report_config.py
class TeamType(Enum):
    MARKETING = "marketing"
    PURCHASING = "purchasing"
    DEVELOPMENT = "development"
    GENERAL_AFFAIRS = "general_affairs"

class ReportType(Enum):
    BRIEF = "brief"         # 500-800단어
    STANDARD = "standard"   # 1000-1500단어
    DETAILED = "detailed"   # 1500-2000단어
    COMPREHENSIVE = "comprehensive"  # 2000-3000단어
```

---

## Features

### 핵심 기능
- **다중 데이터베이스 검색**: PostgreSQL + Neo4j + Vector DB + Web Search
- **팀별 맞춤 템플릿**: 4개 팀 × 4개 복잡도 레벨 = 16가지 템플릿
- **실시간 차트 생성**: JSON 기반 동적 차트 (실제 데이터 우선)
- **피드백 루프**: Agent 간 실시간 피드백으로 검색 품질 향상
- **다국어 지원**: 한국어/영어 (확장 가능)
- **메모리 관리**: 계층적 메모리로 컨텍스트 유지

### 차트 생성 예시
```json
{CHART_START}
{"title": "월별 귀리 가격 변동 (실제 데이터)", "type": "line", "data": {"labels": ["1월", "2월", "3월"], "datasets": [{"label": "가격(원/kg)", "data": [1200, 1350, 1180]}]}, "source": "농진청 RDB", "data_type": "real"}
{CHART_END}
```

---

## Development

### 새로운 팀 타입 추가
```python
# 1. core/config/report_config.py
class TeamType(Enum):
    NEW_TEAM = "new_team"

# 2. utils/analyzers/query_analyzer.py
TEAM_KEYWORDS = {
    TeamType.NEW_TEAM: ["키워드1", "키워드2", "keyword3"]
}

# 3. services/templates/report_templates.py
# 새 템플릿 추가
```

### 새로운 보고서 타입 추가
```python
# 1. ReportType enum에 추가
# 2. 템플릿 정의
# 3. 번역 데이터 추가
```

### 테스트 실행
```bash
# 단위 테스트
python -m pytest tests/

# 연결 테스트
python tests/test_connection.py

# Docker 테스트
docker-compose exec backend python tests/test_connection.py
```

---

## Docker Commands

```bash
# 전체 재빌드 (캐시 무시)
docker-compose build --no-cache

# 로그 확인
docker-compose logs -f backend

# 컨테이너 접속
docker-compose exec backend bash

# 완전 정리
docker-compose down --rmi all && docker system prune -f

# 특정 서비스만 재시작
docker-compose restart backend
```

---

## API Usage Examples

### 구매팀 시세 분석 요청
```python
query = "국내 귀리 시세 분석 보고서를 작성해주세요. 최근 3개월 동향과 향후 전망을 포함해서"
# → 자동으로 PURCHASING 팀 템플릿 적용
# → PostgreSQL 농진청 데이터 우선 검색
# → 시세 차트 자동 생성
```

### 마케팅팀 트렌드 분석
```python
query = "건강식품 마케팅 캠페인 전략 수립을 위한 소비자 트렌드 분석"
# → MARKETING 팀 템플릿 적용
# → 소비자 행동 분석 섹션 자동 포함
# → 트렌드 차트 및 세그먼테이션 차트 생성
```

---

## Project Goals
이 시스템은 크라우드웍스의 산학 협력 프로젝트의 일환으로, 실제 식품회사에서 사용 가능한 B2B AI Agent 플랫폼을 목표로 합니다.

### 기대 효과
- **업무 효율성 30% 향상**: 자동화된 보고서 생성
- **의사결정 속도 50% 개선**: 실시간 데이터 통합 분석
- **데이터 신뢰도 향상**: 다중 소스 검증 및 출처 명시
- **팀별 맞춤 인사이트**: 전문 도메인 특화 분석

---

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Contact
- **Project**: 크라우드웍스 산학 협력 프로젝트
- **Developer**: 이성민 (고려대학교 컴퓨터학과 23학번)
- **Company**: 크라우드웍스
- **GitHub**: [danlee-dev](https://github.com/danlee-dev)

---

## License
This project is part of an industry-academia collaboration with Crowdworks.

---

## Recent Updates
- **v2.0.0**: 모듈화 리팩토링 및 팀별 템플릿 시스템
- **v1.5.0**: Docker 컨테이너화 및 프로덕션 준비
- **v1.0.0**: 초기 Multi-Agent RAG 시스템 구현
