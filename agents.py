from langchain_openai import ChatOpenAI
from models import (
    AgentType,
    DatabaseType,
    QueryPlan,
    AgentMessage,
    MessageType,
    SearchResult,
    CriticResult,
)
import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
import random
import time
from utils import create_agent_message


# 실시간 피드백 채널 (X <-> Y)
class RealTimeFeedbackChannel:
    """
    Agent 간 실시간 피드백 채널
    메모리 기반 큐(asyncio.Queue) 사용
    메시지를 비동기적으로 주고 받음
    """

    def __init__(self):
        self.x_to_y_queue = asyncio.Queue()  # X -> Y
        self.y_to_x_queue = asyncio.Queue()  # Y -> X
        self.active = asyncio.Event()  # 채널 Active 여부
        self.active.set()  # 초기 Actve 됨
        self.message_history: List[AgentMessage] = []  # 모든 송수신 메시지의 로그 기록

    async def send_x_to_y(self, message: AgentMessage):
        """X → Y 메시지 전송"""
        self.message_history.append(message)
        await self.x_to_y_queue.put(message)

    async def send_y_to_x(self, message: AgentMessage):
        """Y → X 메시지 전송"""
        self.message_history.append(message)
        await self.y_to_x_queue.put(message)

    async def get_messages_for_y(
        self,
    ) -> AsyncGenerator[AgentMessage, None]:  # 실시간 스트리밍 지원
        """Y가 X로부터 받을 메시지들 스트리밍"""
        while self.active.is_set():
            try:
                message = await asyncio.wait_for(self.x_to_y_queue.get(), timeout=0.5)
                yield message
            except asyncio.TimeoutError:
                continue

    async def get_messages_for_x(self) -> AsyncGenerator[AgentMessage, None]:
        """X가 Y로부터 받을 메시지들 스트리밍"""
        while self.active.is_set():
            try:
                message = await asyncio.wait_for(self.y_to_x_queue.get(), timeout=0.5)
                yield message
            except asyncio.TimeoutError:
                continue

    def stop(self):
        """채널 종료"""
        self.active.clear()

    def get_message_count(self) -> Dict[str, int]:
        """메시지 통계"""
        x_to_y = sum(
            1 for msg in self.message_history if msg.from_agent == AgentType.RETRIEVER_X
        )
        y_to_x = sum(
            1 for msg in self.message_history if msg.from_agent == AgentType.RETRIEVER_Y
        )
        return {"x_to_y": x_to_y, "y_to_x": y_to_x, "total": len(self.message_history)}


# PlanningAgent: 쿼리 분석 및 작업 플랜 수립 담당
class PlanningAgent:
    def __init__(self):
        self.chat = ChatOpenAI(model="gpt-3.5-turbo")
        self.agent_type = AgentType.PLANNING

    async def plan(self, state):
        print(">> Planning Start")
        print(f"- 원본 쿼리: {state.original_query}")

        complexity_result = await self._judge_complexity(state.original_query)
        print(f"\n>> 복잡도 판단: {complexity_result}")

        if complexity_result == "SIMPLE":
            # 단순 쿼리 - Vector DB 사용 여부만 결정
            needs_vector_db = await self._needs_vector_search(state.original_query)

            if needs_vector_db:
                required_dbs = [DatabaseType.VECTOR_DB]
                reasoning = "단순 쿼리, Vector DB 검색 필요"
            else:
                required_dbs = []
                reasoning = "단순 쿼리, DB 검색 불필요 (인사말/간단한 대화)"

            print(f"- Vector DB 필요: {needs_vector_db}")
            print(f"- 선택된 DB: {required_dbs}")

            state.query_plan = QueryPlan(
                original_query=state.original_query,
                sub_queries=[state.original_query],
                required_databases=required_dbs,
                reasoning=reasoning,
                estimated_complexity="low",
            )
            print("\n>> SIMPLE 쿼리로 처리")

        else:
            # 복잡 쿼리 - Retriever X는 Graph DB 고정, Retriever Y는 선택적
            sub_queries = await self._decompose_query(state.original_query)
            retriever_y_dbs = await self._select_retriever_y_databases(
                state.original_query
            )

            # Retriever X는 항상 Graph DB
            required_dbs = [DatabaseType.GRAPH_DB]

            # Retriever Y가 사용할 DB들 추가
            required_dbs.extend(retriever_y_dbs)

            print(f"\n>> 쿼리 분해 결과: {sub_queries}")
            print(f"- Retriever X: GRAPH_DB (고정)")
            print(f"- Retriever Y: {retriever_y_dbs}")
            print(f"- 전체 선택된 DB: {required_dbs}")

            state.query_plan = QueryPlan(
                original_query=state.original_query,
                sub_queries=sub_queries,
                required_databases=required_dbs,
                reasoning=f"복잡 쿼리로 판단, {len(sub_queries)}개로 분해",
                estimated_complexity="high",
            )
            print("\n>> COMPLEX 쿼리로 처리")

        state.planning_complete = True

        memory = state.get_agent_memory(AgentType.PLANNING)
        memory.add_finding(
            f"Planning 완료 - {len(state.query_plan.sub_queries)}개 세부쿼리, {len(required_dbs)}개 DB"
        )

        print("\n>> Planning Agent 완료")
        return state

    async def _judge_complexity(self, query):
        """복잡도 판단"""
        prompt = f"""
        다음 질문이 단순한지 복잡한지 판단해주세요.

        질문: "{query}"

        SIMPLE: 단순한 사실 질문, 인사말, 간단한 정보 요청
        - 예: "안녕하세요", "고마워", "쌀의 영양성분은?", "퀴노아란 무엇인가요?"

        COMPLEX: 분석, 비교, 예측, 종합, 관계 탐색이 필요한 질문
        - 예: "쌀 가격 상승 원인과 전망", "퀴노아와 귀리 비교 분석", "완두콩 시장 동향"

        SIMPLE 또는 COMPLEX 중 하나만 답변해주세요:
        """

        response = await self.chat.ainvoke(prompt)
        return response.content.strip()

    async def _needs_vector_search(self, query):
        """단순 쿼리에서 Vector DB 검색이 필요한지 판단"""
        prompt = f"""
        다음 단순한 질문이 데이터베이스 검색이 필요한지 판단해주세요.

        질문: "{query}"

        검색이 필요한 경우:
        - 식품/농업 관련 정보 요청 (영양성분, 특징, 정의 등)
        - 예: "쌀의 영양성분은?", "퀴노아란 무엇인가요?", "연어의 특징은?"

        검색이 불필요한 경우:
        - 인사말, 감사 인사, 일반적인 대화
        - 예: "안녕하세요", "고마워요", "안녕", "반가워요", "잘 지내세요?"

        검색이 필요하면 YES, 불필요하면 NO로만 답변해주세요:
        """

        response = await self.chat.ainvoke(prompt)
        return response.content.strip().upper() == "YES"

    async def _select_retriever_y_databases(self, query):
        """복잡 쿼리에서 Retriever Y가 사용할 데이터베이스 선택"""
        prompt = f"""
        다음 복잡한 질문에 답하기 위해 Retriever Y가 사용할 데이터베이스를 선택해주세요.
        (참고: Retriever X는 이미 GRAPH_DB를 사용합니다)

        질문: "{query}"

        Retriever Y가 선택할 수 있는 데이터베이스:

        1. VECTOR_DB (의미 검색 DB)
        - 연구보고서, 시장분석서, 기술동향 문서
        - 문서: 식물성단백질시장, 양식업현황, 기능성식품, AI농업기술, 대체육동향
        - 적합한 질문: "시장 전망", "기술 동향", "심층 분석", "연구 자료"

        2. RDB (관계형 정형 DB)
        - 테이블: 농산물가격, 영양성분, 시장데이터, 지역생산량, 소비자트렌드
        - 적합한 질문: "구체적 가격", "영양 성분", "생산량 통계", "정확한 수치"

        3. WEB (실시간 웹 검색)
        - 최신 뉴스기사, 시장동향, 정책변화, 실시간 정보
        - 적합한 질문: "최신 동향", "뉴스", "실시간 정보", "정책 변화"

        주의사항:
        - 필요하지 않은 데이터베이스는 선택하지 마세요
        - 데이터베이스가 전혀 필요 없다면 NONE을 답변하세요

        다음 중에서 필요한 것만 선택해주세요 (필요 없으면 NONE):
        - VECTOR_DB
        - RDB
        - WEB
        - NONE

        쉼표로 구분하여 답변해주세요 (예: VECTOR_DB, RDB 또는 NONE):
        """

        response = await self.chat.ainvoke(prompt)

        if "NONE" in response.content.upper():
            return []

        db_names = [db.strip() for db in response.content.split(",")]

        # 문자열을 DatabaseType enum으로 변환
        database_mapping = {
            "VECTOR_DB": DatabaseType.VECTOR_DB,
            "RDB": DatabaseType.RDB,
            "WEB": DatabaseType.WEB,
        }

        selected_dbs = []
        for db_name in db_names:
            if db_name in database_mapping:
                selected_dbs.append(database_mapping[db_name])

        return selected_dbs

    async def _decompose_query(self, query):
        """복잡한 쿼리를 세부 쿼리로 분해"""
        prompt = f"""
        다음 복잡한 질문을 3-4개의 독립적인 세부 질문으로 나누어주세요.

        원본 질문: "{query}"

        각 세부 질문은 한 줄씩, 번호 없이 답변해주세요:
        """

        response = await self.chat.ainvoke(prompt)
        sub_queries = [
            line.strip() for line in response.content.split("\n") if line.strip()
        ]
        return sub_queries[:]


# RetrieverAgentXWithFeedback: Graph DB 중심 검색 + 피드백
class RetrieverAgentXWithFeedback:
    """RetrieverX(피드백 가능)"""

    def __init__(self, graph_db, feedback_channel):
        self.graph_db = graph_db
        self.feedback_channel = feedback_channel
        self.chat = ChatOpenAI(model="gpt-3.5-turbo")
        self.agent_type = AgentType.RETRIEVER_X

    async def search_with_feedback(self, state):
        print(">> RETRIEVER_X (피드백 모드) 시작")

        # 기본 검색 수행
        state = await self._basic_search(state)

        # Y에게 힌트 전송
        await self._send_hints_to_y(state)

        # Y로부터 피드백 수신 및 추가 검색 : Graph DB는 정적이기 때문에 최신 정보가 부족하기 때문에 Y로부터 지속적으로 피드백을 받아야함
        await self._receive_feedback_and_search(state)

        print("\n>> RETRIEVER_X (피드백 모드) 완료")
        return state

    async def _basic_search(self, state):
        """기본 Graph DB 검색"""
        print("- 기본 Graph DB 검색 수행")

        sub_queries = state.query_plan.sub_queries
        all_results = []

        for sub_query in sub_queries:
            # LLM으로 키워드 최적화
            keywords = await self._optimize_keywords(sub_query)
            print(f"- 키워드: {keywords}")

            # Graph DB 검색
            for keyword in keywords[:2]:
                graph_result = self.graph_db.search(keyword)

                for node in graph_result["nodes"]:
                    search_result = SearchResult(
                        source="graph_db",
                        content=f"{node['properties'].get('name', 'Unknown')}: {str(node['properties'])}",
                        relevance_score=random.uniform(
                            0.7, 0.95
                        ),  # Graph DB는 유사도 점수가 없기 때문에, 일단 임의로 랜덤 값 부여
                        metadata=node,
                        search_query=keyword,
                    )
                    all_results.append(search_result)

        # 결과를 state에 추가
        for result in all_results:
            state.add_graph_result(result)

        memory = state.get_agent_memory(AgentType.RETRIEVER_X)
        memory.add_finding(f"Graph DB 기본 검색: {len(all_results)}개 결과")

        return state

    async def _send_hints_to_y(self, state):
        """Y에게 힌트 메시지 전송"""
        graph_results = state.graph_results_stream

        if not graph_results:
            return

        # 가장 관련성 높은 결과들로 힌트 생성(상위 3개)
        top_results = sorted(
            graph_results, key=lambda x: x.relevance_score, reverse=True
        )[:3]

        hints = []
        for result in top_results:
            # 메타데이터에서 키워드 추출
            if "properties" in result.metadata:
                props = result.metadata["properties"]
                hint_data = {
                    "entity": props.get("name", ""),
                    "category": props.get("category", ""),
                    "search_query": result.search_query,
                }
                hints.append(hint_data)

        if hints:
            hint_message = create_agent_message(
                from_agent=AgentType.RETRIEVER_X,
                to_agent=AgentType.RETRIEVER_Y,
                message_type=MessageType.REAL_TIME_HINT,
                content=f"Graph DB에서 {len(hints)}개 주요 요소 발견. 관련 최신 정보 검색 요청",
                data={"hints": hints, "priority": "high"},
            )

            await self.feedback_channel.send_x_to_y(hint_message)
            print(f"- Y에게 힌트 전송: {len(hints)}개 요소")

    async def _receive_feedback_and_search(self, state):
        """Y로부터 피드백 받고 추가 검색"""
        print("- Y로부터 피드백 대기 중...")

        feedback_count = 0
        timeout_seconds = 10.0  # 타임아웃 (초)

        try:
            start_time = time.time()

            async for message in self.feedback_channel.get_messages_for_x():
                # 타임아웃 체크
                if time.time() - start_time > timeout_seconds:
                    print(f"- 피드백 대기 타임아웃 ({timeout_seconds}초)")
                    break

                if message.message_type != MessageType.FEEDBACK:
                    continue

                if message.data.get("terminate"):
                    print("- 종료 신호 수신 >> 피드백 루프 종료")
                    break

                print(f"- Y로부터 피드백 수신: {message.content}")

                # 피드백 기반 추가 검색
                additional_results = await self._search_based_on_feedback(
                    message, state
                )
                for result in additional_results:
                    state.add_graph_result(result)

                feedback_count += 1
                if feedback_count >= 2:
                    print("- 최대 피드백 횟수 도달 >> 피드백 루프 종료")
                    break

        except Exception as e:
            print(f"- 피드백 수신 오류: {e}")

        if feedback_count > 0:
            memory = state.get_agent_memory(AgentType.RETRIEVER_X)
            memory.add_finding(f"피드백 기반 추가 검색: {feedback_count}회")
        else:
            print("- 피드백 수신 없음")

    async def _search_based_on_feedback(self, feedback_message, state):
        """피드백 메시지 기반 추가 검색"""
        feedback_data = feedback_message.data
        additional_results = []

        if "suggested_keywords" in feedback_data:
            suggested_keywords = feedback_data["suggested_keywords"]

            for keyword in suggested_keywords[:2]:
                print(f"- 피드백 키워드로 재검색: {keyword}")
                graph_result = self.graph_db.search(keyword)

                for node in graph_result["nodes"]:
                    search_result = SearchResult(
                        source="graph_db_feedback",
                        content=f"[피드백기반] {node['properties'].get('name', 'Unknown')}: {str(node['properties'])}",
                        relevance_score=random.uniform(0.8, 0.95),
                        metadata=node,
                        search_query=keyword,
                    )
                    additional_results.append(search_result)

        return additional_results

    async def _optimize_keywords(self, query):
        """키워드 최적화"""
        prompt = f"""
        다음 질문을 Graph Database 검색용 키워드로 변환해주세요.

        질문: "{query}"

        Graph DB에는 식품재료, 가격정보, 트렌드, 뉴스가 있습니다.
        검색 키워드 2-3개만 쉼표로 구분해서 답변해주세요:
        """

        response = await self.chat.ainvoke(prompt)
        keywords = [kw.strip() for kw in response.content.split(",")]
        return keywords[:3]


# RetrieverAgentYWithFeedback: Multi-Source 검색 + 피드백
class RetrieverAgentYWithFeedback:
    """RetrieverY(피드백 가능)"""

    def __init__(self, vector_db, rdb, web_search, feedback_channel):
        self.vector_db = vector_db
        self.rdb = rdb
        self.web_search = web_search
        self.feedback_channel = feedback_channel
        self.chat = ChatOpenAI(model="gpt-3.5-turbo")
        self.agent_type = AgentType.RETRIEVER_Y

    async def search_with_feedback(self, state):
        print(">> RETRIEVER_Y (피드백 모드) 시작")

        # X로부터 힌트 수신
        hints_received = await self._receive_hints_from_x()

        # 힌트 기반 검색 수행
        state = await self._search_with_hints(state, hints_received)

        # X에게 피드백 전송
        await self._send_feedback_to_x(state)

        print("\n>> RETRIEVER_Y 완료")
        return state

    async def _receive_hints_from_x(self):
        """X로부터 힌트 수신"""
        print("- X로부터 힌트 대기 중...")

        hints_received = []
        timeout_seconds = 10.0
        start_time = time.time()

        try:
            async for message in self.feedback_channel.get_messages_for_y():
                if time.time() - start_time > timeout_seconds:
                    print(f"- 힌트 대기 타임아웃 ({timeout_seconds}초)")
                    break

                if message.message_type != MessageType.REAL_TIME_HINT:
                    continue

                print(f"- X로부터 힌트 수신: {message.content}")

                if "hints" in message.data:
                    hints_received.extend(message.data["hints"])

                if len(hints_received) >= 3:
                    break

        except Exception as e:
            print(f"- 힌트 수신 오류: {e}")

        if not hints_received:
            print("- 힌트 수신 없음")

        return hints_received

    async def _search_with_hints(self, state, hints):
        """힌트 기반 멀티소스 검색"""
        sub_queries = state.query_plan.sub_queries
        all_results = []

        # 기본 검색 키워드 생성
        base_keywords = []
        for sub_query in sub_queries:
            keywords = await self._optimize_keywords(sub_query)
            base_keywords.extend(keywords)

        # 힌트에서 추가 키워드 추출
        hint_keywords = []
        for hint in hints:
            if isinstance(hint, dict):
                hint_keywords.append(hint.get("entity", ""))
                hint_keywords.append(hint.get("category", ""))

        # 중복 제거 및 정리
        all_keywords = list(set([kw for kw in base_keywords + hint_keywords if kw]))[:5]
        print(f"- 통합 검색 키워드: {all_keywords}")

        # 멀티소스 검색 수행
        for keyword in all_keywords:
            # Vector DB 검색
            vector_results = self.vector_db.search(keyword, top_k=2)
            for doc in vector_results:
                search_result = SearchResult(
                    source="vector_db",
                    content=doc["content"],
                    relevance_score=doc.get(
                        "similarity_score", 0.7
                    ),  # similarity score가 있으면 가져오고 아니면 0.7 부여
                    metadata=doc.get("metadata", {}),
                    search_query=keyword,
                )
                all_results.append(search_result)

            # RDB 검색
            rdb_results = self.rdb.search(keyword)
            for data_type, data_list in rdb_results["data"].items():
                for item in data_list[:1]:
                    search_result = SearchResult(
                        source=f"rdb_{data_type}",
                        content=f"{data_type}: {str(item)}",
                        relevance_score=random.uniform(0.8, 0.95),
                        metadata={"table": data_type, "data": item},
                        search_query=keyword,
                    )
                    all_results.append(search_result)

        # Web 검색 (최신 정보)
        if (
            all_keywords
        ):  # “sub_query에서 추출한 키워드” + “X로부터 받은 힌트 키워드”를 합친 최종 검색 키워드 목록
            web_results = self.web_search.search(all_keywords[0])
            for article in web_results["results"][:2]:
                search_result = SearchResult(
                    source="web",
                    content=f"{article['title']}: {article['snippet']}",
                    relevance_score=article["relevance"],
                    metadata={"url": article["url"], "date": article["published_date"]},
                    search_query=all_keywords[0],
                )
                all_results.append(search_result)

        # 결과를 state에 추가
        for result in all_results:
            state.add_multi_source_result(result)

        memory = state.get_agent_memory(AgentType.RETRIEVER_Y)
        memory.add_finding(f"힌트 기반 멀티소스 검색: {len(all_results)}개 결과")

        return state

    async def _send_feedback_to_x(self, state):
        """X에게 피드백 전송"""
        multi_results = state.multi_source_results_stream

        if not multi_results:
            return

        # 검색 결과에서 새로운 키워드 제안
        suggested_keywords = await self._extract_feedback_keywords(multi_results)

        if suggested_keywords:
            feedback_message = create_agent_message(
                from_agent=AgentType.RETRIEVER_Y,
                to_agent=AgentType.RETRIEVER_X,
                message_type=MessageType.FEEDBACK,
                content=f"멀티소스 검색 완료. {len(suggested_keywords)}개 보완 키워드 제안",
                data={"suggested_keywords": suggested_keywords},
            )

            # 실제 피드백 전송
            await self.feedback_channel.send_y_to_x(feedback_message)

            # 종료 신호 전송
            await self.feedback_channel.send_y_to_x(
                create_agent_message(
                    from_agent=AgentType.RETRIEVER_Y,
                    to_agent=AgentType.RETRIEVER_X,
                    message_type=MessageType.FEEDBACK,
                    content="피드백 종료",
                    data={"terminate": True},
                )
            )

            print(f"- X에게 피드백 전송: {len(suggested_keywords)}개 키워드")

    async def _extract_feedback_keywords(self, results):
        """검색 결과에서 피드백 키워드 추출"""
        # 간단한 키워드 추출 로직
        content_texts = [result.content for result in results[:5]]
        combined_text = " ".join(content_texts)

        prompt = f"""
        다음 검색 결과에서 추가 검색이 필요한 핵심 키워드 2-3개를 추출해주세요.

        검색 결과:
        {combined_text[:500]}...

        추가 검색이 필요한 키워드를 쉼표로 구분해서 답변해주세요:
        """

        try:
            response = await self.chat.ainvoke(prompt)
            keywords = [kw.strip() for kw in response.content.split(",")]
            return keywords[:3]
        except:
            return []

    async def _optimize_keywords(self, query):
        """키워드 최적화"""
        prompt = f"""
        다음 질문을 다양한 데이터베이스 검색용 키워드로 변환해주세요.

        질문: "{query}"

        검색 대상:
        - Vector DB: 식품관련 논문, 보고서, 연구자료
        - RDB: 가격정보, 영양성분, 시장데이터
        - Web: 최신 뉴스, 트렌드, 업계동향

        검색 키워드 2-3개만 쉼표로 구분해서 답변해주세요:
        """

        response = await self.chat.ainvoke(prompt)
        keywords = [kw.strip() for kw in response.content.split(",")]
        return keywords[:3]


# CriticAgent1: 정보량 충분성 평가
class CriticAgent1:
    def __init__(self):
        self.chat = ChatOpenAI(model="gpt-4o-mini")
        self.agent_type = AgentType.CRITIC_1

    async def evaluate(self, state):
        print(">> CRITIC_1 시작")

        # 수집된 결과들 분석
        graph_results = state.graph_results_stream
        multi_results = state.multi_source_results_stream

        print(f"- Graph DB 결과: {len(graph_results)}개")
        print(f"- Multi Source 결과: {len(multi_results)}개")

        # LLM으로 정보 충분성 평가
        evaluation_result = await self._evaluate_sufficiency(
            state.original_query, graph_results, multi_results
        )

        print(f"- 평가 결과: {evaluation_result['status']}")
        print(f"- 평가 이유: {evaluation_result['reasoning']}")

        # CriticResult 생성
        state.critic1_result = CriticResult(
            status=evaluation_result["status"],
            suggestion=evaluation_result["suggestion"],
            confidence=evaluation_result["confidence"],
            reasoning=evaluation_result["reasoning"],
        )

        # 충분성 판단
        if evaluation_result["status"] == "sufficient":
            state.info_sufficient = True
            print("- 정보 충분 - 다음 단계 진행")
        else:
            state.info_sufficient = False
            print("- 정보 부족 - 추가 검색 필요")

        # 메모리 기록
        memory = state.get_agent_memory(AgentType.CRITIC_1)
        memory.add_finding(f"정보 충분성 평가: {evaluation_result['status']}")
        memory.update_metric("confidence_score", evaluation_result["confidence"])

        print(">> CRITIC_1 완료")
        return state

    async def _evaluate_sufficiency(self, original_query, graph_results, multi_results):
        """정보 충분성 평가"""

        # 결과 요약 생성
        graph_summary = self._summarize_results(graph_results, "Graph DB")
        multi_summary = self._summarize_results(multi_results, "Multi Source")

        prompt = f"""
        다음 질문에 대한 검색 결과가 충분한지 평가해주세요.

        원본 질문: "{original_query}"

        수집된 정보:
        {graph_summary}
        {multi_summary}

        ### 평가 기준:
        1. 질문에 명시된 핵심 대상(예: 사람, 장소, 상품 등)이 결과 내에서 언급되고 있는가?
        2. 질문에 답하기 위한 핵심 정보가 실제로 포함되어 있는가?
        3. 정보의 다양성과 신뢰성은 충분한가?
        4. 최신성과 정확성은 어떤가?

        다음 형식으로 답변해주세요:
        STATUS: sufficient 또는 insufficient
        REASONING: 판단 근거 (한 줄)
        SUGGESTION: 부족한 경우 개선 제안 (한 줄)
        CONFIDENCE: 0.0-1.0 (신뢰도)
        """

        response = await self.chat.ainvoke(prompt)
        return self._parse_evaluation(response.content)

    def _summarize_results(self, results, source_name):
        """검색 결과 요약"""
        if not results:
            return f"{source_name}: 결과 없음"

        summary = f"{source_name} ({len(results)}개 결과):\n"
        for i, result in enumerate(results[:3], 1):  # 상위 3개만
            content_preview = (
                result.content[:100] + "..."
                if len(result.content) > 100
                else result.content
            )
            summary += f"  {i}. {content_preview}\n"

        if len(results) > 3:
            summary += f"  ... 외 {len(results) - 3}개\n"

        return summary

    def _parse_evaluation(self, response_content):
        """LLM 응답 파싱"""
        try:
            lines = response_content.strip().split("\n")
            result = {}

            for line in lines:
                if line.startswith("STATUS:"):
                    result["status"] = line.split(":", 1)[1].strip()
                elif line.startswith("REASONING:"):
                    result["reasoning"] = line.split(":", 1)[1].strip()
                elif line.startswith("SUGGESTION:"):
                    result["suggestion"] = line.split(":", 1)[1].strip()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        result["confidence"] = float(line.split(":", 1)[1].strip())
                    except:
                        result["confidence"] = 0.7

            # 기본값 설정
            if "status" not in result:
                result["status"] = "insufficient"
            if "reasoning" not in result:
                result["reasoning"] = "정보 부족으로 판단"
            if "suggestion" not in result:
                result["suggestion"] = "추가 검색 필요"
            if "confidence" not in result:
                result["confidence"] = 0.7

            return result

        except Exception as e:
            print(f"- 파싱 실패: {e}, 기본값 사용")
            return {
                "status": "insufficient",
                "reasoning": "평가 파싱 실패",
                "suggestion": "추가 검색 권장",
                "confidence": 0.5,
            }


# CriticAgent2: 컨텍스트 품질/신뢰도 평가
class CriticAgent2:
    def __init__(self):
        self.chat = ChatOpenAI(model="gpt-4o-mini")
        self.agent_type = AgentType.CRITIC_2

    async def evaluate(self, state):
        print(">> CRITIC_2 시작")

        # 통합된 맥락 검토
        integrated_context = state.integrated_context
        original_query = state.original_query

        if not integrated_context:
            print("- 통합된 맥락이 없음")
            state.critic2_result = CriticResult(
                status="insufficient",
                suggestion="통합된 맥락이 없어 평가 불가",
                confidence=0.0,
                reasoning="맥락 통합 단계 미완료",
            )
            state.context_sufficient = False
            return state

        print(f"- 통합 맥락 길이: {len(integrated_context)}자")

        # LLM으로 맥락 완성도 평가
        evaluation_result = await self._evaluate_context_quality(
            original_query, integrated_context
        )

        print(f"- 평가 결과: {evaluation_result['status']}")
        print(f"- 평가 이유: {evaluation_result['reasoning']}")

        # CriticResult 생성
        state.critic2_result = CriticResult(
            status=evaluation_result["status"],
            suggestion=evaluation_result["suggestion"],
            confidence=evaluation_result["confidence"],
            reasoning=evaluation_result["reasoning"],
        )

        # 맥락 충분성 판단
        if evaluation_result["status"] == "sufficient":
            state.context_sufficient = True
            print("- 맥락 완성도 충분 - 보고서 생성 가능")
        else:
            state.context_sufficient = False
            print("- 맥락 완성도 부족 - 추가 보완 필요")

        # 메모리 기록
        memory = state.get_agent_memory(AgentType.CRITIC_2)
        memory.add_finding(f"맥락 완성도 평가: {evaluation_result['status']}")
        memory.update_metric("context_quality_score", evaluation_result["confidence"])

        print("\n>> CRITIC_2 완료")
        return state

    # ... (나머지 메서드들은 동일)

    async def _evaluate_context_quality(self, original_query, integrated_context):
        """통합된 맥락의 품질 평가"""

        prompt = f"""
        다음 질문에 대한 통합된 맥락의 완성도를 평가해주세요.

        원본 질문: "{original_query}"

        통합된 맥락:
        {integrated_context}

        평가 기준:
        1. 논리적 일관성: 정보들이 서로 모순되지 않는가?
        2. 완전성: 질문에 답하기 위한 핵심 요소들이 모두 포함되었는가?
        3. 답변 가능성: 이 맥락으로 질문에 명확히 답할 수 있는가?
        4. 신뢰성: 출처가 명확하고 신뢰할 만한가?
        5. 구체성: 구체적이고 실용적인 정보를 포함하는가?

        다음 형식으로 답변해주세요:
        STATUS: sufficient 또는 insufficient
        REASONING: 판단 근거 (한 줄)
        SUGGESTION: 부족한 경우 개선 제안 (한 줄)
        CONFIDENCE: 0.0-1.0 (평가 신뢰도)
        COMPLETENESS: 0.0-1.0 (완성도 점수)
        """

        response = await self.chat.ainvoke(prompt)
        return self._parse_evaluation(response.content)

    def _parse_evaluation(self, response_content):
        """LLM 응답 파싱"""
        try:
            lines = response_content.strip().split("\n")
            result = {}

            for line in lines:
                if line.startswith("STATUS:"):
                    result["status"] = line.split(":", 1)[1].strip()
                elif line.startswith("REASONING:"):
                    result["reasoning"] = line.split(":", 1)[1].strip()
                elif line.startswith("SUGGESTION:"):
                    result["suggestion"] = line.split(":", 1)[1].strip()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        result["confidence"] = float(line.split(":", 1)[1].strip())
                    except:
                        result["confidence"] = 0.7
                elif line.startswith("COMPLETENESS:"):
                    try:
                        result["completeness"] = float(line.split(":", 1)[1].strip())
                    except:
                        result["completeness"] = 0.7

            # 기본값 설정
            if "status" not in result:
                result["status"] = "insufficient"
            if "reasoning" not in result:
                result["reasoning"] = "맥락 품질 평가 불가"
            if "suggestion" not in result:
                result["suggestion"] = "맥락 보완 필요"
            if "confidence" not in result:
                result["confidence"] = 0.7
            if "completeness" not in result:
                result["completeness"] = 0.7

            return result

        except Exception as e:
            print(f"- 파싱 실패: {e}, 기본값 사용")
            return {
                "status": "insufficient",
                "reasoning": "평가 파싱 실패",
                "suggestion": "맥락 재구성 권장",
                "confidence": 0.5,
                "completeness": 0.5,
            }


# ContextIntegratorAgent: 검색 결과 통합
class ContextIntegratorAgent:
    def __init__(self):
        self.chat = ChatOpenAI(model="gpt-3.5-turbo")
        self.agent_type = AgentType.CONTEXT_INTEGRATOR

    async def integrate(self, state):
        print(">> CONTEXT_INTEGRATOR 시작")

        # 모든 검색 결과 수집
        graph_results = state.graph_results_stream
        multi_results = state.multi_source_results_stream
        all_results = graph_results + multi_results

        print(f"- 총 {len(all_results)}개 결과 통합 시작")
        print(f"- Graph DB: {len(graph_results)}개")
        print(f"- Multi Source: {len(multi_results)}개")

        if not all_results:
            print("- 통합할 결과가 없음")
            state.integrated_context = "검색 결과가 없어 통합할 정보가 부족합니다."
            return state

        # 결과 정리 및 우선순위 정렬
        organized_results = await self._organize_results(
            all_results, state.original_query
        )

        print(f"- {len(organized_results)}개 결과로 정리 완료")

        # 통합된 맥락 생성
        integrated_context = await self._create_integrated_context(
            state.original_query, organized_results
        )

        state.integrated_context = integrated_context

        # 메모리 기록
        memory = state.get_agent_memory(AgentType.CONTEXT_INTEGRATOR)
        memory.add_finding(f"정보 통합 완료 - {len(organized_results)}개 핵심 정보")
        memory.update_metric(
            "integration_ratio", len(organized_results) / len(all_results)
        )

        print("\n>> CONTEXT_INTEGRATOR 완료")
        return state

    async def _organize_results(self, all_results, original_query):
        """검색 결과 정리 및 우선순위 정렬"""

        # 관련성 점수 기준으로 정렬
        sorted_results = sorted(
            all_results, key=lambda x: x.relevance_score, reverse=True
        )

        # 중복 제거 (내용 유사도 기반)
        unique_results = self._remove_duplicates(sorted_results)

        # LLM으로 최종 우선순위 결정
        prioritized_results = await self._prioritize_with_llm(
            unique_results, original_query
        )

        return prioritized_results[:10]  # 상위 10개만

    def _remove_duplicates(self, results):
        """간단한 중복 제거"""
        unique_results = []
        seen_contents = set()

        for result in results:
            # 내용의 첫 50자로 중복 판단 (간단한 방법)
            content_key = result.content[:50].lower().strip()
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                unique_results.append(result)

        return unique_results

    async def _prioritize_with_llm(self, results, original_query):
        """LLM으로 우선순위 결정"""

        if len(results) <= 5:
            return results  # 5개 이하면 그대로

        # 결과 요약본 생성
        results_summary = ""
        for i, result in enumerate(results[:15], 1):  # 상위 15개만
            content_preview = (
                result.content[:150] + "..."
                if len(result.content) > 150
                else result.content
            )
            results_summary += f"{i}. [{result.source}] {content_preview}\n"

        prompt = f"""
        다음 질문에 대한 검색 결과들을 중요도 순으로 우선순위를 매겨주세요.

        질문: "{original_query}"

        검색 결과들:
        {results_summary}

        가장 중요한 결과 8개의 번호만 쉼표로 구분해서 답변해주세요 (예: 1,3,5,7,9,11,13,15):
        """

        try:
            response = await self.chat.ainvoke(prompt)
            selected_numbers = [
                int(x.strip())
                for x in response.content.split(",")
                if x.strip().isdigit()
            ]

            # 선택된 번호에 해당하는 결과만 반환
            prioritized = []
            for num in selected_numbers:
                if 1 <= num <= len(results):
                    prioritized.append(results[num - 1])

            return prioritized if prioritized else results[:8]

        except:
            print("- 우선순위 결정 실패, 점수순 사용")
            return results[:8]

    async def _create_integrated_context(self, original_query, organized_results):
        """통합된 맥락 생성"""

        # 소스별로 그룹핑
        sources_summary = self._group_by_source(organized_results)

        prompt = f"""
        다음 질문에 대한 다양한 검색 결과를 논리적이고 일관성 있게 통합해주세요.

        질문: "{original_query}"

        검색 결과 (소스별):
        {sources_summary}

        통합 지침:
        1. 정보 간 연관관계를 파악하여 논리적 순서로 구성
        2. 모순되는 정보가 있다면 신뢰도 높은 소스 우선
        3. 질문에 직접적으로 답할 수 있도록 핵심 정보 중심으로 정리
        4. 각 정보의 출처를 명시

        통합된 맥락을 자연스러운 문장으로 작성해주세요:
        """

        response = await self.chat.ainvoke(prompt)
        return response.content

    def _group_by_source(self, results):
        """소스별로 결과 그룹핑"""
        source_groups = {}

        for result in results:
            source = result.source
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(result)

        summary = ""
        for source, source_results in source_groups.items():
            summary += f"\n[{source.upper()}]:\n"
            for i, result in enumerate(source_results, 1):
                content_preview = (
                    result.content[:200] + "..."
                    if len(result.content) > 200
                    else result.content
                )
                summary += f"  {i}. {content_preview}\n"

        return summary


# ReportGeneratorAgent: 최종 보고서 생성
class ReportGeneratorAgent:
    def __init__(self):
        self.chat = ChatOpenAI(model="gpt-4o-mini")
        self.agent_type = AgentType.REPORT_GENERATOR

    async def generate(self, state):
        print("\n>> REPORT_GENERATOR 시작")

        # 통합된 맥락 확인
        integrated_context = state.integrated_context
        original_query = state.original_query

        if not integrated_context:
            print("- 통합된 맥락이 없음")
            state.final_answer = (
                "충분한 정보를 수집하지 못해 보고서를 생성할 수 없습니다."
            )
            return state

        print(f"- 통합 맥락 기반 전문 보고서 생성 시작")
        print(f"- 원본 질문: {original_query}")

        # Critic2 메모리에서 컨텍스트 품질 정보 추출
        critic2_insights = self._extract_critic2_insights(state)
        print(f"- Critic2 인사이트: {critic2_insights}")

        # 보고서 유형 결정
        report_type = await self._determine_report_type(original_query)
        print(f"- 보고서 유형: {report_type}")

        # 전문적인 보고서 생성
        final_report = await self._create_professional_report(
            original_query, integrated_context, report_type, critic2_insights, state
        )

        state.final_answer = final_report

        # 보고서 품질 검증
        quality_score = await self._validate_report_quality(
            original_query, final_report
        )
        print(f"- 보고서 품질 점수: {quality_score:.2f}")

        # 메모리 기록
        memory = state.get_agent_memory(AgentType.REPORT_GENERATOR)
        memory.add_finding(f"전문 보고서 생성 완료 - {len(final_report)}자")
        memory.update_metric("report_quality_score", quality_score)
        memory.update_metric("report_length", len(final_report))
        memory.update_metric("critic2_consideration", len(critic2_insights))

        print("\n>> REPORT_GENERATOR 완료")
        return state

    def _extract_critic2_insights(self, state):
        """Critic2 메모리에서 컨텍스트 품질 관련 인사이트 추출"""
        print("- Critic2 메모리 분석 중")

        critic2_memory = state.get_agent_memory(AgentType.CRITIC_2)
        insights = {
            "quality_issues": [],
            "confidence_score": 0.0,
            "completeness_score": 0.0,
            "status": "unknown",
            "suggestions": "",
        }

        # 메모리에서 평가 결과 추출
        if state.critic2_result:
            insights["status"] = state.critic2_result.status
            insights["confidence_score"] = state.critic2_result.confidence
            insights["suggestions"] = state.critic2_result.suggestion

            # 부족한 부분 분석
            if state.critic2_result.status == "insufficient":
                insights["quality_issues"].append("컨텍스트 완성도 부족")

        # 성능 메트릭에서 품질 점수 추출
        if "context_quality_score" in critic2_memory.performance_metrics:
            insights["completeness_score"] = critic2_memory.performance_metrics[
                "context_quality_score"
            ]

        # 발견사항에서 품질 관련 이슈 추출
        for finding in critic2_memory.findings:
            if "부족" in finding or "insufficient" in finding.lower():
                insights["quality_issues"].append(f"품질 이슈: {finding}")

        return insights

    async def _determine_report_type(self, query):
        """보고서 유형 결정 (B2B 전문성 기반)"""

        prompt = f"""
        다음 질문에 가장 적합한 B2B 전문 보고서 유형을 결정해주세요.

        질문: "{query}"

        보고서 유형:
        1. EXECUTIVE_SUMMARY - 경영진 요약 보고서 (핵심 인사이트, 의사결정 지원)
        2. MARKET_INTELLIGENCE - 시장 인텔리전스 (시장 동향, 경쟁 분석, 기회 요소)
        3. FINANCIAL_ANALYSIS - 재무 분석 보고서 (수익성, ROI, 비용 분석)
        4. STRATEGIC_CONSULTING - 전략 컨설팅 (전략 방향, 실행 계획, 리스크 분석)
        5. OPERATIONAL_INSIGHTS - 운영 인사이트 (프로세스 개선, 효율성, 성과 지표)
        6. INDUSTRY_RESEARCH - 산업 연구 보고서 (산업 분석, 트렌드, 예측)

        해당하는 보고서 유형 하나만 답변해주세요 (예: MARKET_INTELLIGENCE):
        """

        try:
            response = await self.chat.ainvoke(prompt)
            report_type = response.content.strip().upper()

            valid_types = [
                "EXECUTIVE_SUMMARY",
                "MARKET_INTELLIGENCE",
                "FINANCIAL_ANALYSIS",
                "STRATEGIC_CONSULTING",
                "OPERATIONAL_INSIGHTS",
                "INDUSTRY_RESEARCH",
            ]
            if report_type in valid_types:
                return report_type
            else:
                return "EXECUTIVE_SUMMARY"  # 기본값

        except:
            print("- 보고서 유형 결정 실패, 기본값 사용")
            return "EXECUTIVE_SUMMARY"

    def _build_quality_context(self, critic2_insights):
        """Critic2 인사이트를 바탕으로 품질 개선 컨텍스트 구성"""

        if not critic2_insights or critic2_insights["status"] == "sufficient":
            return "데이터 품질: 양호 - 신뢰할 수 있는 분석 기반이 확보되었습니다."

        quality_issues = []

        if critic2_insights["status"] == "insufficient":
            quality_issues.append("데이터 완성도에 제약이 있습니다.")

        if critic2_insights["confidence_score"] < 0.7:
            quality_issues.append(
                f"신뢰도 수준: {critic2_insights['confidence_score']:.2f}/1.0"
            )

        if critic2_insights["completeness_score"] < 0.7:
            quality_issues.append(
                f"완성도 점수: {critic2_insights['completeness_score']:.2f}/1.0"
            )

        if critic2_insights["suggestions"]:
            quality_issues.append(f"개선 권고: {critic2_insights['suggestions']}")

        if quality_issues:
            quality_context = f"""
**데이터 품질 평가**
{chr(10).join(f"• {issue}" for issue in quality_issues)}

**보고서 작성 시 고려사항**
• 제한적 데이터는 명시적으로 표기
• 불확실한 정보는 "추가 검증 필요" 표시
• 출처가 확인된 데이터는 하이퍼링크로 연결
• 데이터 부족 영역은 향후 연구 과제로 제안
"""
        else:
            quality_context = (
                "**데이터 품질**: 분석에 충분한 고품질 데이터가 확보되었습니다."
            )

        return quality_context

    def _extract_sources_from_context(self, context):
        """통합 컨텍스트에서 출처 정보와 URL 추출 (Mock DB 구조 대응)"""
        print("- 출처 정보 추출 중")

        import re
        import json

        sources = []

        # URL 패턴 매칭 (http, https)
        url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+[^\s<>"{}|\\^`[\].,;:!?]'
        urls = re.findall(url_pattern, context)

        # Mock DB 특화 출처 정보 추출 패턴들
        db_source_patterns = [
            # Graph DB 뉴스 노드에서 URL 추출
            r'"url":\s*"([^"]+)"[^}]*"title":\s*"([^"]+)"',
            # Vector DB 문서에서 source와 URL 추출
            r'"source":\s*"([^"]+)"[^}]*"url":\s*"([^"]+)"',
            r'"metadata":\s*{[^}]*"source":\s*"([^"]+)"',
            # RDB 웹 검색 결과에서 추출
            r'"source_type":\s*"([^"]+)"',
            # 일반 출처 패턴
            r"출처[:\s]*([^\n]+)",
            r"source[:\s]*([^\n]+)",
            r"참조[:\s]*([^\n]+)",
            r"reference[:\s]*([^\n]+)",
        ]

        # JSON 형태의 데이터에서 구조화된 정보 추출
        try:
            # 컨텍스트에서 JSON 형태 데이터 찾기
            json_matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", context)
            for json_str in json_matches:
                try:
                    data = json.loads(json_str)
                    # Graph DB 뉴스 노드 처리
                    if "properties" in data and "url" in data.get("properties", {}):
                        props = data["properties"]
                        sources.append(
                            {
                                "name": props.get(
                                    "title", props.get("source", "뉴스기사")
                                ),
                                "url": props["url"],
                                "type": "news_article",
                                "domain": self._extract_domain(props["url"]),
                            }
                        )

                    # Vector DB 문서 처리
                    elif "metadata" in data and "url" in data.get("metadata", {}):
                        meta = data["metadata"]
                        sources.append(
                            {
                                "name": data.get(
                                    "title", meta.get("source", "연구문서")
                                ),
                                "url": meta["url"],
                                "type": "research_document",
                                "domain": self._extract_domain(meta["url"]),
                            }
                        )

                    # Web Search 결과 처리
                    elif "url" in data and "title" in data:
                        sources.append(
                            {
                                "name": data["title"][:50]
                                + ("..." if len(data["title"]) > 50 else ""),
                                "url": data["url"],
                                "type": data.get("source_type", "web_article"),
                                "domain": self._extract_domain(data["url"]),
                            }
                        )

                except (json.JSONDecodeError, KeyError):
                    continue
        except Exception as e:
            print(f"- JSON 파싱 중 오류: {e}")

        # URL만 있는 경우 도메인 기반으로 출처명 생성
        for url in urls:
            if not any(source["url"] == url for source in sources):
                domain = self._extract_domain(url)
                source_name = self._get_korean_source_name(domain)
                sources.append(
                    {
                        "name": source_name,
                        "url": url,
                        "type": "web_source",
                        "domain": domain,
                    }
                )

        # 중복 제거 (URL 기준)
        seen_urls = set()
        unique_sources = []
        for source in sources:
            if source["url"] not in seen_urls:
                unique_sources.append(source)
                seen_urls.add(source["url"])

        print(f"- 추출된 출처 개수: {len(unique_sources)}")
        return unique_sources[:10]  # 최대 10개

    def _extract_domain(self, url):
        """URL에서 도메인 추출"""
        import re

        domain_match = re.search(r"(?:https?://)?(?:www\.)?([^./]+\.[^./]+)", url)
        return domain_match.group(1) if domain_match else url

    def _get_korean_source_name(self, domain):
        """도메인별 한국어 출처명 매핑 (Mock DB 특화)"""
        domain_names = {
            # 농식품 전문 사이트들
            "kamis.or.kr": "KAMIS 농산물유통정보",
            "atfis.or.kr": "atFIS 식품산업통계정보",
            "krei.re.kr": "한국농촌경제연구원",
            "kati.net": "KATI 농식품수출정보",
            "foodnews.co.kr": "식품저널",
            "foodbiz.co.kr": "식품비즈니스",
            # 친환경 트렌드 사이트들 (Mock DB 기준)
            "ecofoodnews.co.kr": "친환경식품뉴스",
            "sustainableagri.co.kr": "지속가능농업신문",
            "biofoodtimes.co.kr": "바이오식품타임즈",
            "fairtrademagazine.co.kr": "공정무역매거진",
            "fermentedfoods.co.kr": "발효식품전문지",
            "futurefood.co.kr": "미래식품리포트",
            "tropicalfruits.co.kr": "열대과일뉴스",
            "packaging-innovation.co.kr": "패키징혁신지",
            # 연구기관들
            "fao.org": "FAO 통계",
            "usda.gov": "USDA",
            "tridge.com": "Tridge",
            "indexmundi.com": "IndexMundi",
            "data.kma.go.kr": "기상청",
            "ncei.noaa.gov": "NOAA 기후데이터",
            # 일반 사이트들
            "naver.com": "네이버",
            "google.com": "구글",
            "wikipedia.org": "위키피디아",
        }

        domain_lower = domain.lower()
        for known_domain, korean_name in domain_names.items():
            if known_domain in domain_lower:
                return korean_name

        # 도메인명에서 의미있는 부분 추출
        domain_parts = domain_lower.split(".")
        if len(domain_parts) >= 2:
            return domain_parts[0].capitalize()

        return domain

    def _build_sources_section(self, sources):
        """출처 섹션 구성 (강화된 버전)"""
        if not sources:
            return ""

        sources_md = "\n---\n\n## 📚 참고 자료 및 출처\n\n"
        sources_md += "*본 보고서는 다음과 같은 신뢰할 수 있는 출처를 바탕으로 작성되었습니다.*\n\n"

        # 출처 유형별 분류
        news_sources = [
            s for s in sources if s.get("type") in ["news_article", "web_article"]
        ]
        research_sources = [s for s in sources if s.get("type") == "research_document"]
        data_sources = [s for s in sources if s.get("type") == "web_source"]

        if news_sources:
            sources_md += "### 뉴스 및 기사\n\n"
            sources_md += "| 번호 | 출처명 | 링크 |\n"
            sources_md += "|:----:|--------|------|\n"
            for i, source in enumerate(news_sources, 1):
                link_text = (
                    source["name"]
                    if len(source["name"]) < 40
                    else f"{source['name'][:37]}..."
                )
                sources_md += f"| {i} | {link_text} | [바로가기]({source['url']}) |\n"
            sources_md += "\n"

        if research_sources:
            sources_md += "### 연구 자료\n\n"
            sources_md += "| 번호 | 연구명 | 링크 |\n"
            sources_md += "|:----:|--------|------|\n"
            for i, source in enumerate(research_sources, 1):
                link_text = (
                    source["name"]
                    if len(source["name"]) < 40
                    else f"{source['name'][:37]}..."
                )
                sources_md += f"| {i} | {link_text} | [바로가기]({source['url']}) |\n"
            sources_md += "\n"

        if data_sources:
            sources_md += "### 데이터 출처\n\n"
            sources_md += "| 번호 | 기관명 | 링크 |\n"
            sources_md += "|:----:|--------|------|\n"
            for i, source in enumerate(data_sources, 1):
                sources_md += (
                    f"| {i} | {source['name']} | [바로가기]({source['url']}) |\n"
                )
            sources_md += "\n"

        if not (news_sources or research_sources or data_sources):
            # 단일 테이블로 표시
            sources_md += "| 번호 | 출처명 | 기관/사이트 | 링크 |\n"
            sources_md += "|:----:|--------|-------------|------|\n"
            for i, source in enumerate(sources, 1):
                link_text = (
                    source["name"]
                    if len(source["name"]) < 30
                    else f"{source['name'][:27]}..."
                )
                sources_md += f"| {i} | {link_text} | {source['domain']} | [바로가기]({source['url']}) |\n"
            sources_md += "\n"

        sources_md += "*📌 클릭하면 해당 출처 페이지로 이동합니다.*\n"
        return sources_md

    def _integrate_sources_in_content(self, content, sources):
        """본문에 출처 링크 통합 (Mock DB 구조 대응)"""
        if not sources:
            return content

        import re

        # 농식품 관련 데이터와 출처 매칭을 위한 키워드 맵핑
        agricultural_keywords = {
            "KAMIS": ["농산물", "시세", "가격", "유통"],
            "atFIS": ["식품산업", "통계", "시장규모"],
            "한국농촌경제연구원": ["농촌", "경제", "전망", "관측"],
            "KATI": ["수출", "수입", "무역"],
            "식품저널": ["식품", "업계", "동향"],
            "친환경식품뉴스": ["친환경", "유기농", "지속가능"],
            "바이오식품타임즈": ["바이오", "미래식품", "스피룰리나", "클로렐라"],
            "FAO": ["글로벌", "세계", "국제"],
            "USDA": ["미국", "해외"],
            "Tridge": ["국제가격", "무역"],
        }

        # 수치나 데이터가 포함된 문장 패턴 찾기
        data_patterns = [
            r"(\d+[%억만원달러]+)",  # 숫자+단위
            r"(\d+[\.\,]\d+[%억만원달러]+)",  # 소수점 포함
            r"(증가|감소|상승|하락|성장)\s*(\d+[%]+)",  # 변화율
            r"(\d+년\s*\d+[%억만원달러]+)",  # 연도별 데이터
            r"(시장\s*규모|매출|수익|성장률|점유율)",  # 주요 지표 키워드
            r"(아마란스|테프|햄프시드|모링가|스피룰리나|퀴노아|치아시드)",  # 트렌드 식재료
        ]

        # 각 출처에 대해 관련 키워드가 있는 데이터에 링크 추가
        used_sources = []

        for source in sources[:8]:  # 최대 8개 출처만 사용
            source_name = source["name"]
            source_url = source["url"]

            # 출처별 관련 키워드 찾기
            related_keywords = []
            for src_key, keywords in agricultural_keywords.items():
                if src_key in source_name:
                    related_keywords = keywords
                    break

            # 관련 키워드가 있는 문장에 출처 링크 추가
            if related_keywords:
                for keyword in related_keywords:
                    # 해당 키워드가 포함된 문장에서 첫 번째 수치 데이터에 출처 추가
                    pattern = rf"({keyword}[^.]*?(\d+[%억만원달러][^.]*?))"
                    matches = re.finditer(pattern, content)

                    for match in matches:
                        if source_url not in content:  # 중복 방지
                            original_text = match.group(1)
                            enhanced_text = (
                                f"{original_text} ([{source_name}]({source_url}))"
                            )
                            content = content.replace(original_text, enhanced_text, 1)
                            used_sources.append(source_name)
                            break
                    if source_name in used_sources:
                        break

            # 트렌드 식재료명에 직접 출처 연결
            trend_ingredients = [
                "아마란스",
                "테프",
                "햄프시드",
                "모링가",
                "스피룰리나",
                "퀴노아",
                "치아시드",
            ]
            for ingredient in trend_ingredients:
                if ingredient in content and ingredient in source_name.lower():
                    # 해당 식재료 첫 번째 언급에 출처 추가
                    pattern = rf"({ingredient}(?:[^.]*?(?:증가|성장|인기|주목|트렌드)[^.]*?)?)"
                    match = re.search(pattern, content)
                    if match and source_url not in content:
                        original_text = match.group(1)
                        enhanced_text = (
                            f"{original_text} ([{source_name}]({source_url}))"
                        )
                        content = content.replace(original_text, enhanced_text, 1)
                        used_sources.append(source_name)
                        break

        # 사용되지 않은 중요한 출처들을 주요 섹션에 추가
        unused_sources = [s for s in sources if s["name"] not in used_sources]
        if unused_sources:
            # Executive Summary나 첫 번째 주요 섹션에 출처 추가
            sections = [
                "## Executive Summary",
                "## 시장 현황",
                "## 주요 발견사항",
                "## 현황 분석",
            ]
            for section in sections:
                if section in content and unused_sources:
                    source = unused_sources[0]
                    # 섹션 제목 바로 다음 문단에 출처 추가
                    section_pattern = rf"({section}[^\n]*\n\n[^.]+\.)"
                    match = re.search(section_pattern, content)
                    if match:
                        original_text = match.group(1)
                        enhanced_text = (
                            f"{original_text} ([{source['name']}]({source['url']}))"
                        )
                        content = content.replace(original_text, enhanced_text, 1)
                        unused_sources.pop(0)
                        break

        return content

    async def _create_professional_report(
        self, query, context, report_type, critic2_insights, state
    ):
        """전문적이고 시각적으로 우수한 B2B 보고서 생성"""

        # 출처 정보 추출
        sources = self._extract_sources_from_context(context)

        # 보고서 유형별 전문 템플릿
        templates = {
            "EXECUTIVE_SUMMARY": """# Executive Summary: {query}

## Key Findings & Recommendations

| Category | Finding | Business Impact | Priority |
|----------|---------|-----------------|----------|
| Strategic | [핵심 전략 발견사항] | HIGH | P1 |
| Operational | [운영 관련 발견사항] | MEDIUM | P2 |
| Financial | [재무 관련 발견사항] | HIGH | P1 |

## Financial Overview

```
Revenue Impact Analysis
████████████████████ 85% Positive
████████████ 60% Cost Efficiency
██████████████████ 80% ROI Potential
```

### Performance Metrics

| KPI | Current | Target | Gap | Action Required |
|-----|---------|--------|-----|-----------------|
| Revenue Growth | [현재값] | [목표값] | [차이] | [필요 조치] |
| Market Share | [현재값] | [목표값] | [차이] | [필요 조치] |
| Cost Efficiency | [현재값] | [목표값] | [차이] | [필요 조치] |

## Strategic Recommendations

### Immediate Actions (0-30 days)
1. **[액션 1]** - [구체적 실행 방안]
2. **[액션 2]** - [구체적 실행 방안]

### Short-term Initiatives (1-6 months)
1. **[이니셔티브 1]** - [실행 계획 및 예상 효과]
2. **[이니셔티브 2]** - [실행 계획 및 예상 효과]

### Long-term Strategy (6+ months)
1. **[전략 1]** - [장기 비전 및 목표]
2. **[전략 2]** - [장기 비전 및 목표]

## Risk Assessment

| Risk Factor | Probability | Impact | Mitigation Strategy |
|-------------|-------------|--------|-------------------|
| [리스크 1] | Medium | High | [완화 전략] |
| [리스크 2] | Low | Medium | [완화 전략] |

## Resource Requirements

### Budget Allocation
- **Phase 1**: [예산] - [용도]
- **Phase 2**: [예산] - [용도]
- **Total Investment**: [총 투자 금액]

### Human Resources
- **Dedicated Team**: [팀 구성]
- **External Support**: [외부 지원 필요사항]

---
*Report Generated: {current_date} | Next Review: [리뷰 예정일]*""",
            "MARKET_INTELLIGENCE": """# Market Intelligence Report: {query}

## Market Overview

### Market Size & Growth
| Metric | 2023 | 2024E | 2025F | CAGR |
|--------|------|-------|-------|------|
| Total Addressable Market | [값] | [값] | [값] | [%] |
| Serviceable Market | [값] | [값] | [값] | [%] |
| Market Penetration | [값] | [값] | [값] | [%] |

```
Market Growth Trajectory
2023 ████████████ $[금액]B
2024 ████████████████ $[금액]B (+[%])
2025 ████████████████████ $[금액]B (+[%])
```

## Competitive Landscape

### Market Share Analysis
| Company | Market Share | Revenue | Growth Rate | Competitive Strength |
|---------|-------------|---------|-------------|-------------------|
| [회사 A] | [%] | $[금액] | [%] | Strong |
| [회사 B] | [%] | $[금액] | [%] | Moderate |
| [회사 C] | [%] | $[금액] | [%] | Emerging |

### Competitive Positioning
```
Market Position Matrix

High Growth  │    Stars        │    Question Marks
            │  • [Company A]   │  • [Company B]
            │                  │
────────────┼──────────────────┼──────────────────
            │                  │
Low Growth   │  Cash Cows      │    Dogs
            │  • [Company C]   │  • [Company D]

            Low Share          High Share
```

## Market Trends & Drivers

### Key Growth Drivers
1. **[드라이버 1]** - [영향도: High/Medium/Low]
   - [상세 설명 및 데이터]

2. **[드라이버 2]** - [영향도: High/Medium/Low]
   - [상세 설명 및 데이터]

### Technology Trends
| Technology | Adoption Rate | Impact on Market | Timeline |
|------------|---------------|------------------|----------|
| [기술 1] | [%] | Transformational | 2025-2027 |
| [기술 2] | [%] | Incremental | 2024-2025 |

## Customer Analysis

### Buyer Behavior Insights
- **Primary Decision Makers**: [의사결정자 분석]
- **Purchase Criteria**: [구매 기준 순위]
- **Budget Allocation**: [예산 배정 패턴]

### Customer Segmentation
| Segment | Size | Growth | Profitability | Strategic Priority |
|---------|------|--------|---------------|-------------------|
| Enterprise | [%] | [%] | High | P1 |
| Mid-Market | [%] | [%] | Medium | P2 |
| SMB | [%] | [%] | Low | P3 |

## Opportunities & Recommendations

### Market Opportunities
1. **[기회 1]** - Market Size: $[금액], Timeline: [기간]
2. **[기회 2]** - Market Size: $[금액], Timeline: [기간]

### Strategic Recommendations
- **Market Entry Strategy**: [전략]
- **Product Development**: [권고사항]
- **Partnership Opportunities**: [파트너십 기회]

---
*Data Sources: [출처 리스트] | Analysis Date: {current_date}*""",
            "FINANCIAL_ANALYSIS": """# Financial Analysis Report: {query}

## Executive Financial Summary

### Key Financial Metrics
| Metric | Current Period | Previous Period | Change | Benchmark |
|--------|---------------|----------------|---------|-----------|
| Revenue | $[금액] | $[금액] | [%] | $[업계 평균] |
| EBITDA | $[금액] | $[금액] | [%] | [%] |
| Net Margin | [%] | [%] | [변화] | [업계 평균%] |
| ROI | [%] | [%] | [변화] | [목표%] |

```
Financial Performance Trend
Revenue  ████████████████████ +15%
EBITDA   ████████████████ +12%
Margin   ████████████ +8%
```

## Revenue Analysis

### Revenue Breakdown
| Revenue Stream | Amount | % of Total | YoY Growth | Margin |
|---------------|--------|------------|------------|--------|
| [스트림 1] | $[금액] | [%] | [%] | [%] |
| [스트림 2] | $[금액] | [%] | [%] | [%] |
| [스트림 3] | $[금액] | [%] | [%] | [%] |

### Geographic Performance
```
Regional Revenue Distribution
North America  ████████████████ 45%
Europe        ████████████ 30%
Asia Pacific  ████████ 20%
Others        ██ 5%
```

## Cost Structure Analysis

### Operating Expenses
| Category | Amount | % of Revenue | YoY Change | Industry Average |
|----------|--------|--------------|------------|------------------|
| Personnel | $[금액] | [%] | [%] | [%] |
| Technology | $[금액] | [%] | [%] | [%] |
| Marketing | $[금액] | [%] | [%] | [%] |
| Operations | $[금액] | [%] | [%] | [%] |

### Cost Optimization Opportunities
1. **[영역 1]** - Potential Savings: $[금액] ([%])
2. **[영역 2]** - Potential Savings: $[금액] ([%])

## Profitability Analysis

### Margin Analysis by Segment
| Business Unit | Revenue | Direct Costs | Margin | Margin % |
|---------------|---------|-------------|--------|----------|
| [유닛 1] | $[금액] | $[금액] | $[금액] | [%] |
| [유닛 2] | $[금액] | $[금액] | $[금액] | [%] |

### Profitability Trends
```
Quarterly Margin Progression
Q1 ████████ 18%
Q2 ██████████ 22%
Q3 ████████████ 25%
Q4 ██████████████ 28%
```

## Investment Analysis

### Capital Allocation
- **R&D Investment**: $[금액] ([%] of revenue)
- **Market Expansion**: $[금액]
- **Technology Infrastructure**: $[금액]

### ROI Analysis
| Investment Area | Amount Invested | Expected Return | Payback Period | Risk Level |
|----------------|----------------|----------------|----------------|------------|
| [영역 1] | $[금액] | [%] | [기간] | Low |
| [영역 2] | $[금액] | [%] | [기간] | Medium |

## Financial Projections

### 3-Year Financial Forecast
| Year | Revenue | EBITDA | Net Income | Cash Flow |
|------|---------|--------|------------|-----------|
| 2024E | $[금액] | $[금액] | $[금액] | $[금액] |
| 2025F | $[금액] | $[금액] | $[금액] | $[금액] |
| 2026F | $[금액] | $[금액] | $[금액] | $[금액] |

## Recommendations

### Financial Strategy
1. **Revenue Growth**: [구체적 전략]
2. **Cost Management**: [비용 최적화 방안]
3. **Investment Priorities**: [투자 우선순위]

### Risk Mitigation
- **Financial Risks**: [식별된 리스크]
- **Mitigation Strategies**: [완화 전략]

---
*Financial data as of: {current_date} | Assumptions & Methodology: [상세 설명]*""",
        }

        template = templates.get(report_type, templates["EXECUTIVE_SUMMARY"])

        # Critic2 인사이트를 반영한 프롬프트 구성
        quality_context = self._build_quality_context(critic2_insights)

        from datetime import datetime

        current_date = datetime.now().strftime("%Y년 %m월 %d일")

        # 출처 섹션 구성
        sources_section = self._build_sources_section(sources)

        prompt = f"""
        다음 질문에 대해 최고 수준의 B2B 전문 보고서를 작성해주세요.

        **분석 요청**: "{query}"

        **수집된 정보**:
        {context}

        **{quality_context}**

        **활용 가능한 출처 정보**:
        {chr(10).join([f"• {s['name']}: {s['url']}" for s in sources]) if sources else "출처 정보 없음"}

        **보고서 템플릿**:
        {template.format(query=query, current_date=current_date)}

        **작성 지침**:

        **전문성 요구사항**
        • Fortune 500 기업 수준의 보고서 품질
        • 정확한 데이터와 KPI를 활용한 객관적 분석
        • 논리적이고 체계적인 정보 구성
        • 실행 가능한 비즈니스 인사이트 제공

        **시각화 및 레이아웃**
        • 마크다운 표를 적극 활용하여 데이터 정리
        • ASCII 차트로 시각적 임팩트 강화
        • 계층적 헤딩 구조로 가독성 향상
        • 색상 코딩: 빨강(위험), 노랑(주의), 초록(양호)

        **출처 및 신뢰성 (중요!)**
        • 데이터나 통계를 언급할 때 반드시 출처 링크 포함
        • 형식: "시장 규모는 150억 달러로 추정됩니다 ([McKinsey Report](https://example.com))"
        • 각 핵심 데이터 포인트마다 클릭 가능한 하이퍼링크 제공
        • 추정치와 확정치 명확히 구분
        • 가정과 제한사항 투명하게 공개

        **비즈니스 가치**
        • Executive Summary에 핵심 인사이트 집약
        • 단계별 실행 계획 제시
        • ROI와 비즈니스 임팩트 정량화
        • 리스크 요소와 완화 방안 제공

        **중요**:
        - 이모지는 절대 사용하지 마세요
        - 모든 데이터와 수치에 출처 링크를 반드시 포함하세요
        - 링크는 [텍스트](URL) 마크다운 형식으로 작성하세요

        전문적이고 완성도 높은 보고서를 작성해주세요:
        """

        try:
            response = await self.chat.ainvoke(prompt)
            report_content = response.content

            # 생성된 보고서에 출처 섹션 추가
            if sources_section:
                report_content += sources_section

            # 본문에 출처 링크 통합 (추가 처리)
            report_content = self._integrate_sources_in_content(report_content, sources)

            return report_content

        except Exception as e:
            print(f"- 보고서 생성 중 오류: {e}")
            return f"보고서 생성 중 오류가 발생했습니다: {str(e)}"

    async def _validate_report_quality(self, query, report):
        """보고서 품질 검증 (B2B 전문성 중심)"""

        prompt = f"""
        다음 B2B 보고서의 품질을 0.0-1.0 점수로 평가해주세요.

        질문: "{query}"

        보고서 샘플:
        {report[:3000]}...

        **B2B 전문성 평가 기준:**

        1. **전문성 (30점)**: Fortune 500 수준 품질, 업계 표준 용어, 정확한 분석
        2. **시각화 (25점)**: 표/차트 활용, 데이터 가독성, 구조적 배치
        3. **신뢰성 (20점)**: 출처 명시, 데이터 추적성, 투명성
        4. **실용성 (15점)**: 실행 가능성, 비즈니스 가치, 의사결정 지원
        5. **완성도 (10점)**: 구조적 완결성, 전문적 톤, 이모지 미사용

        0.0-1.0 사이의 점수만 답변해주세요 (예: 0.89):
        """

        try:
            response = await self.chat.ainvoke(prompt)
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except:
            print("- 품질 검증 실패, 기본값 사용")
            return 0.75


# SimpleAnswererAgent: 단순 쿼리 빠른 응답
class SimpleAnswererAgent:
    """단순 질문 전용 Agent - Planning에서 SIMPLE로 분류된 쿼리 처리"""

    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.chat = ChatOpenAI(model="gpt-3.5-turbo")
        self.agent_type = AgentType.SIMPLE_ANSWERER

    async def answer(self, state):
        print("\n>> SIMPLE_ANSWERER 시작")
        print(f"- 단순 질문: {state.original_query}")

        # 1. Planning 단계에서 Vector DB가 필요한지 확인
        needs_vector_db = DatabaseType.VECTOR_DB in state.query_plan.required_databases

        if needs_vector_db:
            # Vector DB 검색 수행
            simple_results = await self._simple_search(state.original_query)
            print(f"- Vector DB 검색: {len(simple_results)}개 결과")

            # 검색 결과 저장
            for result in simple_results:
                state.add_multi_source_result(result)
        else:
            # Vector DB 검색 생략
            simple_results = []
            print("- Vector DB 검색 생략 (Planning 판단에 따라)")

        # 2. LLM으로 답변 생성
        final_answer = await self._generate_simple_answer(
            state.original_query, simple_results
        )
        state.final_answer = final_answer

        # 3. 메모리 기록
        memory = state.get_agent_memory(AgentType.SIMPLE_ANSWERER)
        memory.add_finding(
            f"단순 답변 생성 완료 - Vector DB 사용 여부: {needs_vector_db}"
        )
        memory.update_metric("answer_length", len(final_answer))

        print(f"- 답변 생성 완료: {len(final_answer)}자")
        print("\n>> SIMPLE_ANSWERER 완료")

        return state

    async def _simple_search(self, query):
        """Vector DB 간단 검색"""
        try:
            # Vector DB에서 상위 3개 문서만 검색
            vector_results = self.vector_db.search(query, top_k=3)

            search_results = []
            for doc in vector_results:
                search_result = SearchResult(
                    source="vector_db_simple",
                    content=doc["content"],
                    relevance_score=doc.get("similarity_score", 0.7),
                    metadata=doc.get("metadata", {}),
                    search_query=query,
                )
                search_results.append(search_result)

            return search_results

        except Exception as e:
            print(f"- 검색 오류: {e}")
            return []

    async def _generate_simple_answer(self, query, search_results):
        """단순하고 직접적인 답변 생성"""

        # 검색 결과 요약
        context_summary = ""
        if search_results:
            context_summary = "\n".join(
                [f"- {result.content[:200]}..." for result in search_results[:3]]
            )
        else:
            context_summary = "관련 자료를 찾지 못했습니다."

        prompt = f"""
        다음 질문에 대해 간단하고 명확한 답변을 제공해주세요.

        질문: "{query}"

        참고 자료:
        {context_summary}

        답변 지침:
        1. 질문에 직접적으로 답변
        2. 2-3개 문단으로 간결하게 구성(너무 간단한 질문일 경우 그냥 질문에 맞게 아무렇게나 대답하면 됨)
        3. 구체적인 정보가 있다면 포함
        4. 불확실한 내용은 언급하지 말 것
        5. 실용적이고 도움이 되는 정보 위주

        답변:
        """

        try:
            response = await self.chat.ainvoke(prompt)
            return response.content

        except Exception as e:
            print(f"- 답변 생성 오류: {e}")
            return (
                f"죄송합니다. '{query}'에 대한 답변을 생성하는 중 오류가 발생했습니다."
            )
