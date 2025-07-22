import re
import json
import os
from typing import List, Dict, Any, Tuple
from pathlib import Path
import asyncio

from dotenv import load_dotenv
from neo4j import GraphDatabase, basic_auth
from langchain_openai import ChatOpenAI


class Neo4jSearchService:
    """Neo4j 그래프 데이터베이스 검색 비즈니스 로직을 캡슐화하는 서비스 클래스"""
    def __init__(self):
        """Neo4j 연결 초기화"""
        self.driver = self._init_driver()
        self.llm_client = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def _init_driver(self):
        """Neo4j 드라이버 초기화"""
        env_path = Path(__file__).parent.parent / '.env'
        load_dotenv(dotenv_path=env_path)
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "#cwp2025!")
        print(f"\n>> Neo4j 연결 초기화 (URI: {uri})")

        return GraphDatabase.driver(uri, auth=basic_auth(user, password))

    def _run_cypher_sync(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Cypher 쿼리를 동기적으로 실행합니다."""
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            print(f"- Cypher 쿼리 실행 오류: {e}\n- 쿼리: {query}\n- 파라미터: {parameters}")
            return []


    async def _run_cypher_async(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """동기 DB 호출 함수(_run_cypher_sync)를 별도 스레드에서 비동기적으로 실행합니다."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run_cypher_sync, query, parameters)

    async def _extract_keywords_with_llm(self, query: str) -> Dict[str, List[str]]:
        """LLM을 사용하여 농수산물 관련 키워드를 카테고리별로 추출"""
        try:
            extraction_prompt = f"""
            Extract search keywords from the following question for a food and agriculture database.

            Question: "{query}"

            Respond in the following JSON format:
            {{
                "products": ["item1", "item2"],
                "regions": ["region1", "region2"],
                "categories": ["category1", "category2"],
                "fish_states": ["state1", "state2"]
            }}

            - products: Names of agricultural, marine, or livestock products (e.g., apple, mackerel, pork).
            - regions: Geographic locations (e.g., Seoul, Jeju, Gyeonggi-do).
            - categories: Classification of products (e.g., fruits, leafy vegetables).
            - fish_states: States of marine products (e.g., live, fresh, frozen, dried).

            Respond only with the JSON object.
            """

            response = await self.llm_client.ainvoke(extraction_prompt)
            response_text = response.content.strip()

            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)

            print(f"- LLM 추출 키워드: {response_text}")
            return json.loads(response_text)

        except Exception as e:
            print(f"- LLM 키워드 추출 오류: {e}")
            return self._extract_keywords_fallback(query)

    def _extract_keywords_fallback(self, query: str) -> Dict[str, List[str]]:
        """LLM 실패시 폴백 키워드 추출"""
        stop_words = {'의', '을', '를', '이', '가', '에', '에서', '알려줘', '검색', '찾아', '정보', '어디', '무엇', '언제', '어떤'}
        words = re.sub(r'[^\w\s]', '', query).split()
        filtered_words = [word for word in words if len(word) > 1 and word not in stop_words]

        return {
            "products": filtered_words[:3],
            "regions": [],
            "categories": [],
            "fish_states": [],
            "associations": []
        }

    async def search(self, query: str) -> str:
        """쿼리를 받아 Neo4j 검색을 수행하고 LLM이 이해하기 쉬운 문자열로 결과를 반환합니다."""
        try:
            print(f"\n>> Neo4j 검색 시작")
            keywords = await self._extract_keywords_with_llm(query)

            search_tasks = []
            for product in keywords.get("products", []):
                search_tasks.append(self._search_products_optimized(product))
            for region in keywords.get("regions", []):
                search_tasks.append(self._search_regions_optimized(region))
            for category in keywords.get("categories", []):
                search_tasks.append(self._search_by_category(category))
            for fish_state in keywords.get("fish_states", []):
                search_tasks.append(self._search_by_fish_state(fish_state))
            for association in keywords.get("associations", []):
                search_tasks.append(self._search_by_association(association))


            search_results = await asyncio.gather(*search_tasks)

            all_nodes = [item for sublist in search_results for item in sublist]

            unique_nodes = self._deduplicate_and_rank_nodes(all_nodes)
            relationships = await self._search_relationships_batch(unique_nodes[:5])

            result = self._format_results(query, unique_nodes, relationships)
            print(f"- Neo4j 검색 완료: {len(unique_nodes)}개 항목, {len(relationships)}개 관계")
            return result

        except Exception as e:
            print(f"- Neo4j 검색 서비스 오류: {e}")
            return f"Neo4j 검색 중 오류가 발생했습니다: {str(e)}"


    async def _search_products_optimized(self, keyword: str) -> List[dict]:
        """최적화된 품목 검색 - 인덱스 활용 + 퍼포먼스 튜닝"""
        try:
            cypher_query = """
            CALL db.index.fulltext.queryNodes('prod_idx', $keyword) YIELD node, score
            WHERE score > 0.1
            WITH node, score + 2.0 AS boosted_score, 'fulltext' as search_type
            RETURN node, boosted_score AS final_score, search_type
            LIMIT 5

            UNION

            MATCH (n:농산물|수산물|축산물)
            WHERE n.product STARTS WITH $keyword
            WITH n as node, 3.0 as score, 'exact' as search_type
            RETURN node, score AS final_score, search_type
            LIMIT 3
            """

            results = await self._run_cypher_async(cypher_query, {"keyword": keyword})
            print(f"  품목 최적화 검색 ({keyword}): {len(results)}개")

            formatted_results = []
            for result in results:
                node = result['node']
                labels = node.get('labels', [])
                formatted_results.append({
                    'type': labels[0] if labels else 'Ingredient',
                    'properties': dict(node),
                    'score': result.get('final_score', 0),
                    'search_type': result.get('search_type', 'unknown')
                })
            return formatted_results

        except Exception as e:
            print(f"  품목 검색 오류: {e}")
            return []

    async def _search_regions_optimized(self, keyword: str) -> List[dict]:
        """최적화된 지역 검색 - 계층적 지역 검색"""
        try:
            cypher_query = """
            MATCH (o:Origin)
            WHERE o.city = $keyword OR o.region = $keyword
            WITH o as node, 2.0 as score, 'exact' as search_type
            RETURN node, score as final_score, search_type

            UNION

            CALL db.index.fulltext.queryNodes('org_idx', $keyword) YIELD node, score
            WHERE score > 0.3
            WITH node, score, 'fulltext' as search_type
            RETURN node, score as final_score, search_type

            UNION

            MATCH (o:Origin)
            WHERE o.city CONTAINS $keyword OR o.region CONTAINS $keyword
            WITH o as node, 1.0 as score, 'contains' as search_type
            RETURN node, score as final_score, search_type
            """

            results = await self._run_cypher_async(cypher_query, {"keyword": keyword})
            print(f"  지역 최적화 검색 ({keyword}): {len(results)}개")

            formatted_results = []
            for result in results:
                node = result['node']
                formatted_results.append({
                    'type': 'Origin',
                    'properties': dict(node),
                    'score': result['final_score'],
                    'search_type': result['search_type']
                })
            return formatted_results

        except Exception as e:
            print(f"  지역 검색 오류: {e}")
            return []

    async def _search_by_category(self, category: str) -> List[dict]:
        """농산물 카테고리별 검색"""
        try:
            cypher_query = """
            MATCH (n:농산물)
            WHERE n.category = $category OR n.category CONTAINS $category
            RETURN n as node, 1.5 as score
            ORDER BY n.product
            LIMIT 5
            """

            print(cypher_query)

            results = await self._run_cypher_async(cypher_query, {"category": category})
            print(f"  카테고리 검색 ({category}): {len(results)}개")

            formatted_results = []
            for result in results:
                node = result['node']
                formatted_results.append({
                    'type': '농산물',
                    'properties': dict(node),
                    'score': result['score'],
                    'search_type': 'category'
                })
            return formatted_results

        except Exception as e:
            print(f"  카테고리 검색 오류: {e}")
            return []

    async def _search_by_fish_state(self, fish_state: str) -> List[dict]:
        """수산물 상태별 검색"""
        try:
            cypher_query = """
            MATCH (n:수산물)
            WHERE n.fishState = $fish_state OR n.fishState CONTAINS $fish_state
            RETURN n as node, 1.5 as score
            ORDER BY n.product
            LIMIT 5
            """

            print(cypher_query)

            results = await self._run_cypher_async(cypher_query, {"fish_state": fish_state})
            print(f"  수산물 상태 검색 ({fish_state}): {len(results)}개")

            formatted_results = []
            for result in results:
                node = result['node']
                formatted_results.append({
                    'type': '수산물',
                    'properties': dict(node),
                    'score': result['score'],
                    'search_type': 'fish_state'
                })
            return formatted_results

        except Exception as e:
            print(f"  수산물 상태 검색 오류: {e}")
            return []

    async def _search_by_association(self, association: str) -> List[dict]:
        """수협별 검색 - 관계를 통한 역추적"""
        try:
            cypher_query = """
            MATCH (i:수산물)-[r:isFrom]->(o:Origin)
            WHERE r.association CONTAINS $association
            RETURN DISTINCT i as node, 1.0 as score,
                   r.association as found_association
            ORDER BY i.product
            LIMIT 5
            """

            print(cypher_query)

            results = await self._run_cypher_async(cypher_query, {"association": association})
            print(f"  수협 검색 ({association}): {len(results)}개")

            formatted_results = []
            for result in results:
                node = result['node']
                properties = dict(node)
                properties['found_association'] = result['found_association']
                formatted_results.append({
                    'type': '수산물',
                    'properties': properties,
                    'score': result['score'],
                    'search_type': 'association'
                })
            return formatted_results

        except Exception as e:
            print(f"  수협 검색 오류: {e}")
            return []

    def _deduplicate_and_rank_nodes(self, nodes: List[dict]) -> List[dict]:
        """결과 중복 제거 및 점수별 정렬"""
        node_map = {}

        for node in nodes:
            props = node.get('properties', {})
            # 품목명 + 타입으로 유니크 키 생성
            product = props.get('product', '')
            node_type = node.get('type', '')
            key = f"{node_type}:{product}"

            if key not in node_map:
                node_map[key] = node
            else:
                # 더 높은 점수가 있으면 교체
                if node.get('score', 0) > node_map[key].get('score', 0):
                    node_map[key] = node

        # 점수순 정렬
        unique_nodes = list(node_map.values())
        unique_nodes.sort(key=lambda x: x.get('score', 0), reverse=True)

        return unique_nodes

    async def _search_relationships_batch(self, nodes: List[dict]) -> List[dict]:
        """배치로 관계 검색 - 성능 최적화"""
        if not nodes:
            return []

        try:
            # 품목명들을 배치로 수집
            product_names = []
            for node in nodes:
                props = node.get('properties', {})
                product_name = props.get('product')
                if product_name:
                    product_names.append(product_name)

            if not product_names:
                return []

            print(product_names)

            cypher_query = """
            MATCH (i)-[r:isFrom]->(o:Origin)
            WHERE i.product IN $product_names
            RETURN i.product as product,
                   COALESCE(o.city, o.region) as location,
                   o.region as region,
                   r.farm as farm,
                   r.count as count,
                   r.association as association,
                   r.sold as sold,
                   labels(i)[0] as ingredient_type
            ORDER BY i.product, r.farm DESC
            LIMIT 20
            """

            print(cypher_query)

            results = await self._run_cypher_async(cypher_query, {"product_names": product_names})
            print(f"  배치 관계 검색: {len(results)}개")

            relationships = []
            for rel in results:
                relationship_info = {
                    'start': rel['product'],
                    'end': f"{rel['location']}({rel['region']})" if rel['region'] else rel['location'],
                    'type': 'isFrom',
                    'ingredient_type': rel['ingredient_type']
                }

                # 관계 속성 추가 (null 체크)
                if rel['farm'] is not None:
                    relationship_info['farm'] = rel['farm']
                if rel['count'] is not None:
                    relationship_info['count'] = rel['count']
                if rel['association']:
                    relationship_info['association'] = rel['association']
                if rel['sold']:
                    relationship_info['sold'] = rel['sold']

                relationships.append(relationship_info)

            return relationships

        except Exception as e:
            print(f"  배치 관계 검색 오류: {e}")
            return []

    def _format_results(self, query: str, nodes: List[dict], relationships: List[dict]) -> str:
        """검색된 노드와 관계를 최종 문자열로 포맷팅합니다."""
        if not nodes and not relationships:
            return f"'{query}'에 대한 관련 정보를 Neo4j 그래프 데이터베이스에서 찾을 수 없습니다."

        summary = f"Neo4j Graph Database 검색 결과 ('{query}'):\n"
        summary += f"총 {len(nodes)}개 항목, {len(relationships)}개 연관관계 발견\n\n"

        if nodes:
            summary += "검색된 농수산물 정보:\n"
            for i, node in enumerate(nodes[:], 1):
                props = node.get('properties', {})
                node_type = str(node.get('type', '항목'))
                score = node.get('score', 0)
                search_type = node.get('search_type', 'unknown')

                # 노드 타입별 정보 포맷팅
                if node_type == '농산물':
                    name = props.get('product', 'N/A')
                    category = props.get('category', '')
                    info = f"농산물: {name}"
                    if category:
                        info += f" (분류: {category})"

                elif node_type == '수산물':
                    name = props.get('product', 'N/A')
                    fish_state = props.get('fishState', '')
                    info = f"수산물: {name}"
                    if fish_state:
                        info += f" (상태: {fish_state})"
                    if props.get('found_association'):
                        info += f" (수협: {props['found_association']})"

                elif node_type == '축산물':
                    name = props.get('product', 'N/A')
                    info = f"축산물: {name}"

                elif node_type == 'Origin':
                    city = props.get('city', '')
                    region = props.get('region', '')
                    if city and city != '.':
                        info = f"지역: {city} ({region})"
                    else:
                        info = f"지역: {region}"
                else:
                    name = next(iter(props.values())) if props else 'N/A'
                    info = f"{node_type}: {name}"

                summary += f"{i:2d}. {info} [매칭도: {score:.2f}, 검색타입: {search_type}]\n"

        if relationships:
            summary += f"\n생산지 연관관계 ({len(relationships)}개):\n"
            for i, rel in enumerate(relationships[:], 1):
                rel_info = f"{i:2d}. {rel['start']} → {rel['end']}"

                # 관계 속성 정보 추가
                details = []
                if 'farm' in rel and rel['farm'] is not None:
                    details.append(f"농장수: {rel['farm']}개")
                if 'count' in rel and rel['count'] is not None:
                    details.append(f"사육수: {rel['count']:,}마리")
                if 'association' in rel and rel['association']:
                    details.append(f"수협: {rel['association']}")
                if 'sold' in rel and rel['sold']:
                    details.append(f"위판장: {rel['sold']}")

                if details:
                    rel_info += f" ({', '.join(details)})"

                summary += f"{rel_info}\n"

        return summary

    def close(self):
        """드라이버 연결 종료"""
        if self.driver:
            self.driver.close()
            print("- Neo4j 드라이버 연결 종료")

async def neo4j_graph_search(query: str) -> str:
    """
    RAG 에이전트가 호출할 Neo4j 검색 도구의 메인 진입점 함수

    Args:
        query: 검색 질의
        llm_client: LLM 클라이언트 (OpenAI 등)

    Returns:
        검색 결과 문자열
    """
    service = Neo4jSearchService()
    try:
        return await service.search(query)
    finally:
        service.close()
