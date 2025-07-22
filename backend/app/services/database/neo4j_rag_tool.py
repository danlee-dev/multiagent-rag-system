import re
import json
from typing import List, Dict, Any

from ...tools.query.neo4j_query import run_cypher

class Neo4jSearchService:
    """Neo4j 그래프 데이터베이스 검색 비즈니스 로직을 캡슐화하는 서비스 클래스"""

    def search(self, query: str) -> str:
        """쿼리를 받아 Neo4j 검색을 수행하고 LLM이 이해하기 쉬운 문자열로 결과를 반환합니다."""
        try:
            keywords = self._extract_keywords(query)
            print(f"추출된 키워드: {keywords}")

            all_nodes = []
            for keyword in keywords[:3]: # 성능을 위해 상위 3개 키워드만 사용
                all_nodes.extend(self._search_node("농산물", ["product", "category"], keyword))
                all_nodes.extend(self._search_node("수산물", ["product", "fishState"], keyword))
                all_nodes.extend(self._search_node("Origin", ["city", "region"], keyword))

            unique_nodes = self._deduplicate_nodes(all_nodes)
            relationships = self._search_relationships(unique_nodes[:3])

            return self._format_results(query, unique_nodes, relationships)
        except Exception as e:
            print(f"- Neo4j 검색 서비스 오류: {e}")
            return f"Neo4j 검색 중 오류가 발생했습니다: {str(e)}"

    def _extract_keywords(self, query: str) -> List[str]:
        """쿼리에서 검색 키워드를 추출하고 정제합니다."""
        stop_words = ['의', '을', '를', '이', '가', '에', '에서', '알려줘', '검색', '찾아', '정보']
        words = re.sub(r'[^\w\s]', '', query).split()
        return [word for word in words if len(word) > 1 and word not in stop_words]

    def _search_node(self, label: str, properties: List[str], keyword: str) -> List[dict]:
        """특정 레이블과 속성을 가진 노드를 검색하는 범용 메서드"""
        try:
            where_clauses = " OR ".join([f"n.{prop} CONTAINS $keyword" for prop in properties])
            cypher_query = f"""
            MATCH (n:{label}) WHERE {where_clauses}
            RETURN labels(n)[0] as type, properties(n) as properties
            LIMIT 5
            """
            results = run_cypher(cypher_query, {"keyword": keyword})
            print(f"  '{label}' 검색 ({keyword}): {len(results)}개")
            return results
        except Exception as e:
            print(f"  노드 검색 오류 ({label}): {e}")
            return []

    def _deduplicate_nodes(self, nodes: List[dict]) -> List[dict]:
        """결과 중복 제거"""
        seen = set()
        unique_results = []
        for node in nodes:
            # properties 딕셔너리를 정렬된 JSON 문자열로 만들어 고유 키로 사용
            key = json.dumps(node.get('properties'), sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique_results.append(node)
        return unique_results

    def _search_relationships(self, nodes: List[dict]) -> List[dict]:
        """품목 노드와 관련된 isFrom 관계를 검색"""
        relationships = []
        try:
            for node in nodes:
                props = node.get('properties', {})
                product_name = props.get('product')
                if product_name:
                    rel_query = """
                    MATCH (p {product: $product_name})-[r:isFrom]->(o:Origin)
                    RETURN p.product as product, o.city as city, o.region as region
                    LIMIT 3
                    """
                    rel_results = run_cypher(rel_query, {"product_name": product_name})
                    for rel in rel_results:
                        relationships.append({
                            'start': rel['product'],
                            'end': f"{rel['city']}({rel['region']})",
                            'type': 'isFrom'
                        })
            print(f"  관계 검색: {len(relationships)}개")
            return relationships
        except Exception as e:
            print(f"  관계 검색 오류: {e}")
            return []

    def _format_results(self, query: str, nodes: List[dict], relationships: List[dict]) -> str:
        """검색된 노드와 관계를 최종 문자열로 포맷팅합니다."""
        if not nodes and not relationships:
            return f"'{query}'에 대한 관련 정보를 Neo4j에서 찾을 수 없습니다."

        summary = f"Neo4j Graph DB 검색 결과: {len(nodes)}개 항목, {len(relationships)}개 관계 발견\n\n"
        if nodes:
            summary += "### 검색된 항목:\n"
            for node in nodes[:8]:
                props = node.get('properties', {})
                node_type_label = str(node.get('type', '항목')).replace('Origin', '지역')
                name = props.get('product') or props.get('city') or 'N/A'
                details = ', '.join([f"{k}:'{v}'" for k, v in props.items()])
                summary += f"- {node_type_label}: {name} (속성: {details})\n"

        if relationships:
            summary += "\n### 연관 관계:\n"
            for rel in relationships[:5]:
                summary += f"- {rel['start']}는 {rel['end']}에서 생산됩니다. (관계: {rel['type']})\n"

        print(f"- Neo4j 검색 완료: {len(nodes)}개 항목, {len(relationships)}개 관계")
        return summary

def neo4j_graph_search(query: str) -> str:
    """
    RAG 에이전트가 호출할 Neo4j 검색 도구의 메인 진입점 함수
    복잡한 로직은 Neo4jSearchService 클래스에 위임
    """
    service = Neo4jSearchService()
    return service.search(query)
