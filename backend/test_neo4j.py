#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Neo4j 연결 및 쿼리 테스트 스크립트
사용법: python test_neo4j.py
"""

import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv
from neo4j import GraphDatabase
import json
from datetime import datetime

# 환경변수 로드
load_dotenv('.env')

class Neo4jTester:
    def __init__(self):
        """Neo4j 연결 초기화"""
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        
        print(f"🔗 Neo4j 연결 정보:")
        print(f"   URI: {self.uri}")
        print(f"   Username: {self.username}")
        print(f"   Password: {'*' * len(self.password)}")
        
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            print("✅ Neo4j 드라이버 생성 성공")
        except Exception as e:
            print(f"❌ Neo4j 드라이버 생성 실패: {e}")
            sys.exit(1)
    
    def test_connection(self):
        """연결 테스트"""
        print("\n" + "="*50)
        print("1️⃣ 연결 테스트")
        print("="*50)
        
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS num")
                record = result.single()
                if record and record["num"] == 1:
                    print("✅ Neo4j 연결 성공!")
                    return True
        except Exception as e:
            print(f"❌ Neo4j 연결 실패: {e}")
            return False
    
    def get_database_info(self):
        """데이터베이스 정보 조회"""
        print("\n" + "="*50)
        print("2️⃣ 데이터베이스 정보")
        print("="*50)
        
        with self.driver.session() as session:
            # 노드 개수
            node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            print(f"📊 전체 노드 수: {node_count:,}개")
            
            # 관계 개수
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            print(f"🔗 전체 관계 수: {rel_count:,}개")
            
            # 노드 라벨별 개수
            labels_result = session.run("""
                CALL db.labels() YIELD label
                CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) AS count', {})
                YIELD value
                RETURN label, value.count AS count
                ORDER BY value.count DESC
            """)
            
            print("\n📌 라벨별 노드 수:")
            for record in labels_result:
                print(f"   - {record['label']}: {record['count']:,}개")
            
            # 관계 타입별 개수
            rel_types = session.run("""
                CALL db.relationshipTypes() YIELD relationshipType
                CALL apoc.cypher.run('MATCH ()-[r:' + relationshipType + ']->() RETURN count(r) AS count', {})
                YIELD value
                RETURN relationshipType, value.count AS count
                ORDER BY value.count DESC
            """)
            
            print("\n🔗 관계 타입별 개수:")
            for record in rel_types:
                print(f"   - {record['relationshipType']}: {record['count']:,}개")
    
    def test_basic_queries(self):
        """기본 쿼리 테스트"""
        print("\n" + "="*50)
        print("3️⃣ 기본 쿼리 테스트")
        print("="*50)
        
        test_queries = [
            {
                "name": "회사 노드 조회",
                "query": "MATCH (c:Company) RETURN c.name AS name LIMIT 5"
            },
            {
                "name": "제품 노드 조회",
                "query": "MATCH (p:Product) RETURN p.name AS name LIMIT 5"
            },
            {
                "name": "회사-제품 관계 조회",
                "query": """
                    MATCH (c:Company)-[r:PRODUCES]->(p:Product)
                    RETURN c.name AS company, p.name AS product
                    LIMIT 5
                """
            },
            {
                "name": "식품 카테고리별 제품 수",
                "query": """
                    MATCH (p:Product)
                    WHERE p.category IS NOT NULL
                    RETURN p.category AS category, count(p) AS count
                    ORDER BY count DESC
                    LIMIT 10
                """
            }
        ]
        
        with self.driver.session() as session:
            for test in test_queries:
                print(f"\n📝 {test['name']}:")
                try:
                    result = session.run(test['query'])
                    records = list(result)
                    
                    if records:
                        for i, record in enumerate(records[:5], 1):
                            print(f"   {i}. {dict(record)}")
                    else:
                        print("   (결과 없음)")
                        
                except Exception as e:
                    print(f"   ❌ 쿼리 실행 실패: {e}")
    
    def test_food_queries(self):
        """식품 도메인 특화 쿼리 테스트"""
        print("\n" + "="*50)
        print("4️⃣ 식품 도메인 쿼리 테스트")
        print("="*50)
        
        food_queries = [
            {
                "name": "라면 제조사 조회",
                "query": """
                    MATCH (c:Company)-[:PRODUCES]->(p:Product)
                    WHERE p.name CONTAINS '라면' OR p.category CONTAINS '라면'
                    RETURN DISTINCT c.name AS company
                    LIMIT 10
                """
            },
            {
                "name": "수출 관련 정보",
                "query": """
                    MATCH (c:Company)-[r:EXPORTS_TO]->(country)
                    RETURN c.name AS company, country.name AS country, r.year AS year
                    LIMIT 10
                """
            },
            {
                "name": "식품 안전 인증 정보",
                "query": """
                    MATCH (c:Company)-[r:HAS_CERTIFICATION]->(cert:Certification)
                    WHERE cert.type IN ['HACCP', 'ISO22000', 'FSSC22000']
                    RETURN c.name AS company, cert.type AS certification, r.date AS date
                    LIMIT 10
                """
            },
            {
                "name": "원재료 공급망",
                "query": """
                    MATCH (s:Supplier)-[:SUPPLIES]->(i:Ingredient)-[:USED_IN]->(p:Product)
                    RETURN s.name AS supplier, i.name AS ingredient, p.name AS product
                    LIMIT 10
                """
            }
        ]
        
        with self.driver.session() as session:
            for test in food_queries:
                print(f"\n🍜 {test['name']}:")
                try:
                    result = session.run(test['query'])
                    records = list(result)
                    
                    if records:
                        for i, record in enumerate(records[:5], 1):
                            print(f"   {i}. {dict(record)}")
                        if len(records) > 5:
                            print(f"   ... 외 {len(records)-5}개")
                    else:
                        print("   (결과 없음)")
                        
                except Exception as e:
                    print(f"   ❌ 쿼리 실행 실패: {e}")
    
    def test_rag_search(self, query: str = "오뚜기 라면 제품"):
        """RAG 검색 시뮬레이션"""
        print("\n" + "="*50)
        print("5️⃣ RAG 검색 테스트")
        print("="*50)
        print(f"🔍 검색어: {query}")
        
        # 키워드 추출 (간단한 예시)
        keywords = query.split()
        
        # 다양한 검색 패턴
        search_patterns = [
            # 회사명 검색
            f"""
            MATCH (c:Company)
            WHERE c.name CONTAINS '{keywords[0] if keywords else ""}'
            RETURN 'Company' AS type, c.name AS name, c AS data
            LIMIT 5
            """,
            
            # 제품명 검색
            f"""
            MATCH (p:Product)
            WHERE p.name CONTAINS '{keywords[-1] if keywords else ""}'
            RETURN 'Product' AS type, p.name AS name, p AS data
            LIMIT 5
            """,
            
            # 관계 기반 검색
            f"""
            MATCH (c:Company)-[:PRODUCES]->(p:Product)
            WHERE c.name CONTAINS '{keywords[0] if keywords else ""}' 
               OR p.name CONTAINS '{keywords[-1] if keywords else ""}'
            RETURN 'Relationship' AS type, 
                   c.name + ' -> ' + p.name AS name,
                   {{company: c.name, product: p.name}} AS data
            LIMIT 5
            """
        ]
        
        all_results = []
        with self.driver.session() as session:
            for pattern in search_patterns:
                try:
                    result = session.run(pattern)
                    for record in result:
                        all_results.append({
                            "type": record["type"],
                            "name": record["name"],
                            "score": 0.8  # 실제로는 유사도 계산 필요
                        })
                except Exception as e:
                    print(f"   ⚠️ 패턴 실행 실패: {e[:100]}")
        
        # 결과 출력
        print(f"\n📊 검색 결과: {len(all_results)}개")
        for i, result in enumerate(all_results[:10], 1):
            print(f"   {i}. [{result['type']}] {result['name']} (score: {result['score']:.2f})")
    
    def test_complex_analytics(self):
        """복잡한 분석 쿼리 테스트"""
        print("\n" + "="*50)
        print("6️⃣ 복잡한 분석 쿼리")
        print("="*50)
        
        analytics_queries = [
            {
                "name": "회사별 제품 수 TOP 10",
                "query": """
                    MATCH (c:Company)-[:PRODUCES]->(p:Product)
                    RETURN c.name AS company, count(p) AS product_count
                    ORDER BY product_count DESC
                    LIMIT 10
                """
            },
            {
                "name": "가장 많은 관계를 가진 노드",
                "query": """
                    MATCH (n)
                    RETURN n.name AS name, 
                           labels(n)[0] AS type,
                           size((n)--()) AS degree
                    ORDER BY degree DESC
                    LIMIT 10
                """
            },
            {
                "name": "공급망 깊이 분석",
                "query": """
                    MATCH path = (s:Supplier)-[:SUPPLIES*1..3]->(p:Product)
                    RETURN s.name AS supplier, 
                           p.name AS product,
                           length(path) AS depth
                    ORDER BY depth DESC
                    LIMIT 5
                """
            }
        ]
        
        with self.driver.session() as session:
            for test in analytics_queries:
                print(f"\n📈 {test['name']}:")
                try:
                    result = session.run(test['query'])
                    records = list(result)
                    
                    if records:
                        for i, record in enumerate(records, 1):
                            print(f"   {i}. {dict(record)}")
                    else:
                        print("   (결과 없음)")
                        
                except Exception as e:
                    print(f"   ❌ 쿼리 실행 실패: {e}")
    
    def close(self):
        """연결 종료"""
        self.driver.close()
        print("\n🔒 Neo4j 연결 종료")

def main():
    """메인 실행 함수"""
    print("\n" + "🚀 Neo4j 테스트 시작 " + "🚀")
    print("시간:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # 테스터 초기화
    tester = Neo4jTester()
    
    try:
        # 1. 연결 테스트
        if not tester.test_connection():
            print("연결 실패로 테스트 중단")
            return
        
        # 2. 데이터베이스 정보
        try:
            tester.get_database_info()
        except Exception as e:
            print(f"⚠️ 데이터베이스 정보 조회 실패: {e}")
        
        # 3. 기본 쿼리
        tester.test_basic_queries()
        
        # 4. 식품 도메인 쿼리
        tester.test_food_queries()
        
        # 5. RAG 검색 테스트
        test_queries = [
            "오뚜기 라면",
            "농심 수출 현황",
            "김치 제조 업체",
            "HACCP 인증 기업"
        ]
        for q in test_queries:
            tester.test_rag_search(q)
        
        # 6. 복잡한 분석
        tester.test_complex_analytics()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
    finally:
        tester.close()
    
    print("\n" + "✅ 테스트 완료! " + "✅\n")

if __name__ == "__main__":
    main()