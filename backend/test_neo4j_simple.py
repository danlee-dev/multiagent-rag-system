#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Neo4j 간단 테스트 스크립트
사용법: python test_neo4j_simple.py
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# 환경변수 로드
load_dotenv('.env')

def test_neo4j():
    """Neo4j 간단 테스트"""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    print(f"🔗 Neo4j 연결: {uri}")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        with driver.session() as session:
            # 연결 테스트
            result = session.run("RETURN 1 AS test")
            if result.single()["test"] == 1:
                print("✅ 연결 성공!")
            
            # 노드 수 확인
            node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            print(f"📊 전체 노드 수: {node_count:,}개")
            
            # 관계 수 확인
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            print(f"🔗 전체 관계 수: {rel_count:,}개")
            
            # 샘플 데이터 조회
            print("\n📝 샘플 데이터:")
            sample_result = session.run("""
                MATCH (n) 
                RETURN labels(n)[0] AS type, n.name AS name 
                LIMIT 5
            """)
            
            for i, record in enumerate(sample_result, 1):
                print(f"   {i}. [{record['type']}] {record['name']}")
        
        driver.close()
        print("\n✅ 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 오류: {e}")

if __name__ == "__main__":
    test_neo4j()