# neo4j_structure_analyzer.py

import os
import sys
from pathlib import Path
import json
from collections import defaultdict

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 환경변수 로드
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Neo4j 직접 연결
from neo4j import GraphDatabase, basic_auth

def run_cypher(query: str, parameters=None):
    """Neo4j 쿼리 실행"""
    URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    USER = os.getenv("NEO4J_USER", "neo4j")
    PASSWORD = os.getenv("NEO4J_PASSWORD", "#cwp2025!")

    driver = GraphDatabase.driver(URI, auth=basic_auth(USER, PASSWORD))

    with driver.session() as session:
        result = session.run(query, parameters or {})
        data = [record.data() for record in result]

    driver.close()
    return data

def analyze_neo4j_structure():
    """Neo4j 데이터베이스 구조 완전 분석"""

    print("=" * 80)
    print("NEO4J 데이터베이스 구조 완전 분석")
    print("=" * 80)

    # 1. 기본 통계
    print("\n1. 기본 통계:")
    try:
        count_result = run_cypher("MATCH (n) RETURN count(n) as total_nodes")
        total_nodes = count_result[0]['total_nodes']
        print(f"   총 노드 개수: {total_nodes:,}개")

        rel_count = run_cypher("MATCH ()-[r]->() RETURN count(r) as total_rels")
        total_rels = rel_count[0]['total_rels']
        print(f"   총 관계 개수: {total_rels:,}개")
    except Exception as e:
        print(f"   기본 통계 오류: {e}")

    # 2. 라벨 분석
    print("\n2. 라벨 분석:")
    try:
        labels_result = run_cypher("CALL db.labels()")
        labels = [r['label'] for r in labels_result]
        print(f"   사용 중인 라벨: {labels}")

        for label in labels:
            count_query = f"MATCH (n:{label}) RETURN count(n) as count"
            count_result = run_cypher(count_query)
            count = count_result[0]['count']
            print(f"   - {label}: {count:,}개")
    except Exception as e:
        print(f"   라벨 분석 오류: {e}")

    # 3. 관계 타입 분석
    print("\n3. 관계 타입 분석:")
    try:
        rels_result = run_cypher("CALL db.relationshipTypes()")
        rel_types = [r['relationshipType'] for r in rels_result]
        print(f"   관계 타입: {rel_types}")

        for rel_type in rel_types:
            count_query = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"
            count_result = run_cypher(count_query)
            count = count_result[0]['count']
            print(f"   - {rel_type}: {count:,}개")
    except Exception as e:
        print(f"   관계 타입 분석 오류: {e}")

    # 4. 노드 속성 완전 분석
    print("\n4. 노드 속성 완전 분석:")
    try:
        # 모든 속성 키 찾기
        all_keys_query = """
        MATCH (n)
        UNWIND keys(n) as key
        RETURN DISTINCT key, count(*) as frequency
        ORDER BY frequency DESC
        """
        keys_result = run_cypher(all_keys_query)
        print("   모든 속성 키 (사용 빈도순):")
        for item in keys_result:
            print(f"     {item['key']}: {item['frequency']:,}번 사용")

        # 각 속성의 고유값 샘플
        print("\n   각 속성의 고유값 샘플:")
        for item in keys_result[:10]:  # 상위 10개 속성만
            key = item['key']
            sample_query = f"""
            MATCH (n)
            WHERE n.{key} IS NOT NULL
            RETURN DISTINCT n.{key} as value
            LIMIT 20
            """
            try:
                sample_result = run_cypher(sample_query)
                values = [r['value'] for r in sample_result]
                print(f"     {key} 샘플: {values[:10]}...")
            except Exception as e:
                print(f"     {key} 샘플 조회 실패: {e}")

    except Exception as e:
        print(f"   속성 분석 오류: {e}")

    # 5. 라벨별 속성 패턴 분석
    print("\n5. 라벨별 속성 패턴:")
    try:
        labels_result = run_cypher("CALL db.labels()")
        labels = [r['label'] for r in labels_result]

        for label in labels:
            print(f"\n   라벨 '{label}'의 속성 패턴:")

            # 이 라벨의 모든 속성 찾기
            label_keys_query = f"""
            MATCH (n:{label})
            UNWIND keys(n) as key
            RETURN DISTINCT key, count(*) as frequency
            ORDER BY frequency DESC
            """
            label_keys = run_cypher(label_keys_query)

            for key_info in label_keys:
                key = key_info['key']
                freq = key_info['frequency']

                # 이 속성의 샘플 값들
                sample_query = f"""
                MATCH (n:{label})
                WHERE n.{key} IS NOT NULL
                RETURN DISTINCT n.{key} as value
                LIMIT 10
                """
                try:
                    sample_result = run_cypher(sample_query)
                    values = [str(r['value'])[:50] for r in sample_result]  # 길이 제한
                    print(f"     {key} ({freq}개): {values}")
                except Exception as e:
                    print(f"     {key} ({freq}개): 샘플 조회 실패")

    except Exception as e:
        print(f"   라벨별 속성 분석 오류: {e}")

    # 6. 관계 패턴 분석
    print("\n6. 관계 패턴 분석:")
    try:
        # 관계별로 연결되는 라벨 패턴 찾기
        rel_pattern_query = """
        MATCH (a)-[r]->(b)
        RETURN type(r) as rel_type,
               labels(a) as start_labels,
               labels(b) as end_labels,
               count(*) as frequency
        ORDER BY frequency DESC
        LIMIT 20
        """
        patterns = run_cypher(rel_pattern_query)

        print("   관계 패턴 (빈도순):")
        for pattern in patterns:
            rel_type = pattern['rel_type']
            start_labels = pattern['start_labels']
            end_labels = pattern['end_labels']
            freq = pattern['frequency']
            print(f"     ({start_labels}) -[{rel_type}]-> ({end_labels}): {freq:,}개")

    except Exception as e:
        print(f"   관계 패턴 분석 오류: {e}")

    # 7. 검색 최적화를 위한 인덱스 정보
    print("\n7. 인덱스 정보:")
    try:
        # 현재 인덱스 조회
        index_query = "CALL db.indexes()"
        indexes = run_cypher(index_query)

        if indexes:
            print("   현재 인덱스:")
            for idx in indexes:
                print(f"     {idx}")
        else:
            print("   인덱스 없음")

        # 인덱스 추천
        print("\n   추천 인덱스 (자주 검색되는 속성):")
        frequent_props = ['product', 'city', 'region', 'category', 'name']
        for prop in frequent_props:
            print(f"     CREATE INDEX FOR (n) ON (n.{prop})")

    except Exception as e:
        print(f"   인덱스 정보 조회 실패: {e}")

    # 8. 샘플 데이터 (검색 쿼리 개발용)
    print("\n8. 샘플 데이터 (검색 쿼리 개발용):")
    try:
        # 각 라벨별 샘플 노드
        labels_result = run_cypher("CALL db.labels()")
        labels = [r['label'] for r in labels_result]

        for label in labels:
            sample_query = f"MATCH (n:{label}) RETURN n LIMIT 3"
            samples = run_cypher(sample_query)
            print(f"\n   라벨 '{label}' 샘플:")
            for i, sample in enumerate(samples, 1):
                print(f"     {i}. {sample['n']}")

    except Exception as e:
        print(f"   샘플 데이터 조회 실패: {e}")

    # 9. 저장을 위한 구조 요약
    print("\n9. 구조 요약 (JSON 저장용):")
    try:
        structure_summary = {
            "total_nodes": total_nodes,
            "total_relationships": total_rels,
            "labels": {},
            "relationship_types": {},
            "property_keys": {},
            "common_patterns": []
        }

        # 라벨별 정보
        for label in labels:
            count_query = f"MATCH (n:{label}) RETURN count(n) as count"
            count_result = run_cypher(count_query)
            count = count_result[0]['count']

            structure_summary["labels"][label] = {
                "count": count,
                "properties": {}
            }

            # 각 라벨의 주요 속성
            label_keys_query = f"""
            MATCH (n:{label})
            UNWIND keys(n) as key
            RETURN DISTINCT key, count(*) as frequency
            ORDER BY frequency DESC
            LIMIT 10
            """
            label_keys = run_cypher(label_keys_query)

            for key_info in label_keys:
                key = key_info['key']
                freq = key_info['frequency']
                structure_summary["labels"][label]["properties"][key] = freq

        # JSON으로 저장
        with open('neo4j_structure.json', 'w', encoding='utf-8') as f:
            json.dump(structure_summary, f, ensure_ascii=False, indent=2)

        print("   구조 정보를 'neo4j_structure.json'에 저장했습니다.")
        print(f"   요약: {json.dumps(structure_summary, ensure_ascii=False, indent=2)}")

    except Exception as e:
        print(f"   구조 요약 저장 실패: {e}")

    print("\n" + "=" * 80)
    print("분석 완료!")
    print("=" * 80)

if __name__ == "__main__":
    analyze_neo4j_structure()
