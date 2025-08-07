# neo4j_rag_tool.py
import os
import re
import json
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase, basic_auth
from neo4j.exceptions import ServiceUnavailable, ClientError
from langchain_google_genai import ChatGoogleGenerativeAI


# =========================
# Config & Constants
# =========================
DEFAULT_NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
DEFAULT_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# 풀텍스트 인덱스 이름(존재하면 가산점)
FULLTEXT_PRODUCT_INDEX = "prod_idx"  # ON :농산물|수산물|축산물 (product)
FULLTEXT_ORIGIN_INDEX = "org_idx"    # ON :Origin (city, region)

# 검색 가중치
SCORE_FULLTEXT = 4.0
SCORE_PREFIX = 3.0
SCORE_EXACT = 2.0
SCORE_CONTAINS = 1.0
SCORE_CATEGORY = 1.5
SCORE_FISHSTATE = 1.5
SCORE_ASSOC = 1.0

# 상한
MAX_RELN_PRODUCTS = 8  # 관계 조회 시 상위 N개의 노드만 관계를 긁음(성능 보호)


# =========================
# Utilities
# =========================
def _debug(msg: str):
    print(f"[neo4j-rag] {msg}")


def _load_env():
    # 상위 프로젝트 루트에 .env가 있다면 로드
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


# =========================
# Graph Client
# =========================
class GraphDBClient:
    def __init__(self, uri: str = DEFAULT_NEO4J_URI, user: str = DEFAULT_NEO4J_USER, password: str = DEFAULT_NEO4J_PASSWORD):
        _load_env()
        uri = os.getenv("NEO4J_URI", uri)
        user = os.getenv("NEO4J_USER", user)
        password = os.getenv("NEO4J_PASSWORD", password)

        self._driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
        _debug(f"Connected Neo4j (uri={uri})")

    def close(self):
        try:
            self._driver.close()
            _debug("Driver closed")
        except Exception:
            pass

    def run(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        with self._driver.session() as session:
            res = session.run(query, params or {})
            return [r.data() for r in res]


# =========================
# LLM Wrapper
# =========================
class LLM:
    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.0):
        self.client = ChatGoogleGenerativeAI(model=model, temperature=temperature)

    async def ainvoke(self, prompt: str) -> str:
        resp = await self.client.ainvoke(prompt)
        return (resp.content or "").strip()

    def invoke(self, prompt: str) -> str:
        resp = self.client.invoke(prompt)
        return (resp.content or "").strip()


# =========================
# Search Service
# =========================
class GraphDBSearchService:
    """
    - 키워드 추출(LLM + 폴백)
    - 품목/지역/카테고리/수산상태/수협 기반 노드 검색
    - 상위 노드의 생산지 관계(isFrom), (옵션) 영양소 관계(hasNutrient) 배치 조회
    """
    def __init__(self, client: Optional[GraphDBClient] = None, llm: Optional[LLM] = None):
        self.client = client or GraphDBClient()
        self.llm = llm or LLM()

    # ---------- Public ----------
    async def search(self, user_query: str) -> str:
        _debug("Search start")
        keywords = await self._extract_keywords(user_query)

        tasks = []
        for p in keywords.get("products", []):
            tasks.append(self._search_products(p))
        for r in keywords.get("regions", []):
            tasks.append(self._search_origins(r))
        for c in keywords.get("categories", []):
            tasks.append(self._search_by_category(c))
        for fs in keywords.get("fish_states", []):
            tasks.append(self._search_by_fish_state(fs))
        for assoc in keywords.get("associations", []):
            tasks.append(self._search_by_association(assoc))

        nested = await asyncio.gather(*tasks) if tasks else [[]]
        flat_nodes = [x for sub in nested for x in sub]

        unique_nodes = self._dedupe_and_rank(flat_nodes)
        relns = await self._fetch_relationships(unique_nodes[:MAX_RELN_PRODUCTS])

        # (옵션) 영양소 관계도 있으면 붙이기
        nutrients = await self._fetch_nutrients(unique_nodes[:MAX_RELN_PRODUCTS])

        report = self._format_report(user_query, unique_nodes, relns, nutrients)
        _debug(f"Search done: nodes={len(unique_nodes)} relns={len(relns)} nutrients={len(nutrients)}")
        return report

    def close(self):
        self.client.close()

    # ---------- Keyword Extraction ----------
    async def _extract_keywords(self, q: str) -> Dict[str, List[str]]:
        prompt = f"""
다음 질문에서 그래프 검색에 적합한 키워드를 JSON 으로 뽑아주세요.

질문: "{q}"

JSON 형식:
{{
  "products": ["사과","문어"],
  "regions": ["제주시","경상북도"],
  "categories": ["과실류","엽경채류"],
  "fish_states": ["활어","선어","냉동","건어"],
  "associations": ["통영수협","경인서부수협"]
}}

규칙:
- products: 농산물/수산물/축산물 품목명(예: 사과, 문어, 돼지 등)
- regions: 행정명(시/군/구/도 등). 질문에 나온 지명만.
- categories: 질문에 '분류'가 명시되었거나 '비슷한 품목'을 묻는 경우에만 넣기. 그 외엔 비워두기.
- fish_states: 수산물 상태(활어/선어/냉동/건어). 질문에 있을 때만.
- associations: 수협/조합명. 질문에 있을 때만.
- 반드시 JSON만 출력.
"""
        try:
            txt = await self.llm.ainvoke(prompt)
            # 안전 JSON 추출
            m = re.search(r"\{[\s\S]*\}", txt)
            if m:
                txt = m.group(0)
            data = json.loads(txt)
            _debug(f"LLM keywords: {data}")
            # 필드 보정
            for k in ["products","regions","categories","fish_states","associations"]:
                data.setdefault(k, [])
            return data
        except Exception as e:
            _debug(f"LLM keyword extraction failed: {e}")
            return self._keyword_fallback(q)

    def _keyword_fallback(self, q: str) -> Dict[str, List[str]]:
        stop = {"의","을","를","이","가","에","에서","알려줘","검색","찾아","정보","어디","무엇","언제","어떤"}
        words = re.sub(r"[^\w\s]", " ", q).split()
        terms = [w for w in words if len(w) > 1 and w not in stop]
        return {
            "products": terms[:3],
            "regions": [],
            "categories": [],
            "fish_states": [],
            "associations": [],
        }

    # ---------- Node Searches ----------
    async def _search_products(self, keyword: str) -> List[Dict[str, Any]]:
        """
        :농산물|수산물|축산물 의 product 필드 중심 검색.
        - 풀텍스트 인덱스 + prefix + exact contains 혼합
        """
        query = f"""
// fulltext
CALL {{
  WITH $kw AS kw
  CALL {{
    WITH kw
    CALL db.index.fulltext.queryNodes('{FULLTEXT_PRODUCT_INDEX}', kw) YIELD node, score
    RETURN node, score + {SCORE_FULLTEXT} AS s, 'fulltext' AS t
  }}
  RETURN node, s, t
}}
UNION
// starts with (prefix)
MATCH (n:농산물|수산물|축산물)
WHERE n.product STARTS WITH $kw
RETURN n AS node, {SCORE_PREFIX} AS s, 'prefix' AS t
UNION
// exact equals
MATCH (n:농산물|수산물|축산물)
WHERE n.product = $kw
RETURN n AS node, {SCORE_EXACT} AS s, 'exact' AS t
UNION
// contains (fallback)
MATCH (n:농산물|수산물|축산물)
WHERE n.product CONTAINS $kw
RETURN n AS node, {SCORE_CONTAINS} AS s, 'contains' AS t
        """
        try:
            rows = await self._run_async(query, {"kw": keyword})
        except ClientError:
            # 풀텍스트 인덱스 없을 때: 나머지 블럭만
            query_no_ft = """
// starts with (prefix)
MATCH (n:농산물|수산물|축산물)
WHERE n.product STARTS WITH $kw
RETURN n AS node, 3.0 AS s, 'prefix' AS t
UNION
MATCH (n:농산물|수산물|축산물)
WHERE n.product = $kw
RETURN n AS node, 2.0 AS s, 'exact' AS t
UNION
MATCH (n:농산물|수산물|축산물)
WHERE n.product CONTAINS $kw
RETURN n AS node, 1.0 AS s, 'contains' AS t
"""
            rows = await self._run_async(query_no_ft, {"kw": keyword})
        return [self._format_node_row(r) for r in rows]

    async def _search_origins(self, keyword: str) -> List[Dict[str, Any]]:
        query = f"""
// exact city/region
MATCH (o:Origin)
WHERE o.city = $kw OR o.region = $kw
RETURN o AS node, {SCORE_EXACT} AS s, 'exact' AS t
UNION
// fulltext (if exists)
CALL {{
  WITH $kw AS kw
  CALL {{
    WITH kw
    CALL db.index.fulltext.queryNodes('{FULLTEXT_ORIGIN_INDEX}', kw) YIELD node, score
    RETURN node, score AS s, 'fulltext' AS t
  }}
  RETURN node, s, t
}}
UNION
// contains
MATCH (o:Origin)
WHERE o.city CONTAINS $kw OR o.region CONTAINS $kw
RETURN o AS node, {SCORE_CONTAINS} AS s, 'contains' AS t
        """
        try:
            rows = await self._run_async(query, {"kw": keyword})
        except ClientError:
            query_no_ft = """
MATCH (o:Origin)
WHERE o.city = $kw OR o.region = $kw
RETURN o AS node, 2.0 AS s, 'exact' AS t
UNION
MATCH (o:Origin)
WHERE o.city CONTAINS $kw OR o.region CONTAINS $kw
RETURN o AS node, 1.0 AS s, 'contains' AS t
"""
            rows = await self._run_async(query_no_ft, {"kw": keyword})
        # 강제 타입 명시
        formatted = []
        for r in rows:
            node = r["node"]
            rec = self._format_node_row(r)
            rec["type"] = "Origin"
            # product가 없으니 랭킹 시 동일키 충돌 방지용 식별자 부여
            rec["properties"].setdefault("region", rec["properties"].get("region", ""))
            formatted.append(rec)
        return formatted

    async def _search_by_category(self, category: str) -> List[Dict[str, Any]]:
        q = """
MATCH (n:농산물)
WHERE n.category = $c OR n.category CONTAINS $c
RETURN n AS node, $score AS s, 'category' AS t
"""
        rows = await self._run_async(q, {"c": category, "score": SCORE_CATEGORY})
        return [self._format_node_row(r, forced_type="농산물") for r in rows]

    async def _search_by_fish_state(self, fish_state: str) -> List[Dict[str, Any]]:
        q = """
MATCH (n:수산물)
WHERE n.fishState = $fs OR n.fishState CONTAINS $fs
RETURN n AS node, $score AS s, 'fish_state' AS t
"""
        rows = await self._run_async(q, {"fs": fish_state, "score": SCORE_FISHSTATE})
        return [self._format_node_row(r, forced_type="수산물") for r in rows]

    async def _search_by_association(self, assoc: str) -> List[Dict[str, Any]]:
        q = """
MATCH (i:수산물)-[r:isFrom]->(o:Origin)
WHERE r.association CONTAINS $a
RETURN DISTINCT i AS node, $score AS s, 'association' AS t, r.association AS found_association
"""
        rows = await self._run_async(q, {"a": assoc, "score": SCORE_ASSOC})
        out = []
        for r in rows:
            rec = self._format_node_row(r, forced_type="수산물")
            if r.get("found_association"):
                rec["properties"]["found_association"] = r["found_association"]
            out.append(rec)
        return out

    # ---------- Relationships ----------
    async def _fetch_relationships(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        products = [n["properties"].get("product") for n in nodes if n["properties"].get("product")]
        if not products:
            return []
        q = """
MATCH (i:농산물|수산물|축산물)-[r:isFrom]->(o:Origin)
WHERE i.product IN $names
RETURN i.product AS product,
       labels(i)[0] AS ingredient_type,
       COALESCE(o.city, o.region) AS location,
       o.region AS region,
       r.farm AS farm,
       r.count AS count,
       r.association AS association,
       r.sold AS sold
ORDER BY i.product, r.farm DESC
"""
        rows = await self._run_async(q, {"names": products})
        rels = []
        for r in rows:
            end_loc = r["location"]
            if r.get("region"):
                end_loc = f"{end_loc}({r['region']})" if end_loc else r["region"]
            item = {
                "start": r["product"],
                "end": end_loc,
                "type": "isFrom",
                "ingredient_type": r["ingredient_type"],
            }
            if r.get("farm") is not None:
                item["farm"] = r["farm"]
            if r.get("count") is not None:
                item["count"] = r["count"]
            if r.get("association"):
                item["association"] = r["association"]
            if r.get("sold"):
                item["sold"] = r["sold"]
            rels.append(item)
        return rels

    async def _fetch_nutrients(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """데이터에 영양소가 있을 때만 의미 있음. 없으면 빈 리스트 반환."""
        products = [n["properties"].get("product") for n in nodes if n["properties"].get("product")]
        if not products:
            return []
        q = """
MATCH (i:농산물|수산물|축산물)-[r:hasNutrient]->(n:Nutrient)
WHERE i.product IN $names
RETURN i.product AS product, n.name AS nutrient, r.amount AS amount, r.unit AS unit
ORDER BY i.product, n.name
"""
        try:
            rows = await self._run_async(q, {"names": products})
        except ClientError:
            return []
        out = []
        for r in rows:
            out.append({
                "product": r["product"],
                "nutrient": r["nutrient"],
                "amount": r.get("amount"),
                "unit": r.get("unit"),
            })
        return out

    # ---------- Helpers ----------
    async def _run_async(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.client.run, query, params or {})

    def _format_node_row(self, row: Dict[str, Any], forced_type: Optional[str] = None) -> Dict[str, Any]:
        node = row["node"]
        props = dict(node)  # neo4j.Node -> dict
        labels = list(node.labels)
        ntype = forced_type or (labels[0] if labels else "Item")
        return {
            "type": ntype,
            "properties": props,
            "score": float(row.get("s", 0)),
            "search_type": row.get("t", "unknown"),
        }

    def _dedupe_and_rank(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        bucket: Dict[str, Dict[str, Any]] = {}
        for n in nodes:
            props = n.get("properties", {})
            product = props.get("product") or props.get("city") or props.get("region") or json.dumps(props, ensure_ascii=False)
            key = f"{n.get('type','')}::{product}"
            if key not in bucket or n.get("score", 0) > bucket[key].get("score", 0):
                bucket[key] = n
        out = list(bucket.values())
        out.sort(key=lambda x: x.get("score", 0), reverse=True)
        return out

    def _format_report(self, q: str, nodes: List[Dict[str, Any]], rels: List[Dict[str, Any]], nutrients: List[Dict[str, Any]]) -> str:
        if not nodes and not rels and not nutrients:
            return f"'{q}'에 대한 그래프 검색 결과를 찾지 못했습니다."

        lines = []
        lines.append(f"Neo4j Graph 검색 결과 ('{q}')")
        lines.append(f"- 항목 {len(nodes)}개, 생산지 관계 {len(rels)}개, 영양소 관계 {len(nutrients)}개\n")

        if nodes:
            lines.append("검색된 항목:")
            for i, n in enumerate(nodes, 1):
                t = n["type"]
                p = n["properties"]
                score = n.get("score", 0.0)
                st = n.get("search_type", "unknown")

                if t == "농산물":
                    name = p.get("product", "N/A")
                    cat = p.get("category", "")
                    info = f"농산물: {name}" + (f" (분류: {cat})" if cat else "")
                elif t == "수산물":
                    name = p.get("product", "N/A")
                    fs = p.get("fishState", "")
                    info = f"수산물: {name}" + (f" (상태: {fs})" if fs else "")
                    if p.get("found_association"):
                        info += f" (수협: {p['found_association']})"
                elif t == "축산물":
                    name = p.get("product", "N/A")
                    info = f"축산물: {name}"
                elif t == "Origin":
                    city = p.get("city", "")
                    region = p.get("region", "")
                    info = f"지역: {city} ({region})" if city and city != "." else f"지역: {region or city}"
                else:
                    info = f"{t}: {next(iter(p.values()), 'N/A')}"

                lines.append(f"{i:2d}. {info} [매칭도 {score:.2f}, 검색타입: {st}]")
            lines.append("")

        if rels:
            lines.append(f"생산지 연관관계 ({len(rels)}):")
            for i, r in enumerate(rels, 1):
                detail = []
                if "farm" in r and r["farm"] is not None:
                    detail.append(f"농장수 {r['farm']}개")
                if "count" in r and r["count"] is not None:
                    detail.append(f"사육/양식수 {r['count']:,}")
                if "association" in r and r["association"]:
                    detail.append(f"수협 {r['association']}")
                if "sold" in r and r["sold"]:
                    detail.append(f"위판장 {r['sold']}")
                suffix = f" ({', '.join(detail)})" if detail else ""
                lines.append(f"{i:2d}. {r['start']} → {r['end']}{suffix}")
            lines.append("")

        if nutrients:
            lines.append(f"영양소 관계 ({len(nutrients)}):")
            for i, n in enumerate(nutrients, 1):
                part = f"{i:2d}. {n['product']} - {n['nutrient']}"
                if n.get("amount") is not None and n.get("unit"):
                    part += f" ({n['amount']}{n['unit']})"
                lines.append(part)

        return "\n".join(lines)


# =========================
# Query Optimizer (Graph-friendly phrase)
# =========================
class GraphQueryOptimizer:
    """
    사용자 자연어 질문을 그래프 DB 검색에 유리한 '관계 중심 문구'로 단순화.
    예: "사과의 원산지", "오렌지의 영양소", "오렌지의 원산지와 영양소"
    """
    def __init__(self, llm: Optional[LLM] = None):
        self.llm = llm or LLM()

    async def optimize(self, user_query: str) -> str:
        prompt = f"""
다음 사용자 질문을 Graph DB 검색에 가장 효과적인 핵심 관계 중심 문구로 바꾸세요.
- 그래프 데이터 특성: (품목)-[:isFrom]->(원산지), (품목)-[:hasNutrient]->(영양소)
- 꼭 필요한 엔티티와 관계만 남기고, 한국어로 간결히.
- 예시: "사과의 원산지", "오렌지의 영양소", "오렌지의 원산지와 영양소"
- 불필요한 수식/설명 금지. 결과만 한 줄로.

질문: "{user_query}"
답변:
"""
        try:
            txt = await self.llm.ainvoke(prompt)
            return txt.strip().replace("\n", " ")
        except Exception as e:
            _debug(f"optimize failed: {e}")
            # 실패 시 원문 반납
            return user_query


# =========================
# Public Entrypoints
# =========================
def neo4j_search_sync(query: str) -> str:
    """스레드/프로세스 어디서나 호출 가능한 동기 진입점"""
    svc = GraphDBSearchService()
    try:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(svc.search(query))
        finally:
            loop.close()
    except Exception as e:
        _debug(f"sync search error: {e}")
        return f"Neo4j 동기 검색 오류: {e}"
    finally:
        svc.close()


async def neo4j_graph_search(query: str) -> str:
    """에이전트에서 직접 await할 수 있는 비동기 진입점"""
    svc = GraphDBSearchService()
    try:
        return await svc.search(query)
    finally:
        svc.close()

