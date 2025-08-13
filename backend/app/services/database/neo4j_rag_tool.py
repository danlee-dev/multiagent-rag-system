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
from neo4j.exceptions import ClientError
from langchain_google_genai import ChatGoogleGenerativeAI


# =========================
# Config & Constants
# =========================
DEFAULT_NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
DEFAULT_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# 단일 풀텍스트 인덱스 이름
FULLTEXT_UNIFIED_INDEX = "product_idx"  # 품목/국가/지역/영양소/기업/문서 등 통합 검색


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
# Search Service (single-query)
# =========================
class GraphDBSearchService:
    """
    - 키워드 추출(LLM + 폴백) : 구분 없이 단일 리스트
    - DB 호출은 단 하나의 쿼리 형태만 사용:
        CALL db.index.fulltext.queryNodes('product_idx', $kw)
        YIELD node AS n, score
        MATCH (n)-[r0]-(m)
        RETURN n, r, m, score
        ORDER BY score DESC
    - 관계 가공:
        * isFrom      -> 원산지 정보 (r.count => 농장수)
        * hasNutrient -> 영양성분 정보 (r.value[+unit] => 양)
        * relation    -> 문서 정보 (r.type => 문서 타입)
    """

    SINGLE_QUERY = f"""
CALL db.index.fulltext.queryNodes('{FULLTEXT_UNIFIED_INDEX}', $kw)
YIELD node AS n, score
MATCH (n)-[r]-(m)
RETURN n, r, m, score
ORDER BY score DESC
"""

    def __init__(self, client: Optional[GraphDBClient] = None, llm: Optional[LLM] = None):
        self.client = client or GraphDBClient()
        self.llm = llm or LLM()

    # ---------- Public ----------
    async def search(self, user_query: str) -> str:
        _debug("Search start (single-query)")
        kw = await self._extract_keywords(user_query)
        keywords = kw.get("keywords", [])
        if not keywords:
            return f"'{user_query}'에 사용할 키워드를 찾지 못했습니다."

        # 키워드를 Lucene OR 쿼리로 결합하여 한 번만 질의
        lucene = self._build_fulltext_query(keywords)
        rows = await self._run_single_query(lucene)

        # 누적 컨테이너
        node_bucket: Dict[str, Dict[str, Any]] = {}  # elementId -> node dict (최대 score로 유지)
        isfrom_list: List[Dict[str, Any]] = []
        nutrients_list: List[Dict[str, Any]] = []
        docrels_list: List[Dict[str, Any]] = []

        # 결과 파싱/병합
        self._parse_rows(rows, node_bucket, isfrom_list, nutrients_list, docrels_list)

        # 노드 정렬 (score desc)
        nodes_sorted = sorted(node_bucket.values(), key=lambda x: x.get("score", 0.0), reverse=True)

        # 관계 중복 제거
        def uniq(items, keyfunc):
            seen = set(); out = []
            for it in items:
                k = keyfunc(it)
                if k not in seen:
                    seen.add(k); out.append(it)
            return out

        isfrom_list    = uniq(isfrom_list,    lambda r: (r.get("item"), r.get("origin"), r.get("count")))
        nutrients_list = uniq(nutrients_list, lambda r: (r.get("item"), r.get("nutrient"), r.get("value"), r.get("unit")))
        docrels_list   = uniq(docrels_list,   lambda r: (r.get("source"), r.get("target"), r.get("rel_type")))

        report = self._format_report(user_query, nodes_sorted, isfrom_list, nutrients_list, docrels_list)
        _debug(f"Search done: nodes={len(nodes_sorted)} isFrom={len(isfrom_list)} nutrients={len(nutrients_list)} docs={len(docrels_list)}")
        return report

    def close(self):
        self.client.close()

    # ---------- Keyword Extraction ----------
    async def _extract_keywords(self, q: str) -> Dict[str, List[str]]:
        prompt = f"""
다음 질문에서 그래프 검색에 사용할 핵심 키워드를 JSON으로 뽑아주세요.
- 품목/국가/지역/영양성분/기업/조직 등 구분하지 말고 한 리스트에 담기
- 반드시 JSON만 출력

질문: "{q}"

JSON 예:
{{ "keywords": ["사과","제주시","비타민C","통영수협"] }}
"""
        try:
            txt = await self.llm.ainvoke(prompt)
            m = re.search(r"\{[\s\S]*\}", txt)
            if m:
                txt = m.group(0)
            data = json.loads(txt)
            _debug(f"LLM keywords: {data}")
            if "keywords" not in data or not isinstance(data["keywords"], list):
                data["keywords"] = []
            return data
        except Exception as e:
            _debug(f"LLM keyword extraction failed: {e}")
            return self._keyword_fallback(q)

    def _keyword_fallback(self, q: str) -> Dict[str, List[str]]:
        stop = {"의","을","를","이","가","에","에서","알려줘","검색","찾아","정보","어디","무엇","언제","어떤"}
        words = re.sub(r"[^\w\s]", " ", q).split()
        terms = [w for w in words if len(w) > 1 and w not in stop]
        return {"keywords": terms[:5]}

    # ---------- Fulltext OR query builder ----------
    def _build_fulltext_query(self, keywords: List[str]) -> str:
        def esc(t: str) -> str:
            # Lucene 특수문자 최소 이스케이프: backslash와 큰따옴표
            t = t.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{t}"'
        terms = [esc(k) for k in keywords if k]
        return " OR ".join(terms) if terms else ""

    # ---------- Single-query runner ----------
    async def _run_single_query(self, lucene_query: str) -> List[Dict[str, Any]]:
        return await self._run_async(self.SINGLE_QUERY, {"kw": lucene_query})

    # ---------- Row parsing ----------
    def _parse_rows(
        self,
        rows: List[Dict[str, Any]],
        node_bucket: Dict[str, Dict[str, Any]],
        isfrom_list: List[Dict[str, Any]],
        nutrients_list: List[Dict[str, Any]],
        docrels_list: List[Dict[str, Any]],
    ):
        for r in rows:
            n = r["n"]; m = r["m"]; rel = r["r"]; score = float(r.get("score", 0.0))

            # n 노드만 점수 반영
            n_fmt = self._format_node(n, score)
            node_bucket[n_fmt["id"]] = self._keep_max_score(node_bucket.get(n_fmt["id"]), n_fmt)

            # 관계 분기
            rel_type_label = getattr(rel, "type", None) or ""   # isFrom / hasNutrient / relation ...
            rel_props = dict(rel)  # count, value, unit, type(문서타입) 등

            def has_label(node, label) -> bool:
                try:
                    return label in list(node.labels)
                except Exception:
                    return False

            def node_display(node) -> str:
                p = dict(node)
                labels = list(node.labels)
                if "Origin" in labels:
                    city = p.get("city",""); region = p.get("region","")
                    return f"{city} ({region})" if city and city != "." else (region or city or "N/A")
                for k in ("product","name","title","city","region","id"):
                    if p.get(k): return str(p[k])
                return "N/A"

            if rel_type_label == "isFrom":
                origin_node, item_node = (n, m) if has_label(n, "Origin") else ((m, n) if has_label(m, "Origin") else (None, None))
                if origin_node and item_node:
                    isfrom_list.append({
                        "item": node_display(item_node),
                        "origin": node_display(origin_node),
                        "count": rel_props.get("count"),   # 농장수
                    })

            elif rel_type_label == "hasNutrient":
                nut_node, item_node = (n, m) if has_label(n, "Nutrient") else ((m, n) if has_label(m, "Nutrient") else (None, None))
                if nut_node and item_node:
                    nutrients_list.append({
                        "item": node_display(item_node),
                        "nutrient": dict(nut_node).get("name") or node_display(nut_node),
                        "value": rel_props.get("value"),
                        "unit": rel_props.get("unit"),
                    })

            elif rel_type_label == "relation":
                doc_node, src_node = (n, m) if has_label(n, "Entity") else ((m, n) if has_label(m, "Entity") else (m, n))
                docrels_list.append({
                    "source": node_display(src_node),
                    "target": node_display(doc_node),
                    "rel_type": rel_props.get("type"),  # 문서 타입
                })

            # 기타 관계 타입은 출력 생략

    # ---------- Helpers ----------
    async def _run_async(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.client.run, query, params or {})

    def _format_node(self, node, score: float) -> Dict[str, Any]:
        props = dict(node)
        labels = list(node.labels)
        # elementId 추출
        eid = None
        for attr in ("element_id", "id", "identity"):
            if hasattr(node, attr):
                try:
                    val = getattr(node, attr)
                    eid = str(val) if val is not None else None
                    if eid: break
                except Exception:
                    continue
        ntype = labels[0] if labels else "Node"
        return {
            "id": eid or json.dumps(props, ensure_ascii=False),
            "type": ntype,
            "labels": labels,
            "properties": props,
            "score": score,
            "search_type": "fulltext",  # 본 쿼리는 항상 fulltext
        }

    def _keep_max_score(self, old: Optional[Dict[str, Any]], new: Dict[str, Any]) -> Dict[str, Any]:
        if not old: return new
        return new if new.get("score", 0.0) > old.get("score", 0.0) else old

    def _node_display_for_report(self, n: Dict[str, Any]) -> str:
        t = n.get("type", "")
        p = n.get("properties", {})
        if t in ("농산물","수산물","축산물"):
            return p.get("product", "N/A")
        if t == "Origin":
            city = p.get("city", ""); region = p.get("region", "")
            return f"{city} ({region})" if city and city != "." else (region or city or "N/A")
        for k in ("name","title","product","city","region","id"):
            if p.get(k):
                return str(p[k])
        return "N/A"

    def _format_report(
        self,
        q: str,
        nodes: List[Dict[str, Any]],
        isfrom: List[Dict[str, Any]],
        nutrients: List[Dict[str, Any]],
        docrels: List[Dict[str, Any]],
    ) -> str:
        if not nodes and not isfrom and not nutrients and not docrels:
            return f"'{q}'에 대한 그래프 검색 결과를 찾지 못했습니다."

        lines = []
        lines.append(f"Neo4j Graph 검색 결과 ('{q}')")
        lines.append(f"- 항목 {len(nodes)}개, 원산지 관계 {len(isfrom)}개, 영양성분 관계 {len(nutrients)}개, 문서 관계 {len(docrels)}개\n")

        if nodes:
            lines.append("검색된 항목:")
            for i, n in enumerate(nodes, 1):
                name = self._node_display_for_report(n)
                score = n.get("score", 0.0)
                lines.append(f"{i:2d}. {n['type']}: {name} [매칭도 {score:.2f}, 검색타입: fulltext]")
            lines.append("")

        if isfrom:
            lines.append(f"원산지 정보 (isFrom) — {len(isfrom)}건:")
            for i, r in enumerate(isfrom, 1):
                suffix = f" (농장수 {int(r['count'])}개)" if r.get("count") is not None else ""
                lines.append(f"{i:2d}. {r['item']} → {r['origin']}{suffix}")
            lines.append("")

        if nutrients:
            lines.append(f"영양성분 정보 (hasNutrient) — {len(nutrients)}건:")
            for i, n in enumerate(nutrients, 1):
                amount = ""
                if n.get("value") is not None:
                    amount = f" (양: {n['value']}{n.get('unit','') or ''})"
                lines.append(f"{i:2d}. {n['item']} - {n['nutrient']}{amount}")
            lines.append("")

        if docrels:
            lines.append(f"문서 정보 (relation) — {len(docrels)}건:")
            for i, d in enumerate(docrels, 1):
                rtype = f" (type: {d['rel_type']})" if d.get("rel_type") else ""
                lines.append(f"{i:2d}. {d['source']} - {d['target']}{rtype}")

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
- 그래프 데이터 특성: (품목)-[:isFrom]->(원산지), (품목)-[:hasNutrient]->(영양소), (노드)-[:relation]->(문서)
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
