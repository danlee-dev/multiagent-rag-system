"""
Utility module to run Cypher queries as simple Python function calls.

Usage examples
--------------
>>> from neo4j_query import run_cypher
>>> run_cypher("MATCH (n) RETURN count(n) AS nodes")
[{'nodes': 42}]

If you prefer CLI style:
$ python neo4j_query.py "MATCH (n) RETURN n LIMIT 1" '{}'

Configuration
-------------
The driver reads connection details from environment variables, falling back to sensible defaults.
  - NEO4J_URI        (default: bolt://localhost:7687)
  - NEO4J_USER       (default: neo4j)
  - NEO4J_PASSWORD   (default: password)

Install requirements:
    pip install neo4j
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv

from neo4j import GraphDatabase, basic_auth

from pathlib import Path

# ────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────
# 현재 파일의 부모 디렉토리에서 .env 파일 찾기
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# 또는 네가 쓴 방식
# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER: str = os.getenv("NEO4J_USER", "neo4j")
PASSWORD: str = os.getenv("NEO4J_PASSWORD", "#cwp2025!")

print(f"NEO4J_URI: {os.getenv('NEO4J_URI')}")
print(f"NEO4J_USER: {os.getenv('NEO4J_USER')}")
print(f"NEO4J_PASSWORD: {os.getenv('NEO4J_PASSWORD')}")

# Single shared driver instance (thread‑safe)
driver = GraphDatabase.driver(URI, auth=basic_auth(USER, PASSWORD))

# ────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────

def run_cypher(query: str, parameters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """Execute *query* with optional *parameters* and return list of dicts.

    Parameters
    ----------
    query : str
        Cypher statement. Use `$param` placeholders for parameters.
    parameters : dict, optional
        Mapping of parameter names to values.

    Returns
    -------
    list[dict]
        Each record converted to a plain ``dict``.
    """
    with driver.session() as session:
        result = session.run(query, parameters or {})
        return [record.data() for record in result]


def close_driver() -> None:
    """Close the underlying Neo4j driver (optional)."""
    global driver  # noqa: PLW0603
    if driver is not None:
        driver.close()
        driver = None  # type: ignore[assignment]


# ────────────────────────────────────────────
# CLI entry‑point
# ────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python neo4j_query.py '<CYPHER>' '[JSON_PARAMETERS]'", file=sys.stderr)
        sys.exit(1)

    cypher = sys.argv[1]
    params: Dict[str, Any] | None = None
    if len(sys.argv) > 2:
        params = json.loads(sys.argv[2])

    try:
        data = run_cypher(cypher, params)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    finally:
        close_driver()

