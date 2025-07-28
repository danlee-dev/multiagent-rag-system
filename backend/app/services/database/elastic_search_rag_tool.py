# -*- coding: utf-8 -*-
"""
멀티 인덱스 RAG 검색 엔진 (JSON 결과 버전)
- multi_index_search_engine.py의 모든 로직을 포함
- 결과를 HTML이 아닌 VectorDB.json 파일로 저장
- rag_config.py만 있으면 동작
"""
import sys
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from elasticsearch import Elasticsearch
import openai
from datetime import datetime
import re
from ...core.config.rag_config import RAGConfig

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stdin, 'reconfigure'):
    sys.stdin.reconfigure(encoding='utf-8')

class MultiIndexRAGSearchEngine:
    def __init__(self, openai_api_key: str = None, es_host: str = None, es_user: str = None, es_password: str = None, config: RAGConfig = None):
        if config is None:
            config = RAGConfig()
        api_key = openai_api_key or config.OPENAI_API_KEY
        if api_key == "your-openai-api-key-here":
            raise ValueError("OpenAI API 키를 설정해주세요 (rag_config.py 또는 초기화 파라미터)")
        self.client = openai.OpenAI(api_key=api_key)
        self.es = Elasticsearch(
            es_host or config.ELASTICSEARCH_HOST,
            basic_auth=(es_user or config.ELASTICSEARCH_USER, es_password or config.ELASTICSEARCH_PASSWORD)
        )
        self.TEXT_INDEX = "documents_text"
        self.TABLE_INDEX = "documents_table"
        self.config = config
        self.HYBRID_ALPHA = config.HYBRID_ALPHA
        self.TOP_K_RETRIEVAL = config.TOP_K_RETRIEVAL
        self.TOP_K_RERANK = config.TOP_K_RERANK
        self.TOP_K_FINAL = config.TOP_K_FINAL
        self.USE_HYDE = config.USE_HYDE
        self.USE_RERANKING = config.USE_RERANKING
        self.USE_SUMMARIZATION = config.USE_SUMMARIZATION
        self.HYDE_MAX_TOKENS = config.HYDE_MAX_TOKENS
        self.HYDE_TEMPERATURE = config.HYDE_TEMPERATURE
        self.HYDE_MODEL = config.HYDE_MODEL
        self.SUMMARIZATION_MAX_TOKENS = config.SUMMARIZATION_MAX_TOKENS
        self.SUMMARIZATION_RATIO = config.SUMMARIZATION_RATIO
        self.DOMAIN_KEYWORDS = config.DOMAIN_KEYWORDS

    def embed_text(self, text: str) -> List[float]:
        try:
            safe_text = text.encode('utf-8', errors='ignore').decode('utf-8')
            response = self.client.embeddings.create(
                input=safe_text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"임베딩 생성 오류: {e}")
            return []

    def query_enhancement_hyde_text(self, query: str) -> str:
        try:
            prompt = f"""
식품 도메인에 관련된 답변을 작성해주세요.
다음 질문에 대한 상세하고 정확한 답변을 작성해주세요.
이 답변은 검색을 개선하기 위한 것이므로 가능한 한 구체적이고 전문적으로 작성해주세요.
마크다운 문법을 사용하지 않고 줄글로 써주세요.
최대 {self.HYDE_MAX_TOKENS} 토큰 이내로 답변을 작성하세요.

질문: {query}
답변:"""
            response = self.client.chat.completions.create(
                model=self.HYDE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.HYDE_MAX_TOKENS,
                temperature=self.HYDE_TEMPERATURE
            )
            hypothetical_doc = response.choices[0].message.content.strip()
            enhanced_query = f"{query} {hypothetical_doc}"
            return enhanced_query
        except Exception as e:
            print(f"HyDE(text) 처리 중 오류: {e}")
            return query

    def query_enhancement_hyde_table(self, query: str) -> str:
        try:
            prompt = f"""
아래 질문에 대해 표 형식의 가상 통계 데이터를 만들어주세요.
- 첫 줄에는 표 제목을 써주세요.
- 두 번째 줄에는 열 이름(헤더)을 공백으로 구분해서 써주세요.
- 그 아래에는 각 행의 데이터를 공백으로 구분해서 **최소 10줄 이상** 써주세요.
- 표는 마크다운, HTML, 파이프(|) 없이, 전처리된 텍스트 표 형식(공백 구분)으로만 작성해주세요.
- 가능한 한 많은 정보를 포함하고, 각 행의 내용도 구체적으로 작성해주세요.

질문: {query}

예시:
표제목: 2024년 식품 통계 주요 항목
항목명 수치 단위 비고
쌀 생산량 370만 톤 전국 기준
밀 수입량 250만 톤 주요 5개국
외식업 매출 120조 원 2024년 기준
식품 수출액 80억 달러 전년 대비 5% 증가

이와 같은 형식으로 답변해주세요.
"""
            response = self.client.chat.completions.create(
                model=self.HYDE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.HYDE_MAX_TOKENS,
                temperature=self.HYDE_TEMPERATURE
            )
            hypothetical_doc = response.choices[0].message.content.strip()
            enhanced_query = f"{query} {hypothetical_doc}"
            return enhanced_query
        except Exception as e:
            print(f"HyDE(table) 처리 중 오류: {e}")
            return query

    def dense_retrieval(self, query: str, top_k: int = 100) -> List[Dict]:
        vector = self.embed_text(query)
        if not vector:
            return []
        results = []
        for index in [self.TEXT_INDEX, self.TABLE_INDEX]:
            body = {
                "size": top_k,
                "knn": {
                    "field": "embedding",
                    "query_vector": vector,
                    "k": top_k,
                    "num_candidates": min(top_k * 2, 200)
                },
                "_source": ["page_content", "name", "meta_data"]
            }
            try:
                response = self.es.search(index=index, body=body)
                hits = response.get("hits", {}).get("hits", [])
                for hit in hits:
                    source = hit["_source"]
                    results.append({
                        "score": hit["_score"],
                        "page_content": source.get("page_content", ""),
                        "name": source.get("name", ""),
                        "meta_data": source.get("meta_data", {}),
                        "search_type": "dense",
                        "_index": index
                    })
            except Exception as e:
                print(f"Dense 검색 오류({index}): {e}")
        return results

    def sparse_retrieval(self, query: str, top_k: int = 100) -> List[Dict]:
        results = []
        for index in [self.TEXT_INDEX, self.TABLE_INDEX]:
            search_query = {
                "size": top_k,
                "query": {
                    "bool": {
                        "should": [
                            {"match_phrase": {"page_content": {"query": query, "boost": 5.0}}},
                            {"match": {"name": {"query": query, "boost": 4.0}}},
                            {"match": {"page_content": {"query": query, "boost": 2.0}}},
                            {"match": {"page_content.ngram": {"query": query, "boost": 1.0}}}
                        ],
                        "minimum_should_match": 1
                    }
                },
                "_source": ["page_content", "name", "meta_data"]
            }
            try:
                response = self.es.search(index=index, body=search_query)
                hits = response.get("hits", {}).get("hits", [])
                for hit in hits:
                    source = hit["_source"]
                    results.append({
                        "score": hit["_score"],
                        "page_content": source.get("page_content", ""),
                        "name": source.get("name", ""),
                        "meta_data": source.get("meta_data", {}),
                        "search_type": "sparse",
                        "_index": index
                    })
            except Exception as e:
                print(f"Sparse 검색 오류({index}): {e}")
        return results

    def normalize_scores(self, results: List[Dict], score_field: str = "score") -> List[Dict]:
        if not results:
            return results
        scores = [r[score_field] for r in results]
        if not scores or max(scores) == min(scores):
            return results
        min_score, max_score = min(scores), max(scores)
        score_range = max_score - min_score
        for result in results:
            original_score = result[score_field]
            normalized_score = (original_score - min_score) / score_range
            result[f"normalized_{score_field}"] = normalized_score
        return results

    def hybrid_search(self, query: str, alpha: float = None, top_k: int = 100, enhanced_query_text: str = None, enhanced_query_table: str = None) -> List[Dict]:
        if alpha is None:
            alpha = self.HYBRID_ALPHA
        # HyDE 쿼리 분리
        if self.USE_HYDE and enhanced_query_text and enhanced_query_table:
            pass  # HyDE 쿼리 개선 메시지 생략
        else:
            enhanced_query_text = query
            enhanced_query_table = query
        dense_results = []
        sparse_results = []
        # text 인덱스
        dense_results += self.dense_retrieval_index(enhanced_query_text, self.TEXT_INDEX, top_k)
        sparse_results += self.sparse_retrieval_index(enhanced_query_text, self.TEXT_INDEX, top_k)
        # table 인덱스
        dense_results += self.dense_retrieval_index(enhanced_query_table, self.TABLE_INDEX, top_k)
        sparse_results += self.sparse_retrieval_index(enhanced_query_table, self.TABLE_INDEX, top_k)
        dense_results = self.normalize_scores(dense_results)
        sparse_results = self.normalize_scores(sparse_results)
        doc_scores = {}
        for result in dense_results:
            doc_id = self._get_doc_id(result)
            dense_score = result.get("normalized_score", 0)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "doc": result,
                    "dense_score": dense_score,
                    "sparse_score": 0.0
                }
            else:
                doc_scores[doc_id]["dense_score"] = dense_score
        for result in sparse_results:
            doc_id = self._get_doc_id(result)
            sparse_score = result.get("normalized_score", 0)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "doc": result,
                    "dense_score": 0.0,
                    "sparse_score": sparse_score
                }
            else:
                doc_scores[doc_id]["sparse_score"] = sparse_score
        hybrid_results = []
        for doc_id, scores in doc_scores.items():
            hybrid_score = alpha * scores["sparse_score"] + (1 - alpha) * scores["dense_score"]
            result = scores["doc"].copy()
            result["hybrid_score"] = hybrid_score
            result["dense_component"] = scores["dense_score"]
            result["sparse_component"] = scores["sparse_score"]
            result["search_type"] = "hybrid"
            hybrid_results.append(result)
        hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return hybrid_results[:top_k]

    def dense_retrieval_index(self, query: str, index: str, top_k: int = 100) -> List[Dict]:
        vector = self.embed_text(query)
        if not vector:
            return []
        results = []
        body = {
            "size": top_k,
            "knn": {
                "field": "embedding",
                "query_vector": vector,
                "k": top_k,
                "num_candidates": min(top_k * 2, 200)
            },
            "_source": ["page_content", "name", "meta_data"]
        }
        try:
            response = self.es.search(index=index, body=body)
            hits = response.get("hits", {}).get("hits", [])
            for hit in hits:
                source = hit["_source"]
                results.append({
                    "score": hit["_score"],
                    "page_content": source.get("page_content", ""),
                    "name": source.get("name", ""),
                    "meta_data": source.get("meta_data", {}),
                    "search_type": "dense",
                    "_index": index
                })
        except Exception as e:
            print(f"Dense 검색 오류({index}): {e}")
        return results

    def sparse_retrieval_index(self, query: str, index: str, top_k: int = 100) -> List[Dict]:
        results = []
        search_query = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {"match_phrase": {"page_content": {"query": query, "boost": 5.0}}},
                        {"match": {"name": {"query": query, "boost": 4.0}}},
                        {"match": {"page_content": {"query": query, "boost": 2.0}}},
                        {"match": {"page_content.ngram": {"query": query, "boost": 1.0}}}
                    ],
                    "minimum_should_match": 1
                }
            },
            "_source": ["page_content", "name", "meta_data"]
        }
        try:
            response = self.es.search(index=index, body=search_query)
            hits = response.get("hits", {}).get("hits", [])
            for hit in hits:
                source = hit["_source"]
                results.append({
                    "score": hit["_score"],
                    "page_content": source.get("page_content", ""),
                    "name": source.get("name", ""),
                    "meta_data": source.get("meta_data", {}),
                    "search_type": "sparse",
                    "_index": index
                })
        except Exception as e:
            print(f"Sparse 검색 오류({index}): {e}")
        return results

    def _get_doc_id(self, result: Dict) -> str:
        meta_data = result.get("meta_data", {})
        chunk_id = meta_data.get("chunk_id", "")
        name = result.get("name", "")
        page_content_hash = str(hash(result.get("page_content", "")))
        return f"{chunk_id}_{name}_{page_content_hash}"

    def simple_reranking(self, results: List[Dict], query: str, top_k: int = 20) -> List[Dict]:
        reranked_results = []
        for result in results[:top_k]:
            page_content = result.get("page_content", "")
            name = result.get("name", "")
            query_terms = query.lower().split()
            content_lower = page_content.lower()
            name_lower = name.lower()
            content_matches = sum(content_lower.count(term) for term in query_terms)
            name_matches = sum(name_lower.count(term) for term in query_terms)
            original_score = result.get("hybrid_score", result.get("score", 0))
            rerank_bonus = (content_matches * 0.1) + (name_matches * 0.2)
            rerank_score = original_score + rerank_bonus
            result["rerank_score"] = rerank_score
            result["rerank_bonus"] = rerank_bonus
            reranked_results.append(result)
        reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked_results

    def document_summarization(self, results: List[Dict], query: str) -> List[Dict]:
        summarized_results = []
        for result in results:
            page_content = result.get("page_content", "")
            if len(page_content) > 500:
                try:
                    prompt = f"""
다음 문서를 주어진 질문과 관련된 핵심 내용 위주로 요약해주세요.\n요약 길이: 원본의 {self.SUMMARIZATION_RATIO}% 정도\n질문: {query}\n문서 내용:\n{page_content}\n요약:"""
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=self.SUMMARIZATION_MAX_TOKENS,
                        temperature=0.3
                    )
                    summary = response.choices[0].message.content.strip()
                    result["summarized_content"] = summary
                    result["original_length"] = len(page_content)
                    result["summary_length"] = len(summary)
                except Exception as e:
                    print(f"요약 처리 오류: {e}")
                    result["summarized_content"] = page_content
            else:
                result["summarized_content"] = page_content
            summarized_results.append(result)
        return summarized_results

    def advanced_rag_search(self, query: str) -> Dict:
        start_time = datetime.now()
        # HyDE 쿼리 분리
        if self.USE_HYDE:
            enhanced_query_text = self.query_enhancement_hyde_text(query)
            enhanced_query_table = self.query_enhancement_hyde_table(query)
        else:
            enhanced_query_text = query
            enhanced_query_table = query
        hybrid_results = self.hybrid_search(query, top_k=self.TOP_K_RETRIEVAL, enhanced_query_text=enhanced_query_text, enhanced_query_table=enhanced_query_table)
        reranked_results = hybrid_results
        if self.USE_RERANKING and len(hybrid_results) > 5:
            reranked_results = self.simple_reranking(
                hybrid_results, query, top_k=self.TOP_K_RERANK
            )
        final_results = reranked_results[:self.TOP_K_FINAL]
        if self.USE_SUMMARIZATION:
            final_results = self.document_summarization(final_results, query)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        query_for_result = query if query else (enhanced_query_text if enhanced_query_text else "N/A")
        return {
            "query": query_for_result,
            "enhanced_query": {
                "text": enhanced_query_text,
                "table": enhanced_query_table
            },
            "results": final_results,
            "total_candidates": len(hybrid_results),
            "final_count": len(final_results),
            "processing_time": duration,
            "config": {
                "hybrid_alpha": self.HYBRID_ALPHA,
                "use_hyde": self.USE_HYDE,
                "use_reranking": self.USE_RERANKING,
                "use_summarization": self.USE_SUMMARIZATION
            }
        }

def search(query: str, top_k: int = 50):
    """
    주어진 쿼리(query)로 멀티 인덱스 RAG 검색을 수행하고,
    상위 top_k개의 결과를 VectorDB.json 파일로 저장하며,
    결과를 json 객체로 반환합니다.
    """
    config = RAGConfig()
    if config.OPENAI_API_KEY == "your-openai-api-key-here":
        print("⚠️ rag_config.py에서 OpenAI API 키가 설정되지 않았습니다.")
        api_key = input("OpenAI API 키를 입력하세요 (또는 Enter로 스킵): ").strip()
        if not api_key:
            print("❌ API 키 없이는 테스트할 수 없습니다.")
            return None
    else:
        api_key = config.OPENAI_API_KEY
    search_engine = MultiIndexRAGSearchEngine(openai_api_key=api_key, config=config)
    if not query:
        print("검색어가 입력되지 않았습니다.")
        return None
    print("검색 실행 중...")
    results = search_engine.advanced_rag_search(query)
    # 상위 top_k개만 저장
    top_results = results.get('results', [])[:top_k]
    with open('VectorDB.json', 'w', encoding='utf-8') as f:
        json.dump(top_results, f, ensure_ascii=False, indent=2)
    print(f"검색 결과가 VectorDB.json 파일로 저장되었습니다. (총 {len(top_results)}개)")
    return top_results
