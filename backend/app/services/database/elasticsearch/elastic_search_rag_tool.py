# -*- coding: utf-8 -*-
"""
멀티 인덱스 RAG 검색 엔진 (JSON 결과 버전)
- multi_index_search_engine.py의 모든 로직을 포함
- 결과를 HTML이 아닌 VectorDB.json 파일로 저장
- rag_config.py만 있으면 동작
"""
import sys
import json
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from elasticsearch import Elasticsearch
from datetime import datetime
import re
from ....core.config.rag_config import RAGConfig
from sentence_transformers import SentenceTransformer
import cohere

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
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
import google.generativeai as genai
from datetime import datetime
import re
from ....core.config.rag_config import RAGConfig
from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stdin, 'reconfigure'):
    sys.stdin.reconfigure(encoding='utf-8')

print(">> CPU 전용으로 모델 로드...")
hf_model = SentenceTransformer("dragonkue/bge-m3-ko", device="cpu", trust_remote_code=True)
print(">> CPU 모델 로드 성공")

class MultiIndexRAGSearchEngine:
    def __init__(self, google_api_key: str = None, cohere_api_key: str = None, hf_model = hf_model, es_host: str = None, es_user: str = None, es_password: str = None, config: RAGConfig = None):
        if config is None:
            config = RAGConfig()
        api_key = google_api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google API 키를 설정해주세요 (환경변수 GOOGLE_API_KEY)")

        # Cohere API 키 설정
        cohere_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if not cohere_key:
            raise ValueError("Cohere API 키를 설정해주세요 (환경변수 COHERE_API_KEY 또는 초기화 파라미터)")

        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel('gemini-2.5-flash')
        self.cohere_client = cohere.Client(cohere_key)
        self.es = Elasticsearch(
            es_host or config.ELASTICSEARCH_HOST,
            basic_auth=(es_user or config.ELASTICSEARCH_USER, es_password or config.ELASTICSEARCH_PASSWORD)
        )
        # SentenceTransformer 안전한 초기화 (meta 장치 문제 완전 회피)
        
        # if not hf_model:
        #     try:
        #         print(">> SentenceTransformer 모델 로드 시작...")

        #         # 환경변수로 meta 장치 사용 방지
        #         os.environ["TRANSFORMERS_OFFLINE"] = "0"

        #         # 1단계: CPU 전용으로 로드
        #         print(">> CPU 전용으로 모델 로드...")
        #         self.hf_model = SentenceTransformer("dragonkue/bge-m3-ko", device="cpu", trust_remote_code=True)
        #         print(">> CPU 모델 로드 성공")

        #     except Exception as e:
        #         print(f">> 한국어 모델 로드 실패: {e}")
        #         print(">> 영어 fallback 모델 시도...")
        #         try:
        #             # 더 안정적인 영어 모델로 fallback
        #             self.hf_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        #             print(">> Fallback 모델 로드 성공")
        #         except Exception as final_error:
        #             print(f">> 모든 모델 로드 실패: {final_error}")
        #             print(">> 임베딩 기능 비활성화")
        #             self.hf_model = None
        # else:
        self.hf_model = hf_model
            
        self.TEXT_INDEX = "bge_text"
        self.TABLE_INDEX = "bge_table"
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

        # 동의어 사전 로드 (synonym.json)
        syn_path = os.path.join(os.path.dirname(__file__), "synonym.json")
        if os.path.exists(syn_path):
            with open(syn_path, encoding='utf-8') as f:
                self.synonym_dict = json.load(f)
        else:
            self.synonym_dict = {}

    def embed_text(self, text: str) -> List[float]:
        try:
            if self.hf_model is None:
                print(">> 임베딩 모델이 없어 더미 벡터 반환")
                return [0.0] * 1024  # 더미 임베딩 벡터

            safe_text = text.encode('utf-8', errors='ignore').decode('utf-8')
            # Hugging Face 모델로 임베딩 생성
            embedding = self.hf_model.encode(safe_text)
            return embedding.tolist()
        except Exception as e:
            print(f"임베딩 생성 오류: {e}")
            return [0.0] * 1024  # 오류 시 더미 벡터 반환

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
            response = self.client.generate_content(prompt)
            hypothetical_doc = response.text.strip()
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
            response = self.client.generate_content(prompt)
            hypothetical_doc = response.text.strip()
            enhanced_query = f"{query} {hypothetical_doc}"
            return enhanced_query
        except Exception as e:
            print(f"HyDE(table) 처리 중 오류: {e}")
            return query

    # ========== 동의어 확장 헬퍼 ==========
    def expand_terms(self, tokens: List[str]) -> Dict[str, List[str]]:
        """
        토큰 리스트를 받아서 동의어 사전 기반으로 확장된 variants 딕셔너리 반환
        {원어: [원어, 동의어1, 동의어2, ...], ...}
        """
        expanded: Dict[str, List[str]] = {}
        for t in tokens:
            variants = [t]
            if t in self.synonym_dict:
                variants.extend(self.synonym_dict[t])
            expanded[t] = variants
        return expanded

    def build_synonym_expanded_query(self, query: str, top_k: int) -> Dict[str, Any]:
        """
        원본 쿼리와 동의어 사전을 기반으로 bool should 확장 쿼리 생성
        """
        tokens = query.strip().split()
        expanded = self.expand_terms(tokens)

        # 기본 cross_fields 매칭
        should_clauses: List[Dict[str, Any]] = [
            {
                "multi_match": {
                    "query": query,
                    "type": "cross_fields",
                    "fields": [
                        "page_content^2",
                        "page_content.ngram^1",
                        "name^3",
                        "meta_data.document_title^2"
                    ],
                    "operator": "and"
                }
            }
        ]
        # 각 토큰과 variants로 match/phrase 추가
        for variants in expanded.values():
            for v in variants:
                should_clauses.append({"match": {"name": {"query": v, "boost": 2.5}}})
                should_clauses.append({"match_phrase": {"meta_data.document_title": {"query": v, "boost": 2.0}}})
                should_clauses.append({"match": {"page_content.ngram": {"query": v, "boost": 1.0, "operator": "and"}}})

        bool_query = {"bool": {"should": should_clauses, "minimum_should_match": 1}}
        return {
            "size": top_k,
            "query": bool_query,
            "_source": ["page_content", "name", "meta_data"]
        }

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
        """
        index별로 query-time 동의어 확장 쿼리를 사용한 sparse 검색
        """
        results = []
        for index in [self.TEXT_INDEX, self.TABLE_INDEX]:
            body = self.build_synonym_expanded_query(query, top_k)
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
                metadata = source.get("meta_data", {})
                
                # 문서 제목/출처 정보 추출 개선
                doc_title = source.get("name", "")
                if not doc_title and metadata:
                    doc_title = metadata.get("document_title", metadata.get("file_name", metadata.get("title", f"Document {len(results)+1}")))
                
                # 출처 URL 정보 설정
                source_url = doc_title
                if metadata.get("url"):
                    source_url = metadata["url"]
                elif metadata.get("source"):
                    source_url = metadata["source"] 
                elif metadata.get("file_name"):
                    source_url = f"문서: {metadata['file_name']}"
                
                results.append({
                    "score": hit["_score"],
                    "page_content": source.get("page_content", ""),
                    "name": doc_title,
                    "meta_data": metadata,
                    "search_type": "dense",
                    "_index": index,
                    "source_url": source_url  # 출처 정보 추가
                })
        except Exception as e:
            print(f"Dense 검색 오류({index}): {e}")
        return results

    def sparse_retrieval_index(self, query: str, index: str, top_k: int = 100) -> List[Dict]:
        # multi-index RAG hybrid_search 내에서 사용되는 sparse 검색
        results = []
        # Query-time 동의어 확장 쿼리
        body = self.build_synonym_expanded_query(query, top_k)
        try:
            response = self.es.search(index=index, body=body)
            hits = response.get("hits", {}).get("hits", [])
            for hit in hits:
                source = hit["_source"]
                metadata = source.get("meta_data", {})
                
                # 문서 제목/출처 정보 추출 개선
                doc_title = source.get("name", "")
                if not doc_title and metadata:
                    doc_title = metadata.get("document_title", metadata.get("file_name", metadata.get("title", f"Document {len(results)+1}")))
                
                # 출처 URL 정보 설정
                source_url = doc_title
                if metadata.get("url"):
                    source_url = metadata["url"]
                elif metadata.get("source"):
                    source_url = metadata["source"] 
                elif metadata.get("file_name"):
                    source_url = f"문서: {metadata['file_name']}"
                
                results.append({
                    "score": hit["_score"],
                    "page_content": source.get("page_content", ""),
                    "name": doc_title,
                    "meta_data": metadata,
                    "search_type": "sparse",
                    "_index": index,
                    "source_url": source_url  # 출처 정보 추가
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

    def cohere_reranking(self, results: List[Dict], query: str, top_k: int = 20) -> List[Dict]:
        """
        Cohere reranker를 사용한 재순위화
        """
        if not results or len(results) == 0:
            return results

        try:
            # Cohere reranker에 전달할 문서들 준비
            documents = []
            for r in results[:top_k]:
                content = r.get("page_content", "")
                name = r.get("name", "")
                # 문서 제목과 내용을 결합
                doc_text = f"제목: {name}\n내용: {content}"
                documents.append(doc_text)

            if not documents:
                return results

            # Cohere reranker 호출
            response = self.cohere_client.rerank(
                query=query,
                documents=documents,
                top_n=len(documents),
                model="rerank-v3.5"  # 다국어 지원 모델
            )

            # 결과 재정렬
            reranked_results = []
            for i, result in enumerate(response.results):
                original_index = result.index
                if original_index < len(results):
                    r = results[original_index].copy()
                    r["rerank_score"] = result.relevance_score
                    r["rerank_rank"] = i + 1
                    reranked_results.append(r)

            # rerank_score로 정렬
            reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)

            return reranked_results

        except Exception as e:
            print(f"Cohere reranker 오류: {e}")
            # 오류 발생시 원본 결과 반환
            return results[:top_k]

    def simple_reranking(self, results: List[Dict], query: str, top_k: int = 20) -> List[Dict]:
        """
        Cohere reranker를 사용한 재순위화 (기존 simple_reranking 대체)
        """
        return self.cohere_reranking(results, query, top_k)

    def document_summarization(self, results: List[Dict], query: str) -> List[Dict]:
        summarized_results = []
        for result in results:
            page_content = result.get("page_content", "")
            if len(page_content) > 500:
                try:
                    prompt = f"""
다음 문서를 주어진 질문과 관련된 핵심 내용 위주로 요약해주세요.\n요약 길이: 원본의 {self.SUMMARIZATION_RATIO}% 정도\n질문: {query}\n문서 내용:\n{page_content}\n요약:"""
                    response = self.client.generate_content(prompt)
                    summary = response.text.strip()
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

    # Google API 키 확인
    if config.GOOGLE_API_KEY == "your-google-api-key-here":
        print("⚠️ rag_config.py에서 Google API 키가 설정되지 않았습니다.")
        api_key = input("Google API 키를 입력하세요 (또는 Enter로 스킵): ").strip()
        if not api_key:
            print("❌ Google API 키 없이는 테스트할 수 없습니다.")
            return None
    else:
        api_key = config.GOOGLE_API_KEY

    # Cohere API 키 확인
    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        print("⚠️ 환경변수에서 Cohere API 키가 설정되지 않았습니다.")
        cohere_key = input("Cohere API 키를 입력하세요 (또는 Enter로 스킵): ").strip()
        if not cohere_key:
            print("❌ Cohere API 키 없이는 reranking을 사용할 수 없습니다.")
            return None

    search_engine = MultiIndexRAGSearchEngine(google_api_key=api_key, cohere_api_key=cohere_key, config=config)
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
