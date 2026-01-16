"""
Hybrid Search Service.

Combines multiple retrieval strategies for better results:
1. Semantic Search (vector similarity) - finds conceptually similar documents
2. Keyword Search (BM25) - finds exact term matches
3. Reranking (cross-encoder) - reorders results by relevance

Research shows hybrid search significantly outperforms either method alone:
- Semantic alone misses exact keyword matches
- Keyword alone misses semantic similarity
- Combining them captures both types of relevance

Based on best practices from:
- Pinecone hybrid search documentation
- ColBERT and other neural retrieval research
- RAG optimization papers
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.models.schemas import Document, RetrievedContext
from src.services.embedding import EmbeddingService, get_embedding_service
from src.services.vector_store import FAISSVectorStore, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """Result from hybrid search with scores from each method."""
    document: Document
    semantic_score: float
    keyword_score: float
    combined_score: float
    rerank_score: Optional[float] = None


class BM25:
    """
    BM25 (Best Matching 25) keyword search implementation.
    
    BM25 is a bag-of-words retrieval function that ranks documents
    based on query term frequency, document length, and corpus statistics.
    
    It's the standard baseline for keyword search and complements
    semantic search by finding exact term matches.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25.
        
        Args:
            k1: Term frequency saturation parameter.
            b: Document length normalization parameter.
        """
        self.k1 = k1
        self.b = b
        
        self.corpus: List[List[str]] = []
        self.doc_ids: List[str] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0
        self.doc_freqs: Counter = Counter()
        self.idf: dict = {}
        self.n_docs: int = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase and split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def fit(self, documents: List[Document]) -> None:
        """
        Fit BM25 on a corpus of documents.
        
        Args:
            documents: List of documents to index.
        """
        self.corpus = []
        self.doc_ids = []
        self.doc_lengths = []
        self.doc_freqs = Counter()
        
        for doc in documents:
            text = f"{doc.title} {doc.content}"
            tokens = self._tokenize(text)
            
            self.corpus.append(tokens)
            self.doc_ids.append(doc.id)
            self.doc_lengths.append(len(tokens))
            
            # Count document frequency for each unique term
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
        
        self.n_docs = len(documents)
        self.avg_doc_length = sum(self.doc_lengths) / max(self.n_docs, 1)
        
        # Compute IDF for all terms
        self.idf = {}
        for term, df in self.doc_freqs.items():
            # IDF with smoothing
            self.idf[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
        
        logger.info(f"BM25 fitted on {self.n_docs} documents, {len(self.idf)} unique terms")
    
    def score(self, query: str) -> List[Tuple[str, float]]:
        """
        Score all documents against a query.
        
        Args:
            query: Search query.
            
        Returns:
            List of (doc_id, score) tuples, sorted by score descending.
        """
        query_tokens = self._tokenize(query)
        scores = []
        
        for i, doc_tokens in enumerate(self.corpus):
            score = 0.0
            doc_len = self.doc_lengths[i]
            
            # Count term frequencies in this document
            term_freqs = Counter(doc_tokens)
            
            for term in query_tokens:
                if term not in self.idf:
                    continue
                
                tf = term_freqs.get(term, 0)
                idf = self.idf[term]
                
                # BM25 scoring formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                
                score += idf * (numerator / denominator)
            
            scores.append((self.doc_ids[i], score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class CrossEncoderReranker:
    """
    Cross-encoder based reranking.
    
    Cross-encoders are more accurate than bi-encoders (used in vector search)
    because they process query and document together, allowing for
    fine-grained interaction between terms.
    
    However, they're slower, so we use them only for reranking
    the top-k results from initial retrieval.
    
    Note: This is a simplified implementation. In production,
    use sentence-transformers CrossEncoder or a dedicated model.
    """
    
    def __init__(self, embedding_service: EmbeddingService | None = None):
        """
        Initialize the reranker.
        
        Args:
            embedding_service: Service for embeddings.
        """
        self.embedding_service = embedding_service or get_embedding_service()
    
    def rerank(
        self,
        query: str,
        results: List[HybridSearchResult],
        top_k: int | None = None
    ) -> List[HybridSearchResult]:
        """
        Rerank results using cross-encoder style scoring.
        
        This simplified version uses embedding similarity of
        concatenated query+document vs query alone.
        
        Args:
            query: The search query.
            results: Initial search results.
            top_k: Number of results to return.
            
        Returns:
            Reranked results.
        """
        if not results:
            return []
        
        # Get query embedding
        query_emb = self.embedding_service.embed_text(query)
        
        # Score each result
        for result in results:
            # Create combined text
            combined = f"{query} [SEP] {result.document.title} {result.document.content[:500]}"
            combined_emb = self.embedding_service.embed_text(combined)
            
            # Score is similarity between query and combined representation
            # Higher score means better relevance
            similarity = np.dot(query_emb, combined_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(combined_emb) + 1e-8
            )
            
            result.rerank_score = float(similarity)
        
        # Sort by rerank score
        results.sort(key=lambda x: x.rerank_score or 0, reverse=True)
        
        if top_k:
            results = results[:top_k]
        
        return results


class HybridSearchService:
    """
    Hybrid search combining semantic and keyword search with reranking.
    
    Pipeline:
    1. Semantic search (FAISS) → top_k * 2 candidates
    2. BM25 keyword search → top_k * 2 candidates
    3. Merge and dedupe results
    4. Combine scores with weighted average
    5. Rerank top results with cross-encoder
    6. Return final top_k
    
    This approach is used by:
    - Pinecone hybrid search
    - Elasticsearch with vector plugin
    - ColBERT and other neural IR systems
    """
    
    def __init__(
        self,
        vector_store: FAISSVectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        use_reranking: bool = True
    ):
        """
        Initialize hybrid search.
        
        Args:
            vector_store: FAISS vector store.
            embedding_service: Embedding service.
            semantic_weight: Weight for semantic scores (0-1).
            keyword_weight: Weight for keyword scores (0-1).
            use_reranking: Whether to use cross-encoder reranking.
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service or get_embedding_service()
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.use_reranking = use_reranking
        
        self.bm25 = BM25()
        self.reranker = CrossEncoderReranker(self.embedding_service)
        
        self._documents: List[Document] = []
        self._doc_map: dict = {}
        self._initialized = False
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents for hybrid search.
        
        Args:
            documents: Documents to index.
        """
        self._documents = documents
        self._doc_map = {doc.id: doc for doc in documents}
        
        # Fit BM25
        self.bm25.fit(documents)
        
        # Initialize vector store if not provided
        if self.vector_store is None:
            self.vector_store = FAISSVectorStore(self.embedding_service)
        
        # Add to vector store if empty
        if self.vector_store.get_document_count() == 0:
            self.vector_store.add_documents(documents)
        
        self._initialized = True
        logger.info(f"Hybrid search indexed {len(documents)} documents")
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return []
        
        min_s = min(scores)
        max_s = max(scores)
        
        if max_s == min_s:
            return [0.5] * len(scores)
        
        return [(s - min_s) / (max_s - min_s) for s in scores]
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        semantic_threshold: float = 0.3
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query.
            top_k: Number of results to return.
            semantic_threshold: Min threshold for semantic search.
            
        Returns:
            List of HybridSearchResult objects.
        """
        if not self._initialized:
            logger.warning("Hybrid search not initialized")
            return []
        
        # Get more candidates than needed for merging
        candidate_k = top_k * 3
        
        # 1. Semantic search
        semantic_results = self.vector_store.search(
            query,
            top_k=candidate_k,
            threshold=semantic_threshold
        )
        
        semantic_scores = {
            r.document.id: r.similarity_score
            for r in semantic_results
        }
        
        # 2. BM25 keyword search
        bm25_scores_raw = self.bm25.score(query)[:candidate_k]
        
        # Normalize BM25 scores
        if bm25_scores_raw:
            raw_scores = [s for _, s in bm25_scores_raw]
            normalized = self._normalize_scores(raw_scores)
            bm25_scores = {
                doc_id: score
                for (doc_id, _), score in zip(bm25_scores_raw, normalized)
            }
        else:
            bm25_scores = {}
        
        # 3. Merge candidates
        all_doc_ids = set(semantic_scores.keys()) | set(bm25_scores.keys())
        
        results = []
        for doc_id in all_doc_ids:
            if doc_id not in self._doc_map:
                continue
            
            sem_score = semantic_scores.get(doc_id, 0.0)
            kw_score = bm25_scores.get(doc_id, 0.0)
            
            # Weighted combination
            combined = (
                self.semantic_weight * sem_score +
                self.keyword_weight * kw_score
            )
            
            results.append(HybridSearchResult(
                document=self._doc_map[doc_id],
                semantic_score=sem_score,
                keyword_score=kw_score,
                combined_score=combined
            ))
        
        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Take top candidates for reranking
        top_candidates = results[:top_k * 2]
        
        # 4. Rerank (optional)
        if self.use_reranking and top_candidates:
            top_candidates = self.reranker.rerank(query, top_candidates, top_k)
        else:
            top_candidates = top_candidates[:top_k]
        
        logger.info(f"Hybrid search returned {len(top_candidates)} results")
        return top_candidates
    
    def to_retrieved_context(
        self,
        results: List[HybridSearchResult]
    ) -> List[RetrievedContext]:
        """Convert hybrid results to RetrievedContext for RAG pipeline."""
        return [
            RetrievedContext(
                document=r.document,
                similarity_score=r.rerank_score if r.rerank_score else r.combined_score
            )
            for r in results
        ]


# Singleton
_hybrid_search: HybridSearchService | None = None


def get_hybrid_search() -> HybridSearchService:
    """Get or create singleton hybrid search service."""
    global _hybrid_search
    if _hybrid_search is None:
        _hybrid_search = HybridSearchService()
    return _hybrid_search


def initialize_hybrid_search(
    documents: List[Document],
    vector_store: FAISSVectorStore | None = None
) -> HybridSearchService:
    """Initialize hybrid search with documents."""
    global _hybrid_search
    _hybrid_search = HybridSearchService(vector_store=vector_store)
    _hybrid_search.index_documents(documents)
    return _hybrid_search
