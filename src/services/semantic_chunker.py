"""
Semantic Chunking Service.

Unlike simple character-based chunking, semantic chunking splits documents
based on meaning and topic boundaries using embeddings.

Approach:
1. Split text into sentences
2. Generate embeddings for each sentence
3. Calculate semantic similarity between adjacent sentences
4. Split at points where similarity drops significantly (topic change)
5. Group similar sentences into coherent chunks

This preserves context and meaning better than arbitrary character splits.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.services.embedding import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


@dataclass
class SemanticChunk:
    """A semantically coherent chunk of text."""
    text: str
    start_sentence_idx: int
    end_sentence_idx: int
    avg_embedding: Optional[np.ndarray] = None


class SemanticChunker:
    """
    Splits text into semantically coherent chunks based on topic boundaries.
    
    How it works:
    1. Tokenize text into sentences
    2. Embed each sentence
    3. Compute cosine similarity between adjacent sentences
    4. Find "breakpoints" where similarity drops below threshold
    5. Create chunks between breakpoints
    
    This is superior to character-based chunking because:
    - Preserves complete thoughts and ideas
    - Doesn't break mid-sentence or mid-topic
    - Creates more coherent context for retrieval
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1500,
        buffer_size: int = 1  # Sentences to look ahead/behind for smoothing
    ):
        """
        Initialize the semantic chunker.
        
        Args:
            embedding_service: Service for generating embeddings.
            similarity_threshold: Threshold below which to split (0-1).
                                 Lower = fewer splits, larger chunks.
            min_chunk_size: Minimum characters per chunk.
            max_chunk_size: Maximum characters per chunk (force split if exceeded).
            buffer_size: Number of sentences to consider for smoothing similarity.
        """
        self.embedding_service = embedding_service or get_embedding_service()
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.buffer_size = buffer_size
    
    def _tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Uses regex-based splitting that handles common edge cases
        like abbreviations, decimals, etc.
        """
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split on sentence boundaries
        # Handles: . ! ? and newlines, but not abbreviations like Mr. Dr. etc.
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*(?=\S)'
        
        sentences = re.split(sentence_pattern, text)
        
        # Filter empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _compute_similarities(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between adjacent sentence embeddings.
        
        Uses a sliding window approach with buffer for smoothing.
        """
        n = len(embeddings)
        if n < 2:
            return np.array([])
        
        similarities = []
        
        for i in range(n - 1):
            # Get window of embeddings for smoothing
            start = max(0, i - self.buffer_size)
            end = min(n, i + self.buffer_size + 2)
            
            # Compare current sentence's context with next sentence's context
            left_emb = np.mean(embeddings[start:i+1], axis=0)
            right_emb = np.mean(embeddings[i+1:end], axis=0)
            
            # Cosine similarity
            similarity = np.dot(left_emb, right_emb) / (
                np.linalg.norm(left_emb) * np.linalg.norm(right_emb) + 1e-8
            )
            similarities.append(similarity)
        
        return np.array(similarities)
    
    def _find_breakpoints(self, similarities: np.ndarray) -> List[int]:
        """
        Find indices where semantic breakpoints should occur.
        
        A breakpoint occurs when:
        1. Similarity drops below threshold, OR
        2. There's a significant local minimum in similarity
        """
        if len(similarities) == 0:
            return []
        
        breakpoints = []
        
        # Method 1: Below threshold
        below_threshold = np.where(similarities < self.similarity_threshold)[0]
        
        # Method 2: Local minima (significant drops)
        # Find points that are lower than both neighbors
        local_minima = []
        for i in range(1, len(similarities) - 1):
            if (similarities[i] < similarities[i-1] and 
                similarities[i] < similarities[i+1] and
                similarities[i] < self.similarity_threshold + 0.1):
                local_minima.append(i)
        
        # Combine breakpoints
        breakpoints = sorted(set(below_threshold.tolist() + local_minima))
        
        return breakpoints
    
    def _merge_small_chunks(
        self,
        sentences: List[str],
        breakpoints: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Merge chunks that are too small based on min_chunk_size.
        
        Returns list of (start_idx, end_idx) tuples.
        """
        if not breakpoints:
            return [(0, len(sentences))]
        
        # Create initial chunk boundaries
        boundaries = [0] + [bp + 1 for bp in breakpoints] + [len(sentences)]
        chunks = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]
        
        # Merge small chunks with neighbors
        merged = []
        current_start = chunks[0][0]
        current_end = chunks[0][1]
        current_text_len = sum(len(sentences[i]) for i in range(current_start, current_end))
        
        for start, end in chunks[1:]:
            chunk_text_len = sum(len(sentences[i]) for i in range(start, end))
            
            if current_text_len < self.min_chunk_size:
                # Merge with current chunk
                current_end = end
                current_text_len += chunk_text_len
            else:
                # Save current and start new
                merged.append((current_start, current_end))
                current_start = start
                current_end = end
                current_text_len = chunk_text_len
        
        # Don't forget the last chunk
        merged.append((current_start, current_end))
        
        return merged
    
    def _split_large_chunks(
        self,
        sentences: List[str],
        chunk_boundaries: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Split chunks that exceed max_chunk_size.
        """
        result = []
        
        for start, end in chunk_boundaries:
            chunk_text = ' '.join(sentences[start:end])
            
            if len(chunk_text) <= self.max_chunk_size:
                result.append((start, end))
            else:
                # Need to split this chunk
                current_start = start
                current_len = 0
                
                for i in range(start, end):
                    sent_len = len(sentences[i])
                    
                    if current_len + sent_len > self.max_chunk_size and current_len > 0:
                        result.append((current_start, i))
                        current_start = i
                        current_len = sent_len
                    else:
                        current_len += sent_len
                
                if current_start < end:
                    result.append((current_start, end))
        
        return result
    
    def chunk(self, text: str) -> List[SemanticChunk]:
        """
        Split text into semantic chunks.
        
        Args:
            text: The text to chunk.
            
        Returns:
            List of SemanticChunk objects.
        """
        # Step 1: Tokenize into sentences
        sentences = self._tokenize_sentences(text)
        
        if not sentences:
            return []
        
        if len(sentences) == 1:
            return [SemanticChunk(
                text=sentences[0],
                start_sentence_idx=0,
                end_sentence_idx=1
            )]
        
        logger.info(f"Semantic chunking: {len(sentences)} sentences")
        
        # Step 2: Generate embeddings for each sentence
        embeddings = self.embedding_service.embed_texts(sentences)
        
        # Step 3: Compute similarities between adjacent sentences
        similarities = self._compute_similarities(embeddings)
        
        # Step 4: Find breakpoints
        breakpoints = self._find_breakpoints(similarities)
        logger.info(f"Found {len(breakpoints)} semantic breakpoints")
        
        # Step 5: Create chunk boundaries
        chunk_boundaries = self._merge_small_chunks(sentences, breakpoints)
        
        # Step 6: Split any chunks that are too large
        chunk_boundaries = self._split_large_chunks(sentences, chunk_boundaries)
        
        # Step 7: Create SemanticChunk objects
        chunks = []
        for start, end in chunk_boundaries:
            chunk_text = ' '.join(sentences[start:end])
            chunk_embedding = np.mean(embeddings[start:end], axis=0) if start < end else None
            
            chunks.append(SemanticChunk(
                text=chunk_text,
                start_sentence_idx=start,
                end_sentence_idx=end,
                avg_embedding=chunk_embedding
            ))
        
        logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    def chunk_with_overlap(
        self,
        text: str,
        overlap_sentences: int = 1
    ) -> List[SemanticChunk]:
        """
        Chunk with sentence overlap between chunks.
        
        This helps maintain context across chunk boundaries.
        
        Args:
            text: Text to chunk.
            overlap_sentences: Number of sentences to overlap.
            
        Returns:
            List of overlapping chunks.
        """
        base_chunks = self.chunk(text)
        
        if len(base_chunks) <= 1 or overlap_sentences == 0:
            return base_chunks
        
        sentences = self._tokenize_sentences(text)
        overlapped_chunks = []
        
        for i, chunk in enumerate(base_chunks):
            start = chunk.start_sentence_idx
            end = chunk.end_sentence_idx
            
            # Add overlap from previous chunk
            if i > 0:
                prev_end = base_chunks[i-1].end_sentence_idx
                overlap_start = max(prev_end - overlap_sentences, base_chunks[i-1].start_sentence_idx)
                start = overlap_start
            
            # Add overlap to next chunk
            if i < len(base_chunks) - 1:
                next_start = base_chunks[i+1].start_sentence_idx
                overlap_end = min(next_start + overlap_sentences, base_chunks[i+1].end_sentence_idx)
                end = max(end, overlap_end)
            
            chunk_text = ' '.join(sentences[start:end])
            overlapped_chunks.append(SemanticChunk(
                text=chunk_text,
                start_sentence_idx=start,
                end_sentence_idx=end
            ))
        
        return overlapped_chunks


# Singleton
_semantic_chunker: SemanticChunker | None = None


def get_semantic_chunker() -> SemanticChunker:
    """Get or create the singleton semantic chunker."""
    global _semantic_chunker
    if _semantic_chunker is None:
        _semantic_chunker = SemanticChunker()
    return _semantic_chunker
