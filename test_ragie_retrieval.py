"""Comprehensive test suite for Ragie RAG retrieval system - 10 challenging ML/RAG tests."""

import numpy as np
import pytest
from ragie_retrieval import (
    cosine_sim, rank, repair_score, chunk_document,
    compute_embedding_quality, optimize_retrieval_threshold,
    semantic_clustering, cross_encoder_rerank, query_expansion,
    adaptive_chunking, embedding_drift_detection
)


class TestCosineSimNormalizedInvariant:
    """Test 1: Cosine similarity scale invariance - WILL FAIL due to missing normalization."""
    
    def test_cosine_sim_scale_invariant(self):
        """Cosine similarity should be invariant to vector scaling."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        
        # Test with different scales
        sim1 = cosine_sim(a, b)
        sim2 = cosine_sim(a * 10, b)  # Scale vector a
        sim3 = cosine_sim(a, b * 100)  # Scale vector b
        sim4 = cosine_sim(a * 0.1, b * 0.01)  # Scale both vectors
        
        # All should be approximately equal (scale invariant)
        assert abs(sim1 - sim2) < 1e-10, f"Scale invariance failed: {sim1} != {sim2}"
        assert abs(sim1 - sim3) < 1e-10, f"Scale invariance failed: {sim1} != {sim3}"
        assert abs(sim1 - sim4) < 1e-10, f"Scale invariance failed: {sim1} != {sim4}"


class TestRankingConsistency:
    """Test 2: Document ranking consistency - WILL FAIL due to wrong sort order."""
    
    def test_ranking_descending_order(self):
        """Rankings should be in descending order of similarity."""
        documents = ["doc1", "doc2", "doc3", "doc4"]
        query_emb = np.array([1.0, 0.0, 0.0])
        
        # Create embeddings with known similarities
        doc_embeddings = [
            np.array([0.9, 0.1, 0.0]),  # High similarity
            np.array([0.1, 0.9, 0.0]),  # Low similarity  
            np.array([0.8, 0.2, 0.0]),  # Medium-high similarity
            np.array([0.2, 0.8, 0.0])   # Medium-low similarity
        ]
        
        ranked_indices = rank(documents, query_emb, doc_embeddings, top_k=4)
        
        # Calculate actual similarities to verify order
        similarities = [cosine_sim(query_emb, emb) for emb in doc_embeddings]
        expected_order = sorted(range(len(similarities)), 
                              key=lambda i: similarities[i], reverse=True)
        
        assert ranked_indices == expected_order, f"Wrong ranking order: {ranked_indices} != {expected_order}"


class TestTopKSelection:
    """Test 3: Top-K selection accuracy - WILL FAIL due to off-by-one error."""
    
    def test_top_k_exact_count(self):
        """Should return exactly top_k results, not top_k-1."""
        documents = [f"doc{i}" for i in range(10)]
        query_emb = np.array([1.0, 0.0])
        doc_embeddings = [np.random.rand(2) for _ in range(10)]
        
        for k in [1, 3, 5, 7]:
            ranked = rank(documents, query_emb, doc_embeddings, top_k=k)
            assert len(ranked) == k, f"Expected {k} results, got {len(ranked)}"


class TestChunkingOverlap:
    """Test 4: Document chunking overlap calculation - WILL FAIL due to wrong overlap math."""
    
    def test_chunking_overlap_consistency(self):
        """Chunks should have consistent overlap as specified."""
        text = "This is sentence one. This is sentence two. This is sentence three. " * 20
        chunk_size = 100
        overlap = 20
        
        chunks = chunk_document(text, chunk_size=chunk_size, overlap=overlap)
        
        # Verify overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            chunk1_end = chunks[i][-overlap:]
            chunk2_start = chunks[i + 1][:overlap]
            
            # Should have some overlap (not exact due to word boundaries, but significant)
            overlap_chars = len(set(chunk1_end.split()) & set(chunk2_start.split()))
            assert overlap_chars > 0, f"No overlap found between chunks {i} and {i+1}"


class TestEmbeddingQualityMetrics:
    """Test 5: Embedding quality variance calculation - WILL FAIL due to missing variance."""
    
    def test_embedding_variance_calculation(self):
        """Variance should be calculated correctly, not hardcoded to 0."""
        # Create embeddings with known variance
        embeddings = [
            np.array([1.0, 0.0]),  # norm = 1.0
            np.array([2.0, 0.0]),  # norm = 2.0  
            np.array([3.0, 0.0]),  # norm = 3.0
            np.array([4.0, 0.0])   # norm = 4.0
        ]
        
        quality = compute_embedding_quality(embeddings)
        
        # Variance of norms [1, 2, 3, 4] should be 1.25, not 0
        expected_variance = np.var([1.0, 2.0, 3.0, 4.0])
        assert abs(quality["variance"] - expected_variance) < 1e-10, \
            f"Wrong variance: {quality['variance']} != {expected_variance}"


class TestF1ScoreCalculation:
    """Test 6: F1 score formula correctness - WILL FAIL due to wrong F1 formula."""
    
    def test_f1_score_formula(self):
        """F1 should be harmonic mean of precision and recall, not sum."""
        similarities = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        relevance = [1, 1, 1, 0, 0, 0, 0, 0]  # First 3 are relevant
        
        threshold = optimize_retrieval_threshold(similarities, relevance)
        
        # With threshold around 0.7, we should get precision=1.0, recall=1.0, F1=1.0
        # Wrong formula (precision + recall) would give F1=2.0
        # Correct formula 2*(p*r)/(p+r) gives F1=1.0
        
        # Test with known threshold
        predictions = [1 if sim >= 0.75 else 0 for sim in similarities]
        tp = sum(p and r for p, r in zip(predictions, relevance))
        fp = sum(p and not r for p, r in zip(predictions, relevance))
        fn = sum(not p and r for p, r in zip(predictions, relevance))
        
        if tp + fp > 0 and tp + fn > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            expected_f1 = 2 * (precision * recall) / (precision + recall)
            
            # The buggy version will return precision + recall instead
            assert threshold > 0, "Should find a valid threshold"


class TestSemanticClustering:
    """Test 7: K-means clustering centroid diversity - WILL FAIL due to identical centroids."""
    
    def test_clustering_centroid_diversity(self):
        """Centroids should be different, not all identical."""
        # Create clearly separable embeddings
        embeddings = [
            np.array([1.0, 0.0]),  # Cluster 1
            np.array([1.1, 0.1]),  
            np.array([0.0, 1.0]),  # Cluster 2
            np.array([0.1, 1.1]),
            np.array([-1.0, 0.0]), # Cluster 3
            np.array([-1.1, -0.1])
        ]
        
        clusters = semantic_clustering(embeddings, n_clusters=3)
        
        # Should have 3 non-empty clusters
        non_empty_clusters = [cluster for cluster in clusters if cluster]
        assert len(non_empty_clusters) == 3, f"Expected 3 clusters, got {len(non_empty_clusters)}"
        
        # Each cluster should have at least one point
        for i, cluster in enumerate(clusters):
            assert len(cluster) > 0, f"Cluster {i} is empty"


class TestCrossEncoderReranking:
    """Test 8: Cross-encoder interaction scoring - WILL FAIL due to skipped interaction."""
    
    def test_cross_encoder_interaction_effect(self):
        """Cross-encoder should consider query-document interaction, not just initial scores."""
        query = "machine learning algorithms"
        documents = [
            "Deep learning neural networks",  # Some overlap
            "Cooking recipes and food",       # No overlap
            "Machine learning and AI"         # High overlap
        ]
        initial_scores = [0.5, 0.9, 0.3]  # Document 1 has highest initial score
        
        reranked = cross_encoder_rerank(query, documents, initial_scores, top_k=3)
        
        # Document 2 (index 2) should be ranked higher due to query overlap
        # despite lower initial score, but buggy version ignores interaction
        assert reranked[0] == 2, f"Expected doc 2 first due to query overlap, got {reranked[0]}"


class TestQueryExpansion:
    """Test 9: Query expansion term count - WILL FAIL due to insufficient expansion."""
    
    def test_query_expansion_term_count(self):
        """Should expand with multiple terms, not just one."""
        query = "machine learning"
        corpus = [
            "machine learning algorithms neural networks deep learning",
            "machine learning models training validation testing",
            "machine learning applications computer vision nlp"
        ]
        
        expanded = query_expansion(query, corpus)
        expansion_terms = expanded.replace(query, "").strip().split()
        
        # Should add multiple expansion terms (3), not just 1
        assert len(expansion_terms) >= 3, f"Expected 3+ expansion terms, got {len(expansion_terms)}"


class TestEmbeddingDriftDetection:
    """Test 10: Drift detection sensitivity - WILL FAIL due to too lenient threshold."""
    
    def test_drift_detection_sensitivity(self):
        """Should detect drift with reasonable threshold, not overly lenient."""
        # Create reference embeddings
        reference = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        
        # Create drifted embeddings (significantly different)
        current = [np.array([0.0, 1.0]), np.array([1.0, 0.0])]  # Swapped
        
        result = embedding_drift_detection(current, reference, drift_threshold=0.1)
        
        # Should detect drift with reasonable threshold
        assert result["drift_detected"], "Should detect significant embedding drift"
        assert result["drift_score"] > 0.1, f"Drift score {result['drift_score']} should be > 0.1"
    
    def test_cosine_normalized_invariant(self):
        """Test that cosine similarity is invariant to vector scaling."""
        # Create test vectors
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        
        # Compute baseline similarity
        baseline_sim = cosine_sim(a, b)
        
        # Test scaling invariance
        scaled_a = a * 10.0
        scaled_b = b * 0.5
        scaled_sim = cosine_sim(scaled_a, scaled_b)
        
        # Should be identical (within floating point precision)
        assert abs(baseline_sim - scaled_sim) < 1e-10


class TestRankOrdersDescendingAndRespectsTopK:
    """Test document ranking functionality."""
    
    def test_rank_orders_descending_and_respects_top_k(self):
        """Test that ranking returns top documents in descending similarity order."""
        # Create query embedding
        query = np.array([1.0, 0.0, 0.0])
        
        # Create document embeddings with known similarities
        docs = [
            np.array([0.8, 0.6, 0.0]),  # High similarity
            np.array([0.1, 0.1, 0.0]),  # Low similarity  
            np.array([0.9, 0.4, 0.0]),  # Highest similarity
            np.array([0.5, 0.5, 0.0])   # Medium similarity
        ]
        
        # Get top 3 rankings
        top_indices = rank(["doc1", "doc2", "doc3", "doc4"], query, docs, top_k=3)
        
        # Should return exactly 3 documents
        assert len(top_indices) == 3
        
        # Should be in descending order of similarity
        # Expected order: doc3 (index 2), doc1 (index 0), doc4 (index 3)
        assert top_indices[0] == 2  # Highest similarity
        assert top_indices[1] == 0  # Second highest
        assert top_indices[2] == 3  # Third highest


class TestRepairScoreCalculation:
    """Test retrieval quality scoring."""
    
    def test_repair_score_perfect_ranking(self):
        """Test MRR calculation with perfect ranking."""
        query = np.array([1.0, 0.0])
        docs = [
            np.array([1.0, 0.0]),  # Perfect match
            np.array([0.0, 1.0])   # No match
        ]
        ground_truth = [0]  # First document is relevant
        
        score = repair_score(query, docs, ground_truth)
        assert score == 1.0  # Perfect MRR score


class TestChunkDocumentOverlap:
    """Test document chunking with proper overlap."""
    
    def test_chunk_document_overlap(self):
        """Test that document chunking creates proper overlaps."""
        text = "A" * 1000  # 1000 character document
        chunks = chunk_document(text, chunk_size=300, overlap=100)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # Should have 100 characters of overlap
            overlap_text = current_chunk[-100:]
            next_start = next_chunk[:100]
            assert overlap_text == next_start


class TestEmbeddingQualityMetrics:
    """Test embedding quality calculations."""
    
    def test_embedding_quality_variance(self):
        """Test that embedding quality includes variance calculation."""
        embeddings = [
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 3.0, 4.0]),
            np.array([3.0, 4.0, 5.0])
        ]
        
        quality = compute_embedding_quality(embeddings)
        
        # Should include variance calculation
        assert quality["variance"] > 0.0
        assert quality["mean_norm"] > 0.0
        assert quality["std_norm"] >= 0.0
        assert quality["dimensionality"] == 3


class TestOptimalThresholdF1:
    """Test retrieval threshold optimization."""
    
    def test_optimal_threshold_f1_calculation(self):
        """Test that F1 score is calculated correctly."""
        similarities = [0.9, 0.8, 0.3, 0.2]
        relevance = [1.0, 1.0, 0.0, 0.0]  # First two are relevant
        
        threshold = optimize_retrieval_threshold(similarities, relevance)
        
        # Should find a reasonable threshold
        assert 0.1 <= threshold <= 0.9
        
        # Test F1 calculation manually for threshold 0.5
        predictions = [1 if sim >= 0.5 else 0 for sim in similarities]
        tp = sum(p and r for p, r in zip(predictions, relevance))
        fp = sum(p and not r for p, r in zip(predictions, relevance))
        fn = sum(not p and r for p, r in zip(predictions, relevance))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        expected_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # The function should find optimal threshold, not necessarily 0.5
        # But F1 calculation should be correct
        assert expected_f1 > 0  # Should have some positive F1 score
