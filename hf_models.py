"""
HuggingFace Model Integration for Audit Copilot
Provides fine-tuned models for:
- Named Entity Recognition (NER) - Extract design elements
- Token Classification - Identify design issues
- Text Classification - Categorize audit findings
- Semantic Search - Find similar audits
"""

import structlog
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from pathlib import Path

log = structlog.get_logger(__name__)

# Cache models locally
HF_MODELS_CACHE = Path("/tmp/audit_copilot_hf_models")
HF_MODELS_CACHE.mkdir(exist_ok=True)


@dataclass
class ModelConfig:
    """Configuration for HuggingFace models."""
    
    model_id: str
    task: str  # ner, token_classification, text_classification, semantic_search
    device: str = "cpu"
    cache_dir: str = str(HF_MODELS_CACHE)
    confidence_threshold: float = 0.5
    
    def __post_init__(self):
        if torch.cuda.is_available() and self.device == "auto":
            self.device = "cuda"
        elif self.device == "auto":
            self.device = "cpu"


class NERModel:
    """Named Entity Recognition for extracting design elements."""
    
    def __init__(self, model_id: str = "inklingScholar/bert-finetuned-ner", device: str = "auto"):
        """
        Initialize NER model for extracting entities from audit text.
        
        Args:
            model_id: HuggingFace model ID
            device: "auto", "cpu", or "cuda"
        """
        self.config = ModelConfig(
            model_id=model_id,
            task="ner",
            device=device
        )
        
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            
            log.info("loading_ner_model", model_id=model_id, device=self.config.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=self.config.cache_dir
            )
            
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_id,
                cache_dir=self.config.cache_dir
            )
            
            self.model.to(self.config.device)
            
            # Use pipeline for easier inference
            self.pipeline = pipeline(
                "token-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.config.device == "cuda" else -1,
                aggregation_strategy="simple"
            )
            
            log.info("ner_model_loaded", model_id=model_id)
            self.available = True
            
        except Exception as e:
            log.error("ner_model_load_failed", error=str(e), model_id=model_id)
            self.available = False
            self.pipeline = None
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        Args:
            text: Audit rationale or other text
        
        Returns:
            List of extracted entities with labels and scores
        
        Example output:
        [
            {
                "entity": "Design Principle",
                "score": 0.98,
                "word": "alignment",
                "start": 15,
                "end": 24
            },
            ...
        ]
        """
        if not self.available or not self.pipeline:
            log.warning("ner_model_not_available")
            return []
        
        try:
            results = self.pipeline(text)
            
            # Filter by confidence threshold
            filtered = [
                r for r in results
                if r.get("score", 0) >= self.config.confidence_threshold
            ]
            
            log.debug("entities_extracted", count=len(filtered), text_length=len(text))
            
            return filtered
        
        except Exception as e:
            log.error("entity_extraction_failed", error=str(e))
            return []
    
    def extract_design_elements(self, audit_rationale: str) -> Dict[str, List[str]]:
        """
        Extract specific design elements from audit text.
        Groups entities by type (typography, color, layout, etc).
        
        Args:
            audit_rationale: Full audit rationale text
        
        Returns:
            Dictionary of design elements by category
        """
        entities = self.extract_entities(audit_rationale)
        
        # Group by entity type
        elements = {}
        for entity in entities:
            entity_type = entity.get("entity", "OTHER")
            word = entity.get("word", "")
            
            if entity_type not in elements:
                elements[entity_type] = []
            
            if word and word not in elements[entity_type]:
                elements[entity_type].append(word)
        
        return elements


class TextClassificationModel:
    """Classify audit findings (severity, category)."""
    
    def __init__(self, model_id: str = "distilbert-base-uncased-finetuned-sst-2-english", device: str = "auto"):
        """
        Initialize text classification model.
        
        Args:
            model_id: HuggingFace model ID
            device: "auto", "cpu", or "cuda"
        """
        self.config = ModelConfig(
            model_id=model_id,
            task="text_classification",
            device=device
        )
        
        try:
            from transformers import pipeline
            
            log.info("loading_classification_model", model_id=model_id)
            
            self.pipeline = pipeline(
                "text-classification",
                model=model_id,
                device=0 if self.config.device == "cuda" else -1,
                top_k=None  # Return all scores
            )
            
            log.info("classification_model_loaded", model_id=model_id)
            self.available = True
            
        except Exception as e:
            log.error("classification_model_load_failed", error=str(e))
            self.available = False
            self.pipeline = None
    
    def classify_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Classify text into categories.
        
        Args:
            text: Text to classify (audit section)
        
        Returns:
            List of classifications with scores
        """
        if not self.available or not self.pipeline:
            log.warning("classification_model_not_available")
            return []
        
        try:
            results = self.pipeline(text)
            log.debug("text_classified", count=len(results))
            return results
        
        except Exception as e:
            log.error("text_classification_failed", error=str(e))
            return []
    
    def categorize_findings(self, audit_rationale: str) -> Dict[str, float]:
        """
        Categorize overall audit finding (positive/negative).
        
        Args:
            audit_rationale: Full audit text
        
        Returns:
            Category probabilities
        """
        # Split into sections and classify each
        sections = audit_rationale.split('\n\n')[:3]  # First 3 sections
        
        all_scores = {}
        
        for section in sections:
            if not section.strip():
                continue
            
            results = self.classify_text(section)
            for result in results:
                label = result.get("label", "neutral")
                score = result.get("score", 0)
                
                if label not in all_scores:
                    all_scores[label] = []
                all_scores[label].append(score)
        
        # Average scores by category
        avg_scores = {
            label: sum(scores) / len(scores)
            for label, scores in all_scores.items()
        }
        
        return avg_scores


class EmbeddingModel:
    """Generate embeddings for semantic search and similarity."""
    
    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "auto"):
        """
        Initialize embedding model for semantic search.
        
        Args:
            model_id: HuggingFace model ID (SentenceTransformers)
            device: "auto", "cpu", or "cuda"
        """
        self.config = ModelConfig(
            model_id=model_id,
            task="semantic_search",
            device=device
        )
        
        try:
            from sentence_transformers import SentenceTransformer
            
            log.info("loading_embedding_model", model_id=model_id)
            
            self.model = SentenceTransformer(
                model_id,
                cache_folder=self.config.cache_dir,
                device=self.config.device
            )
            
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            log.info(
                "embedding_model_loaded",
                model_id=model_id,
                embedding_dim=self.embedding_dim
            )
            self.available = True
            
        except Exception as e:
            log.error("embedding_model_load_failed", error=str(e))
            self.available = False
            self.model = None
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector or None
        """
        if not self.available or not self.model:
            log.warning("embedding_model_not_available")
            return None
        
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        
        except Exception as e:
            log.error("embedding_failed", error=str(e))
            return None
    
    def find_similar_audits(
        self,
        query: str,
        audit_database: List[Dict[str, Any]],
        top_k: int = 5,
        threshold: float = 0.5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find similar audits using semantic search.
        
        Args:
            query: Query text (audit or rationale)
            audit_database: List of audit documents with 'rationale' field
            top_k: Number of results to return
            threshold: Minimum similarity score (0-1)
        
        Returns:
            List of (audit, similarity_score) tuples
        """
        if not self.available or not self.model:
            log.warning("semantic_search_unavailable")
            return []
        
        try:
            # Embed query
            query_embedding = self.embed_text(query)
            if not query_embedding:
                return []
            
            # Embed all audits
            scores = []
            for audit in audit_database:
                rationale = audit.get("rationale", "")
                if not rationale:
                    continue
                
                audit_embedding = self.embed_text(rationale)
                if not audit_embedding:
                    continue
                
                # Calculate cosine similarity
                import numpy as np
                similarity = np.dot(query_embedding, audit_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(audit_embedding) + 1e-8
                )
                
                if similarity >= threshold:
                    scores.append((audit, float(similarity)))
            
            # Sort by score and return top_k
            results = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
            
            log.debug("similar_audits_found", count=len(results), query_length=len(query))
            
            return results
        
        except Exception as e:
            log.error("similarity_search_failed", error=str(e))
            return []


class HFModelManager:
    """
    Central manager for all HuggingFace models.
    Lazy-loads models on demand.
    """
    
    def __init__(self, device: str = "auto"):
        self.device = device
        self.ner_model: Optional[NERModel] = None
        self.classification_model: Optional[TextClassificationModel] = None
        self.embedding_model: Optional[EmbeddingModel] = None
    
    def get_ner_model(self) -> Optional[NERModel]:
        """Get or load NER model."""
        if self.ner_model is None:
            self.ner_model = NERModel(device=self.device)
        return self.ner_model if self.ner_model.available else None
    
    def get_classification_model(self) -> Optional[TextClassificationModel]:
        """Get or load classification model."""
        if self.classification_model is None:
            self.classification_model = TextClassificationModel(device=self.device)
        return self.classification_model if self.classification_model.available else None
    
    def get_embedding_model(self) -> Optional[EmbeddingModel]:
        """Get or load embedding model."""
        if self.embedding_model is None:
            self.embedding_model = EmbeddingModel(device=self.device)
        return self.embedding_model if self.embedding_model.available else None
    
    def extract_entities_from_audit(self, audit_rationale: str) -> Dict[str, Any]:
        """
        Extract entities and design elements from audit.
        
        Args:
            audit_rationale: Full audit text
        
        Returns:
            Extracted entities grouped by type
        """
        ner = self.get_ner_model()
        if not ner:
            return {}
        
        return ner.extract_design_elements(audit_rationale)
    
    def categorize_audit(self, audit_rationale: str) -> Dict[str, float]:
        """
        Categorize audit findings.
        
        Args:
            audit_rationale: Full audit text
        
        Returns:
            Category scores
        """
        clf = self.get_classification_model()
        if not clf:
            return {}
        
        return clf.categorize_findings(audit_rationale)
    
    def embed_audit(self, audit_rationale: str) -> Optional[List[float]]:
        """
        Generate embedding for audit.
        
        Args:
            audit_rationale: Full audit text
        
        Returns:
            Embedding vector
        """
        emb = self.get_embedding_model()
        if not emb:
            return None
        
        return emb.embed_text(audit_rationale)
    
    def find_similar_audits(
        self,
        query: str,
        audit_database: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find similar audits.
        
        Args:
            query: Query text
            audit_database: List of audits
            top_k: Number of results
        
        Returns:
            Similar audits with scores
        """
        emb = self.get_embedding_model()
        if not emb:
            return []
        
        return emb.find_similar_audits(query, audit_database, top_k=top_k)


# Global manager instance
_hf_manager: Optional[HFModelManager] = None


def get_hf_manager(device: str = "auto") -> HFModelManager:
    """Get or create global HuggingFace model manager."""
    global _hf_manager
    if _hf_manager is None:
        _hf_manager = HFModelManager(device=device)
        log.info("hf_manager_initialized", device=device)
    return _hf_manager

