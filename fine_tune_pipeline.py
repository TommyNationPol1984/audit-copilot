"""
Fine-Tuning Pipeline for Audit Copilot
Stores LLM outputs in standardized formats for training smaller models.

Strategies:
1. Collect high-quality audit rationales from gemini-1.5-flash
2. Store as JSON-L for OpenAI/Anthropic fine-tuning
3. Store as Parquet for HuggingFace datasets
4. Track metadata for quality filtering and versioning
5. Enable easy export to fine-tuning services
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib
import structlog
from enum import Enum

log = structlog.get_logger(__name__)

# Storage directories
PIPELINE_DATA_DIR = Path("/tmp/audit_copilot_pipeline")
PIPELINE_DATA_DIR.mkdir(exist_ok=True)

RAW_OUTPUTS_DIR = PIPELINE_DATA_DIR / "raw_outputs"  # Raw LLM responses
RAW_OUTPUTS_DIR.mkdir(exist_ok=True)

TRAINING_DATA_DIR = PIPELINE_DATA_DIR / "training_data"  # Formatted for training
TRAINING_DATA_DIR.mkdir(exist_ok=True)

VALIDATION_DIR = PIPELINE_DATA_DIR / "validation"  # Quality-checked outputs
VALIDATION_DIR.mkdir(exist_ok=True)

METADATA_DIR = PIPELINE_DATA_DIR / "metadata"  # Versioning and tracking
METADATA_DIR.mkdir(exist_ok=True)


class DatasetFormat(Enum):
    """Output formats for different fine-tuning services."""
    JSONL = "jsonl"  # OpenAI format
    PARQUET = "parquet"  # HuggingFace format
    CHAT = "chat"  # Chat completion format
    COMPLETION = "completion"  # Text completion format
    CUSTOM = "custom"  # Custom format


@dataclass
class AuditSample:
    """Single training sample from an audit."""
    
    # Input
    pdf_name: str
    guidelines: str
    metrics_summary: Dict[str, float]
    num_slides: int
    
    # Output
    rationale: str
    
    # Metadata
    model: str  # "gemini-1.5-flash"
    timestamp: str
    processing_time: float
    
    # Quality scoring
    quality_score: Optional[float] = None  # 0-10
    is_validated: bool = False
    validation_notes: Optional[str] = None
    
    # Training metadata
    split: str = "train"  # train/validation/test
    tags: List[str] = None
    version: str = "1.0"
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_openai_format(self) -> Dict[str, str]:
        """Convert to OpenAI fine-tuning format."""
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a design auditor. Analyze PDF presentations against design guidelines."
                },
                {
                    "role": "user",
                    "content": f"""PDF: {self.pdf_name}
Guidelines: {self.guidelines}
Slides: {self.num_slides}
Metrics: {json.dumps(self.metrics_summary)}"""
                },
                {
                    "role": "assistant",
                    "content": self.rationale
                }
            ]
        }
    
    def to_completion_format(self) -> Dict[str, str]:
        """Convert to completion fine-tuning format."""
        prompt = f"""PDF: {self.pdf_name}
Guidelines: {self.guidelines}
Metrics: {json.dumps(self.metrics_summary)}

Analysis:"""
        
        return {
            "prompt": prompt,
            "completion": f" {self.rationale}\n"
        }
    
    def calculate_quality_score(self) -> float:
        """
        Auto-score quality based on heuristics.
        Higher is better (0-10 scale).
        """
        score = 7.0  # Base score
        
        # Length check (good audits are substantial)
        word_count = len(self.rationale.split())
        if word_count < 50:
            score -= 2
        elif word_count > 500:
            score -= 1
        elif 100 <= word_count <= 400:
            score += 1
        
        # Structure check (should have multiple paragraphs)
        paragraph_count = len(self.rationale.split('\n\n'))
        if paragraph_count >= 3:
            score += 0.5
        
        # Specificity check (mentions specific slides/metrics)
        if "slide" in self.rationale.lower():
            score += 0.5
        if "metric" in self.rationale.lower() or any(str(v) in self.rationale for v in self.metrics_summary.values()):
            score += 0.5
        
        # Avoid extreme scores
        return max(1.0, min(10.0, score))


class PipelineStore:
    """
    Centralized pipeline for storing and managing fine-tuning datasets.
    """
    
    def __init__(self):
        self.samples: List[AuditSample] = []
        self.metadata = {
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "total_samples": 0,
            "validated_samples": 0
        }
    
    def add_sample(self, sample: AuditSample) -> str:
        """
        Add an audit to the pipeline.
        
        Returns:
            Sample ID for tracking
        """
        # Auto-score if not provided
        if sample.quality_score is None:
            sample.quality_score = sample.calculate_quality_score()
        
        # Generate sample ID
        sample_id = self._generate_sample_id(sample)
        
        self.samples.append(sample)
        self._save_raw_output(sample_id, sample)
        
        log.info("sample_added", sample_id=sample_id, quality=sample.quality_score)
        
        return sample_id
    
    def _generate_sample_id(self, sample: AuditSample) -> str:
        """Generate unique sample ID."""
        content = f"{sample.pdf_name}{sample.timestamp}{sample.rationale}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _save_raw_output(self, sample_id: str, sample: AuditSample):
        """Save raw output to JSON."""
        output_file = RAW_OUTPUTS_DIR / f"{sample_id}.json"
        with open(output_file, 'w') as f:
            json.dump(sample.to_dict(), f, indent=2)
    
    def validate_sample(
        self,
        sample_id: str,
        is_valid: bool,
        notes: str = ""
    ) -> bool:
        """Manually validate a sample."""
        for sample in self.samples:
            if self._generate_sample_id(sample) == sample_id:
                sample.is_validated = is_valid
                sample.validation_notes = notes
                log.info("sample_validated", sample_id=sample_id, valid=is_valid)
                return True
        return False
    
    def filter_by_quality(self, min_score: float = 7.0) -> List[AuditSample]:
        """Get samples meeting quality threshold."""
        return [s for s in self.samples if s.quality_score >= min_score]
    
    def filter_validated(self) -> List[AuditSample]:
        """Get only validated samples."""
        return [s for s in self.samples if s.is_validated]
    
    def export_for_training(
        self,
        format: DatasetFormat = DatasetFormat.JSONL,
        min_quality: float = 7.0,
        only_validated: bool = False
    ) -> str:
        """
        Export dataset in specified format for fine-tuning.
        
        Args:
            format: Output format
            min_quality: Minimum quality score (0-10)
            only_validated: Only include manually validated samples
        
        Returns:
            Path to exported dataset
        """
        # Filter samples
        samples = self.samples
        if only_validated:
            samples = self.filter_validated()
        samples = [s for s in samples if s.quality_score >= min_quality]
        
        if not samples:
            log.warning("no_samples_for_export", quality_threshold=min_quality, validated_only=only_validated)
            return None
        
        if format == DatasetFormat.JSONL:
            return self._export_jsonl(samples)
        elif format == DatasetFormat.COMPLETION:
            return self._export_completion(samples)
        elif format == DatasetFormat.CHAT:
            return self._export_chat(samples)
        elif format == DatasetFormat.PARQUET:
            return self._export_parquet(samples)
        else:
            return self._export_custom(samples)
    
    def _export_jsonl(self, samples: List[AuditSample]) -> str:
        """Export as OpenAI JSONL format."""
        timestamp = datetime.utcnow().isoformat()
        output_file = TRAINING_DATA_DIR / f"training_openai_{timestamp}.jsonl"
        
        with open(output_file, 'w') as f:
            for sample in samples:
                line = json.dumps(sample.to_openai_format())
                f.write(line + '\n')
        
        log.info("exported_jsonl", file=str(output_file), samples=len(samples))
        return str(output_file)
    
    def _export_completion(self, samples: List[AuditSample]) -> str:
        """Export as completion format."""
        timestamp = datetime.utcnow().isoformat()
        output_file = TRAINING_DATA_DIR / f"training_completion_{timestamp}.jsonl"
        
        with open(output_file, 'w') as f:
            for sample in samples:
                line = json.dumps(sample.to_completion_format())
                f.write(line + '\n')
        
        log.info("exported_completion", file=str(output_file), samples=len(samples))
        return str(output_file)
    
    def _export_chat(self, samples: List[AuditSample]) -> str:
        """Export as chat format."""
        timestamp = datetime.utcnow().isoformat()
        output_file = TRAINING_DATA_DIR / f"training_chat_{timestamp}.jsonl"
        
        with open(output_file, 'w') as f:
            for sample in samples:
                line = json.dumps(sample.to_openai_format())
                f.write(line + '\n')
        
        log.info("exported_chat", file=str(output_file), samples=len(samples))
        return str(output_file)
    
    def _export_parquet(self, samples: List[AuditSample]) -> str:
        """Export as Parquet for HuggingFace."""
        try:
            import pandas as pd
            
            # Convert to DataFrame
            data = [s.to_dict() for s in samples]
            df = pd.DataFrame(data)
            
            timestamp = datetime.utcnow().isoformat()
            output_file = TRAINING_DATA_DIR / f"training_hf_{timestamp}.parquet"
            
            df.to_parquet(output_file, index=False)
            
            log.info("exported_parquet", file=str(output_file), samples=len(samples))
            return str(output_file)
        except ImportError:
            log.error("pandas_required_for_parquet")
            return None
    
    def _export_custom(self, samples: List[AuditSample]) -> str:
        """Export as custom JSON structure."""
        timestamp = datetime.utcnow().isoformat()
        output_file = TRAINING_DATA_DIR / f"training_custom_{timestamp}.json"
        
        export = {
            "metadata": {
                "exported_at": timestamp,
                "num_samples": len(samples),
                "format_version": "1.0"
            },
            "samples": [s.to_dict() for s in samples]
        }
        
        with open(output_file, 'w') as f:
            json.dump(export, f, indent=2)
        
        log.info("exported_custom", file=str(output_file), samples=len(samples))
        return str(output_file)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        validated = [s for s in self.samples if s.is_validated]
        quality_scores = [s.quality_score for s in self.samples if s.quality_score]
        
        return {
            "total_samples": len(self.samples),
            "validated_samples": len(validated),
            "validation_rate": len(validated) / len(self.samples) if self.samples else 0,
            "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "quality_score_range": (min(quality_scores), max(quality_scores)) if quality_scores else (0, 0),
            "models_used": list(set(s.model for s in self.samples)),
            "sample_distribution": {
                "train": len([s for s in self.samples if s.split == "train"]),
                "validation": len([s for s in self.samples if s.split == "validation"]),
                "test": len([s for s in self.samples if s.split == "test"])
            }
        }


# Global pipeline instance
_pipeline_instance = None


def get_pipeline() -> PipelineStore:
    """Get or create global pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = PipelineStore()
    return _pipeline_instance


def create_training_sample(
    audit_result: Dict[str, Any],
    quality_score: Optional[float] = None
) -> Optional[AuditSample]:
    """
    Create a training sample from an audit result.
    
    Args:
        audit_result: Output from run_full_audit_fast()
        quality_score: Optional manual quality score
    
    Returns:
        AuditSample or None if invalid
    """
    if audit_result.get("status") != "success":
        log.warning("audit_failed_sample_skipped", status=audit_result.get("status"))
        return None
    
    try:
        sample = AuditSample(
            pdf_name=audit_result.get("pdf_name", "unknown"),
            guidelines=audit_result.get("guidelines", ""),
            metrics_summary=audit_result.get("metrics_summary", {}),
            num_slides=audit_result.get("total_slides", 0),
            rationale=audit_result.get("rationale", ""),
            model=audit_result.get("model", "unknown"),
            timestamp=datetime.utcnow().isoformat(),
            processing_time=audit_result.get("total_processing_time_seconds", 0),
            quality_score=quality_score
        )
        
        return sample
    except Exception as e:
        log.error("sample_creation_failed", error=str(e))
        return None

