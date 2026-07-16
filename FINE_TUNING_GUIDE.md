# Fine-Tuning Guide: Building Custom Audit Models

## Overview

This guide explains how to use the audit-copilot fine-tuning pipeline to build custom, optimized models for your specific audit needs.

**Benefits:**
- 10x faster inference (5-10s vs 60s)
- 90% cheaper (self-hosted vs API calls)
- Data privacy (models run locally)
- Customized to your audit criteria

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Run Audits (Collect Training Data)                       │
│    POST /analyze/deck → Auto-stores to pipeline              │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Quality Control                                           │
│    /pipeline/stats → Monitor                                 │
│    /pipeline/validate → Mark good/bad samples               │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Export Dataset                                            │
│    /pipeline/export → Format for training                   │
│    Formats: JSONL, Chat, Completion, Parquet               │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Fine-Tune Model                                          │
│    OpenAI API, HuggingFace, or Local                        │
│    (Llama 2, Mistral, etc)                                  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Deploy Custom Model                                      │
│    Self-hosted, faster, cheaper                            │
│    Optional: Keep Gemini as fallback                       │
└─────────────────────────────────────────────────────────────┘
```

## Step 1: Collect Training Data

### Automatic Collection
Every successful audit is automatically stored for training.

```bash
# All these automatically save to pipeline
POST /analyze/deck
{
  "pdf_path": "/tmp/deck.pdf",
  "guidelines": "Your design guidelines",
  "store_for_training": true  # Default
}

Response includes:
{
  "training_sample_id": "abc123def456",  # For reference
  "status": "success",
  "rationale": "...",
  ...
}
```

### How Many Samples Do I Need?

| Samples | Time | Cost | Quality |
|---------|------|------|---------|
| 50 | 1 hour | $0.05 | Minimal |
| 100 | 2 hours | $0.10 | OK |
| **500** | **8 hours** | **$0.50** | **Good** |
| **1000** | **16 hours** | **$1.00** | **Excellent** |

**Recommendation**: Start with 100, validate heavily, then expand.

## Step 2: Quality Control

### Monitor Pipeline Health
```bash
GET /pipeline/stats

{
  "pipeline": {
    "total_samples": 156,
    "validated_samples": 42,
    "validation_rate": 0.269,
    "avg_quality_score": 7.82,
    "quality_score_range": [5.2, 9.9],
    "models_used": ["gemini-1.5-flash"],
    "sample_distribution": {
      "train": 100,
      "validation": 40,
      "test": 16
    }
  }
}
```

### Validate Samples
```bash
# Mark a high-quality sample
POST /pipeline/validate
{
  "sample_id": "abc123def456",
  "is_valid": true,
  "notes": "Excellent analysis with specific examples"
}

# Reject a poor sample
POST /pipeline/validate
{
  "sample_id": "xyz789",
  "is_valid": false,
  "notes": "Too generic, missing evidence"
}
```

### Quality Scoring (Auto-Calculated)

Samples are auto-scored 0-10 based on:
- **Length**: 100-400 words optimal
- **Structure**: 3+ paragraphs
- **Specificity**: Mentions slides/metrics

You can manually override with `is_valid` field.

## Step 3: Export Dataset

### Export for OpenAI Fine-Tuning

```bash
POST /pipeline/export
{
  "format": "jsonl",
  "min_quality": 8.0,
  "only_validated": false
}

Response:
{
  "status": "success",
  "format": "jsonl",
  "file": "/tmp/audit_copilot_pipeline/training_data/training_openai_2024-01-15T10:30:00.jsonl",
  "instructions": "Use with OpenAI Fine-tuning API"
}
```

### Export for HuggingFace

```bash
POST /pipeline/export
{
  "format": "parquet",
  "min_quality": 7.0,
  "only_validated": true
}
```

### Available Formats

| Format | Use | Provider |
|--------|-----|----------|
| `jsonl` | OpenAI Chat/GPT | OpenAI |
| `chat` | OpenAI Chat Completion | OpenAI |
| `completion` | OpenAI Text Completion | OpenAI |
| `parquet` | HuggingFace/PyArrow | HuggingFace |
| `custom` | Custom pipelines | Self-hosted |

## Step 4: Fine-Tune a Model

### Option A: OpenAI Fine-Tuning (Easiest)

```python
from openai import OpenAI

client = OpenAI(api_key="your-key")

# Upload training file
with open("/tmp/training_openai_*.jsonl") as f:
    response = client.files.create(
        file=f,
        purpose="fine-tune"
    )
    file_id = response.id

# Create fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=file_id,
    model="gpt-3.5-turbo",
    hyperparameters={
        "n_epochs": 3,
        "batch_size": 16
    }
)

print(f"Job ID: {job.id}")
# Wait for job to complete (~1-4 hours)
```

### Option B: HuggingFace Fine-Tuning (Most Flexible)

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Load your exported parquet
dataset = load_dataset("parquet", data_files="/tmp/training_hf_*.parquet")

# Load base model (Llama 2 7B example)
model_name = "meta-llama/Llama-2-7b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Fine-tune
training_args = TrainingArguments(
    output_dir="./audit-copilot-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

trainer.train()
```

### Option C: Local Fine-Tuning (Self-Hosted)

```python
import torch
from transformers import pipeline

# Use your fine-tuned model
generator = pipeline(
    "text-generation",
    model="./audit-copilot-finetuned",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Inference
prompt = """
PDF: design.pdf
Guidelines: Use sans-serif fonts
Metrics: {"avg_score": 7.8, "avg_contrast": 5.2}

Analysis:"""

result = generator(prompt, max_length=500)
print(result[0]["generated_text"])
```

## Step 5: Deploy Custom Model

### Fallback Strategy (Recommended)

Use your custom model + Gemini as fallback:

```python
def audit_with_fallback(pdf_path, guidelines):
    try:
        # Try custom model (fast, cheap)
        result = custom_model.audit(pdf_path, guidelines)
        return result
    except:
        # Fallback to Gemini
        return gemini_audit(pdf_path, guidelines)
```

### Production Deployment

Deploy fine-tuned model with:
- **vLLM** - Optimized inference server
- **TensorRT-LLM** - NVIDIA optimized serving
- **Ollama** - Local model server
- **FastAPI** - Custom inference endpoint

## Monitoring & Iteration

### Track Model Performance

```python
# Compare models
results = {
    "custom": benchmark_model(custom_model),
    "gemini": benchmark_model(gemini),
}

for metric in ["speed", "quality", "cost"]:
    custom_score = results["custom"][metric]
    gemini_score = results["gemini"][metric]
    improvement = (custom_score / gemini_score - 1) * 100
    print(f"{metric}: {improvement:+.0f}%")
```

### Continuous Improvement

1. **Collect more audits** (new designs, edge cases)
2. **Monitor errors** (where does custom model fail?)
3. **Validate samples** (ensure training data quality)
4. **Re-fine-tune** (quarterly or monthly)

## Cost Breakdown

### Scenario: 1000 Audits/Month

**Gemini-Only Approach:**
```
1000 audits × $0.001 = $1.00/month
Infrastructure: $10-50/month
Total: ~$15-55/month
```

**Fine-Tuned Model Approach:**
```
Initial setup:
- Collect 1000 training samples: $1.00
- Fine-tune (OpenAI): $100-200
- Subtotal: $101-201

Per month:
- Self-hosted inference: $0-10 (owned hardware)
- Or serverless (AWS): $0.50-2.00
- Gemini fallback (1%): $0.01
Total: ~$0.51-12.01/month

Breakeven: ~20 months
Long-term savings: ~90%
```

## Best Practices

### 1. Data Quality > Quantity
- 100 high-quality samples > 1000 poor samples
- Always validate
- Remove duplicates and poor examples

### 2. Start Small
- Begin with 50-100 samples
- Test fine-tuning
- Measure performance
- Expand if worthwhile

### 3. Monitor Drift
- Track performance over time
- Audit models regularly
- A/B test new versions

### 4. Version Control
- Tag each exported dataset with version
- Track which model trained on which data
- Keep audit logs

## Troubleshooting

### "Not enough samples"
**Solution**: Collect 100+ samples minimum. Start collecting and export when ready.

### "Low quality scores"
**Solution**: Audit process issues. Check if:
- Guidelines are clear
- PDFs are readable
- Model is working correctly

### "Fine-tuned model performs worse"
**Solution**: 
1. Increase training samples
2. Lower learning rate
3. Train for more epochs
4. Validate training data quality

### "Inference is still slow"
**Solution**:
1. Use quantization (int8, int4)
2. Use faster models (Mistral vs Llama)
3. Use specialized hardware (RTX, TPU)
4. Use batching

## Next Steps

1. **Start audits now** - All will be collected automatically
2. **Wait for 100+ samples** - Takes ~2 hours of audit time
3. **Validate samples** - Mark good ones
4. **Export dataset** - Use `/pipeline/export`
5. **Fine-tune model** - Choose OpenAI, HuggingFace, or local
6. **Test & deploy** - Compare with Gemini
7. **Iterate** - Add more data, improve quality

## Additional Resources

- [OpenAI Fine-Tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- [HuggingFace Fine-Tuning](https://huggingface.co/docs/transformers/training)
- [Llama 2 Fine-Tuning](https://github.com/facebookresearch/llama)
- [vLLM Inference Server](https://github.com/lm-sys/vllm)

