from rq import get_current_job
from config import redis_client, default_queue, high_priority_queue
import structlog
from audit_logic import run_full_audit_with_quantitative_metrics
import time

log = structlog.get_logger(__name__)

def process_audit_job(audit_job_id: str, deck_id: str, guidelines: str, api_key: str, pdf_path: str):
    job = get_current_job()
    job.meta.update({
        "status": "processing", 
        "deck_id": deck_id,
        "started_at": time.time()
    })
    job.save_meta()

    log.info("audit_job_started", job=audit_job_id, deck=deck_id)

    try:
        # Run the full audit with quantitative metrics (font size + contrast + AAA)
        result = run_full_audit_with_quantitative_metrics(
            pdf_path=pdf_path,
            guidelines=guidelines
        )

        job.meta.update({
            "status": "finished",
            "result_summary": {
                "total_slides": result["total_slides"],
                "average_design_score": result.get("deck_summary", {}).get("average_design_score"),
            }
        })
        job.save_meta()

        return {"status": "success", "audit_job_id": audit_job_id, "result": result}

    except Exception as e:
        log.error("audit_job_failed", job=audit_job_id, error=str(e))
        job.meta.update({"status": "failed", "error": str(e)})
        job.save_meta()
        raise