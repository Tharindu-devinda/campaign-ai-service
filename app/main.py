import os
import json
import tempfile
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from converter.converter import CampaignConverter

app = FastAPI(
    title="Campaign AI Service",
    version="1.0.0",
)

data_root = os.environ.get("DATA_ROOT", "/tmp/campaign_data")
events_dir = os.environ.get("EVENTS_DIR", None)
converter = CampaignConverter(data_root=data_root)


class ChatRequest(BaseModel):
    customer_id: int = 1
    campaign_id: Optional[int] = None
    query: str
    target_date: Optional[str] = None


@app.get("/health")
def health() -> dict:
    model_path = Path("/app/converter/model.joblib")
    return {
        "status": "ok",
        "service": "campaign-ai",
        "model_file_exists": model_path.exists(),
        "data_root": data_root,
        "events_dir": events_dir,
    }


@app.post("/v1/chat")
def chat(req: ChatRequest) -> dict:
    """Processes a user question from the chatbot UI and returns the full analysis and formatted response."""
    result = converter.convert(
        customer_id=req.customer_id,
        campaign_id=req.campaign_id,
        events_dir=events_dir,
        target_date=req.target_date,
        query=req.query,
    )
    # Include all top-level keys for both chat and upload processing
    result["status"] = "success"
    result["diagnosis"] = result.get("xai_diagnosis", {})
    return result


@app.post("/v1/campaigns/analyze")
async def analyze_campaign(
    customer_id: int = Form(...),
    campaign_id: int = Form(...),
    numbers_file: UploadFile = File(...),
    campaign_log: UploadFile | None = File(None),
    settings: str | None = Form(None),
) -> dict:
    if not numbers_file.filename:
        raise HTTPException(status_code=422, detail="numbers_file is required")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        numbers_path = temp_path / Path(numbers_file.filename).name
        numbers_path.write_bytes(await numbers_file.read())

        log_path = None
        if campaign_log and campaign_log.filename:
            log_path = temp_path / Path(campaign_log.filename).name
            log_path.write_bytes(await campaign_log.read())

        settings_path = None
        if settings:
            try:
                json.loads(settings)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=422, detail="settings must be valid JSON") from exc

            settings_path = temp_path / "settings.json"
            settings_path.write_text(settings, encoding="utf-8")

        result = converter.convert(
            customer_id=customer_id,
            campaign_id=campaign_id,
            file_path=str(numbers_path),
            log_path=str(log_path) if log_path else None,
            settings_path=str(settings_path) if settings_path else None,
        )

        result["status"] = "success"
        result["diagnosis"] = result.get("xai_diagnosis", {})
        return result
