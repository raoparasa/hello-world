from fastapi import FastAPI
from pydantic import BaseModel, Field, AnyHttpUrl
from typing import Optional, List
from enum import Enum

# 1. DEFINE YOUR MODELS FIRST
class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class WordPressPlugin(BaseModel):
    # ... (other fields)
    name: str = Field(..., example="Contact Form 7")
    slug: str = Field(..., description="The unique folder name of the plugin")
    version: str = Field(..., example="5.9.3")
    is_active: bool = True
    has_valid_license: bool = False
    last_updated_days_ago: int = Field(ge=0)
    author_url: Optional[AnyHttpUrl] = None
    documentation_url: Optional[AnyHttpUrl] = None

class PluginAuditReport(BaseModel):
    site_url: str
    total_plugins_scanned: int
    vulnerable_plugins: List[WordPressPlugin]
    risk_assessment: RiskLevel
    recommendation: str

# 2. INITIALIZE THE APP (This defines 'app')
app = FastAPI()

# 3. DEFINE YOUR ROUTES (These use '@app')
@app.post("/audit/plugins")
async def run_audit(report: PluginAuditReport):
    # Your AI Logic here (e.g., calling your CrewAI agent)
    if report.risk_assessment == RiskLevel.CRITICAL:
        return {"action": "Immediate Update Required", "details": report.recommendation}
    
    return {"status": "success", "message": f"Audit complete for {report.site_url}"}

