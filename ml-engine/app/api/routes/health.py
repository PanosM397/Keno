from fastapi import APIRouter

from app.api.schemas import HealthResponse
from app.core.config import settings
from app.services.subtraction_model import engine

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return HealthResponse(status="ok", device=settings.device, model_loaded=engine.model is not None)
