from fastapi import APIRouter

from app.api.schemas import HealthResponse
from app.core.config import settings
from app.services.subtraction_model import engine

router = APIRouter()


def _health_payload() -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=settings.device,
        model_loaded=engine.model is not None,
        checkpoint_loaded=engine.checkpoint_loaded,
        checkpoint_path=settings.model_checkpoint_path,
        checkpoint_sha256=engine.checkpoint_sha256,
        checkpoint_matches_freeze=engine.checkpoint_matches_freeze,
    )


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return _health_payload()


@router.post("/health/reload-model", response_model=HealthResponse)
def reload_model() -> HealthResponse:
    """Reload weights from disk without restarting the server.

    Useful right after `python -m app.training.train` finishes, so the
    running API process picks up the freshly retrained checkpoint.
    """
    engine.reload_checkpoint()
    return _health_payload()
