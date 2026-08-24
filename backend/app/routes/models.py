"""GET /models — which recognizers this box can actually run."""

from __future__ import annotations

from fastapi import APIRouter

from ..config import settings
from ..models import SHERPA_AVAILABLE, SHERPA_VERSION, registry
from ..schemas import ModelInfo, ModelsResponse

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelsResponse)
def list_models(rescan: bool = False) -> ModelsResponse:
    """List every known model and whether its files are on disk.

    Pass ``?rescan=true`` after downloading a model to re-stat the models dir
    without restarting the server.
    """
    if rescan:
        registry.scan()
    return ModelsResponse(
        sherpa_onnx_available=SHERPA_AVAILABLE,
        sherpa_onnx_version=SHERPA_VERSION,
        models_dir=str(settings.models_dir),
        num_threads=settings.num_threads,
        models=[ModelInfo(**info) for info in registry.availability()],
    )
