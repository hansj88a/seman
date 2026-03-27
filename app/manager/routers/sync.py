"""동기화 API: S3 → OpenSearch 등록."""
from fastapi import APIRouter, HTTPException

from app.models import SyncStartResponse
from app.manager.services.sync_service import run_sync

router = APIRouter()
 

@router.post(
    "/sync/start",
    summary="동기화 시작",
    description="S3에서 상품 정보를 가져와 OpenSearch에 등록합니다.",
    response_model=SyncStartResponse,
)
async def sync_start() -> SyncStartResponse:
    try:
        result = run_sync()
        return SyncStartResponse(
            status=result["status"],
            indexed=result["indexed"],
            total_fetched=result["total_fetched"],
            errors=result["errors"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"동기화 실패: {str(e)}") from e
    