"""Health check endpoint — used by UptimeRobot to keep Render warm."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}
