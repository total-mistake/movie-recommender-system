from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Проверка работоспособности сервера
    """
    return {"status": "healthy"} 