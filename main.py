import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from service.selfie_urls import UnifiedSelfieService

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI Application
app = FastAPI(title="Advanced Selfie Validation API")
service = UnifiedSelfieService()

class SelfieURLs(BaseModel):
    urls: List[str]


@app.post("/validate_selfies/")
async def validate_selfies_urls(
        data: SelfieURLs
):
    logger.info("Request data: %s", data)

    if not data.urls:
        raise HTTPException(
            status_code=400,
            detail="URLs list cannot be empty"
        )
    try:
        results = await service.process_images(files=None, urls=data.urls)
        return results
    finally:
        await service.cleanup()


@app.post("/validate_selfies/upload/")
async def validate_selfies_files(
        files: List[UploadFile] = File(...)
):
    logger.info("Request files: %s", files)
    try:
        results = await service.process_images(files=files, urls=[])
        return results
    finally:
        await service.cleanup()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
