from fastapi import FastAPI, File, UploadFile
from typing import List, Optional
import logging

from service.selfie_urls import SelfieRequest, UnifiedSelfieService
from service.upload_service import SelfieUploadService

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI Application
app = FastAPI(title="Advanced Selfie Validation API")
selfie_upload_service = SelfieUploadService()


@app.post("/validate_selfies/")
async def validate_selfies(
    files: Optional[List[UploadFile]] = File(None),
    urls: Optional[SelfieRequest] = None
):
    service = UnifiedSelfieService()
    try:
        url_list = urls.urls if urls else None
        results = await service.process_images(files=files, urls=url_list)
        return results
    finally:
        await service.cleanup()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
