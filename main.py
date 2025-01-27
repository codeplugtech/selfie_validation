from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import logging
from service.upload_service import SelfieUploadService

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI Application
app = FastAPI(title="Advanced Selfie Validation API")
selfie_upload_service = SelfieUploadService()


@app.post("/validate_advanced_selfies/")
async def validate_advanced_selfies(files: List[UploadFile] = File(...)):
    try:
        results = selfie_upload_service.process_images(files)
        return results
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
