from fastapi import UploadFile, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, HttpUrl
import aiohttp
import asyncio
import aiofiles
import os
import logging
from urllib.parse import urlparse
from service.selfie_validator import AdvancedSelfieValidator

logger = logging.getLogger(__name__)


class UnifiedSelfieService:
    def __init__(self):
        self.validator = AdvancedSelfieValidator()
        self.session = None
        self.UPLOAD_DIR = "temp_uploads"
        self.MAX_IMAGES = 25

    async def _get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def download_image(self, url: str) -> tuple[str, str]:
        """Download image from URL and save to temporary file"""
        try:
            parsed_url = urlparse(str(url))
            filename = os.path.basename(parsed_url.path) or f"image_{os.urandom(8).hex()}.jpg"
            file_path = os.path.join(self.UPLOAD_DIR, filename)

            session = await self._get_session()
            async with session.get(url) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to download image from {url}. Status: {response.status}"
                    )
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    raise HTTPException(
                        status_code=400,
                        detail=f"URL {url} does not point to an image. Content-Type: {content_type}"
                    )
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(await response.read())
            return file_path, filename
        except aiohttp.ClientError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download image from {url}: {str(e)}"
            )

    async def save_upload_file(self, file: UploadFile) -> tuple[str, str]:
        """Save uploaded file to temporary directory"""
        try:
            file_path = os.path.join(self.UPLOAD_DIR, file.filename)
            async with aiofiles.open(file_path, 'wb') as buffer:
                while chunk := await file.read(8192):  # 8KB chunks
                    await buffer.write(chunk)
            return file_path, file.filename
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to save uploaded file {file.filename}: {str(e)}"
            )

    def validate_image_sync(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Synchronous image validation"""
        try:
            result = self.validator.validate_image(file_path)
            result['filename'] = filename
            return result
        except Exception as e:
            return {
                'filename': filename,
                'valid': False,
                'reason': str(e),
                'score': 0
            }

    async def process_images(self, files: Optional[List[UploadFile]] = None, urls: Optional[List[str]] = None) -> Dict[
        str, Any]:
        """Process and validate images from both files and URLs"""
        # Create temporary directory
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        try:
            # Initialize storage for all images
            all_files = []  # List of (file_path, filename) tuples

            # Validate total number of images
            total_images = (len(files) if files else 0) + (len(urls) if urls else 0)
            if total_images > self.MAX_IMAGES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Maximum {self.MAX_IMAGES} images allowed, received {total_images}"
                )

            # Process uploaded files
            if files:
                file_tasks = [self.save_upload_file(file) for file in files]
                uploaded_files = await asyncio.gather(*file_tasks, return_exceptions=True)
                all_files.extend([f for f in uploaded_files if not isinstance(f, Exception)])

            # Process URLs
            if urls:
                url_tasks = [self.download_image(url) for url in urls]
                downloaded_files = await asyncio.gather(*url_tasks, return_exceptions=True)
                all_files.extend([f for f in downloaded_files if not isinstance(f, Exception)])

            # Validate all images and track invalid ones
            invalid_images = []
            valid_count = 0

            for file_path, filename in all_files:
                result = self.validate_image_sync(file_path, filename)
                if not result['valid']:
                    invalid_images.append({
                        "filename": filename,
                        "reason": result.get('reason', 'Unknown validation failure'),
                        "score": result.get('score', 0)
                    })
                else:
                    valid_count += 1

            # Validate success rate
            valid_percentage = valid_count / total_images
            if valid_percentage < 0.9:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Insufficient valid selfies",
                        "required": "90%",
                        "current": f"{valid_percentage * 100:.1f}%",
                        "invalid_images": invalid_images,
                        "success": False
                    }
                )

            # Return appropriate response based on validation results
            if len(invalid_images) == 0:
                return {
                    "message": "All images are valid",
                    "total_images": total_images,
                    "success": True
                }

            return {
                "total_images": total_images,
                "invalid_count": len(invalid_images),
                "invalid_images": invalid_images,
                "success": False
            }

        finally:
            # Cleanup temporary files
            if os.path.exists(self.UPLOAD_DIR):
                for filename in os.listdir(self.UPLOAD_DIR):
                    try:
                        os.remove(os.path.join(self.UPLOAD_DIR, filename))
                    except Exception as e:
                        logger.error(f"Error removing temporary file {filename}: {str(e)}")
                try:
                    os.rmdir(self.UPLOAD_DIR)
                except Exception as e:
                    logger.error(f"Error removing temporary directory: {str(e)}")

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None