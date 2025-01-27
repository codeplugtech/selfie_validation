import os
from typing import List, Dict, Any

from fastapi import UploadFile, HTTPException

from service.selfie_validator import AdvancedSelfieValidator


class SelfieUploadService:
    def __init__(self):
        self.validator = AdvancedSelfieValidator()

    def process_images(self, image_files: List[UploadFile]) -> Dict[str, Any]:
        """
        Process and validate selfie images with detailed error reporting
        """
        if len(image_files) > 20:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum 20 images allowed, received {len(image_files)}"
            )

        upload_dir = "temp_uploads"
        os.makedirs(upload_dir, exist_ok=True)

        try:
            validation_results = {
                "total_images": len(image_files),
                "valid_images": [],
                "invalid_images": [],
                "average_score": 0.0
            }

            for file in image_files:
                file_path = os.path.join(upload_dir, file.filename)
                with open(file_path, "wb") as buffer:
                    buffer.write(file.file.read())

                result = self.validator.validate_image(file_path)
                result['filename'] = file.filename

                if result['valid']:
                    validation_results['valid_images'].append(result)
                else:
                    validation_results['invalid_images'].append(result)

            # Validate selfie shot composition
            total_valid = len(validation_results['valid_images'])
            invalid_images = validation_results['invalid_images']

            if total_valid / len(image_files) < 0.9:
                error_details = [
                    {
                        "filename": img['filename'],
                        "reason": img.get('reason', 'Unknown validation failure'),
                        "score": img.get('score', 0)
                    } for img in invalid_images
                ]

                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Insufficient selfie shots",
                        "required": "90%",
                        "current": f"{total_valid / len(image_files) * 100:.1f}%",
                        "invalid_images": error_details
                    }
                )

            # Calculate average score
            if total_valid > 0:
                validation_results['average_score'] = round(
                    sum(img['score'] for img in validation_results['valid_images']) / total_valid, 2
                )

            return validation_results

        finally:
            # Clean up temporary files
            for filename in os.listdir(upload_dir):
                os.remove(os.path.join(upload_dir, filename))
            os.rmdir(upload_dir)