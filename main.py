import os
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
import mediapipe as mp
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedSelfieValidator:
    def __init__(self):
        """
        Initialize ML models for selfie validation
        """
        try:
            self.face_detector = YOLO('ultralytics/models/yolov8n-face.pt')

            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5,
                refine_landmarks=False  # Add this parameter
            )

        except Exception as e:
            logger.error(f"Model initialization error: {e}")
            raise

    def validate_image(self, image_path: str) -> Dict[str, Any]:
        """
        Validate if image is a selfie shot
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"valid": False, "reason": "Unable to read image"}

            # Check face detection
            face_results = self.face_detector(image)[0]

            # Insufficient or too many faces
            if len(face_results.boxes) != 1:
                return {
                    "valid": False,
                    "reason": f"Invalid face count: {len(face_results.boxes)}"
                }

            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Face mesh analysis
            face_mesh_results = self.mp_face_mesh.process(rgb_image)

            if not face_mesh_results.multi_face_landmarks:
                return {
                    "valid": False,
                    "reason": "No clear face detected"
                }

            # Selfie shot validation
            score = self._calculate_selfie_score(image, face_results)

            return {
                "valid": score >= 0.7,  # 70% threshold for selfie shot
                "score": float(score),  # Explicitly convert to float
                "reason": "Selfie shot" if score >= 0.7 else "Not a clear selfie shot"
            }

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {"valid": False, "reason": str(e)}

    def _calculate_selfie_score(self, image, face_results) -> float:
        """
        Calculate selfie shot score based on various factors
        """
        # Safely extract face box and confidence
        face_box = face_results.boxes[0]

        # Convert tensor values to float
        x1, y1, x2, y2 = [float(coord) for coord in face_box.xyxy[0]]
        confidence = float(face_box.conf[0])

        # Face size relative to image
        face_area = (x2 - x1) * (y2 - y1)
        image_area = image.shape[0] * image.shape[1]
        face_ratio = face_area / image_area

        # Image composition and brightness
        brightness = cv2.meanStdDev(image)[0][0][0] / 255.0
        contrast = cv2.meanStdDev(image)[1][0][0] / 100.0

        # Scoring components
        face_size_score = min(face_ratio * 5, 0.5)  # Face should occupy 20-40% of image
        brightness_score = min(brightness, 0.2)
        contrast_score = min(contrast, 0.3)

        # Combine scores
        total_score = face_size_score + brightness_score + contrast_score + (confidence * 0.5)
        return min(total_score, 1.0)


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
