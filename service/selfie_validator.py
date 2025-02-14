import mediapipe as mp
from ultralytics import YOLO
import cv2
from typing import Dict, Any
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedSelfieValidator:
    def __init__(self):
        """
        Initialize ML models for selfie validation.
        """
        try:
            self.face_detector = YOLO('models/yolov11n-face.pt')
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5,
                refine_landmarks=False
            )
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
            raise

    def validate_image(self, image_path: str) -> Dict[str, Any]:
        """
        Validate if image is a selfie shot.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"valid": False, "reason": "Unable to read image"}

            # Validate image resolution; require a minimum of 1000*1000 pixels.
            height, width = image.shape[:2]
            min_width, min_height = 1024, 1024
            if width < min_width or height < min_height:
                return {
                    "valid": False,
                    "reason": f"Low resolution image. Minimum required: {min_width}x{min_height}, but got {width}x{height}"
                }

            # Check face detection
            face_results = self.face_detector(image)[0]

            # Insufficient or too many faces
            if len(face_results.boxes) != 1:
                return {
                    "valid": False,
                    "reason": f"Invalid face count: {len(face_results.boxes)}"
                }

            # Convert to RGB for MediaPipe processing
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Face mesh analysis
            face_mesh_results = self.mp_face_mesh.process(rgb_image)

            if not face_mesh_results.multi_face_landmarks:
                return {
                    "valid": False,
                    "reason": "No clear face detected"
                }

            # Selfie shot validation based on a calculated score
            score = self._calculate_selfie_score(image, face_results)

            return {
                "valid": score >= 0.7,  # 70% threshold for selfie shot
                "score": float(score),
                "reason": "Selfie shot" if score >= 0.7 else "Not a clear selfie shot"
            }

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {"valid": False, "reason": str(e)}

    def _calculate_selfie_score(self, image, face_results) -> float:
        """
        Calculate selfie shot score based on various factors.
        """
        # Safely extract face box and confidence
        face_box = face_results.boxes[0]
        x1, y1, x2, y2 = [float(coord) for coord in face_box.xyxy[0]]
        confidence = float(face_box.conf[0])

        # Face size relative to image
        face_area = (x2 - x1) * (y2 - y1)
        image_area = image.shape[0] * image.shape[1]
        face_ratio = face_area / image_area

        # Evaluate image brightness and contrast
        brightness = cv2.meanStdDev(image)[0][0][0] / 255.0
        contrast = cv2.meanStdDev(image)[1][0][0] / 100.0

        # Scoring components
        face_size_score = min(face_ratio * 5, 0.5)  # Ideal if face occupies 20-40% of image area
        brightness_score = min(brightness, 0.2)
        contrast_score = min(contrast, 0.3)

        # Combine scoring components and factor in the detection confidence
        total_score = face_size_score + brightness_score + contrast_score + (confidence * 0.5)
        return min(total_score, 1.0)