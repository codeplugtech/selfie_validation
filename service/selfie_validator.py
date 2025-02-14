import math
import mediapipe as mp
from ultralytics import YOLO
import cv2
import numpy as np
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedSelfieValidator:
    def __init__(self):
        """
        Initialize ML models for selfie validation with quality metrics
        """
        try:
            self.face_detector = YOLO('models/yolov11n-face.pt')
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5,
                refine_landmarks=False
            )
            # Quality thresholds
            self.quality_thresholds = {
                'min_resolution': (640, 480),
                'min_brightness': 0.3,
                'max_brightness': 0.85,
                'min_contrast': 0.4,
                'min_sharpness': 50.0,
                'face_ratio_range': (0.15, 0.65)
            }

        except Exception as e:
            logger.error(f"Model initialization error: {e}")
            raise

    def validate_image(self, image_path: str) -> Dict[str, Any]:
        """
        Validate if image is a quality selfie shot
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"valid": False, "reason": "Unable to read image"}

            # Basic image quality checks
            quality_check = self._check_image_quality(image)
            if not quality_check["valid"]:
                return quality_check

            # Face detection
            face_results = self.face_detector(image)[0]
            if not hasattr(face_results, 'boxes') or len(face_results.boxes) != 1:
                return {
                    "valid": False,
                    "reason": f"Invalid face count: {len(face_results.boxes) if hasattr(face_results, 'boxes') else 0}"
                }

            # Face mesh analysis
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_mesh_results = self.mp_face_mesh.process(rgb_image)
            if not face_mesh_results.multi_face_landmarks:
                return {
                    "valid": False,
                    "reason": "No clear face detected"
                }

            # Detailed quality analysis
            quality_metrics = self._analyze_image_quality(image, face_results)

            # Calculate final score
            final_score = self._calculate_final_score(quality_metrics)

            return {
                "valid": final_score >= 0.7,
                "score": final_score,
                "quality_metrics": quality_metrics,
                "reason": self._generate_feedback(quality_metrics, final_score)
            }

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {"valid": False, "reason": str(e)}

    def _check_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform basic image quality checks
        """
        height, width = image.shape[:2]

        if width < self.quality_thresholds['min_resolution'][0] or \
           height < self.quality_thresholds['min_resolution'][1]:
            return {
                "valid": False,
                "reason": f"Image resolution too low: {width}x{height}"
            }

        return {"valid": True}

    def _analyze_image_quality(self, image: np.ndarray, face_results) -> Dict[str, float]:
        """
        Analyze various image quality metrics
        """
        # Face position and size
        face_box = face_results.boxes[0]
        x1, y1, x2, y2 = [float(coord) for coord in face_box.xyxy[0]]
        face_confidence = float(face_box.conf[0]) if hasattr(face_box, 'conf') else 0.0

        # Calculate face ratio
        face_area = (x2 - x1) * (y2 - y1)
        image_area = image.shape[0] * image.shape[1]
        face_ratio = face_area / image_area

        # Brightness and contrast
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_image) / 255.0
        contrast = np.std(gray_image) / 255.0

        # Sharpness
        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()

        # Noise estimation
        noise_level = self._estimate_noise(gray_image)

        return {
            'face_ratio': face_ratio,
            'face_confidence': face_confidence,
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': laplacian_var,
            'noise_level': noise_level
        }

    def _generate_feedback(self, metrics: Dict[str, float], final_score: float) -> str:
        """
        Generate detailed feedback based on quality metrics
        """
        if final_score >= 0.7:
            return "Good quality selfie"

        issues = []
        if metrics['face_ratio'] < self.quality_thresholds['face_ratio_range'][0]:
            issues.append("Face too small")
        elif metrics['face_ratio'] > self.quality_thresholds['face_ratio_range'][1]:
            issues.append("Face too close")

        if metrics['brightness'] < self.quality_thresholds['min_brightness']:
            issues.append("Image too dark")
        elif metrics['brightness'] > self.quality_thresholds['max_brightness']:
            issues.append("Image too bright")

        if metrics['contrast'] < self.quality_thresholds['min_contrast']:
            issues.append("Low contrast")

        if metrics['sharpness'] < self.quality_thresholds['min_sharpness']:
            issues.append("Image not sharp enough")

        return "Quality issues: " + ", ".join(issues)

    def _estimate_noise(self, gray_image: np.ndarray) -> float:
        """
        Estimate image noise level
        """
        H, W = gray_image.shape
        M = np.array([[1, -2, 1],
                      [-2, 4, -2],
                      [1, -2, 1]], dtype=np.float32)

        sigma = np.sum(np.abs(cv2.filter2D(gray_image, -1, M)))
        sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2)) if (W > 2 and H > 2) else 0.0
        return sigma

    def _calculate_final_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate final quality score based on all metrics
        """
        scores = {
            'face_ratio': self._score_face_ratio(metrics['face_ratio']),
            'brightness': self._score_brightness(metrics['brightness']),
            'contrast': self._score_contrast(metrics['contrast']),
            'sharpness': self._score_sharpness(metrics['sharpness']),
            'noise': self._score_noise(metrics['noise_level'])
        }

        weights = {
            'face_ratio': 0.3,
            'brightness': 0.2,
            'contrast': 0.2,
            'sharpness': 0.2,
            'noise': 0.1
        }

        final_score = sum(score * weights[metric] for metric, score in scores.items())
        return min(max(final_score, 0.0), 1.0)

    def _score_face_ratio(self, ratio: float) -> float:
        min_ratio, max_ratio = self.quality_thresholds['face_ratio_range']
        return max(0.0, 1.0 - abs((ratio - ((max_ratio + min_ratio) / 2)) / (max_ratio - min_ratio)))

    def _score_brightness(self, brightness: float) -> float:
        return max(0.0, min(1.0, 1.0 - abs((brightness - ((self.quality_thresholds['max_brightness'] + self.quality_thresholds['min_brightness']) / 2)))))

    def _score_contrast(self, contrast: float) -> float:
        return min(contrast / self.quality_thresholds['min_contrast'], 1.0)

    def _score_sharpness(self, sharpness: float) -> float:
        return min(sharpness / self.quality_thresholds['min_sharpness'], 1.0)

    def _score_noise(self, noise: float) -> float:
        return max(1.0 - (noise / 0.1), 0.0)
