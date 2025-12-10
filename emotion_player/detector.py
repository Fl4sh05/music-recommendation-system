from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import numpy as np
from pathlib import Path

import cv2

try:
    from deepface import DeepFace
except Exception as e:  # pragma: no cover
    DeepFace = None  # type: ignore


@dataclass
class EmotionResult:
    dominant_emotion: str
    emotions: Dict[str, float]
    score: float  # normalized probability for dominant emotion (0-1)


class EmotionDetector:
    def __init__(self, detector_backend: str = "opencv", enforce_detection: bool = False, use_calibration: bool = False):
        if DeepFace is None:
            raise RuntimeError("DeepFace is not installed. Please `pip install deepface`.\n"  # noqa: E501
                           )
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection
        self.use_calibration = use_calibration
        self.calibration_data = None
        self._calibration_counter = 0
        self._calibration_check_interval = 5  # Only check calibration every 5 frames
        
        # Load calibration data if available
        if use_calibration:
            self._load_calibration()
        
        # Warm up (optional): build the emotion model to reduce first-call latency
        try:
            DeepFace.build_model("Emotion")
        except Exception:
            pass
    
    def _load_calibration(self):
        """Load personalized calibration data"""
        calibration_file = Path("calibration_data/embeddings.json")
        if calibration_file.exists():
            try:
                with open(calibration_file, 'r') as f:
                    self.calibration_data = json.load(f)
                print(f"✓ Loaded personalized calibration for {len(self.calibration_data)} emotions")
            except Exception as e:
                print(f"Warning: Could not load calibration data: {e}")
                self.calibration_data = None
    
    def _match_with_calibration(self, bgr_image) -> Optional[str]:
        """Match current face with calibrated emotion embeddings"""
        if not self.calibration_data:
            return None
        
        try:
            # Get embedding for current frame
            rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            embedding_obj = DeepFace.represent(
                img_path=rgb,
                model_name="Facenet",
                enforce_detection=False
            )
            
            if isinstance(embedding_obj, list):
                embedding_obj = embedding_obj[0]
            
            current_embedding = embedding_obj.get('embedding')
            if not current_embedding:
                return None
            
            # Compare with calibrated emotions using cosine similarity
            current_emb = np.array(current_embedding)
            best_emotion = None
            best_similarity = -1
            
            for emotion, data in self.calibration_data.items():
                calib_emb = np.array(data['mean_embedding'])
                # Cosine similarity
                similarity = np.dot(current_emb, calib_emb) / (np.linalg.norm(current_emb) * np.linalg.norm(calib_emb))
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_emotion = emotion
            
            # Only return if similarity is high enough
            if best_similarity > 0.75:  # Higher threshold for better accuracy
                return best_emotion
                
        except Exception:
            pass
        
        return None

    def analyze(self, bgr_image) -> Optional[EmotionResult]:
        # Try personalized calibration first if available (but skip most frames to avoid lag)
        if self.use_calibration and self.calibration_data:
            self._calibration_counter += 1
            if self._calibration_counter % self._calibration_check_interval == 0:
                calibrated_emotion = self._match_with_calibration(bgr_image)
                if calibrated_emotion:
                    # Return high confidence result from calibration
                    return EmotionResult(
                        dominant_emotion=calibrated_emotion,
                        emotions={calibrated_emotion: 100.0},
                        score=1.0
                    )
        
        # Fall back to standard DeepFace detection
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        try:
            result: Any = DeepFace.analyze(
                img_path=rgb,
                actions=['emotion'],
                enforce_detection=False,  # Don't fail if face not detected
                detector_backend=self.detector_backend,
                silent=True,
            )
        except Exception as e:
            import sys
            print(f"\n[DEBUG] Detection error: {type(e).__name__}: {str(e)[:100]}", file=sys.stderr)
            return None

        if isinstance(result, list):
            result = result[0] if result else None
        if not result or 'dominant_emotion' not in result:
            return None

        emotions = result.get('emotion') or result.get('emotions') or {}
        dom = str(result['dominant_emotion']).lower()

        score = 0.0
        if isinstance(emotions, dict) and emotions:
            total = float(sum(emotions.values()))
            if total > 0:
                score = float(emotions.get(dom, 0.0)) / total

        return EmotionResult(dominant_emotion=dom, emotions=emotions, score=score)
