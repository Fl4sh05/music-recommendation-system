import argparse
import time
import sys
import logging
from collections import deque, Counter
from typing import Optional, Tuple

import cv2

from emotion_player.camera import CameraStream
from emotion_player.detector import EmotionDetector
from emotion_player.playlist import emotion_to_query, emotion_to_spotify, normalize_emotion_label
from emotion_player.player import open_youtube, open_spotify


def resize_for_model(img, max_width: int = 480):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / float(w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


class LabelSmoother:
    def __init__(self, window: int = 7):
        self.window = max(1, int(window))
        self.buf = deque(maxlen=self.window)

    def update(self, label: Optional[str]) -> Tuple[Optional[str], float]:
        if label:
            self.buf.append(label)
        if not self.buf:
            return None, 0.0
        counts = Counter(self.buf)
        top_label, top_count = counts.most_common(1)[0]
        confidence = top_count / len(self.buf)
        return top_label, confidence


def parse_args():
    p = argparse.ArgumentParser(description="Emotion-Based Music Recommendation")
    p.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    p.add_argument("--interval", type=int, default=5, help="Analyze every N frames (default 5)")
    p.add_argument("--window", type=int, default=10, help="Smoothing window size (default 10)")
    p.add_argument("--threshold", type=float, default=0.7, help="Consensus threshold 0-1 (default 0.7)")
    p.add_argument("--cooldown", type=int, default=45, help="Seconds between auto-opens (default 45)")
    p.add_argument("--detector-backend", type=str, default="opencv", help="DeepFace detector backend (default opencv)")
    p.add_argument("--show", action="store_true", help="Show OpenCV preview window")
    p.add_argument("--ignore-fear", action="store_true", help="Ignore 'fear' detections (treat as neutral)")
    p.add_argument("--use-calibration", action="store_true", help="Use personalized calibration data (slower but more accurate)")
    p.add_argument("--platform", type=str, default="spotify", choices=["spotify", "youtube"], help="Music platform (default: spotify)")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    detector = EmotionDetector(
        detector_backend=args.detector_backend,
        enforce_detection=False,
        use_calibration=False  # Disable by default for speed
    )

    cam = CameraStream(index=args.camera, width=640, height=480)

    smoother = LabelSmoother(window=args.window)
    last_played_label: Optional[str] = None
    last_play_time = 0.0
    snapshot_mode = True  # Use snapshot mode by default

    print("Press 'q' to quit, 'SPACE' to capture & detect emotion.")
    print("Manual controls: 'h'=happy, 's'=sad, 'a'=angry, 'n'=neutral, 'p'=surprise")

    frame_idx = 0
    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                logging.error("Failed to read from webcam.")
                break

            display = frame.copy() if args.show else None
            
            # Show status
            if frame_idx % 30 == 0:
                sys.stdout.write(f"\rCamera ready. Press SPACE to capture...    ")
                sys.stdout.flush()

            if args.show and display is not None:
                # Add instruction text
                cv2.putText(display, "Press SPACE to capture", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Emotion-Based Music Recommendation", display)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                    
                elif key == ord(' '):  # SPACEBAR - Capture and detect
                    print("\n" + "="*60)
                    print("Capturing snapshot...")
                    print("="*60)
                    
                    # Analyze captured frame
                    small = resize_for_model(frame, max_width=480)
                    result = detector.analyze(small)
                    
                    if result is not None:
                        emotion = normalize_emotion_label(result.dominant_emotion)
                        
                        # Handle fear
                        if args.ignore_fear and emotion == 'fear':
                            print("Note: Treating 'fear' as 'neutral'")
                            emotion = 'neutral'
                        
                        # Show all emotions
                        print(f"\nDetected Emotion: {emotion.upper()}")
                        print("All emotions:")
                        for emo, score in result.emotions.items():
                            bar = '█' * int(score / 10)
                            print(f"  {emo:10s}: {score:5.1f}% {bar}")
                        
                        # Play music on selected platform
                        if args.platform == 'spotify':
                            query = emotion_to_spotify(emotion)
                            print(f"\nOpening Spotify: {query}")
                            print("="*60 + "\n")
                            open_spotify(query)
                        else:
                            query = emotion_to_query(emotion)
                            print(f"\nOpening YouTube: {query}")
                            print("="*60 + "\n")
                            open_youtube(query)
                        last_played_label = emotion
                        last_play_time = time.time()
                    else:
                        print("\n✗ No face detected in snapshot. Please try again.\n")
                
                # Manual emotion selection
                elif key == ord('h'):
                    print("\n[Manual] Playing happy songs")
                    if args.platform == 'spotify':
                        open_spotify(emotion_to_spotify('happy'))
                    else:
                        open_youtube(emotion_to_query('happy'))
                    last_played_label = 'happy'
                    last_play_time = time.time()
                elif key == ord('s'):
                    print("\n[Manual] Playing sad songs")
                    if args.platform == 'spotify':
                        open_spotify(emotion_to_spotify('sad'))
                    else:
                        open_youtube(emotion_to_query('sad'))
                    last_played_label = 'sad'
                    last_play_time = time.time()
                elif key == ord('a'):
                    print("\n[Manual] Playing angry/rock songs")
                    if args.platform == 'spotify':
                        open_spotify(emotion_to_spotify('angry'))
                    else:
                        open_youtube(emotion_to_query('angry'))
                    last_played_label = 'angry'
                    last_play_time = time.time()
                elif key == ord('n'):
                    print("\n[Manual] Playing neutral/lofi songs")
                    if args.platform == 'spotify':
                        open_spotify(emotion_to_spotify('neutral'))
                    else:
                        open_youtube(emotion_to_query('neutral'))
                    last_played_label = 'neutral'
                    last_play_time = time.time()
                elif key == ord('p'):
                    print("\n[Manual] Playing party/surprise songs")
                    if args.platform == 'spotify':
                        open_spotify(emotion_to_spotify('surprise'))
                    else:
                        open_youtube(emotion_to_query('surprise'))
                    last_played_label = 'surprise'
                    last_play_time = time.time()

            frame_idx += 1
    finally:
        cam.release()
        if args.show:
            cv2.destroyAllWindows()
        print()  # newline after status line


if __name__ == "__main__":
    main()
