from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import base64
import numpy as np
from emotion_player.detector import EmotionDetector
from emotion_player.playlist import emotion_to_spotify, normalize_emotion_label
from emotion_player.player import open_spotify
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# Initialize detector
detector = EmotionDetector(detector_backend="ssd", enforce_detection=False)

def resize_for_model(img, max_width: int = 480):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / float(w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get image from request
        data = request.json
        image_data = data.get('image', '').split(',')[1]  # Remove data:image/jpeg;base64,
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Resize and analyze
        small = resize_for_model(frame, max_width=480)
        result = detector.analyze(small)
        
        if result is None:
            return jsonify({
                'success': False,
                'error': 'No face detected. Please ensure your face is visible and well-lit.'
            })
        
        emotion = normalize_emotion_label(result.dominant_emotion)
        spotify_query = emotion_to_spotify(emotion)
        
        # Ensure we have a proper URL
        if spotify_query.startswith('http'):
            spotify_url = spotify_query
        else:
            # Convert search query to URL
            from emotion_player.playlist import spotify_search_url
            spotify_url = spotify_search_url(spotify_query)
        
        return jsonify({
            'success': True,
            'emotion': emotion,
            'emotions': result.emotions,
            'spotify_url': spotify_url,
            'score': result.score
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Run on all network interfaces so phone can access it
    print("\n" + "="*60)
    print("🎵 Emotion-Based Music Player - Web Interface")
    print("="*60)
    print("\nStarting server...")
    
    # Try to use ngrok for public access
    try:
        from pyngrok import ngrok
        public_url = ngrok.connect(5000)
        print("\n✅ Public URL (works from anywhere):")
        print(f"   {public_url}")
        print("\n📱 Open this URL on your phone's browser")
        print("   No WiFi/firewall issues!")
    except Exception as e:
        print("\n⚠️  Ngrok not available, using local network only")
        print("\nAccess from your phone:")
        print("  1. Make sure phone and PC are on same WiFi")
        print("  2. Go to: http://10.25.144.107:5000")
    
    print("\n" + "="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
