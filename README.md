# 🎵 Emotion-Based Music Player

A web application that detects your emotion from a photo and recommends personalized Spotify playlists!

## Features
- 📱 **Mobile-friendly web interface** - Access from any device
- 🤖 **AI-powered emotion detection** - Uses DeepFace for accurate emotion recognition
- 🎧 **Spotify integration** - Curated playlists for each emotion
- 📸 **Photo upload** - Works with camera or uploaded photos
- ⚡ **Real-time analysis** - Instant emotion detection and music recommendations

## Supported Emotions
- 😊 **Happy** → Happy Hits!
- 😢 **Sad** → Life Sucks
- 😠 **Angry** → Rage Beats
- 😐 **Neutral** → Chill Lofi Study Beats
- 😲 **Surprise** → Party Hits
- 😨 **Fear** → Dark & Stormy

## Quick Start

### 1. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the Web App
```bash
python web_app.py
```

### 3. Access from Your Device

**On PC:**
- Open browser and go to: `http://localhost:5000`

**On Phone (same WiFi):**
- Find your PC's IP address (run `ipconfig` on Windows)
- Open browser and go to: `http://YOUR_PC_IP:5000`
- Example: `http://192.168.1.100:5000`

## Usage

1. Open the web app in your browser
2. Click **"Upload Photo"** button
3. Take a photo or select from gallery
4. Your emotion will be detected automatically
5. Click **"🎵 Open Spotify Playlist"** to start listening!

## Desktop Version (Optional)

For desktop webcam capture with OpenCV:
```bash
python app.py --platform spotify --show
```

**Options:**
- `--show` - Display camera preview window
- `--platform spotify` - Use Spotify (default) or `youtube`
- `--detector-backend ssd` - Use SSD backend for better accuracy
- `--ignore-fear` - Treat fear detections as neutral

**Controls:**
- `SPACE` - Capture and detect emotion
- `h` - Manually play happy songs
- `s` - Manually play sad songs
- `a` - Manually play angry songs
- `n` - Manually play neutral songs
- `p` - Manually play party/surprise songs
- `q` - Quit

## Notes

- First run may take longer as DeepFace downloads AI models
- Requires active internet connection for Spotify links
- Best results with well-lit, clear face photos
- For camera access in browser, HTTPS is required (use Upload Photo as fallback)

## Project Structure

```
demo/
├── web_app.py              # Flask web application
├── app.py                  # Desktop version with webcam
├── emotion_player/         # Core emotion detection module
│   ├── detector.py         # DeepFace emotion detection
│   ├── playlist.py         # Emotion to Spotify mapping
│   ├── player.py           # Music player functions
│   └── camera.py           # Camera capture utilities
├── templates/
│   └── index.html          # Web interface
└── requirements.txt        # Python dependencies
```

## Technologies Used

- **Flask** - Web framework
- **DeepFace** - Emotion detection AI
- **OpenCV** - Image processing
- **TensorFlow** - Deep learning backend
- **Spotify** - Music streaming integration
