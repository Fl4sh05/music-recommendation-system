from urllib.parse import quote_plus

# Canonical mapping for YouTube
_EMOTION_TO_QUERY_YOUTUBE = {
    'happy': 'happy mood Bollywood songs',
    'sad': 'sad emotional songs',
    'angry': 'rock songs playlist',
    'neutral': 'lofi chill songs',
    'surprise': 'party songs playlist',
}

# Spotify playlist mapping (use full URLs or search queries)
# Using official Spotify curated playlists for better autoplay
_EMOTION_TO_SPOTIFY = {
    'happy': 'https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC',  # Happy Hits!
    'sad': 'https://open.spotify.com/playlist/37i9dQZF1DX3YSRoSdA634',  # Life Sucks
    'angry': 'https://open.spotify.com/playlist/37i9dQZF1DX1tyCD9QhIWF',  # Rage Beats
    'neutral': 'https://open.spotify.com/playlist/37i9dQZF1DX8Uebhn9wzrS',  # Chill Lofi Study Beats
    'surprise': 'https://open.spotify.com/playlist/37i9dQZF1DX0XUsuxWHRQd',  # Party Hits
}

# Reasonable defaults for labels DeepFace may emit beyond the above
# Note: 'fear' often gets misdetected, so we map it to neutral/lofi
_FALLBACKS = {
    'fear': 'https://open.spotify.com/playlist/37i9dQZF1DWXLeA8Omikj7',  # Dark & Stormy
    'disgust': 'https://open.spotify.com/playlist/37i9dQZF1DX1tyCD9QhIWF',  # Rage Beats
}

_SYNONYMS = {
    'anger': 'angry',
    'happiness': 'happy',
    'sadness': 'sad',
    'surprised': 'surprise',
}


def normalize_emotion_label(label: str) -> str:
    lbl = (label or '').strip().lower()
    return _SYNONYMS.get(lbl, lbl)


def emotion_to_query(label: str) -> str:
    """Get YouTube query for emotion (legacy)"""
    lbl = normalize_emotion_label(label)
    return _EMOTION_TO_QUERY_YOUTUBE.get(lbl) or _FALLBACKS.get(lbl) or _EMOTION_TO_QUERY_YOUTUBE['neutral']


def emotion_to_spotify(label: str) -> str:
    """Get Spotify playlist/query for emotion"""
    lbl = normalize_emotion_label(label)
    return _EMOTION_TO_SPOTIFY.get(lbl) or _FALLBACKS.get(lbl) or _EMOTION_TO_SPOTIFY['neutral']


def youtube_search_url(query: str) -> str:
    return f"https://www.youtube.com/results?search_query={quote_plus(query)}"


def spotify_search_url(query: str) -> str:
    return f"https://open.spotify.com/search/{quote_plus(query)}"
