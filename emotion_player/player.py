import webbrowser

from .playlist import youtube_search_url, spotify_search_url


def open_youtube(query: str) -> None:
    # Try pywhatkit first (opens the first result directly), else fallback to a search page
    try:
        import pywhatkit  # type: ignore
        pywhatkit.playonyt(query)
        return
    except Exception:
        pass

    webbrowser.open(youtube_search_url(query), new=2)


def open_spotify(query_or_url: str) -> None:
    """Open Spotify playlist or search"""
    # If it's already a Spotify URL, open directly
    if query_or_url.startswith('http'):
        webbrowser.open(query_or_url, new=2)
    else:
        # Otherwise search Spotify
        webbrowser.open(spotify_search_url(query_or_url), new=2)
