import os
from typing import List, Dict
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi

# Load YouTube API key from environment
API_KEY = os.getenv("YOUTUBE_API_KEY", "")


def search_videos(queries: List[str], max_results: int = 5) -> List[Dict]:
    """
    Search YouTube videos by query terms.
    
    Args:
        queries (List[str]): List of search terms.
        max_results (int): Maximum number of results to return.
    
    Returns:
        List[Dict]: List of video metadata (video_id, title, channel, url).
    """
    if not API_KEY:
        return []
    
    yt = build("youtube", "v3", developerKey=API_KEY)
    q = " ".join(queries)[:200]  # Limit query length
    
    resp = yt.search().list(
        q=q, part="snippet", type="video", maxResults=max_results
    ).execute()
    
    items = []
    for it in resp.get("items", []):
        vid = it["id"]["videoId"]
        title = it["snippet"]["title"]
        channel = it["snippet"]["channelTitle"]
        items.append({
            "video_id": vid,
            "title": title,
            "channel": channel,
            "url": f"https://www.youtube.com/watch?v={vid}"
        })
    
    return items


def best_timestamps(video_id: str, keywords: List[str], top_k: int = 3) -> List[int]:
    """
    Extract timestamps from captions where keywords appear.
    
    Args:
        video_id (str): YouTube video ID.
        keywords (List[str]): Keywords to search for.
        top_k (int): Maximum number of timestamps to return.
    
    Returns:
        List[int]: List of matching timestamps in seconds.
    """
    try:
        caps = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    except Exception:
        return []
    
    hits = []
    keyset = {k.lower() for k in keywords}
    
    for c in caps:
        t = c.get("start", 0)
        text = (c.get("text") or "").lower()
        if any(k in text for k in keyset):
            hits.append(int(t))
    
    return hits[:top_k]
