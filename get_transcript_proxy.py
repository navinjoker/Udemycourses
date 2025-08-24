#!/usr/bin/env python3
"""
Script to fetch YouTube video transcript using proxy configuration
"""

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import ProxyConfig
import requests
import sys

def get_youtube_transcript_with_proxy(video_id):
    """
    Fetch transcript using different approaches to bypass IP blocking
    """
    try:
        # First try with custom headers to appear more like a regular browser
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Create API instance with custom session
        api = YouTubeTranscriptApi(http_client=session)
        
        print("Attempting to fetch transcript with custom headers...")
        transcript_data = api.fetch(video_id, languages=['en'])
        
        # Format the transcript
        formatted_transcript = ""
        for entry in transcript_data:
            start_time = entry['start']
            text = entry['text']
            
            # Convert seconds to minutes:seconds format
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            
            formatted_transcript += f"[{minutes:02d}:{seconds:02d}] {text}\n"
        
        return formatted_transcript
        
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

def try_direct_url_approach(video_id):
    """
    Try to get transcript by directly accessing YouTube's caption API
    """
    try:
        import re
        import urllib.parse
        
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Get the video page
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        response = session.get(video_url)
        
        if response.status_code == 200:
            # Look for caption tracks in the HTML
            caption_pattern = r'"captionTracks":\[(.*?)\]'
            match = re.search(caption_pattern, response.text)
            
            if match:
                return "Found caption tracks in video page HTML (parsing would require additional implementation)"
            else:
                return "No caption tracks found in video page"
        else:
            return f"Failed to access video page: HTTP {response.status_code}"
            
    except Exception as e:
        return f"Error in direct URL approach: {str(e)}"

def main():
    video_id = "CKlfpgtiyj8"
    
    print(f"Attempting to fetch transcript for YouTube video ID: {video_id}")
    print("=" * 60)
    
    # Try the proxy/custom headers approach
    print("Method 1: Custom headers approach")
    transcript = get_youtube_transcript_with_proxy(video_id)
    print(transcript)
    print("\n" + "="*60)
    
    # Try direct URL approach
    print("Method 2: Direct URL analysis")
    result = try_direct_url_approach(video_id)
    print(result)
    
    # Save results
    with open(f"/workspace/transcript_attempts_{video_id}.txt", "w", encoding="utf-8") as f:
        f.write(f"YouTube Video Transcript Attempts - Video ID: {video_id}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Method 1 (Custom Headers):\n")
        f.write(transcript + "\n\n")
        f.write("Method 2 (Direct URL):\n")
        f.write(result + "\n")
    
    print(f"\nResults saved to: transcript_attempts_{video_id}.txt")

if __name__ == "__main__":
    main()