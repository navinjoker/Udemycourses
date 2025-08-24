#!/usr/bin/env python3
"""
Script to fetch YouTube video transcript using youtube-transcript-api
"""

from youtube_transcript_api import YouTubeTranscriptApi
import sys

def get_youtube_transcript(video_id):
    """
    Fetch transcript for a YouTube video
    
    Args:
        video_id (str): YouTube video ID
        
    Returns:
        str: Formatted transcript text
    """
    try:
        # Create API instance and get the transcript
        api = YouTubeTranscriptApi()
        transcript_data = api.fetch(video_id, languages=['en'])
        
        # Format the transcript
        formatted_transcript = ""
        for entry in transcript_data:
            start_time = entry['start']
            duration = entry['duration']
            text = entry['text']
            
            # Convert seconds to minutes:seconds format
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            
            formatted_transcript += f"[{minutes:02d}:{seconds:02d}] {text}\n"
        
        return formatted_transcript
        
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

def main():
    video_id = "CKlfpgtiyj8"
    
    print(f"Fetching transcript for YouTube video ID: {video_id}")
    print("=" * 60)
    
    transcript = get_youtube_transcript(video_id)
    print(transcript)
    
    # Also save to file
    with open(f"/workspace/transcript_{video_id}.txt", "w", encoding="utf-8") as f:
        f.write(f"YouTube Video Transcript - Video ID: {video_id}\n")
        f.write("=" * 60 + "\n\n")
        f.write(transcript)
    
    print(f"\nTranscript also saved to: transcript_{video_id}.txt")

if __name__ == "__main__":
    main()