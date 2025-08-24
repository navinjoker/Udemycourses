#!/usr/bin/env python3
"""
Alternative script to fetch YouTube video transcript
"""

from youtube_transcript_api import YouTubeTranscriptApi
import sys

def get_youtube_transcript_v2(video_id):
    """
    Fetch transcript for a YouTube video using alternative approach
    
    Args:
        video_id (str): YouTube video ID
        
    Returns:
        str: Formatted transcript text
    """
    try:
        # Try to list available transcripts first
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        
        print("Available transcripts:")
        for transcript in transcript_list:
            print(f"- Language: {transcript.language} ({transcript.language_code})")
            print(f"  Generated: {transcript.is_generated}")
            print(f"  Translation languages available: {len(transcript.translation_languages)}")
            print()
        
        # Try to get any available transcript
        try:
            # Try English first
            transcript = transcript_list.find_transcript(['en'])
            transcript_data = transcript.fetch()
        except:
            try:
                # Try auto-generated English
                transcript = transcript_list.find_generated_transcript(['en'])
                transcript_data = transcript.fetch()
            except:
                # Get the first available transcript
                transcript = list(transcript_list)[0]
                transcript_data = transcript.fetch()
                print(f"Using transcript in: {transcript.language}")
        
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

def main():
    video_id = "CKlfpgtiyj8"
    
    print(f"Fetching transcript for YouTube video ID: {video_id}")
    print("=" * 60)
    
    transcript = get_youtube_transcript_v2(video_id)
    print(transcript)
    
    # Also save to file
    with open(f"/workspace/transcript_{video_id}_v2.txt", "w", encoding="utf-8") as f:
        f.write(f"YouTube Video Transcript - Video ID: {video_id}\n")
        f.write("=" * 60 + "\n\n")
        f.write(transcript)
    
    print(f"\nTranscript also saved to: transcript_{video_id}_v2.txt")

if __name__ == "__main__":
    main()