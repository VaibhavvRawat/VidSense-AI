"""
downloader.py
Handles downloading YouTube audio and compressing it for transcription.
"""

import os
import yt_dlp


def download_audio(link: str, file_name: str = "audio.mp3") -> str:
    """Download audio from a YouTube URL using yt-dlp."""
    ydl_opts = {
        "extract_audio": True,
        "format": "worstaudio",
        "overwrites": True,
        "outtmpl": file_name,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(link, download=True)
    return file_name


def compress_audio(input_file: str, output_file: str = "compressed.mp3") -> str:
    """Compress and resample audio to 16kHz mono using ffmpeg."""
    os.system(f"ffmpeg -y -i {input_file} -ar 16000 -ac 1 {output_file}")
    return output_file
