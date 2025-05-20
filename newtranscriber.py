import io
import tempfile
import moviepy.editor as mp
import whisper
import json
from groq import Groq
import ffmpeg
import traceback
import os

class VideoTranscriber:
    def __init__(self, video_file, audio_path, json_path):
        self.video_file = video_file
        self.audio_path = audio_path
        self.json_path = json_path 
        self.client = Groq()
        self.model = whisper.load_model("small")
        self.target_size_kb = 50000

    def extract_audio(self):
        """ 
        Extracts audio from a video file and compresses it to a specific file size.
        Saves the compressed audio to self.audio_path.
        """
        video_file = self.video_file
        temp_video_file_path = None
        temp_audio_path = None

        try:
            if isinstance(video_file, str):
                temp_video_file_path = video_file
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
                    temp_video_file.write(video_file.read())
                    temp_video_file_path = temp_video_file.name
                print(f"Created temp video file: {temp_video_file_path}")

            video_clip = mp.VideoFileClip(temp_video_file_path)
            
            if video_clip.audio is None:
                raise ValueError(f"The video file {temp_video_file_path} has no audio track.")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                video_clip.audio.write_audiofile(temp_audio_file.name)
                temp_audio_path = temp_audio_file.name
            video_clip.audio.close() 
            video_clip.close() 
            print(f"Extracted audio to: {temp_audio_path}")

            audio_clip = mp.AudioFileClip(temp_audio_path)
            duration = audio_clip.duration
            audio_clip.close()
            target_bitrate = (self.target_size_kb * 8) / duration 
            min_bitrate = 32
            if target_bitrate < min_bitrate:
                target_bitrate = min_bitrate

            try:
                ffmpeg.input(temp_audio_path).output(
                    self.audio_path,
                    format="mp3",
                    audio_bitrate=f"{int(target_bitrate)}k",
                    acodec="libmp3lame"
                ).run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
                audio_size_kb = os.path.getsize(self.audio_path) / 1024
                print(f"Compressed audio saved to {self.audio_path}, size: {audio_size_kb:.2f} KB")
            except ffmpeg.Error as e:
                raise

            if temp_video_file_path and not isinstance(video_file, str) and os.path.exists(temp_video_file_path):
                try:
                    os.remove(temp_video_file_path)
                    print(f"Deleted temp video file: {temp_video_file_path}")
                except PermissionError as e:
                    print(f"Could not delete {temp_video_file_path}: {e}")
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                    print(f"Deleted temp audio file: {temp_audio_path}")
                except PermissionError as e:
                    print(f"Could not delete {temp_audio_path}: {e}")

        except Exception as e:
            raise RuntimeError(f"Failed to extract audio: {str(e)}") from e

    def transcribe(self):
        """Transcribe audio and save results to self.audio_path and self.json_path."""
        self.extract_audio()

        with open(self.audio_path, 'rb') as audio_file:
            results = self.client.audio.transcriptions.create(
                file=("audio.mp3", audio_file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
                language='en'
            )

        transcription_output = []
        data = ""
        for segment in results.segments:
            start = segment['start']
            end = segment['end']
            text = segment['text']
            print(f"[{start:.2f}s - {end:.2f}s] {text}")
            data += f"[{start:.2f}s - {end:.2f}s] {text}"
            transcription_output.append({
                'start': start,
                'end': end,
                'text': text
            })

        json_data = json.dumps(transcription_output, ensure_ascii=False, indent=4)
        with open(self.json_path, 'w', encoding='utf-8') as json_file:
            json_file.write(json_data)

        print(f"Transcription completed, JSON saved to {self.json_path}")
        return data