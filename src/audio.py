import ffmpeg
from .utils import named_tempfile

def extract_wav_from_video(video_bytes: bytes, in_suffix: str = ".mp4", sr: int = 16000):
    """Return path to a temp .wav extracted from video bytes (mono, target sr)."""
    with named_tempfile(suffix=in_suffix) as in_path, named_tempfile(suffix=".wav") as out_path:
        with open(in_path, "wb") as f:
            f.write(video_bytes)
        try:
            (
                ffmpeg
                .input(in_path)
                .output(out_path, format='wav', acodec='pcm_s16le', ac=1, ar=sr, af='loudnorm')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            print("FFMPEG ERROR - STDOUT:", e.stdout.decode('utf8'))
            print("FFMPEG ERROR - STDERR:", e.stderr.decode('utf8'))
            raise e
            
        return out_path