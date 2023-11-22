import feedparser
import requests
from pydub import AudioSegment
from pyAudioAnalysis import audioSegmentation as aS

from io import BytesIO

# URL of the podcast RSS feed
rss_url = 'https://www.whatbitcoindid.com/podcast?format=RSS'

# Parse the RSS feed
feed = feedparser.parse(rss_url)

podcast_mp3_url = feed.entries[0].media_content[0]["url"]


r = requests.get(podcast_mp3_url)

audio_buffer = BytesIO(r.content)
audio_buffer_wav = BytesIO()

# Load buffer into Pydub (assuming the format is mp3)
audio = AudioSegment.from_mp3(audio_buffer)
audio[0:20000].export(audio_buffer_wav, format="wav")
audio[0:20000].export("output.wav", format="wav")



segments = aS.speaker_diarization("output.wav", n_speakers=3, mid_window=1.0, mid_step=0.2, short_window=0.15, lda_dim=0)


from whispercpp import Whisper
w = Whisper.from_pretrained("base.en",basedir="/home/skorn/Documents/ai/whisper-stuff/whisper.cpp/models")



import ffmpeg
import numpy as np

try:
y, z = (
    ffmpeg.input("blah.wav", threads=0)
    .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16_000)
    .run(
        cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
    )
)
except ffmpeg.Error as e:
    raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

#  ./main -m ./models/ggml-base.en.bin -f ../../../remove-podcast-ads-with-ai/blah2.wav -ml 1 -sow

arr = np.frombuffer(y, np.int16).flatten().astype(np.float32) / 32768.0

w.transcribe(arr)

# look at w.params!
