import numpy as np
from pathlib import Path

from dorothy import Dorothy

dot = Dorothy()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = ROOT / "models" / "musicnet.ts"
AUDIO = ROOT / "audio"/"nature"/"nature1.wav"
LATENT_DIM = 128


class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)

    def setup(self):
        if not MODEL.exists():
            raise FileNotFoundError(f"Model not found: {MODEL}")
        if not AUDIO.exists():
            raise FileNotFoundError(f"Audio not found: {AUDIO}")

        self.rave_id = dot.music.start_rave_stream(
            model_path=str(MODEL),
            fft_size=1024,
            buffer_size=2048,
            sr=44100,
            latent_dim=LATENT_DIM,
            output_device=None,
        )

        device_id = dot.music.start_file_stream(str(AUDIO))
        dot.music.update_rave_from_stream(device_id)

        dot.start_record(audio_output=self.rave_id, end=10000)

        print(f"Model: {MODEL}")
        print(f"Input: {AUDIO}")

        

    def draw(self):
        dot.background((255, 255, 255))


MySketch()
