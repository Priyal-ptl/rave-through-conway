from pathlib import Path

import torch
from dorothy import Dorothy

dot = Dorothy()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = ROOT / "models" / "musicnet.ts"



class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)

    def setup(self):
        self.rave_id = dot.music.start_rave_stream(
            model_path=str(MODEL),
            fft_size=1024,
            buffer_size=2048,
            sr=44100,
            output_device=None,
        )

        z = torch.randn((1, 16, 1))
        print(f"My latent variable: {z}")

        dot.music.audio_outputs[self.rave_id].current_latent = z
        
        
        dot.start_record(audio_output=self.rave_id, end=10000)

    def draw(self):
        dot.background((255, 255, 255))


MySketch()
