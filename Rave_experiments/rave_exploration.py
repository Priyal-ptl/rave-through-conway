from pathlib import Path

import torch
from dorothy import Dorothy

dot = Dorothy()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = ROOT / "models" / "musicnet.ts"
LATENT_DIM = 128

active_dim = 0
value = 2.0


class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)

    def setup(self):
        rave_id = dot.music.start_rave_stream(
            model_path=str(MODEL),
            fft_size=1024,
            buffer_size=2048,
            sr=44100,
            latent_dim=LATENT_DIM,
            output_device=None,
        )

        z = torch.zeros(1, LATENT_DIM, 1)
        z[0, active_dim, 0] = value

        print(f"active dim: {active_dim}, value: {value}")
        print(f"My latent variable: {z}")

        dot.music.audio_outputs[rave_id].current_latent = z

    def draw(self):
        dot.background((255, 255, 255))


MySketch()
