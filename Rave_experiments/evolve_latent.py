from pathlib import Path

import numpy as np
import torch
from dorothy import Dorothy

dot = Dorothy()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = ROOT / "models" / "birds2.ts"
LATENT_DIM = 16


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

        self.ptr = 0
        target = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        self.t = target[np.random.randint(len(target))]

        def on_new_frame(buffer=np.zeros(2048)):
            n = len(buffer)

            dot.music.audio_outputs[rave_id].z_bias = torch.randn(1, LATENT_DIM, 1) * 0.1

            if self.ptr > (self.t * 2048):
                self.t = target[np.random.randint(len(target))]
                z = torch.randn((1, LATENT_DIM, 1))
                dot.music.audio_outputs[rave_id].current_latent = z
                print("new z!", self.t)
                self.ptr = 0

            self.ptr += n

        dot.music.audio_outputs[rave_id].on_new_frame = on_new_frame

    def draw(self):
        dot.background((255, 255, 255))


MySketch()
