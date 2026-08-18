from pathlib import Path
import numpy as np
import torch
from dorothy import Dorothy

dot = Dorothy()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = ROOT / "models" / "birds2.ts"


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

        self.z = torch.zeros((1, 16, 1))

        self.top_left = torch.randn((1, 16, 1))
        self.top_right = torch.randn((1, 16, 1))
        self.bottom_left = torch.randn((1, 16, 1))
        self.bottom_right = torch.randn((1, 16, 1))

        def on_new_frame(buffer=np.zeros(2048)):
            mx = dot.mouse_x / dot.width
            my = dot.mouse_y / dot.height

            top = self.top_left * (1 - mx) + self.top_right * mx
            bottom = self.bottom_left * (1 - mx) + self.bottom_right * mx
            target_z = top * (1 - my) + bottom * my

            self.z = self.z * 0.8 + target_z * 0.02

            dot.music.audio_outputs[self.rave_id].current_latent = self.z


        dot.music.audio_outputs[self.rave_id].on_new_frame = on_new_frame

    def draw(self):
        dot.background((255, 255, 255))
        dot.circle((dot.mouse_x, dot.mouse_y), 20)


MySketch()
