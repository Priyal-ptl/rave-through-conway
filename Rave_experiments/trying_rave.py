from pathlib import Path

from dorothy import Dorothy

dot = Dorothy()

HERE = Path(__file__).resolve().parent
MODEL = ROOT / "models"/"humpback.ts"


def setup():
    dot.music.start_rave_stream(
        model_path=str(MODEL),
        latent_dim=20,
        buffer_size=64,
        sr=48000,
        output_device=None,
    )


def draw():
    dot.background((255, 255, 255))


dot.start_loop(setup, draw)
