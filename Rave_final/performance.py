import collections
import importlib.util
import random
import sys
from pathlib import Path

import torch
from dorothy import Dorothy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAVE_MODEL = ROOT / "models"/"musicnet.ts"
AUDIO_DIR = ROOT / "audio"

dot = Dorothy(width=384, height=384, title="Conway RAVE Final")

GRID_ROWS = 48
GRID_COLS = 48
EVOLVE_EVERY = 8

RULE_BIRTH = 3
RULE_SURVIVE_MIN = 2
RULE_SURVIVE_MAX = 3

NATURE_MAX_ALIVE = 140
COMPUTER_MIN_ALIVE = 220
DOOM_MIN_ALIVE = 420
MOOD_CROSSFADE_SPEED = 0.08
MOOD_FILE_SWITCH_EVERY = 240


def check_rave_environment():
    missing = []
    if importlib.util.find_spec("torch") is None:
        missing.append("torch")
    if importlib.util.find_spec("rave") is None:
        missing.append("rave")
    if missing:
        print("Missing packages:", ", ".join(missing))
        print("Use: conda activate conway-rave")
        sys.exit(1)


class Sketch:
    alive_cells: set[tuple[int, int]]
    counter: int = 0

    def setup(self):
        self.counter = 0
        self.running = True
        self.generation = 0
        self.evolve_every = EVOLVE_EVERY
        self.current_rules = (RULE_BIRTH, RULE_SURVIVE_MIN, RULE_SURVIVE_MAX)
        self.prev_alive = -1
        self.current_mood = "nature"
        self.target_mood = "nature"
        self.current_stream_id = None
        self.target_stream_id = None
        self.last_file_switch_frame = 0

        self.alive_cells = set([tuple(cell) for cell in [[24, 24], [24, 25], [24, 26]]])
        dot.on_key_press = self.on_key

        if not RAVE_MODEL.exists():
            raise FileNotFoundError(f"RAVE model not found: {RAVE_MODEL}")
        self.rave_id = dot.music.start_rave_stream(
            model_path=str(RAVE_MODEL),
            fft_size=2048,
            buffer_size=2048,
            sr=44100,
            latent_dim=16,
            output_device=None,
        )
        self.rave = dot.music.audio_outputs[self.rave_id]
        self.rave.gain = 0.0
        self.z = torch.zeros(1, 16, 1)
        self.rave.current_latent = self.z.clone()

        self.loop_paths = {
            "nature": self.get_wavs(AUDIO_DIR / "nature"),
            "industrial": self.get_wavs(AUDIO_DIR / "industrial"),
            "computer": self.get_wavs(AUDIO_DIR / "computer"),
            "doom": self.get_wavs(AUDIO_DIR / "doom"),
        }

        self.loop_ids = {"nature": [], "industrial": [], "computer": [], "doom": []}
        self.loop_outputs = {"nature": [], "industrial": [], "computer": [], "doom": []}
        self.stream_gains = {}
        self.setup_mood_streams()
        self.apply_mood_source("nature")

    def on_key(self, key, action, mods):
        if action != dot.keys.ACTION_PRESS:
            return
        if key == dot.keys.SPACE:
            self.running = not self.running
            if not self.running:
                self.mute_audio()
            else:
                self.apply_mood_source(self.current_mood)
        if key in (dot.keys.R, ord("r"), ord("R")):
            self.drop_random_seed()
        if key == ord("["):
            self.evolve_every = min(64, self.evolve_every + 1)
            print(f"evolve_every -> {self.evolve_every}")
        if key == ord("]"):
            self.evolve_every = max(1, self.evolve_every - 1)
            print(f"evolve_every -> {self.evolve_every}")

    def draw(self):
        dot.background((0, 0, 0))

        if self.running and self.counter % self.evolve_every == 0:
            self.evolve()
            self.generation += 1

        self.counter += 1
        self.update_audio()
        self.draw_cells()

    def evolve(self):
        neighbors = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
        num_neighbors = collections.defaultdict(int)
        for row, col in self.alive_cells:
            for drow, dcol in neighbors:
                nr = (row + drow) % GRID_ROWS
                nc = (col + dcol) % GRID_COLS
                num_neighbors[(nr, nc)] += 1

        birth, smin, smax = self.current_rules
        stay_alive = {
            cell for cell in self.alive_cells if smin <= num_neighbors[cell] <= smax
        }
        come_alive = {
            cell for cell, num in num_neighbors.items() if num == birth
        } - self.alive_cells
        self.alive_cells = stay_alive | come_alive

    def drop_random_seed(self):
        pattern = [[0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [1, 1, 0, 0, 1, 1, 1]]
        row = random.randrange(GRID_ROWS)
        col = random.randrange(GRID_COLS)
        for r, row_vals in enumerate(pattern):
            for c, v in enumerate(row_vals):
                if int(v) == 1:
                    self.alive_cells.add(((row + r) % GRID_ROWS, (col + c) % GRID_COLS))

    def get_wavs(self, folder):
        return sorted(folder.glob("*.wav")) if folder.exists() else []

    def pick_mood(self, alive):
        if alive <= NATURE_MAX_ALIVE:
            return "nature"
        if alive < COMPUTER_MIN_ALIVE:
            return "industrial"
        if alive < DOOM_MIN_ALIVE:
            return "computer"
        return "doom"

    def setup_mood_streams(self):
        for mood, paths in self.loop_paths.items():
            for path in paths:
                sid = dot.music.start_file_stream(str(path), fft_size=2048, buffer_size=2048, sr=44100)
                out = dot.music.audio_outputs[sid]
                out.gain = 0.0
                self.loop_ids[mood].append(sid)
                self.loop_outputs[mood].append(out)
                self.stream_gains[sid] = 0.0

    def apply_mood_source(self, mood):
        ids = self.loop_ids.get(mood, [])
        if not ids:
            return
        self.target_mood = mood
        self.target_stream_id = random.choice(ids)

    def pick_next_stream_in_mood(self, mood):
        ids = self.loop_ids.get(mood, [])
        if not ids:
            return None
        if len(ids) == 1:
            return ids[0]
        options = [sid for sid in ids if sid != self.target_stream_id]
        return random.choice(options) if options else random.choice(ids)

    def update_mood_crossfade(self):
        speed = max(0.01, min(0.35, float(MOOD_CROSSFADE_SPEED)))
        for mood, outs in self.loop_outputs.items():
            ids = self.loop_ids.get(mood, [])
            for i, out in enumerate(outs):
                sid = ids[i]
                target = 1.0 if (mood == self.target_mood and sid == self.target_stream_id) else 0.0
                g = self.stream_gains.get(sid, 0.0)
                g = g + (target - g) * speed
                if g < 0.001:
                    g = 0.0
                if g > 0.999:
                    g = 1.0
                self.stream_gains[sid] = g
                out.gain = g

        best_mood = self.current_mood
        best_sid = self.current_stream_id
        best_gain = -1.0
        for mood, ids in self.loop_ids.items():
            for sid in ids:
                g = self.stream_gains.get(sid, 0.0)
                if g > best_gain:
                    best_gain = g
                    best_mood = mood
                    best_sid = sid
        if best_sid is not None:
            self.rave.current_file = best_sid
            self.current_mood = best_mood
            self.current_stream_id = best_sid

    def mute_audio(self):
        self.rave.gain = 0.0
        for mood, outs in self.loop_outputs.items():
            ids = self.loop_ids.get(mood, [])
            for i, out in enumerate(outs):
                sid = ids[i]
                out.gain = 0.0
                self.stream_gains[sid] = 0.0

    def update_audio(self):
        if not self.running:
            self.mute_audio()
            return

        alive = len(self.alive_cells)
        new_mood = self.pick_mood(alive)
        if new_mood != self.target_mood:
            self.apply_mood_source(new_mood)
            self.last_file_switch_frame = self.counter
        elif self.counter - self.last_file_switch_frame >= MOOD_FILE_SWITCH_EVERY:
            next_sid = self.pick_next_stream_in_mood(self.target_mood)
            if next_sid is not None:
                self.target_stream_id = next_sid
            self.last_file_switch_frame = self.counter

        self.update_mood_crossfade()

        target = 0.05 + min(alive / 250.0, 1.0) * 0.75
        self.rave.gain = self.rave.gain * 0.85 + target * 0.15

        if alive != self.prev_alive:
            self.z = self.z + torch.randn(1, 16, 1) * 0.02
            self.prev_alive = alive
        self.z = torch.clamp(self.z + torch.randn(1, 16, 1) * 0.001, -0.4, 0.4)
        self.rave.current_latent = self.z

    def draw_cells(self):
        w = dot.renderer.width
        h = dot.renderer.height
        for row, col in self.alive_cells:
            dot.fill((255, 255, 255))
            dot.circle((int((col / GRID_COLS) * w), int((row / GRID_ROWS) * h)), 4)

        dot.fill((180, 180, 180))
        dot.text(
            f"RUN:{self.running}  alive:{len(self.alive_cells)}  gen:{self.generation}  mood:{self.current_mood}",
            10,
            24,
            14,
        )


if __name__ == "__main__":
    check_rave_environment()
    sketch = Sketch()
    dot.start_loop(sketch.setup, sketch.draw)
