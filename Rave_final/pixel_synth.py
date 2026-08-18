import collections
import random
from pathlib import Path

from dorothy import Dorothy

HERE = Path(__file__).resolve().parent

GRID_ROWS = 48
GRID_COLS = 48

EVOLVE_EVERY = 24

#evolution rate. 
#higher no.= slower.
#stable range: 4-16. default: 8.

DETECT_EVERY = 3
#audio check every N frames. higher = fewer notes.
#stable range: 2-6. default: 3.

FIXED_NUM_ZONES = 7
#lane count: C D E F G A B




RULE_BIRTH = 3
#re-birth rule: dead cell need a miinimum of this many neighbours to turn alive
#lower = more births. Stable value: 3.

RULE_SURVIVE_MIN = 2
#survival : alive cell dies below this neighbor count. 
#lower = more birth. Stable value: 2.

RULE_SURVIVE_MAX = 3
#death by overpopulation: alive cell dies above this neighbor count. 
#lower = more death. Stable value: 3.

# stable: 3,2,3




BASE_SEED_PATTERN = [ [0, 1, 1], [1, 1, 0], [0, 1, 0],   ]

DROP_SEED_PATTERN = [ [0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [1, 1, 0, 0, 1, 1, 1],   ]

#seed pattern cheat code hehe;
# glider (moves sideways a lot): [0,1,0], [0,0,1], [1,1,1]
# r-pentomino (good growth): [0,1,1], [1,1,0], [0,1,0]
# acorn:[0,1,0,0,0,0,0], [0,0,0,1,0,0,0], [1,1,0,0,1,1,1]
# small exploder:[0,1,0], [1,1,1], [1,0,1], [0,1,0]


#Formula: midi = 12 * (octave + 1) + semitone_index
#so C3=48, D3=50, E3=52, F3=53, G3=55, A3=57, B3=5
ZONE_ROOTS = [48, 50, 52, 53, 55, 57, 59]  # C D E F G A B


ZONE_CHORDS = [
    [0, 4, 7],  # C
    [0, 3, 7],  # Dm
    [0, 3, 7],  # Em
    [0, 4, 7],  # F
    [0, 4, 7],  # G
    [0, 3, 7],  # Am
    [0, 3, 6],  # Bdim
]

#jazzy: Cmaj7 [0,4,7,11], Dm7 [0,3,7,10], Em7 [0,3,7,10], Fmaj7 [0,4,7,11], G7 [0,4,7,10], Am7 [0,3,7,10], Bm7b5 [0,3,6,10]

dot = Dorothy(width=384, height=384, title="Conway Pixel Synth")


class Sketch:
    alive_cells: set[tuple[int, int]]
    counter: int = 0

    def setup(self):

        # base state
        self.counter = 0
        self.running = True
        self.generation = 0
        self.scan_row = 0
        self.zone_was_active = {}
        self.last_reload_mtime = 0.0

        self.evolve_every = EVOLVE_EVERY
        self.detect_every = DETECT_EVERY
        self.num_zones = FIXED_NUM_ZONES
        self.base_seed_pattern = [row[:] for row in BASE_SEED_PATTERN]
        self.drop_seed_pattern = [row[:] for row in DROP_SEED_PATTERN]
        self.current_rules = (RULE_BIRTH, RULE_SURVIVE_MIN, RULE_SURVIVE_MAX)

        self.alive_cells = set()
        self.basic_seed()

        # key input
        dot.on_key_press = self.on_key

        # audio part
        idx = dot.music.start_poly_synth_stream(
            n_voices=12,
            waveform="sine",
            attack=0.005,
            decay=0.09,
            sustain=0.0,
            release=0.12,
        )
        self.synth = dot.music.audio_outputs[idx]

        dot.start_record(audio_output=idx, end=30000)

    def on_key(self, key, action, mods):
        if action != dot.keys.ACTION_PRESS:
            return
        if key == dot.keys.SPACE:
            self.running = not self.running
        if key in (dot.keys.R, ord("r"), ord("R")):
            self.drop_random_seed()
        if key == ord("["):
            self.evolve_every = min(64, self.evolve_every + 1)
            print(f"evolve_every -> {self.evolve_every}")
        if key == ord("]"):
            self.evolve_every = max(1, self.evolve_every - 1)
            print(f"evolve_every -> {self.evolve_every}")
        if key in (ord("c"), ord("C")):
            self.alive_cells = set()
            self.generation = 0

    def draw(self):
        dot.background((0, 0, 0))
        self.refresh_live_values()

        # evolve step
        if self.running and self.counter % self.evolve_every == 0:
            self.evolve()
            self.generation += 1

        # detector step
        if self.running and self.counter % self.detect_every == 0:
            self.run_detector()

        self.counter += 1

        # draw world
        self.draw_grid()

    def evolve(self):
        neighbors = (
            (-1, -1),  # Above left
            (-1, 0),  # Above
            (-1, 1),  # Above right
            (0, -1),  # Left
            (0, 1),  # Right
            (1, -1),  # Below left
            (1, 0),  # Below
            (1, 1),  # Below right
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


    #seed drops

    def basic_seed(self):
        mid_r, mid_c = GRID_ROWS // 2, GRID_COLS // 2
        self.alive_cells = set()
        for r, pattern_row in enumerate(self.base_seed_pattern):
            for c, value in enumerate(pattern_row):
                if int(value) == 1:
                    self.alive_cells.add(((mid_r + r) % GRID_ROWS, (mid_c + c) % GRID_COLS))
        self.counter = 0
        self.generation = 0
        self.scan_row = 0

    def drop_random_seed(self):

        row = random.randrange(GRID_ROWS)
        col = random.randrange(GRID_COLS)
        before = len(self.alive_cells)
        added = 0
        for r, pattern_row in enumerate(self.drop_seed_pattern):
            for c, value in enumerate(pattern_row):
                if int(value) != 1:
                    continue
                rr = (row + r) % GRID_ROWS
                cc = (col + c) % GRID_COLS
                if (rr, cc) not in self.alive_cells:
                    added += 1
                self.alive_cells.add((rr, cc))
        after = len(self.alive_cells)
        print(f"seed drop @ ({row}, {col}) added {added} cells ({after - before} net)")

    #scanning audio zone
    def run_detector(self):
        row = self.scan_row
        zone_w = GRID_COLS / self.num_zones
        for zone_idx in range(self.num_zones):
            start_col = int(zone_idx * zone_w)
            end_col = int((zone_idx + 1) * zone_w)
            alive_in_zone = 0
            for col in range(start_col, max(start_col + 1, end_col)):
                if (row, col) in self.alive_cells:
                    alive_in_zone += 1
            width = max(1, end_col - start_col)
            density = alive_in_zone / width
            active_now = alive_in_zone > 0
            was_active = self.zone_was_active.get(zone_idx, False)
            if active_now and not was_active:
                self.trigger_zone_chord(zone_idx, row, density)
            self.zone_was_active[zone_idx] = active_now
        self.scan_row = (self.scan_row + 1) % GRID_ROWS

    def trigger_zone_chord(self, zone_idx, row, density):
        # pick root + chord shape for this zone, then play notes.
        # MIDI note -> Hz conversion happens right before note_on().
        root = ZONE_ROOTS[zone_idx % len(ZONE_ROOTS)]
        chord = ZONE_CHORDS[zone_idx % len(ZONE_CHORDS)]
        octave_shift = ((GRID_ROWS - 1 - row) // 16) * 12
        vel = 0.30 + min(density, 1.0) * 0.35

        for step in chord:
            note = max(24, min(108, root + octave_shift + step))
            freq = 440.0 * (2.0 ** ((note - 69) / 12.0))
            self.synth.note_on(freq, vel=vel, attack=0.005)




#grid drawing
    def draw_grid(self):
        w = dot.renderer.width
        h = dot.renderer.height
        for row, col in self.alive_cells:
            x = int((col / GRID_COLS) * w)
            y = int((row / GRID_ROWS) * h)
            dot.fill((255, 255, 255))
            dot.circle((x, y), 4)

        zone_w = w / self.num_zones
        dot.stroke((45, 45, 55))
        dot.set_stroke_weight(1)
        for i in range(1, self.num_zones):
            x = int(i * zone_w)
            dot.line((x, 0), (x, h))
        dot.no_stroke()

        scan_y = int((self.scan_row / GRID_ROWS) * h)
        dot.stroke((80, 130, 255))
        dot.set_stroke_weight(2)
        dot.line((0, scan_y), (w, scan_y))
        dot.no_stroke()

        b, smin, smax = self.current_rules
        status = "RUN" if self.running else "PAUSED"
        dot.fill((180, 180, 180))
        dot.text(
            f"{status}  alive:{len(self.alive_cells)}  gen:{self.generation} ",
            10, 24, 16
        )
        dot.text(f"B{b} S{smin}-{smax}  R=drop  SPACE=pause", 10, 44, 12)

    # live coding refresh
    def refresh_live_values(self):
        path = HERE / "conway_pixel_midi.py"
        if not path.exists():
            return

        mtime = path.stat().st_mtime
        if mtime <= self.last_reload_mtime:
            return
        self.last_reload_mtime = mtime
        text = path.read_text(encoding="utf-8")

        new_evolve = self.read_int_from_text(text, "EVOLVE_EVERY")
        if new_evolve is not None:
            self.evolve_every = max(1, new_evolve)

        new_detect = self.read_int_from_text(text, "DETECT_EVERY")
        if new_detect is not None:
            self.detect_every = max(1, new_detect)

        old_rules = self.current_rules
        b = self.read_int_from_text(text, "RULE_BIRTH")
        smin = self.read_int_from_text(text, "RULE_SURVIVE_MIN")
        smax = self.read_int_from_text(text, "RULE_SURVIVE_MAX")
        if b is not None and smin is not None and smax is not None:
            self.current_rules = (b, smin, smax)
            if self.current_rules != old_rules:
                print(f"rules reloaded -> B{self.current_rules[0]} S{self.current_rules[1]}-{self.current_rules[2]}")

        base = self.read_pattern(text, "BASE_SEED_PATTERN")
        if base:
            self.base_seed_pattern = base

        drop = self.read_pattern(text, "DROP_SEED_PATTERN")
        if drop:
            self.drop_seed_pattern = drop




    # reads file text
    def read_int_from_text(self, text, name):
        lines = text.splitlines()
        for line in lines:
            clean = line.strip()
            if not clean.startswith(name):
                continue
            if "=" not in clean:
                continue
            right = clean.split("=", 1)[1].strip()
            if "#" in right:
                right = right.split("#", 1)[0].strip()
            if right == "":
                return None
            try:
                return int(right)
            except ValueError:
                return None
        return None

    # pulls matrix rows
    def read_pattern(self, text, name):
        lines = text.splitlines()
        start = -1
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{name} = ["):
                start = i
                break
        if start == -1:
            return None

        rows = []
        depth = 0
        started = False
        for line in lines[start:]:
            depth += line.count("[")
            depth -= line.count("]")
            if "[" in line:
                started = True
            stripped = line.strip()
            if stripped.startswith("["):
                row_text = stripped
                if row_text.endswith(","):
                    row_text = row_text[:-1]
                if row_text.startswith("[") and row_text.endswith("]"):
                    inside = row_text[1:-1].strip()
                    if inside:
                        pieces = inside.split(",")
                        row = []
                        ok = True
                        for p in pieces:
                            p = p.strip()
                            if p == "":
                                continue
                            if p.lstrip("-").isdigit():
                                row.append(int(p))
                            else:
                                ok = False
                                break
                        if ok and row:
                            rows.append(row)
            if started and depth == 0:
                break

        if not rows:
            return None
        return rows

if __name__ == "__main__":
    sketch = Sketch()
    dot.start_loop(sketch.setup, sketch.draw)
