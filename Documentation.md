*Project: An audio-visual simulation system based on Conway's Game of Life*

*Group: Nicolas Urrego Sandoval, Priyal Patel*




# Project Description
‘Life’ is a live audiovisual performance built around a modified version of Conway’s Game of Life. Using a custom system developed in Python, TouchDesigner and RAVE machine learning models, we manipulate the rules governing a digital organism in real time, allowing complex visual and sonic behaviors to emerge.

The simulation acts as a living system whose visual states directly generate sound. During the performance, we use a MIDI controller to alter behavioral rules, influence growth patterns and steer the evolution of the organism. While we can shape the conditions of the system, we never have complete control over the outcome.

The work explores the relationship between authorship and emergence, asking what it means to perform with systems that can surprise their creators. Rather than presenting a fixed composition, each performance becomes a unique negotiation between human decision-making, machine learning and computational behavior. Through evolving visuals, generative sound and real-time interaction, the piece transforms code into a living audiovisual instrument.



# System Diagram showing data/signal flow

<img width="4587" alt="Frame 2606849" src="https://git.arts.ac.uk/user-attachments/assets/2d81d915-7780-4b32-b375-fac751abdccd" />

![System](https://git.arts.ac.uk/user-attachments/assets/b0990e71-963a-4fd2-b6b9-45c39183d005)


# Video documentation of performance(s)

**Class Performance**: https://drive.google.com/file/d/1evnyeDV2BynUqbprZSsdDx-CWlSMPgA2/view?usp=sharing

**Crypt Performance**: https://drive.google.com/file/d/186PmDy9jZQAGyYUiyY2Ej4KavS78KiEJ/view?usp=sharing





# Process blog/Development


## Concept development

The project started with figuring out the game logic itself and understanding which Conway rules could realistically be manipulated in real time to produce interesting behaviours.

Alongside that, we were trying to define an overall thematic progression. 
We landed on four broad states - 'nature, industrial, computer and doom' as a way of representing a cycle of growth, development, technological acceleration and eventual collapse.

At the same time, we were also exploring what form the visuals should take and whether it made sense to keep everything in Python or integrate it with TouchDesigner. A lot of the early experimentation involved testing communication between the two environments and understanding what each was better suited for.

Once those pieces started falling into place, we built on a base implementation of Conway's Game of Life and gradually adapted it into a system that could be performed with in real time.


----

## Collaboration & Workflow

The project was developed collaboratively, with responsibilities divided across different parts of the system.

Priyal primarily focused on the audio generation side of the project. This involved exploring different RAVE models, experimenting with latent-space manipulation, testing different mappings between Conway simulation data and audio output and developing the final mood-based audio system used in the performance.

Niko focused mainly on the core Conway simulation and Touch Designer part of the project; including implementing the game logic, experimenting with Dorothy and other Live Coding approaches, integrating Python and TouchDesigner, cleaning and processing the simulation's data, and developing the final visual language that allowed the simulation data to drive the visuals in real time. 

That said, all of the major decisions were made collaboratively. Both of us were really excited about the audio and visual elements both and regularly switched up the tasks to experiment, try out different approaches, debug all the integration issues and code blockers and iterated on the system as a whole rather than treating the audio and visual components as completely separate pieces.

----

## Game simulation


### Finding the best way to simulate

Once we got the concept we started testing different ways to simulate Conway's Game of Life in a way that allowed us to do live changes to the rules while having a flexible enough output to process it in different ways. At first, I thought of having using [educ8's example](<./GoL_experiments/educ8s example>) which uses Pygame to run the simulation and edit the grid in real time, then capturing snapshots of the game through a script to process the visuals and the audio; this was a compelling option as it allowed us to focus on audio, with TouchDesigner working as a VFx system rather than a runtime. However, we discarded this option as it did not allow live coding and had a high-performance overhead to manage the game state.

Meanwhile, I figured that our ideal output would only comprise a list of coordinates with cells, there was no need to have a visual output at the simulation stage. I found an [example from realpython](https://realpython.com/conway-game-of-life-python/#step-3-define-and-load-the-life-patterns) which used sets to manage the game logic in a clever and lightweight way, allowing for an infinite grid to grow and not needing any external libraries to run. I used the core logic of it to have a [quick Dorothy version](<./GoL_experiments/Dorothy light version>) in which I could live-code the rules and see if it had the intended effects. The test was successful, the grow rate and patterns emergent on the cells changed a lot with changes to three rules: span requirements and minimum/maximum neighbouring cells to survive.
 

### Initial visual system

Because the simulation now only outputs a set of coordinates there were two ways forward to make the visuals, either doing a simple visual in Dorothy and capturing the video feed in TD to process it or having TD as the runtime environment and using the coordinates to create the visuals. We decided to go for the second option as it gave us greater control over the pipeline while reducing technical complexity.

This meant we now needed to run the simulation from TD, I tried to have the same script with Dorothy to run on demand through TD’s script function but it crashed the application every time (apparently TD can’t handle blocking scripts). So, I only implemented the evolution function and had a textbox with the current state to transform it into a visual element. This was surprising, as it showed we could control the simulation speed, parameters, and rules live without needing a live-coding environment.

Finally, I constructed a grid with a shape being instantiated in every coordinate the simulation’s current state had, added some custom shaders, rendered the image and applied some visual effects to make it a bit more interesting. We had a functional visual system which we could build on later.

----

## Audio Generation


### Rave_exploration/trying different models;

Started by trying different RAVE models: vintage, birds, humpback, etc. just to hear the general vibe and see what matched what we were going for. Then added a manual layer in `rave_exploration.py` where I changed one latent dimension at a time and re-ran to see if any dimensions did anything interesting. Hard to tell much without file input, often sounds different for a second then goes back to the same constant noise.

----

### Random_latent;
Got tired of manually changing dimension values every run. In this I let the model pick random latent values instead, to quickly sample the vibe of each model. Ended up choosing musicnet because it had more variety in the kinds of outputs we wanted.

Link to model: https://acids-ircam.github.io/rave_models_download

----

### Evolve_latent

Tried learning how to evolve the model using `on_new_frame`. Liked the emergent behaviour, sound keeps moving on its own. Made me think if we linked a Conway/simulation parameter here instead of random timers, RAVE could evolve based on the simulation.

----

### Evolve trial with mouse coordinates

We had an idea to map the 4 moods as different axes, move the audio output toward a specific mood region. Tried it with mouse coordinates: simplified the latent space to 2D, put a random preset at each corner, and **interpolated** between them based on mouse position (blending top-left/top-right, then blending that with bottom, so moving the mouse smoothly mixes between the four corner sounds instead of jumping).

I didnt work as well I tohught it would. Maybe because I squashed it to 2D? Not sure if that’s technically the right way to say it, but the moods didn’t read clearly, still felt like abstract drone-y stuff, and the smoothing made it slow and mushy. The crossfade idea was useful later though, when we did it with actual WAV files.

```python
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
```

https://git.arts.ac.uk/user-attachments/assets/97133f4e-78ba-4ac9-9ecf-a114583c224d


----

### Detour : conway_pixel_midi

Brief detour with `conway_pixel_midi.py` : split the grid into vertical zones(C D E F G A B lanes) and triggered MIDI notes/chords based on where cells were. 
While initially discussing the conway idea with Louis, we were wondering how to make the audio evolve organically alongside the visual state(pixel growth), he suggested scanning the frame at every interval and using the positios of the pixels as audio notes.

This was a really fun idea to explore as it sounded extremely musical and also evolved organically with the system.

```python
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
```



https://git.arts.ac.uk/user-attachments/assets/d2598d8b-068c-4448-af16-293cfb8e1715


It worked very well technically but we decided to not go ahead with it. Felt basically like a **sequencer**, pitch changes on a grid pattern, not techically connected to what the simulation was *doing* (population, growth, collapse). Didn’t feel like the data was driving the sound in a meaningful way, just translating cell positions into notes.

----

### File input
None of the pretrained models alone gave us the dynamic, distinct audio we wanted. So we switched to **file input**: feed RAVE our own WAVs and hear how musicnet modifies them. That was the turning point. We could actually choose the mood material (nature sounds, industrial loops, etc.) and RAVE adds the ML layer on top.

----

### Multiple audio inputs / moods
Built out from there, separate loop folders per mood, Conway population count picks the phase. 
Started with **3 moods: nature, industrial, computer**.

Lots of trial and error: crossfade speed, population thresholds (mood either flickering or never reaching the next phase), balancing RAVE gain vs loop gain, lag when too many streams running, conda/env setup headaches.

----

### Final_multiple file input

Finally got RAVE working in a way that felt conceptually interesting.
The audio pipeline we followed was:

Audio file loops (one mood folder at a time) → RAVE neural model (musicnet.ts) → output audio

The model morphs timbre through the latent vector z. Population data from the Conway simulation controls mood and output volume. Changes in alive cell count also nudge z, with continuous drift every frame so the timbre keeps evolving.


#### Mood System

We ended up with 4 moods: Nature-->Industrial-->Computer-->Doom (tried to be a little poetic and make it reflect the phases of civilisation)


alive count: ≤ 140	--> Nature

alive count: 141–219  -->	Industrial

alive count: 220–419  -->	Computer

alive count: ≥ 420  -->	Doom

The pick_mood() function decides which mood should be active based on the current alive cell count.
I start with the gain set to 0 for all moods. The different audio streams are always running in the background, but only the active mood is brought up through a gradual crossfade, and within each mood, only one file stream is fully up at a time.
For example, when the simulation transitions from Nature to Industrial:

-Nature gradually fades down

-Industrial gradually fades up

This ended up feeling much smoother than abruptly switching between states.



#### Mapping Conway to Audio
The volume of the final RAVE output is also controlled by the state of the simulation. We wanted the mood to get louder and more intense as the simulation progressed.
When there are very few cells alive, the output stays relatively quiet. As the population increases, the output gets louder. We added some smoothing here as well so the volume changes wouldn't feel too abrupt.
We also added:

-A timbral kick every time the alive count changes.

-A tiny amount of drift every frame so the music keeps moving and feels dynamic.

-Drift is clamped to [-0.4, 0.4] so it doesn't go completely unstable.


#### Adding Variation Within Each Mood
We wanted some dynamism inside each mood as well instead of just looping a single audio file forever.
To do this, we added multiple audio files inside each mood folder and added file-switching logic inside update_audio(). Every 240 frames (~4 seconds), it switches to another random audio file within that mood's folder.
We wanted to avoid as much manual polishing as possible and just let the model do its job, so we didn't crop or trim any of the source audio files.
File switches within a mood also use the same gradual crossfade, so changes don't feel abrupt.

https://git.arts.ac.uk/user-attachments/assets/e37d7826-52b0-45eb-9f8a-e623259e1798


## Finding the Right Thresholds
A surprising amount of time went into figuring out the exact thresholds for each mood.

The question was basically: at what alive cell count should the system transition into the next phase?

The problem was that the grid size we were using in TouchDesigner was considerably larger than the one we had originally tested in Python. Because of this, all of our original number ranges became useless and we had to spend a lot of time manually noting down population counts and testing which ranges felt right for each mood.



----


## Integrating audio with TouchDesigner
From our previous experience trying to get a Dorothy environment to run via TouchDesigner, we knew we had to have separate runtime environments for each purpose, making sure to communicate the simulation results with the audio generation environment. I figured we could have a simple version of local storage to work as the source of truth for both programmes to ensure consistent results, other options included having a live communication between them, but previous experiences have showed me this approach is unreliable and unnecessary for our requirements. That meant that we were running the Rave and audio output through python while the simulation and visuals ran in the TouchDesigner environment.

![Frame 2606847](https://git.arts.ac.uk/user-attachments/assets/2c68b6b9-d187-4404-8863-a6d922fa26f3)


----


## Live coding parameters-figuring out triggers and dynamic parameters
Out of the 4 standard rules of Conway, there were 3 rules that we could realistically modify in real time to have interesting evolutions in the simulation.  
-reproduction rate(the number of neighbors a cell needs to reproduce)
-survival min
-survival max

```python
RULE_BIRTH = 3
RULE_SURVIVE_MIN = 2
RULE_SURVIVE_MAX = 3
```

Even when we altered these in real time there weren’t a lot of interesting evolutions.
We decided to add some unpredictability in the system. We added a random seed drop which would drop a new seed at some random coordinate at our trigger. This was really fun as it added a whole new dimension of emergent behavior.

A few other interesting parameters that produced interesting outputs were the simulation/evolution speed, etc.

We ran a lot of internal tests to find out what was the right combinations to make interesting patterns and behaviors emerge.
The emergent patterns were the most interesting part of the concept for me, but for performance purposes we did need a few cheat-sheet values ready at hand to make sure the visuals never got boring.

We also added a lot of fail-safes for the performance.

* a reset parameters function that makes the 3 primary values go back to the safe default, which is 3,2,3(so this would instantly stabilize the entire simulation when pressed)
* a reset simulation function which makes the entire simulation go back to the single base seed and also resets all the parameters to its stable default(this came in really handy during the performance)
* a reset visuals function which reset all the visuals to the default values

----


## Adding a MIDI controller
We originally started with the base simulation and audio output entirely in Python. For ease of testing, we had a key press activated random seed drop.
This is the point where we started wondering about the performance aspect of it all. we got a midi controller from orb and identified various triggers.
Tbh this was more for ‘performance’ aesthetics so we won’t have to open touch designer for changing each trigger and value during the performance.

<img width="870" alt="image" src="https://git.arts.ac.uk/user-attachments/assets/25f02760-41fd-496e-a0be-fb0fb27bb3ef" />


# Performance
We performed with the piece 2 times.

## First performance: In Class

Did a trial performance in class. Concept was there but the audio wasn’t interesting enough over the full run, moods didn’t feel distinct or dramatic enough. After that we added a **4th mood (doom)** for overload/collapse, added more loops per folder, kept tuning thresholds.

* the audio outputs did not sound as interesting. The different zones we had decided seemed too disconnected and did not seem like a part of the same simulation.
Maybe smoother transitions. Or more gain would be helpful?
* the audio output and video output did not seem connected yet. 
* there was not enough context provided for users to make sense of the simulation/system. (Included text)
* niko and I hadn’t yet decided how to split up the controllers so this led to a lot of confusion during the live performance. (In the revision we decided to clearly split up our respective controls)
* there were some interesting presets we had discovered during our explorations and we weren’t able to show them during the performance (we were thinking of having shortcuts to the specific interesting ones)
<img width="973" alt="image" src="https://git.arts.ac.uk/user-attachments/assets/4cb635fd-86ed-4f1d-b723-0dd840fd5288" />

----

## Second performance: At the crypt
* there were a few points where the rest collided with the least reproduction rule set, this ended up killing the brand new cells spawned to instantly die.
* the audio was improved a lot from the earlier class performance. We actually cleaed up the earlier audio input files for each mood and adjusted the transitions to be better.
* adding the live stats really help the visual output look way better and offered more system feedback to covery the conceptual element of the performance better.
![Frame 2606848](https://git.arts.ac.uk/user-attachments/assets/a4392eb1-c2ed-40fb-8afc-0653896c1a81)




# Research Plan
We followed the user research and testing framework from class and had the following results filled out after discussing in class;


### Risks

**What is the riskiest moment in my piece, and what could go wrong?**

* Not being able to retain control over the system/simulation.
* Technical issues with the Python–TouchDesigner / Python–RAVE pipeline.
* RAVE models failing; audio output being undistiguishable from noise.
* Device failures because of overloads.
* New untested simulations might break the system (even after stress testing).

---

### Audience Perception

**Where am I guessing about how the audience will respond?**

* We are assuming that a higher understanding of the system, its controls, and subsequent behaviour would result in more interest and engagement from the audience.
* How much effect does revealing the intent and concept behind the performance have on audience perception? Does the critical message get conveyed?


**What relationship am I asking the audience to have with the data/system/medium?**

* To gather an understanding of the data and system.
* To notice the correlation between the data and how it drives the system.


**What kind of changes do we trigger, and when, to make the change coherent to the audience?**

* Define specific moments of discovery to establish the rules, controls, parameters, and intentionality of the piece/ecosystem.


**What does the technology do that an audience may or may not perceive as intentional?**

* How much of the system's behaviour is understood as a deliberate performance choice versus emergent behaviour?

---

### Transparency & Revealing the Process

**Questions**
* Is the intentionality of specific moments in our performance evident to the audience? How does it affect their perception/experience?
* How much does revealing the process/behind-the-scenes affect perception?
* Revealing the failings: if something goes wrong, what happens if we actually show the process of debugging?
* Revealing the dialogue between us and the system: what kind of effect does it have, and how does it change perception?
* Revealing the actual parameters we are modifying to control the changes happening in the system.

---

### Success Criteria

**If a stranger watched this and described it back to me, what would I be horrified to hear?**

* It's awful.
* These people suck.
* I don't understand anything that happened.
* The critical message wasn't conveyed.
* It was reduced to just an audiovisual piece.
* Did they actually perform?

**What would I be delighted to hear?**

* They could identify a relationship between our actions and the behaviour of the system.
* They understood that we were shaping conditions rather than directly controlling outcomes.
* They became interested in how the system worked.
* They recognised the tension between performer agency and emergent behaviour.

---

### Performer Agency & Emergence

**What is the difference between what I think is happening and what is actually happening on stage?**

* We are controlling the narrative, but we don't have complete control over the outcome.
* We can drive specific parameters, but leave room for emergent behaviour.
* The performance exists as a negotiation between our intentions and the system's responses.


----



## Survey

We did a small survey during the class performance;

<img width="200" alt="image" src="https://git.arts.ac.uk/user-attachments/assets/89b8cf77-22bd-45a9-8671-a66fcec9f5a8" />

### Feedback;

1.Human v/s Machine Agency (0:machine, 10:human)
   
<img width="691" alt="image" src="https://git.arts.ac.uk/user-attachments/assets/1a563955-0350-43ca-8dbc-61f7986b75f8" />


2.Flow-how did the performance feel? 

<img width="693" alt="image" src="https://git.arts.ac.uk/user-attachments/assets/b80ad0ad-f959-4614-b0d8-97cb38a7c889" />


3.Friction-did anything feel off?
   
<img width="687" alt="image" src="https://git.arts.ac.uk/user-attachments/assets/0d4fb86c-8ffe-4393-b10d-a045e7cef320" />

---

# Project files

Final code file: [Final code file](Rave_final/performance.py)

Final touchdesigner file: [Touchdesigner file](Final.toe)


---



# Priyal's Reflection & Evaluation

### Performance Evaluation: Audience Feedback and Personal Observations

We performed the piece twice, first in class and then later at the Crypt.

There were a lot of things that didn't go completely right...while these weren't explicit failures they were important observations which helped us redirect our approach at various stages of development.


* One of the most useful pieces of feedback from the class performance was that 40% of respondents felt the visuals and sound did not seem connected. This was honestly one of my biggest fears throughout the project. We had decided to split up the tasks quite naturally, with one person focusing a little more on the audio and the other on the visuals, and I was scared they wouldn't sound coherent enough when mashed together. More like, the association between the cell growth and behaviour and the audio change would not be visible at all.
In the feedback survey from class, people did explicitly say this as well, which was a little heartbreaking. But it also validated our own observations after the performance and gave us a clear direction for improvement. We ended up tweaking the audio mappings and transitions quite a bit before the Crypt performance and it did feel noticeably better.

* The feedback was otherwise pretty encouraging. Most participants felt there was a balance between human and machine agency, which was nice because that was one of the main things we were trying to explore. We wanted to create a system that could be influenced and guided, but never completely controlled.

* The open-ended responses were pretty interesting as well. Several people described the visuals as bacteria, algae, growth or evolution, while others commented on how it felt both natural and digital at the same time. I actually really liked this because it meant people were engaging with the behaviours of the system itself rather than just seeing it as a Conway replica.

* Another thing that came up was that people wanted more context, which does make a lot of sense. We had spent weeks building and testing the simulation, so a lot of the behaviours felt obvious to us. For someone seeing it for the first time though, there wasn't always enough information to understand what exactly we were controlling and why certain changes were happening.
We ended up adding live stats for the crypt version to give better system feedback to the audience, and it ended up adding a lot of aethetic value as a bonus as well

<img width="339" alt="image" src="https://git.arts.ac.uk/user-attachments/assets/e6723e4c-efba-414e-a448-9c535872583c" />


* Internally, one of the biggest struggles was performer agency. Niko and I hadn't yet decided how to split up the controllers during the first performance, which led to a lot of confusion during the live performance itself. We were also simplifying a lot of controls out of fear that something might go wrong, but that also meant the performers(us) weren't visibly doing much to change the state of the system. The human vs machine balance was honestly one of the constant struggles throughout the project.

* The Crypt performance exposed a few technical edge cases as well. There were a few points where the reset collided with the lowest reproduction rule set, which caused newly spawned cells to instantly die. Nothing completely broke, but it was a good reminder that even after stress testing, emergent systems will always find new ways to surprise you.

* Overall, both performances were really useful. Some of our assumptions were validated, some weren't, and a lot of the audience feedback ended up matching concerns we already had ourselves. More than anything, the performances helped us understand what parts of the system people were actually noticing, what was getting lost, and how much context audiences need to engage with an emergent system.


### General Reflection

* The most interesting part of the project for me was the tension between control and emergence.
Although we could modify the rules governing the simulation, trigger seed drops and influence the overall behaviour of the system, we could never fully predict the outcome. The performance became less about directly creating visuals and sounds and more about shaping the conditions from which they could emerge.

* One thing that really stuck with me was the discussion around predictive systems. The Markov prediction exercises were actually what got me thinking about Conway's Game of Life in the first place. I was initially interested in using Markov prediction layers alongside classification systems to drive audio generation. That eventually led me down a rabbit hole of thinking about systems that evolve over time, probabilities, prediction and emergence, which is how I ended up becoming interested in Conway's Game of Life as the foundation for this project.

* I also found myself thinking a lot about performer agency throughout the unit. Whether it was machine learning models, generative systems or audiovisual tools, there was always this question of how much control is being handed over to the system. How much creativity is being forfeited? How much of the performer's own biases are already embedded in the choice of dataset, model or parameters? That tension between control and emergence ended up becoming one of the central themes of our project.

* On a technical level, the course pushed me to become much more comfortable moving between Python, TouchDesigner and machine learning tools. More importantly, it taught me to think in terms of parameter spaces, mappings and relationships between inputs and outputs rather than individual effects or features.

* Another really random learning; having a folder structure for every mood instead of linking individual input files by name was such a lifesaver. It helped us out so much when we were experimenting with different audio files trying to find the best moods

* Finally, the course reinforced something I keep running into across design, coding and art: constraints are usually more interesting than unlimited freedom. Conway's Game of Life is probably the perfect example of that. A handful of simple rules can create surprisingly complex, unpredictable and performable behaviour.

--- 

# Niko's Reflection & Evaluation

## Performance Evaluation: Audience Feedback and Personal Observations

#### First performance: Class
* The performance didn’t come out the way we wanted but that was good: Before the performance I feared that people would think that we had a preset video and just played it, but once we got there and started playing with the system, we were in enough control to manage pacing, rhythm and overall experience. This also meant that we could make mistakes and we did, which showed the liveliness was definitely there. This was supported by the feedback, with a majority of responses stating they found a was a balance between human and machine agency, the goal we set of creating a system we could steer In a direction but not fully control was achieved.
* From the feedback we gathered we noticed that around 40% of respondents felt the visuals and sound were not connected. I feel like the splitting of tasks we did, which was helpful to achieve the ambitious result we wanted, had a big part on this perceived split; another factor is the fact that the visual changes happen much faster than the audio changes, which means that if not handled correctly the audio can lag behind the actual simulation on screen. To solve this, we proposed working together in some final adjustments to the project before the performance at the crypt.
* From the open-ended responses my main takeaway is that the system wasn’t as visible as it should and the audience felt that lack of information, some answers asked for further context on the changes we were making, the effects they had and the general message of the piece. However, a positive is that the intuitive analysis of the piece mentioned natural elements being balanced with digital elements, the visuals evoking bacteria or cells, and control being the topic being questioned.

We had a meeting to share our reflections and noticed similar observations, so we decided to make changes to both the audio and visual systems: The audio mappings were modified to have more variety, a more rhythmic base was added and transitions were adjusted to have smoother but more noticeable changes; while visually the focus was in making the performance feel more cohesive, we wanted the state of the game to be more visible so we added live stats to give better system feedback to the audience, remapped some settings and states, and added more backgrounds and tweaked the shaders to have a 1:1 relationship between the audio states and visual states.

#### Second performance: The Crypt
* For this performance we decided to split the tasks up, Priyal would manage visuals only and I would manage the simulation rules only. This meant we had more defined tasks and could communicate our wants in a more straightforward way. This had a great effect, firstly because it created a dialogue between us that helped to build a narrative, and secondly because it meant we could sit back and analyse the state of the system between changes; this was also aided by having a more interesting audio generation and visual feedback without our full intervention.
* Still, even with the gained agency from us, the system still surprised us with the behaviours emerging from both the simulation and the audio; for example, the ruleset and state got to a point where adding seed didn’t change the state of play. However, by simply having more experience performing we were able to incorporate those moments into our set, which surprised (and pleased) me greatly.
* After the set, plenty of people approached us to ask more about the project, the technical aspects behind it, and some showed a deep conceptual understanding of the performance. Overall, I think the changes we made, informed by both the audience’s feedback and our own experience performing, had an incredible effect on the end result. I am shocked at how much we were able to adjust in two days, which shows how projects are also emergent beings, with aesthetic qualities developing slowly and meaning layering as you mature the project more.


### General Reflection

One of the most interesting things that happened during development was how we dialled back the human agency in the performance. At first, we wanted to control the ruleset, the seed, and some audiovisual characteristics; in part because I feared without direct human intervention the performance wouldn’t be interesting. But as the project matured, we noticed this would’ve had the opposite effect; the main dynamic of ‘Life’ is a cycle of changing, evaluating, and reacting, which can only be achieved by having limited control and stimulating emergent behaviours. 

An overarching topic that rose from this project was the importance of shaping the conditions for thriving rather than just forcing things to be a certain way. In our latest iteration, I noticed the performance was about giving direction for the system to evolve and grow while complementing its behaviour through visual elements with a bit of flair. It also showed during development, we had to work heavily on testing and calibrating everything so it fitted the system’s emergent behaviours; it showed me a huge element in performative arts is giving yourself the proper tools to create in a focused manner and that requires a lot of setups, in a funny way, it works like our project does. Even before the performance aspect, while developing we made some conscious decisions to have an environment where we could experiment and iterate quickly, and it allowed us to achieve a result which differed from our initial vision but fulfilled its direction completely.

From a technical perspective, this piece helped me notice the importance of having a clear information and systems view of the project. Thanks to our constant consideration of the bigger picture and concern with how the different parts would play along we were able to develop the project quickly and without major reprocesses, which translated in much more time to experiment, refine and play with the tool we built. 

Finally, I noticed that complexity and richness emerge from unexpected points, although our vision had a big part on making this a successful project, it was experimentation and reacting to hurdles which made the end result look the way it did. On the other hand, simplicity can come naturally, the project asked questions constantly and surprised me by having desirable properties in stages that I thought were too early for them to surge. It makes me reflect on my own practice and motivates me to incorporate more rules and play in my creative method.


# References

https://tinkerings.org/2016/09/24/cellular-automata-explorer/ 

https://www.are.na/agnes-cameron/src-cellular-automata 

https://www.synthtopia.com/content/2008/05/29/glitchds-cellular-automaton-sequencer-for-the-nintendo-ds/ 

https://www.synthtopia.com/content/2009/04/29/game-of-life-music-sequencer/ 

https://www.synthtopia.com/content/2011/01/12/game-of-life-music-sequencer-for-ios-runxt-life/

https://realpython.com/conway-game-of-life-python/#step-3-define-and-load-the-life-patterns

