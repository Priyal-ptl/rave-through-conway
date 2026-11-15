<img width="1277" alt="Life, the performance" src="https://git.arts.ac.uk/user-attachments/assets/06e6e13d-73ed-4029-a89c-1bb022ae6c9a" />

# Life

A live audiovisual performance built on Convay's Game of Life. The simulation is the score. Population, growth and collapse pick the sound, and a MIDI controller lets us push the rules around while it is running.

Made with **Nicolas Urrego Sandoval**. I worked mainly on the audio: RAVE, the latent vector, and the mood system that ties cell count to sound. Niko worked mainly on the Game of Life and the TouchDesigner visuals. The split was never clean. The mappings, the thresholds and the two performances were decided together.

Python runs the audio. TouchDesigner runs the simulation and the image. RAVE (IRCAM's neural audio model) sits between a folder of loops and the speakers.

Longer development notes: [Documentation.md](Documentation.md)

Performance files: [Rave_final/performance.py](Rave_final/performance.py) · [Final.toe](Final.toe)

**First performance:** https://drive.google.com/file/d/1evnyeDV2BynUqbprZSsdDx-CWlSMPgA2/view?usp=sharing

**Crypt performance:** https://drive.google.com/file/d/186PmDy9jZQAGyYUiyY2Ej4KavS78KiEJ/view?usp=sharing

## Concept

Four states, in order: nature, industrial, computer, doom. A small colony, a machine, a system running too hot, a collapse. Conway already does that if you leave it alone. Birth, survival and overcrowding are the whole rule set, and changing one number changes whether the pattern grows, sits, or dies.

We wanted to perform that, not press play on it. The piece is about how much of a system you can steer and how much you have to react to. We can drop a seed, slow the generation, or push the rules off the stable 3/2/3 setting. We cannot decide which cells live.

![Still from the piece](https://git.arts.ac.uk/user-attachments/assets/4155f601-bcd6-4355-956a-705253d0ca22)

## How the system is split

TouchDesigner owns the simulation and the visuals. Python, through Dorothy, owns RAVE and the audio. They do not share a process.

We tried running Dorothy from inside TouchDesigner's script operator. It crashed. TouchDesigner does not like a blocking script sitting in its frame. So the simulation writes its state, and the audio side reads the thing it actually needs, which is the alive-cell count. A live socket would have been another thing to fail on stage. A shared count was enough.

![Signal flow between TouchDesigner and the RAVE process](https://git.arts.ac.uk/user-attachments/assets/2c68b6b9-d187-4404-8863-a6d922fa26f3)

![System diagram](https://git.arts.ac.uk/user-attachments/assets/2d81d915-7780-4b32-b375-fac751abdccd)

![System diagram, second pass](https://git.arts.ac.uk/user-attachments/assets/b0990e71-963a-4fd2-b6b9-45c39183d005)

## The simulation

The game state is a set of `(row, col)` cells, not a dense grid and not a rendered frame. Each generation counts the eight neighbours, then applies three numbers:

```python
RULE_BIRTH = 3        # dead cell with this many neighbours comes alive
RULE_SURVIVE_MIN = 2  # alive cell dies below this
RULE_SURVIVE_MAX = 3  # alive cell dies above this
```

`3, 2, 3` is the classical Game of Life and the safe preset. Anything else is a performance choice. The grid in `performance.py` is 48×48 and wraps. The generation runs every `EVOLVE_EVERY` frames (8 by default), so speed is a dial rather than "as fast as the loop goes".

A key press drops an acorn seed at a random origin. That was the useful kind of unpredictability. Changing the three rules alone did not always produce a new pattern. Throwing a new shape into a quiet grid did.

Early on we tried a Pygame build (the educ8s example in `GoL_experiments`) and captured frames out of it. It was fine as a Game of Life and wrong as an instrument: no live coding, and too much time spent on the game window. A Dorothy version of the set-based logic was enough to hear what the three rules did. From there the coordinates went to TouchDesigner, which instances a shape per live cell, runs shaders, and is what the audience actually sees.

The Python file still draws a debug view, white dots and a line of live stats. On stage the image is the TouchDesigner patch, [Final.toe](Final.toe). The `.toe` history in `TD_change_history/` is the patch as it moved, not a second project.

## RAVE

RAVE is a variational autoencoder for audio. You hand it a latent tensor `z` and it decodes a sound. You can also encode an incoming buffer into that latent and let the model reshape it. Dorothy wraps this as `start_rave_stream`.

The pretrained models were too big to commit. Put the `audio` and `models` folders next to the code. Both are in this Drive folder:

https://drive.google.com/drive/folders/1cOLiZIYgVCfwNzLbNU3qZ5WQ9h5QGzvF?usp=sharing

Models are from the IRCAM RAVE model list: https://acids-ircam.github.io/rave_models_download

The loops are from Pixabay: https://pixabay.com/music/

The model files and the loops belong to their authors. They are not part of this repo.

### Getting to a model that was usable

`Rave_experiments/` is the path, in order.

`trying_rave.py` and `rave_exploration.py` load a `.ts` model and poke one latent dimension at a time. I tried vintage, birds, humpback, musicnet. With no file coming in, a dimension will colour the noise for a moment and then the decoder settles back to a constant. Not a performance.

`random_latent.py` samples `z` from a normal distribution so I could hear a model without hand-editing 16 numbers. Musicnet was the one with more than one kind of output. The others were closer to a single creature.

`evolve_latent.py` writes a new `z_bias` on every audio frame, with a random hold time. The sound moves on its own. That was the first version that felt like it could be attached to a simulation instead of to a timer.

`evolve_mouse.py` was the four-mood idea done badly on purpose, to see if it worked. Four random latent presets sit in the corners of the screen. The mouse blends the top edge, blends the bottmm edge, then blends those two results. `z` eases toward that target (`0.8 * old + 0.02 * target`). It did not read as four moods. Squashing the latent to a mouse x/y made everything a slow drone. The crossfade itself was worth keeping. The corner trick was not.

### The sequencer detour

`Rave_final/pixel_synth.py` splits the grid into seven vertical lanes, C D E F G A B, and fires MIDI notes from where the live cells are. Scan the frame every few ticks, turn cell positions into pitch. Louis had suggested this while we were still deciding how the sound should follow the picture. It is musical, and it grows when the pattern grows. It is also a sequencer. Pitch follows where the cells are, not what the colony is doing. Population, takeover and collapse all sound like notes on a grid. We stopped there.

### File input, which is the version we performed

A pretrained decoder on its own never gave us four distinct states. Feeding it our own loops did. `file_input.py` is the small version: one wav, encoded into musicnet, recorded for ten seconds. `performance.py` is the full one.

The chain is:

loop files, one mood folder at a time → RAVE (`models/musicnet.ts`, latent dim 16, 44.1kHz, fft 2048) → speakers

The alive count picks the mood:

| Alive cells | Mood |
|---:|---|
| ≤ 140 | nature |
| 141–219 | industrial |
| 220–419 | computer |
| ≥ 420 | doom |

Those edges were found by ear, after the TouchDesigner grid turned out larger than the Python grid we had tuned against. The first set of thresholds never reached the later moods, or flickered between them.

Every loop in every folder is already streaming, with gain at 0. `pick_mood` chooses a target. Gain eases toward 1 for the active file and toward 0 for the rest, at `MOOD_CROSSFADE_SPEED = 0.08`. A hard cut read as four separate tracks. The fade reads as one system getting worse.

Inside a mood, another file is chosen every 240 frames, about four seconds, and it uses the same fade. The source files are not trimmed. The model is doing that job.

Output level follows population, smoothed so a single generation does not pump the volume:

```python
target = 0.05 + min(alive / 250.0, 1.0) * 0.75
self.rave.gain = self.rave.gain * 0.85 + target * 0.15
```

The latent moves in two ways. If the alive count changed since last frame, a small random kick (`randn * 0.02`) marks the change in timbre. Every frame, a smaller drift (`randn * 0.001`) keeps the tone from freezing. `z` is clamped to `[-0.4, 0.4]`. Past that, musicnet stops sounding like the loop and starts sounding like a fault.

```python
if alive != self.prev_alive:
    self.z = self.z + torch.randn(1, 16, 1) * 0.02
    self.prev_alive = alive
self.z = torch.clamp(self.z + torch.randn(1, 16, 1) * 0.001, -0.4, 0.4)
self.rave.current_latent = self.z
```

Folders rather than named files mattered more than it should have. Dropping another wav into `audio/doom/` was enough. No path edits in the middle of a rehearsal.

## TouchDesigner and the controller

The patch takes the live cells and instances geometry on that set of coordinates, then shaders and a few post effects. Three rules are exposed live: birth, survival minimum, survival maximum. So is the generation speed.

Three resets, because an emergent system will find a state you do not want in front of an audience:

- parameters back to 3, 2, 3, which settles the simulation
- the whole grid back to the original seed, parameters included
- the visual settings back to their defaults

We started on key presses. For the shows we moved the triggers to a MIDI controller so neither of us had to open the patch to change a number.

![MIDI controller layout](https://git.arts.ac.uk/user-attachments/assets/25f02760-41fd-496e-a0be-fb0fb27bb3ef)

## The two performances

The first showing had the concept and not the mix. The four states were not distinct enough, the sound and the image felt like separate pieces, and we had not decided who held which controls, so a lot of the set was us staying out of the way of the system. About 40% of the people we asked said the visuals and the sound did not seem connected. That was the note we were afraid of, and it was specific enough to use. Before the Crypt we retuned the thresholds, added more loops per mood, and put live stats on the image so a change in the rules had something to point at.

![First showing](https://git.arts.ac.uk/user-attachments/assets/4cb635fd-86ed-4f1d-b723-0dd840fd5288)

At the Crypt the roles were split on purpose. I took the visuals. Niko took the rules. The audio sat in the same world as the image more of the time. One failure mode showed up that rehearsal had not: a reset landing on the lowest birth rule, so the new seed died on the next generation. The system did not crash. It just declined to be reseeded. We played through it.

![Crypt performance](https://git.arts.ac.uk/user-attachments/assets/a4392eb1-c2ed-40fb-8afc-0653896c1a81)

People described the image as bacteria, algae, growth. A few called it both natural and digital. That was closer to the piece than "a Game of Life demo". The agency question landed where we wanted it: most people felt the control was shared between us and the system, not owned by either.

## Afterwards

The part I keep is the limit. We could change the rules, drop a seed, and ride the volume with the population. We could not aim the pattern. The set became a loop of change something, look, change something else.

I had arrived at Convay from the other direction, from Markov-style prediction and the idea of a system that only knows what tends to follow. Life is that idea with the prediction taken out. The rules are tiny, the state is the whole history, and the performance is whatever the state does next.

A practical note, because it saved the last week: one folder per mood, gains at zero until a file is chosen, and a clamp on `z`. Most of the "the model sounds broken" afternoons were an unclamped latent or a threshold copied from a smaller grid.

![Audience notes on agency](https://git.arts.ac.uk/user-attachments/assets/1a563955-0350-43ca-8dbc-61f7986b75f8)

## Layout

```
Rave_final/performance.py     the instrument: Life + RAVE + moods
Rave_final/pixel_synth.py     the MIDI-lane detour
Rave_experiments/             latent studies, file input, the mouse blend
GoL_experiments/              pygame and the early Dorothy simulation
Final.toe                     TouchDesigner patch
TD_change_history/            patch backups
```

Run the audio side with the conda env that has PyTorch, RAVE and Dorothy (`conway-rave` in our setup), from the repo root, with `models/` and `audio/` in place:

```bash
python Rave_final/performance.py
```

Space pauses. `R` drops a seed. `[` and `]` change how many frames pass between generations. The image on stage is `Final.toe`, not this window.
