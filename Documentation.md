# Life — development notes

Priyal Patel and Nicolas Urrego Sandoval.

This is the build log behind the project page in [README.md](README.md). The performance files are [Rave_final/performance.py](Rave_final/performance.py) and [Final.toe](Final.toe).

## What we were making

A performance where Conway's Game of Life is the thing being played, and the sound is a reading of that system rather than a soundtrack laid on top.

The arc we settled on was nature → industrial → computer → doom. Growth, a built-up phase, a system running past its size, collapse. The simulation already produces those if the rules are allowed to move. The work was making that playable, and making the audio follow the state instead of the coordinates of individual cells.

We can shape the conditions. We do not get to pick the outcome. That was the point of using Life rather than a timeline.

## Who did what

Priyal: RAVE, the latent experiments, the file-input pipeline, the mood thresholds and the crossfades in `performance.py`.

Niko: the Game of Life implementation, the Dorothy and TouchDesigner tests, the visual system, the patch that turns a set of coordinates into the stage image.

Decisions about the four states, the thresholds, the MIDI layout and what to change between the two shows were made together. We swapped sides often enough that "audio person" and "visual person" was only a starting split.

## Simulation

The question at the start was which Conway rules can be changed live and still produce something worth watching. Birth count, minimum neighbours to survive, maximum neighbours to survive. Those three, plus how often a generation is stepped.

We looked at a Pygame implementation first (`GoL_experiments/educ8s example`). You can edit the grid, which was attractive, and then screenshot it for TouchDesigner to treat as footage. We dropped it. No live coding, and too much overhead for a state that is really just a set of cells.

The version that stuck comes from the set-based Life pattern (the Real Python walkthrough was the reference): an infinite-style set of live coordinates, eight neighbour offsets, no picture required at the simulation stage. `GoL_experiments/Dorothy light version` was that logic in Dorothy, so we could change a rule and see the growth rate change. It did. The interesting range really was those three numbers. Stable default: 3, 2, 3.

From a set of coordinates there were two ways to make an image. Draw it in Dorothy and capture the window, or let TouchDesigner be the runtime and instance geometry on the coordinates. We took the second. More control, less to keep in sync.

Running the Dorothy script on demand from TouchDesigner crashed the patch. TouchDesigner will not sit on a blocking script. So only the evolution step lives in the patch, with the current state in a form the visual network can read. That turned out to be a better instrument than live-coding the rules in a text editor. Speed, parameters and the three rules were all on the patch.

Niko then built the grid: a shape at every live coordinate, shaders, a rendered image, post effects. That was the visual system the audio had to catch up with.

## Audio, in the order it was tried

### Listening to models

`Rave_experiments/trying_rave.py`, `rave_exploration.py`. Vintage, birds, humpback, musicnet, one latent dimension at a time. Without an input file the result is mostly a steady noise that changes for a second when you touch a dimension. Hard to judge a model that way.

### Random z

`random_latent.py` draws `z` from a normal distribution and records ten seconds. This is how musicnet got chosen. It had more than one behaviour in it. Link to the model list: https://acids-ircam.github.io/rave_models_download

### Evolving z

`evolve_latent.py` uses Dorothy's `on_new_frame` to keep biasing `z`. The sound moves without anyone turning a knob. That suggested the next step: drive the same update from the simulation instead of from a random timer.

### Mouse, as a fake four-mood space

`evolve_mouse.py` puts a random latent in each corner and interpolates with the mouse. Top edge blend, bottom edge blend, then blend those. Smoothing made it slower. The four corners did not become four moods. It stayed a drone. We kept the idea of a crossfade and threw away the 2D latent map.

### Cells as notes

`Rave_final/pixel_synth.py`. The grid is cut into seven lanes, C D E F G A B. Every few frames the live cells in a lane trigger a note. While we were still talking about the piece, Louis suggested scanning the frame and using pixel position as pitch, so the sound would grow when the pattern grew.

It works, and it is pleasant. It is a sequencer. The pitch does not know if the colony is stable, exploding or dying. Only where the cells happen to sit. We wanted the sound to follow population, so this stayed an experiment.

### Files into musicnet

`file_input.py` is the turn. A wav goes in, musicnet reshapes it, you can hear the source and the model at the same time. That was the first time we could choose a mood on purpose: nature loops, industrial loops, and so on, with RAVE as the layer on top rather than the whole composition.

`performance.py` is that idea at show scale.

- All mood folders stream from the start. Gain starts at 0.
- `pick_mood(alive)` selects nature / industrial / computer / doom.
- The active file fades up and the previous one fades down (`MOOD_CROSSFADE_SPEED = 0.08`).
- Every 240 frames the active mood picks a different file in its folder, same fade.
- RAVE output gain tracks alive count, smoothed (`0.85 * previous + 0.15 * target`).
- A change in alive count kicks `z` by a small random vector. Every frame adds a smaller drift. `z` stays inside `[-0.4, 0.4]`.

Thresholds, after the TouchDesigner grid turned out bigger than the one we had tested in Python:

| Alive cells | Mood |
|---:|---|
| ≤ 140 | nature |
| 141–219 | industrial |
| 220–419 | computer |
| ≥ 420 | doom |

A lot of the tuning was writing down population counts during a run and moving these edges until a mood could actually be reached, and until it stopped flickering.

The source loops are in the Drive folder with the models, not in the repo. They come from Pixabay. https://drive.google.com/drive/folders/1cOLiZIYgVCfwNzLbNU3qZ5WQ9h5QGzvF?usp=sharing

## Joining the two programs

Dorothy inside TouchDesigner had already failed, so we did not try to make one process do both jobs. TouchDesigner holds the simulation and the image. Python holds RAVE. The value that has to agree is the alive count.

A direct live link was the other option. We had already watched that kind of link drop. A count both sides can read was dull and reliable, which is what we wanted on stage.

## Playing it

Out of the classical rule set, three numbers are worth changing live: birth, survive-min, survive-max. Even then, a running pattern can sit there and refuse to become interesting. A random seed drop (an acorn, placed at a random origin) was the control that actually opened a new behaviour.

Other useful dials: how many frames between generations, and the visual resets.

Failsafes, added once we knew we would be in front of people:

- reset the three rules to 3, 2, 3
- reset the grid to the base seed and the rules to the default
- reset the visual parameters

Keys were fine at the desk. A MIDI controller meant we could change a value without looking at the patch. That was for the show, not for the model.

![Controller](https://git.arts.ac.uk/user-attachments/assets/25f02760-41fd-496e-a0be-fb0fb27bb3ef)

## Shows

We played it twice.

**First showing.** The structure was there. The audio was not carrying a full-length set. Moods felt disconnected from each other, and the sound and the image did not yet feel like one system. People also did not have enough context to know what a gesture was doing. We had not split the controller, which meant both of us hesitating. Clips and stills: https://drive.google.com/file/d/1evnyeDV2BynUqbprZSsdDx-CWlSMPgA2/view?usp=sharing

![First showing](https://git.arts.ac.uk/user-attachments/assets/4cb635fd-86ed-4f1d-b723-0dd840fd5288)

**The Crypt.** We cleaned the loops, tightened the transitions, and added a fourth mood, doom, for the overload. Live stats went on the image. Controls were split: Priyal on visuals, Niko on the rules. The sound finally felt related to the colony. One new edge case: reset combined with a very low birth rule, and the fresh cells died immediately. https://drive.google.com/file/d/186PmDy9jZQAGyYUiyY2Ej4KavS78KiEJ/view?usp=sharing

![Crypt](https://git.arts.ac.uk/user-attachments/assets/a4392eb1-c2ed-40fb-8afc-0653896c1a81)

## What people said, and what we thought

We asked a few questions at the first showing. The one that changed the piece: about 40% said the image and the sound did not seem connected. We had split the build along those lines, and the feedback confirmed they still looked split. The Crypt version was mostly a response to that.

On agency, most people put the performance somewhere between machine and human, which is where we were aiming. Open answers described bacteria, algae, growth, something natural and digital at once. Several people wanted more context. The stats overlay was the practical answer, and it helped the image as well.

![Notes from the first showing](https://git.arts.ac.uk/user-attachments/assets/89b8cf77-22bd-45a9-8671-a66fcec9f5a8)

![Agency](https://git.arts.ac.uk/user-attachments/assets/1a563955-0350-43ca-8dbc-61f7986b75f8)

![Flow](https://git.arts.ac.uk/user-attachments/assets/b80ad0ad-f959-4614-b0d8-97cb38a7c889)

![Friction](https://git.arts.ac.uk/user-attachments/assets/0d4fb86c-8ffe-4393-b10d-a045e7cef320)

![Live stats](https://git.arts.ac.uk/user-attachments/assets/e6723e4c-efba-414e-a448-9c535872583c)

Priyal: the interesting tension is that we could modify the rules and still not predict the frame. A lot of the early interest came from predictive systems, Markov chains in particular, and from the question of how much of a performance you hand to a model. Life kept the "what tends to happen next" feeling and dropped the prediction. The other lasting practical lesson was the mood folders. Constraints did more for the piece than adding controls did.

Niko: at the first showing the fear was that it would look like a video with two people standing near it. It did not. There was enough control to pace the set, and enough mistakes that it was obviously live. The 40% on sound versus image matched what we were hearing, and the visual changes were also simply faster than the audio, so the sound lagged the picture if we were not careful. For the Crypt, splitting the controls (Priyal on the image, Niko on the rules) gave us a way to talk to each other during the set and to watch the system between changes. The system still surprised us. There was a state where adding a seed did nothing. Having already played it once, we could use that moment instead of treating it as a fault. People came up afterwards and asked about the actual mechanics, which had not happened the first time. The larger point, for him, was that dialling human control back made the piece more like what it was about: set conditions, then respond.

## References

- https://realpython.com/conway-game-of-life-python/#step-3-define-and-load-the-life-patterns
- https://acids-ircam.github.io/rave_models_download
- https://tinkerings.org/2016/09/24/cellular-automata-explorer/
- https://www.are.na/agnet-cameron/src-cellular-automata
- https://www.synthtopia.com/content/2008/05/29/glitchds-cellular-automaton-sequencer-for-the-nintendo-ds/
- https://www.synthtopia.com/content/2009/04/29/game-of-life-music-sequencer/
- https://www.synthtopia.com/content/2011/01/12/game-of-life-music-sequencer-for-ios-runxt-life/
