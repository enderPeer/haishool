# life8 3D viewer ("Haishool Living Worlds")

`haishool/life8/view3d.py` turns one life8 run archive into one self-contained HTML page
that replays the recorded world in 3D in a browser. The page template is
`haishool/life8/view3d_template.html`. The viewer only shows what the archive recorded.
It adds no behaviour, and nothing in it feeds back into the engine.

## Build a page

```
# a run with the living preset and snapshots often enough for smooth playback
python -c "import json; from haishool.life8.config import LIVING; json.dump({**LIVING, 'log': 'compact'}, open('living.json', 'w'))"
python -m haishool.life8 run --seed 11 --steps 1200 --config living.json --sample-every 1 --out runs/living-11

# the page
python -m haishool.life8.view3d build runs/living-11 --out living-world-11.html [--every K] [--max-ticks N]
```

| Option | Meaning |
|---|---|
| `--every K` | Keep one snapshot per K ticks (default 1: every snapshot in the archive). |
| `--max-ticks N` | Only the first N ticks of the archive. |
| `--replay auto\|on\|off` | Recover predator positions by replaying the run (below). `auto` (default) uses the replay only if it matches the archive; `on` fails otherwise; `off` skips it. |
| `--max-mb M` | Size bound (default 12 MB). If the page would be larger, K is doubled until it fits. As a last resort, the list of heard calls per individual is reduced to counts. |

The command prints a JSON report with the path, size in bytes, frame count, the K it used and
where the predator positions came from. Numbers are rounded: positions to 0.01 world units,
energy to 0.1, health and food opening to 0.01, food to 0.1 units.

The page needs two pinned scripts from public CDNs, so it needs an internet connection:
`three.js r128` (cdnjs) and `OrbitControls` from `three@0.128.0` (jsdelivr). Everything else,
including all data, is inside the file. `tests/life8/test_view3d.py` checks that the template
loads nothing else.

## What the page shows

- **Space.** The world is a flat 2D plane (32 x 24 units by default) whose edges wrap around.
  The page draws it in perspective. Heights only make things visible, and no terrain height is
  invented. The legend says this, and says that a real 3D world is the next round.
- **Inhabitants.** Each body is a sphere whose radius is proportional to the square root of
  its body mass. A small cone shows the direction of its last move. Colour shows either the
  founder lineage (the eight lineages with the most individual-frames get fixed colours, all
  others are grey; the legend lists them with their live counts and can highlight one) or
  energy (a single-hue scale). A thin ring on the ground shows energy as the outer arc and
  health as the inner arc. Health lost for good to scars is drawn in red.
- **Food.** Plant-like markers sized by the food left. Soft patches are green cones. Hard,
  shelled food is a brown "nut", and it turns greener as the shell is opened. Blooms are pink
  flowers.
- **Objects by material.** Stone (grey polyhedron), wood (log), fiber (coil), clay
  (flattened ball), metal (steel octahedron), and composites (violet box). Held objects ride
  beside their holder at a smaller size.
- **Predators.** Red spikes with a faint circle showing how far they see. They are pale while
  sated.
- **Calls.** Rings at the sender, coloured by symbol (0-3). Each symbol also has its own dash
  pattern: solid, then 2, 3 and 4 dashes. A ring grows to the call's reach within one tick
  and fades over the call's life (message_ttl + 1 ticks). Rings that cross an edge are drawn
  again on the far side, as on the torus. Symbols are arbitrary, and the page says so.
- **Events.** A birth is a rising spark. A death is a cross on the ground, coloured by cause.
  A strike is a short amber line from the striker to its target (food, an object or another
  individual), with a small burst at the target. Shares are teal lines, and teaching and
  imitation are violet lines. A predator strike is a red burst, with a line from the predator
  when its position is known. A combine or dismantle is a small violet box pulse.
- **Visitor.** If an agent record ever has `"visitor": true` (optional `visitor_label`), that
  individual gets a gold halo and a floating label. No code sets this field yet. It is
  reserved for a later round.

## Controls

- Play and pause, step back and forward, speed (2-80 ticks per second), a timeline scrubber
  with the tick number, and a population sparkline that can also be dragged to scrub.
- Orbit, zoom and pan with the mouse or touch (OrbitControls), and **Reset view**.
- Click (or tap) an individual, or choose one from **Inspect**, to open the side panel. On
  phones the panel becomes a bottom sheet. The panel shows id, parent, founder lineage,
  generation, birth tick and lifespan, age, energy, health and its scar ceiling, wounds (by
  predators and by others), offspring (as links), what it holds, anatomy (body mass, speed,
  sensing and sight radius, manipulation, metabolism, voice, hearing, call power and reach),
  the genome (learning rate, exploration, number of innate bias weights, innate P(eat) and
  P(call) with food in reach), recent actions, interactions, calls sent (by symbol) and calls
  heard (with the sent symbol when noise changed it), and the visitor flag. **Follow** keeps
  the camera on the individual.
- Toggles for calls and events, and colouring by lineage or energy.
- Keys: Space (play/pause), Left/Right (step; Shift for 10), Home/End, F (follow), C (calls),
  E (events), M (colour), `,` and `.` (rotate), `+` and `-` (zoom), Esc (close the panel).
  Keyboard focus is always visible.
- Light and dark themes follow `prefers-color-scheme`. With `prefers-reduced-motion` the page
  does not start playing by itself and the camera moves without easing.

## What comes from where

| Shown | Source |
|---|---|
| Positions, energy, health, anatomy, genome, last action, food, objects, live calls | `snapshots.jsonl` |
| Births, deaths and causes, calls sent and heard, strikes, shares, teaching, imitation, blooms, predator strikes | `events.jsonl` (compact or full log) |
| Founder lineage and generation | the parent chain (snapshots' `parent_id`, birth events) |
| Blooms and their spoil tick | `bloom` events (else food ids above `food_patches` when blooms are on); spoil = bloom tick + `bloom_ttl` |
| Permanent scars | `permanent_injury` x each wound: predation events, and successful `strike_agent` acts (wound = `strike_injury` x damage, times attacker/target body mass with `injury_mass_scaling`), capped at 1 |
| Call power | a call's range / `call_reach` (signal events), else the final checkpoint |
| Predator positions | a replay of the run (below), else only predation events |
| Innate P(eat), P(call) | computed from each genome with `genome.innate_values` (the contexts of `genome.NEAR_CONTEXTS`) |

**Predator replay.** Snapshots do not contain the ecology record, so predator positions are
not in them. When the archive starts at tick 0 (or its resume checkpoint still exists),
`view3d` re-runs the world from the manifest's seed and config with the installed engine. It
uses the result only if every snapshot matches exactly (same individuals, positions, energy
and health). If the engine changed since the run, the replay is rejected. The page then says
that predators are shown only where they struck, and draws a fading predator marker at each
strike. The replay costs about as much as the original run's engine time (no snapshots or
validation): about 1 minute for 1200 ticks of the living preset on this machine.

## Limits

- The world is 2D. The 3D view is a presentation only.
- Between frames, positions are interpolated (on the torus). With `--sample-every 1` and
  `--every 1` every tick is a frame. With sparser snapshots, short-lived individuals and
  in-between actions can be missed. "Recent actions" are the `last_action` of each snapshot
  plus the logged interactions.
- Snapshots hold no controller or memory, so the panel shows no learned weights and no social
  memory. Calls heard come from compact `heard` events (`message_received` in full logs).
- Event markers sit where the participants were in the nearest snapshot at or before the
  event's tick.
- The CLI's own `viewer.html` (written by `run_world`) embeds the last 500 raw snapshots. With
  `--sample-every 1` in a living world it reaches 50-140 MB. The 3D page does not use it.
- Adding `view3d.py` to the package's top level changes `checkpoint.source_identity()`, as
  any new or edited top-level module does. Checkpoints written by `cli run` before the file
  existed are refused by `resume`/`verify` unless that file is absent.
