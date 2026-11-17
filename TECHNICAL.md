# Technical document - Team Data Wizards

## Approach
A transparent rule-based possession detector over the provided tracking. No training, so no labelled
data of any kind was used (none from the provided game, none from other games).
1. Parse `atd.txt` (46 anonymous tracks + ball, normalised pitch coords; scaled to 105x68 m).
   Team of each track comes from `atd.xml` (Barcelona = local team = `H`, Athletic = `A`).
2. Interpolate ball gaps up to 2 s linearly; smooth ball path (5-frame mean); ball speed from positions.
3. Per frame: distance from the ball to the nearest player of each team.
4. Control: nearest team within 2.5 m and ball speed <= 10 m/s. Control states shorter than 4 frames are
   dropped. Dead ball `D` (stoppage, ball out of play, as the scorer defines it): ball more than 0.5 m
   beyond a pitch line for at least 12 frames.
5. Spell logic: possession persists through passes, loose balls and missing-ball stretches until the
   other team gains control (so abstentions are mostly avoided; `X` only outside active play).
6. `smart_tagging.xml`: half extents (to write `X` before kick-off, at half time, after the end) and the
   team label of corners, goal kicks and free kicks (the tagged team holds the ball at the restart).

## Assets
Used: tracking txt/xml, smart tagging. Not used: homography (tracking is already in pitch coordinates),
broadcast files, video.

## Training data
None. The constants come from physical reasoning plus a look at the ball-to-player distance distribution
of the provided tracking (inspecting data is allowed). No ground truth was used, and the public 5
minutes were not used for tuning.

## Possession rules
Loose ball and flight keep the last controlling team; a contested ball goes to the nearer team; `D` is
only written when the ball is clearly out of the pitch; throw-ins carry no team tag; a ball missing for
more than 2 s keeps the last label.

## Limitations
- Dead ball is only detected when the ball leaves the pitch; stoppages with the ball inside the lines
  (fouls, injuries, waiting for a restart) are labelled as possession, so `D` is under-reported.
- The ball is missing in ~43% of frames, so many stretches are carry-forward, not evidence.
- Constants (2.5 m, 10 m/s, 4 frames) are untuned; identity switches and ball noise cause wrong flips.
- The restart override is weak: it only sets the tagged team on the restart frame when no control was detected there, then the normal spell logic takes over.
- `H` = Barcelona is assumed from `idLocalTeam`. Next steps: infer ball position from player
  movement when missing, self-supervised pass/touch detection, tune on other games.

## Reproduction
`pip install -r requirements.txt` (numpy==2.5.4, Python 3.12.3), put the data bundle in `data/tactical/`
(not committed), then run `python possession.py` (data folder is auto-detected; `--data-dir` is optional).
It writes `Data Wizards.csv` and checks its format.
The output is byte-identical across runs.
