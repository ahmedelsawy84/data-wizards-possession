#!/usr/bin/env python3
"""Rule-based possession detector (H/A/D/X per frame). No learning, no hand labels.
Usage: python possession.py --data-dir data/tactical --out submission.csv
"""
import argparse, os, re
import numpy as np
import xml.etree.ElementTree as ET

N_FRAMES, FPS = 199319, 25.0
PITCH_L, PITCH_W = 105.0, 68.0
# Physical priors (NOT tuned on any ground truth)
R_CTRL = 2.5        # m: player within this of ball can control it
OUT_MARGIN = 0.5    # m: ball this far beyond a touch/goal line is out of play (dead ball, label D)
MIN_DEAD = 12       # frames (0.5 s) the ball must stay out of the pitch
V_BALL_MAX = 10.0   # m/s: faster ball is in flight, not controlled
MIN_RUN = 4         # frames a control state must persist (0.16 s)
GAP_MAX = 50        # frames: max ball gap (2 s) to interpolate
SET_PIECE_WIN = 100 # frames after a team-tagged restart start

def load_tracking(txt, xml):
    s = open(xml, encoding="utf-8").read()
    pl = re.findall(r'<Player id="(\d+)" teamId="([^"]+)">', s)
    home = re.search(r'idLocalTeam="([^"]+)"', s).group(1)
    team = np.array([0 if t == home else 1 for _, t in pl])
    P = np.full((N_FRAMES, len(pl), 2), np.nan, np.float32)
    B = np.full((N_FRAMES, 2), np.nan, np.float32)
    with open(txt) as f:
        for k, line in enumerate(f):
            if k >= N_FRAMES: break
            _, pp, bb = line.rstrip("\n").split(":")
            P[k] = np.array([v.split(",")[:2] for v in pp.split(";")], dtype=np.float32)
            B[k] = np.array(bb.split(",")[:2], dtype=np.float32)
    P[..., 0] *= PITCH_L; P[..., 1] *= PITCH_W
    B[:, 0] *= PITCH_L; B[:, 1] *= PITCH_W
    return P, B, team

def interp_ball(B):
    B = B.copy(); ok = np.isfinite(B[:, 0]); idx = np.where(ok)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        g = b - a - 1
        if 0 < g <= GAP_MAX:
            w = (np.arange(1, g + 1) / (g + 1))[:, None]
            B[a + 1:b] = B[a] * (1 - w) + B[b] * w
    return B

def smooth(x, k=5):
    pad = k // 2; xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        out[:, c] = np.convolve(xp[:, c], np.ones(k) / k, mode="valid")
    return out

def runs_filter(lab, min_run):
    """Drop runs shorter than min_run (set to 0 = none)."""
    out = lab.copy(); n = len(lab); i = 0
    while i < n:
        j = i
        while j < n and lab[j] == lab[i]: j += 1
        if lab[i] != 0 and j - i < min_run: out[i:j] = 0
        i = j
    return out

def parse_tags(path):
    root = ET.parse(path).getroot(); sp = []; half_ranges = {}
    for ins in root.iter("instance"):
        L = {l.find("group").text: l.find("text").text for l in ins.findall("label")}
        s = float(ins.find("start").text) * FPS; e = float(ins.find("end").text) * FPS
        h = L.get("Half")
        if h:
            lo, hi = half_ranges.get(h, (1e18, -1)); half_ranges[h] = (min(lo, s), max(hi, e))
        if ins.find("code").text == "SET PIECES" and L.get("Team") in ("Barcelona", "Atletic Club"):
            sp.append((int(s), 0 if L["Team"] == "Barcelona" else 1))
    return sp, half_ranges

def run(data_dir, out):
    U = os.path.join(data_dir, "BAR-ATH_tactical_")
    P, B, team = load_tracking(U + "atd.txt", U + "atd.xml")
    sp, halves = parse_tags(U + "smart_tagging.xml")
    B = interp_ball(B); Bs = smooth(np.nan_to_num(B, nan=0.0)); Bs[~np.isfinite(B[:, 0])] = np.nan
    v = np.zeros(N_FRAMES); d = np.diff(Bs, axis=0) * FPS
    v[1:] = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
    dist = np.sqrt((P[..., 0] - Bs[:, 0:1]) ** 2 + (P[..., 1] - Bs[:, 1:2]) ** 2)
    big = np.float32(1e6); dist = np.where(np.isfinite(dist), dist, big)
    dH = np.where(team[None, :] == 0, dist, big).min(1)
    dA = np.where(team[None, :] == 1, dist, big).min(1)
    has_ball = np.isfinite(Bs[:, 0]); slow = ~(v > V_BALL_MAX)
    cH = has_ball & slow & (dH <= R_CTRL); cA = has_ball & slow & (dA <= R_CTRL)
    cand = np.zeros(N_FRAMES, np.int8)           # 0 none, 1 H, 2 A
    cand[cH & (dH <= dA)] = 1; cand[cA & (dA < dH)] = 2
    cand = runs_filter(cand, MIN_RUN)
    # dead ball (D in the organisers' scorer = stoppage / ball out of play): ball beyond the pitch lines
    oob = has_ball & ((Bs[:, 0] < -OUT_MARGIN) | (Bs[:, 0] > PITCH_L + OUT_MARGIN) |
                      (Bs[:, 1] < -OUT_MARGIN) | (Bs[:, 1] > PITCH_W + OUT_MARGIN))
    dead = runs_filter(oob.astype(np.int8), MIN_DEAD) == 1
    # team-tagged restarts: the tagged team holds the ball at the start of the window
    for s, t in sp:
        w = cand[s:s + SET_PIECE_WIN]; z = np.where(w == 0)[0]
        if len(z): cand[s + z[0]] = t + 1 if z[0] == 0 else cand[s + z[0]]
    # active play windows (from smart tagging half ranges)
    h1 = halves.get("1st Half"); h2 = halves.get("2nd Half")
    act = np.zeros(N_FRAMES, bool)
    for h in (h1, h2):
        if h: act[int(h[0]):int(h[1]) + int(10 * FPS)] = True
    # spell state: possession persists through flight/loose ball until the other team controls it
    lab = np.full(N_FRAMES, "X", dtype="<U1"); last = None
    for i in range(N_FRAMES):
        if not act[i]: last = None; continue
        c = cand[i]
        if dead[i]: lab[i] = "D"; continue
        if c in (1, 2): last = "H" if c == 1 else "A"
        if last: lab[i] = last
    with open(out, "w", newline="\n") as f:
        f.write("frame,label\n")
        for i in range(N_FRAMES): f.write(f"{i},{lab[i]}\n")
    u, c = np.unique(lab, return_counts=True); print(dict(zip(u, c)))

TEAM = "Data Wizards"
FILES = ("BAR-ATH_tactical_atd.txt", "BAR-ATH_tactical_atd.xml", "BAR-ATH_tactical_smart_tagging.xml")

def find_data_dir(arg):
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [arg] if arg else []
    env = os.environ.get("MTAH_DATA_DIR")
    if env: cands.append(os.path.join(env, "tactical"))
    cands += [os.path.join(os.getcwd(), "data", "tactical"), os.path.join(here, "data", "tactical"), here, os.getcwd()]
    for c in cands:
        if c and all(os.path.isfile(os.path.join(c, "BAR-ATH_tactical_" + f.split("tactical_")[1])) for f in FILES):
            return c
    raise SystemExit("Data files not found. Put these in data/tactical/ (or next to this script) or pass --data-dir:\n  "
                     + "\n  ".join(FILES) + "\nTried:\n  " + "\n  ".join(str(c) for c in cands))

def validate(path):
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    if lines[-1] == "": lines = lines[:-1]
    assert lines[0] == "frame,label", "bad header"
    assert len(lines) - 1 == N_FRAMES, f"expected {N_FRAMES} rows, got {len(lines) - 1}"
    for i, ln in enumerate(lines[1:]):
        fr, lb = ln.split(",")
        assert int(fr) == i and lb in ("H", "A", "D", "X"), f"bad row {i}: {ln}"
    print(f"OK: {path} passes the format checks ({N_FRAMES} rows).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, help="folder with the 3 BAR-ATH_tactical_* files (auto-detected if omitted)")
    ap.add_argument("--out", default=TEAM + ".csv")
    a = ap.parse_args()
    run(find_data_dir(a.data_dir), a.out); validate(a.out)
