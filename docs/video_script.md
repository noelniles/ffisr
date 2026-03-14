# FFISR Explainer Video — Script & Scene Plan

**Target runtime:** 5:00
**Format:** Screen captures + stock video, no live presenter

---

## Toolchain

| Purpose | Tool |
|---|---|
| Screen recording | OBS Studio |
| Video editing / assembly | Kdenlive |
| Animated diagrams | Manim (Python) |
| Stock footage | Pexels / Pixabay (free, no attribution required) |
| Audio narration | Mic + Audacity, or Piper TTS (offline) |
| Charts | `eval/results/fig_drops.pdf`, `eval/results/fig_throughput.pdf` |

---

## Scene 1 — The Problem (0:00–0:45)

**Visuals:**
Stock aerial drone footage (Pexels: "drone surveillance", "military aerial").
Cut to a video feed freezing/pixelating — screen-record a video player being
throttled, or apply a glitch overlay in Kdenlive.

**Narration:**
> "A small drone is watching a target area. The operator on the ground depends
> entirely on that video feed — but the radio link is narrow, contested, and
> unreliable. When the link degrades, the video freezes. Frames are dropped.
> At 200 kilobits per second — a realistic constrained tactical link — a standard
> video stream delivers nothing. The operator goes blind at the worst possible moment."

**On-screen text:** `200 kbps → 0% of video delivered`

---

## Scene 2 — The Idea (0:45–1:30)

**Visuals:**
Animated diagram (Manim or static slide in Kdenlive). Show two paths: a fat
"VIDEO" pipe getting throttled to zero, and a thin "SEMANTICS" pipe passing
through cleanly.

**Narration:**
> "Feature-First ISR flips the priority. Instead of streaming every pixel, the
> edge computer on the drone runs an object detector and tracker in real time.
> What gets transmitted isn't video — it's a structured description of the scene:
> which objects are present, where they are, how fast they're moving, and how
> confident the system is. A full scene update for ten tracked objects is a few
> hundred bytes. That fits through any link that can carry a voice call."

**On-screen text (animated):**
- `Track update: ~300 bytes`
- `One video frame: ~50,000 bytes`

**Manim snippet — packet size comparison:**
```python
from manim import *

class PacketComparison(Scene):
    def construct(self):
        sem = Rectangle(width=0.6, height=3, color=BLUE, fill_opacity=0.8)
        base = Rectangle(width=10, height=3, color=GRAY, fill_opacity=0.5)
        sem_label = Text("Semantic\n~300 B", font_size=24).next_to(sem, DOWN)
        base_label = Text("Video frame\n~50 KB", font_size=24).next_to(base, DOWN)
        sem.shift(LEFT * 5)
        sem_label.shift(LEFT * 5)
        self.play(GrowFromEdge(base, LEFT), Write(base_label))
        self.play(GrowFromEdge(sem, LEFT), Write(sem_label))
        self.wait(2)
```
Run: `manim -pql packet_comparison.py PacketComparison`

---

## Scene 3 — How It Works (1:30–2:30)

**Visuals:**
Screen capture of the live FFISR demo app. Load a VisDrone sequence. Show the
two-panel layout — BASELINE left, FEATURE-FIRST right — at 5 Mbps first so
both panels are working normally.

**Narration:**
> "Here's the system running live. On the left, the baseline channel — continuous
> video, the traditional approach. On the right, the Feature-First channel. The
> drone's edge processor detects and tracks every object in each frame. Track IDs,
> bounding boxes, class labels, and velocities are serialized and sent as compact
> messages. The operator's display reconstructs the scene from those messages and
> shows annotated keyframe images on demand or when an anomaly is detected."

**Action:** Click a track row in the TRACKS table → hit "Keyframe" → show the
keyframe appear on the right panel.

---

## Scene 4 — The Stress Test (2:30–3:30)

**Visuals:**
Continued screen capture. Hit the **1M preset**, pause ~5s, then **200k preset**.
Baseline panel goes blank. Semantic panel continues with live tracks.

**Narration:**
> "Now we stress the link. At one megabit per second, baseline video starts
> dropping packets. At 200 kilobits — a severely congested or jammed channel —
> the baseline channel delivers zero bytes. Nothing. But the semantic channel
> keeps operating normally, because its messages are small enough to fit through
> the constraint. The operator still knows what's in the scene."

**Action:** Point to metrics panel — `Sem. Drop%` low, `Base. Drop%` at 100%.

Then hit **"Trigger RF Silence"** for 20 seconds.

> "We can also simulate a complete RF blackout. The system buffers updates
> locally. When the link comes back, it replays the buffered state and resumes —
> track IDs preserved, no restart required."

---

## Scene 5 — The Numbers (3:30–4:15)

**Visuals:**
Cut to `eval/results/fig_drops.pdf` full screen, then `eval/results/fig_throughput.pdf`.
Export as high-DPI PNG first: `pdftoppm -r 300 fig_drops.pdf fig_drops`

**Narration (over fig_drops):**
> "We ran this across ten sequences from the VisDrone aerial benchmark under
> three link conditions. The drop rate chart tells the story clearly. As the link
> degrades from five megabits down to 200 kilobits, baseline packet drop climbs
> from ten percent to one hundred percent. Semantic drop stays below four percent
> across every regime."

**Narration (over fig_throughput):**
> "The throughput chart shows why. Full video streaming requires two to five
> megabits — far beyond what a constrained link can carry. The semantic channel
> needs roughly 100 kilobits. That's a 96% reduction in bandwidth, consistent
> across all three link settings."

**On-screen text:** `96% bandwidth reduction. Near-zero semantic drop at 200 kbps.`

---

## Scene 6 — What This Means (4:15–5:00)

**Visuals:**
Return to stock aerial footage. Optionally overlay the FFISR demo UI as a
picture-in-picture in the corner showing live tracks.

**Narration:**
> "Feature-First ISR is a software-only change to the transport layer. No new
> sensors, no new radios, no changes to the ground station. The detector and
> tracker are configurable — a program can bring its own domain-trained model.
> And the system is designed to be radio-agnostic, so it works over whatever
> link the platform already has. In environments where spectrum is contested and
> link budget is measured in kilobits, the ability to maintain situational
> awareness is a force multiplier. FFISR makes that possible."

**Closing title card:** `Feature-First ISR — Semantic Transport for Tactical UAS`

---

## Production Checklist

- [ ] Install OBS Studio, Kdenlive, Manim (`pip install manim`)
- [ ] Download stock footage from Pexels: "drone aerial surveillance", "military vehicle convoy aerial"
- [ ] Pre-load VisDrone sequence in demo — pick one with dense vehicle/pedestrian traffic
- [ ] Record demo in one take at 1080p: 5M → 1M → 200k → RF silence → keyframe request
- [ ] Export charts as PNG: `pdftoppm -r 300 eval/results/fig_drops.pdf fig_drops`
- [ ] Export charts as PNG: `pdftoppm -r 300 eval/results/fig_throughput.pdf fig_throughput`
- [ ] Render Manim animation for Scene 2 (optional but high impact)
- [ ] Assemble in Kdenlive: Stock → Diagram → Screen capture → Charts → Stock outro
- [ ] Record or generate narration audio, sync to cuts
- [ ] Add lower-third text overlays for key stats in Kdenlive titles

## Kdenlive Assembly Order

1. Scene 1: Stock drone footage (0:00–0:45)
2. Scene 2: Manim animation or static diagram slide (0:45–1:30)
3. Scene 3: OBS screen capture — demo at 5M (1:30–2:30)
4. Scene 4: OBS screen capture — 1M → 200k → RF silence (2:30–3:30)
5. Scene 5: fig_drops PNG → fig_throughput PNG (3:30–4:15)
6. Scene 6: Stock footage with optional PiP of demo (4:15–5:00)
