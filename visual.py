from manim import *
import numpy as np


def get_word_points(word: str, max_points: int = 50) -> np.ndarray:
    """
    Use Manim's own Text renderer to extract 2D outline points — no matplotlib needed.
    """
    mob = Text(word, font="Noto Sans SemiCondensed")
    pts3d = mob.get_all_points()   # all bezier control points
    anchors = pts3d[::3]           # every 3rd = on-curve anchor points
    pts2d = anchors[:, :2]         # drop z column

    if len(pts2d) > max_points:
        idx = np.round(np.linspace(0, len(pts2d) - 1, max_points)).astype(int)
        pts2d = pts2d[idx]

    return pts2d


class VisualizeConvexHull(Scene):
    WORD = "hello"
    FONT_PATH = "fonts/Noto_Sans/static/NotoSans_SemiCondensed-Regular.ttf"

    def construct(self):
        # ── 1. Get 2D points from the word's font geometry ───────────────────
        pts2d = get_word_points(self.WORD)

        # ── 2. Center + scale to fit Manim's ~14×8 viewport ─────────────────
        center = pts2d.mean(axis=0)
        pts2d  = pts2d - center
        scale  = np.max(np.abs(pts2d))
        pts2d  = pts2d / scale * 5.0        # fits within ±5 units

        # Make 3-D (z = 0) — required by Manim
        pts3d = np.hstack([pts2d, np.zeros((len(pts2d), 1))])
        n = len(pts3d)

        # ── 3. Draw all points (AFTER scaling — this is the critical fix) ────
        dots = VGroup(*[Dot(p, color=BLUE, radius=0.08) for p in pts3d])
        labels = VGroup(*[
            Text(str(i), font_size=14).next_to(dots[i], UR, buff=0.05)
            for i in range(n)
        ])
        title = Text(f'Graham Scan  –  "{self.WORD}"', font_size=28).to_edge(UP)

        self.play(Write(title))
        self.play(Create(dots), run_time=1)
        self.play(Write(labels), run_time=1)
        self.wait(0.5)

        # ── 4. Highlight start point (lowest y) ─────────────────────────────
        start_idx = int(np.argmin(pts3d[:, 1]))
        self.play(
            dots[start_idx].animate.set_color(YELLOW).scale(1.5),
            run_time=0.5,
        )
        self.play(Indicate(dots[start_idx]))

        # ── 5. Sort all points by polar angle from start point ───────────────
        angles = np.arctan2(
            pts3d[:, 1] - pts3d[start_idx, 1],
            pts3d[:, 0] - pts3d[start_idx, 0],
        )
        # Force the pivot to always sort first (arctan2(0,0) == 0 is ambiguous)
        angles[start_idx] = -np.inf
        sorted_indices = list(np.argsort(angles))  # original indices in angle order

        # Sweep-line animation
        sweep = Line(
            pts3d[start_idx],
            pts3d[start_idx] + RIGHT * 6,
            color=YELLOW_A,
            stroke_opacity=0.5,
        )
        self.add(sweep)
        self.play(
            Rotate(sweep, angle=PI, about_point=pts3d[start_idx],
                   run_time=2, rate_func=linear)
        )
        self.remove(sweep)

        # ── 6. Graham Scan animation ─────────────────────────────────────────
        # hull_indices = list of *original* point indices currently on the hull
        hull_indices = [sorted_indices[0]]   # start with the pivot
        hull_lines   = VGroup()
        self.add(hull_lines)

        for step in range(1, len(sorted_indices)):
            next_idx = sorted_indices[step]

            # Highlight candidate in orange
            self.play(dots[next_idx].animate.set_color(ORANGE), run_time=0.1)

            # Pop while the last turn is not a strict left turn
            while len(hull_indices) > 1:
                p1 = pts3d[hull_indices[-2]]
                p2 = pts3d[hull_indices[-1]]
                p3 = pts3d[next_idx]
                cross = (
                    (p2[0] - p1[0]) * (p3[1] - p1[1])
                    - (p2[1] - p1[1]) * (p3[0] - p1[0])
                )
                if cross <= 0:
                    # Right turn or collinear — remove last hull point
                    self.play(
                        dots[hull_indices[-1]].animate.set_color(RED),
                        FadeOut(hull_lines[-1]),
                        run_time=0.1,
                    )
                    hull_indices.pop()
                    hull_lines.remove(hull_lines[-1])
                else:
                    break

            # Add edge from last hull point to the new point
            new_edge = Line(pts3d[hull_indices[-1]], pts3d[next_idx], color=GOLD)
            hull_lines.add(new_edge)
            hull_indices.append(next_idx)
            self.play(Create(new_edge), run_time=0.1)
            self.play(dots[next_idx].animate.set_color(GOLD), run_time=0.1)

        # ── 7. Close the hull ────────────────────────────────────────────────
        closing = Line(pts3d[hull_indices[-1]], pts3d[hull_indices[0]], color=GOLD)
        hull_lines.add(closing)
        self.play(Create(closing))
        self.play(hull_lines.animate.set_stroke(width=5))

        done = Text("Convex Hull Complete!", font_size=26, color=GREEN).to_edge(DOWN)
        self.play(Write(done))
        self.wait(2)
