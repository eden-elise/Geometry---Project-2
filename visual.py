from manim import *
import numpy as np

class VisualizeConvexHull(Scene):
    def construct(self):
        # 1. Setup Points
        seed_points = [
          [211.496826171875, -24.0],
          [218.106201171875, -20.203125],
          [222.699951171875, -12.796875],
          [224.496826171875, -7.703125],
          [244.903076171875, 53.59375],
          [140.70309448242188, 76.0],
          [27.90625, 76.5],
          [19.90625, 76.5],
          [13.0, 72.296875],
          [1.296875, 51.0],
          [1.296875, 46.796875],
          [9.59375, 0.0],
          [204.903076171875, -24.0]
        ]
        points_2d = np.array(seed_points)

        # 2. Pad with a column of zeros to make them 3D (x, y, 0)
        # This satisfies Manim's requirement for 3-element vectors
        points = np.hstack([points_2d, np.zeros((points_2d.shape[0], 1))])

        # 3. Create the dots (this will now work!)
        dot_group = VGroup(*[Dot(p, color=BLUE) for p in points])
        labels = VGroup(*[Text(str(i), font_size=18).next_to(dot_group[i], UR, buff=0.1) for i in range(len(points))])
        # 1. Pad with zeros (You already have this)
        points = np.hstack([points_2d, np.zeros((points_2d.shape[0], 1))])

        # 2. NEW: Center the points (Move them to the middle of the screen)
        center_offset = np.mean(points, axis=0)
        points = points - center_offset

        # 3. NEW: Scale the points (Shrink them to fit in the 14x8 window)
        # We divide by a factor (e.g., 40) so the max values are around 3 or 4 units
        points = points / 40.0
        self.play(Create(dot_group), Write(labels))
        self.wait(1)

        # 2. Find Start Point (Min Y)
        start_idx = np.argmin(points[:, 1])
        start_dot = dot_group[start_idx]
        
        self.play(start_dot.animate.set_color(YELLOW).scale(1.5))
        self.play(Indicate(start_dot))
        
        # 3. Sort by Angle
        angles = np.arctan2(points[:, 1] - points[start_idx, 1], points[:, 0] - points[start_idx, 0])
        sorted_indices = np.argsort(angles)
        points_sorted = points[sorted_indices]
        
        # Visual sorting feedback
        sweep_line = Line(points[start_idx], points[start_idx] + RIGHT * 6, color=YELLOW_A, stroke_opacity=0.5)
        self.add(sweep_line)
        self.play(Rotate(sweep_line, angle=PI, about_point=points[start_idx], run_time=2, rate_func=linear))
        self.remove(sweep_line)

        # 4. Graham Scan Animation
        hull_indices = [sorted_indices[0]]
        hull_line = VGroup().set_color(GOLD)
        self.add(hull_line)

        for i in range(len(points_sorted) - 1):
            next_point_idx = sorted_indices[i + 1]
            
            # Highlight current point being considered
            current_dot = dot_group[next_point_idx]
            self.play(current_dot.animate.set_color(ORANGE), run_time=0.3)

            # Check for left turn (Cross product > 0)
            while len(hull_indices) > 1:
                p1 = points[hull_indices[-2]]
                p2 = points[hull_indices[-1]]
                p3 = points[next_point_idx]
                
                # np.cross in 2D
                val = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                
                if val <= 0:
                    # Right turn or collinear: Remove last point
                    self.play(
                        dot_group[hull_indices[-1]].animate.set_color(RED),
                        FadeOut(hull_line[-1]),
                        run_time=0.4
                    )
                    hull_indices.pop()
                    hull_line.remove(hull_line[-1])
                else:
                    break

            # Add next point to hull
            new_line = Line(points[hull_indices[-1]], points[next_point_idx], color=GOLD)
            hull_line.add(new_line)
            hull_indices.append(next_point_idx)
            self.play(Create(new_line), run_time=0.4)
            self.play(dot_group[next_point_idx].animate.set_color(GOLD), run_time=0.2)

        # Close the hull
        final_line = Line(points[hull_indices[-1]], points[hull_indices[0]], color=GOLD)
        hull_line.add(final_line)
        self.play(Create(final_line))
        
        # Success state
        self.play(hull_line.animate.set_stroke(width=6))
        self.wait(2)