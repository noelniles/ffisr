from manim import *


class PacketComparison(Scene):
    def construct(self):
        self.camera.background_color = "#040810"

        title = Text("What gets transmitted?", font_size=36, color=WHITE)
        title.to_edge(UP, buff=0.6)
        self.play(Write(title))
        self.wait(0.3)

        base_bar = Rectangle(width=10.0, height=1.0, color="#6b7280", fill_opacity=0.7)
        sem_bar  = Rectangle(width=0.6,  height=1.0, color="#3b82f6", fill_opacity=0.9)

        base_bar.move_to(UP * 0.6 + LEFT * 0.0)
        sem_bar.move_to(DOWN * 0.8)
        sem_bar.align_to(base_bar, LEFT)

        base_label = Text("Video frame  ≈ 50,000 bytes", font_size=26, color="#d1d5db")
        sem_label  = Text("Semantic update  ≈ 300 bytes", font_size=26, color="#93c5fd")
        base_label.next_to(base_bar, RIGHT, buff=0.3)
        sem_label.next_to(sem_bar,  RIGHT, buff=0.3)

        self.play(GrowFromEdge(base_bar, LEFT), run_time=1.2)
        self.play(Write(base_label), run_time=0.8)
        self.wait(0.4)

        self.play(GrowFromEdge(sem_bar, LEFT), run_time=0.25)
        self.play(Write(sem_label), run_time=0.8)
        self.wait(0.5)

        ratio = Text("166× smaller", font_size=32, color="#34d399", weight=BOLD)
        ratio.move_to(DOWN * 2.4)
        self.play(Write(ratio))
        self.wait(2.5)
