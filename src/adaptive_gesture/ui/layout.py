from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowLayout:
    width: int
    height: int
    min_width: int
    min_height: int
    x: int
    y: int

    @property
    def geometry(self) -> str:
        return f"{self.width}x{self.height}+{self.x}+{self.y}"


@dataclass(frozen=True)
class ResponsiveProfile:
    camera_weight: int
    workspace_weight: int
    outer_padding: tuple[int, int, int, int]
    inner_padding: int
    column_gap: int
    hide_secondary_header: bool


def choose_window_layout(
    screen_width: int,
    screen_height: int,
    *,
    preferred_width: int = 1500,
    preferred_height: int = 900,
) -> WindowLayout:
    """Choose a safe initial Tk window that stays inside the visible screen.

    Tk's screen dimensions include areas that may be partly occupied by the OS
    taskbar/dock, so the initial window intentionally uses a little less than the
    full reported height. The user can still maximize normally afterwards.
    """
    screen_width = max(640, int(screen_width))
    screen_height = max(480, int(screen_height))

    available_width = max(640, screen_width - 40)
    available_height = max(480, screen_height - 80)

    width = min(preferred_width, int(screen_width * 0.94), available_width)
    height = min(preferred_height, int(screen_height * 0.90), available_height)

    width = max(640, width)
    height = max(480, height)

    # Keep the app resizable to smaller laptop screens without allowing a size
    # at which the two-column layout becomes unusable.
    min_width = min(width, max(900, int(width * 0.70)))
    min_height = min(height, max(560, int(height * 0.70)))

    x = max(0, (screen_width - width) // 2)
    # Leave a little more room at the bottom for taskbars on common desktops.
    y = max(0, min(30, (screen_height - height) // 3))

    return WindowLayout(
        width=width,
        height=height,
        min_width=min_width,
        min_height=min_height,
        x=x,
        y=y,
    )


def fit_size(
    source_width: int,
    source_height: int,
    box_width: int,
    box_height: int,
    *,
    max_upscale: float = 1.5,
) -> tuple[int, int]:
    """Return an aspect-ratio-preserving size that fits inside a UI box."""
    source_width = max(1, int(source_width))
    source_height = max(1, int(source_height))
    box_width = max(1, int(box_width))
    box_height = max(1, int(box_height))

    scale = min(box_width / source_width, box_height / source_height)
    scale = min(scale, max(1.0, float(max_upscale)))

    return (
        max(1, int(round(source_width * scale))),
        max(1, int(round(source_height * scale))),
    )


def responsive_profile(window_width: int) -> ResponsiveProfile:
    """Return spacing and split ratios for the current window width."""
    width = max(1, int(window_width))

    if width < 1120:
        return ResponsiveProfile(
            camera_weight=1,
            workspace_weight=1,
            outer_padding=(10, 4, 10, 10),
            inner_padding=8,
            column_gap=5,
            hide_secondary_header=True,
        )

    if width < 1420:
        return ResponsiveProfile(
            camera_weight=6,
            workspace_weight=5,
            outer_padding=(14, 5, 14, 14),
            inner_padding=10,
            column_gap=7,
            hide_secondary_header=True,
        )

    return ResponsiveProfile(
        camera_weight=7,
        workspace_weight=5,
        outer_padding=(20, 6, 20, 18),
        inner_padding=12,
        column_gap=10,
        hide_secondary_header=False,
    )
