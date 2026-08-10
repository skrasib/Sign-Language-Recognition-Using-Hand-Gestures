from adaptive_gesture.ui.layout import (
    choose_window_layout,
    fit_size,
    responsive_profile,
)


def test_initial_window_stays_inside_common_laptop_screen():
    layout = choose_window_layout(1366, 768)
    assert layout.width <= 1366
    assert layout.height <= 768
    assert layout.min_width <= layout.width
    assert layout.min_height <= layout.height


def test_large_screen_keeps_preferred_window_size():
    layout = choose_window_layout(1920, 1080)
    assert layout.width == 1500
    assert layout.height == 900


def test_camera_fit_preserves_landscape_aspect_ratio():
    width, height = fit_size(960, 540, 700, 500)
    assert width == 700
    assert abs((width / height) - (16 / 9)) < 0.01
    assert height <= 500


def test_camera_fit_respects_height_limited_box():
    width, height = fit_size(960, 540, 1000, 300)
    assert height == 300
    assert width <= 1000


def test_narrow_window_uses_compact_profile():
    compact = responsive_profile(1000)
    wide = responsive_profile(1600)
    assert compact.camera_weight == compact.workspace_weight
    assert compact.inner_padding < wide.inner_padding
    assert compact.hide_secondary_header is True
    assert wide.hide_secondary_header is False
