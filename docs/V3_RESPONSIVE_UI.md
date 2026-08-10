# V3.6.3 Responsive Desktop UI

This patch changes only desktop presentation/layout behavior. Recognition, teaching,
tracking, EVT, metric embedding, exemplar memory, and temporal-prototype logic are
unchanged.

## Problems addressed

The previous Tkinter window used a fixed 1500x900 geometry and a fixed maximum
camera preview size. On smaller laptop displays the window could extend beyond the
usable screen height, lower controls could be clipped, and resizing/maximizing did
not make the camera preview follow the available space.

## Changes

- Initial window size is chosen from the current screen dimensions instead of a
  fixed 1500x900 size.
- Minimum size is also screen-aware, allowing normal restored-window resizing.
- The live preview now renders on a Tk Canvas and is fitted to the actual current
  camera area while preserving aspect ratio.
- The camera image no longer controls the requested Tk widget size.
- Main camera/workspace column ratios and padding adapt at smaller widths.
- Secondary header/footer text is hidden in compact mode to avoid collisions.
- Wrapped explanatory text adapts to the current workspace width.
- Every workspace page can vertically scroll when its content is taller than the
  available screen. On larger windows it expands normally with no forced scrolling.
- Mouse-wheel scrolling works over the workspace while leaving Treeview scrolling
  alone.

## Expected behavior

At startup the app should fit inside the current display rather than extending below
it. Maximizing should expand both the camera and workspace. Restoring/shrinking the
window should reduce the camera preview and keep the right-side controls usable. If
a page (especially Dynamic or Library) becomes taller than the available workspace,
a vertical scrollbar appears instead of clipping the bottom cards.
