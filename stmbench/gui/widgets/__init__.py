"""The viewer widgets of the desktop console: :class:`FrameView` (a frame as an image with nm axes, a
colour bar, numbered markers and a line profile), :func:`frame_thumbnail` and :class:`PlotView` (a
line plot with a cursor that reads out the point under the pointer). They need PySide6;
``stmbench.gui`` itself does not import this package until the console is started."""
from .frame_view import FrameView, frame_thumbnail
from .plot_view import PlotView

__all__ = ["FrameView", "PlotView", "frame_thumbnail"]
