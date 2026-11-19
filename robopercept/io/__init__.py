"""Frame sources and result sinks."""

from .sinks import JsonlSink, SnapshotSink, VideoSink
from .sources import Frame, FrameSource, open_source

__all__ = ["Frame", "FrameSource", "JsonlSink", "SnapshotSink", "VideoSink", "open_source"]
