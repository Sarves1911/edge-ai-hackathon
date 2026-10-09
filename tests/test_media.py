import numpy as np

import edge_detector.media as media_module
from edge_detector.media import MediaSource


class _FakeCapture:
    def __init__(self, frame_count: int = 5) -> None:
        self._frames = [
            np.full((8, 8, 3), index, dtype=np.uint8)
            for index in range(frame_count)
        ]
        self._index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        return 30.0

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self) -> None:
        self.released = True


class _FakeCv2:
    CAP_PROP_FPS = 1
    CAP_PROP_POS_MSEC = 2

    def __init__(self, capture: _FakeCapture) -> None:
        self._capture = capture

    def VideoCapture(self, source):
        return self._capture


def test_live_camera_latest_mode_delivers_newest_frame_and_reports_replacement(
    monkeypatch,
) -> None:
    capture = _FakeCapture(frame_count=5)
    monkeypatch.setattr(media_module, "_cv2", lambda: _FakeCv2(capture))

    with MediaSource("0") as source:
        packets = list(source.frames(latest_only=True))
    summary = source.capture_summary()

    assert packets[-1].index == 4
    assert summary["mode"] == "latest"
    assert summary["captured_frames"] == 5
    assert summary["delivered_frames"] == len(packets)
    assert summary["replaced_frames"] == 5 - len(packets)
    assert capture.released is True
