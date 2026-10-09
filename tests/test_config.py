from pathlib import Path

import pytest

from edge_detector.config import apply_overrides, load_config


def test_loads_default_config() -> None:
    config = load_config(Path("config/default.yaml"))
    assert config.model.path == "yolov8n.pt"
    assert config.model.imgsz == (256, 320)
    assert config.model.classes == (0,)
    assert config.motion.enabled is False
    assert config.motion.width == 160
    assert config.motion.height == 120
    assert config.scheduler.mode == "always"
    assert config.scheduler.idle_poll_ms == 2000.0
    assert config.pipeline.live_capture_mode == "sequential"
    assert config.pipeline.simulated_detector_delay_ms == 0.0
    assert config.output.save_jsonl is True


def test_loads_adaptive_config() -> None:
    config = load_config(Path("config/adaptive.yaml"))
    assert config.motion.enabled is True
    assert config.scheduler.mode == "adaptive"
    assert config.scheduler.active_interval_ms == 100.0
    assert config.pipeline.live_capture_mode == "latest"


def test_rejects_unknown_key(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  confidnce: 0.4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="confidnce"):
        load_config(path)


def test_applies_cli_overrides() -> None:
    config = load_config(Path("config/default.yaml"))
    changed = apply_overrides(
        config,
        confidence=0.4,
        classes=(0, 2),
        max_frames=5,
        output_directory="runs/test",
        motion_enabled=True,
        scheduler_mode="adaptive",
        live_capture_mode="latest",
        simulated_detector_delay_ms=100.0,
    )
    assert changed.model.confidence == 0.4
    assert changed.model.classes == (0, 2)
    assert changed.pipeline.max_frames == 5
    assert changed.output.directory == "runs/test"
    assert changed.motion.enabled is True
    assert changed.scheduler.mode == "adaptive"
    assert changed.pipeline.live_capture_mode == "latest"
    assert changed.pipeline.simulated_detector_delay_ms == 100.0


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_rejects_invalid_confidence(value: float) -> None:
    config = load_config(Path("config/default.yaml"))
    with pytest.raises(ValueError, match="confidence"):
        apply_overrides(config, confidence=value)

