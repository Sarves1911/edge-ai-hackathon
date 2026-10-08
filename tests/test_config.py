from pathlib import Path

import pytest

from edge_detector.config import apply_overrides, load_config


def test_loads_default_config() -> None:
    config = load_config(Path("config/default.yaml"))
    assert config.model.path == "yolov8n.pt"
    assert config.model.imgsz == (256, 320)
    assert config.model.classes is None
    assert config.output.save_jsonl is True


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
    )
    assert changed.model.confidence == 0.4
    assert changed.model.classes == (0, 2)
    assert changed.pipeline.max_frames == 5
    assert changed.output.directory == "runs/test"


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_rejects_invalid_confidence(value: float) -> None:
    config = load_config(Path("config/default.yaml"))
    with pytest.raises(ValueError, match="confidence"):
        apply_overrides(config, confidence=value)

