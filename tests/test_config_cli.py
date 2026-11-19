import json

import pytest
from pydantic import ValidationError

from robopercept.cli import main
from robopercept.config import Config, apply_overrides, list_presets, load_config


def test_defaults_and_overrides():
    cfg = load_config(None, ["detector.conf=0.45", "safety.profile=robot", "detector.classes=[person, car]"])
    assert cfg.detector.conf == 0.45 and cfg.safety.profile == "robot"
    assert cfg.detector.classes == ["person", "car"]


def test_unknown_key_is_rejected():
    with pytest.raises(ValidationError):
        Config.model_validate({"detector": {"confidence": 0.3}})
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equal_sign"])


def test_presets_are_valid():
    assert {"urban", "warehouse"} <= set(list_presets())
    assert load_config("warehouse").safety.profile == "robot"


def test_example_configs_are_valid():
    from pathlib import Path

    for path in sorted(Path(__file__).resolve().parent.parent.glob("configs/*.yaml")):
        load_config(path)


def test_cli_eval_json(capsys):
    assert main(["eval", "--scenario", "urban", "--frames", "60", "--json"]) == 0
    metrics = json.loads(capsys.readouterr().out)
    assert "MOTA" in metrics and "latency_p95_ms" in metrics


def test_cli_bench_and_config(capsys):
    assert main(["bench", "-c", "warehouse", "--frames", "30"]) == 0
    assert "total" in capsys.readouterr().out
    assert main(["config", "-c", "urban", "--set", "server.port=9000"]) == 0
    assert "port: 9000" in capsys.readouterr().out
