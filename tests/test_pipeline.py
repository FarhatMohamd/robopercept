import json

import numpy as np

from robopercept.config import load_config
from robopercept.eval import evaluate_synthetic
from robopercept.pipeline import Pipeline


def test_urban_end_to_end_quality():
    m = evaluate_synthetic(load_config("urban"), frames=450)
    assert m["MOTA"] > 0.8
    assert m["IDF1"] > 0.8
    assert m["range_absrel_mean"] < 0.08
    assert m["lateral_mae_m"] < 0.2


def test_warehouse_end_to_end_quality():
    m = evaluate_synthetic(load_config("warehouse"), frames=450)
    assert m["MOTA"] > 0.8
    assert m["range_absrel_mean"] < 0.08


def test_closed_loop_braking_behind_lead_vehicle(tmp_path):
    cfg = load_config("urban")
    cfg.server.enabled = False
    cfg.source.max_frames = 420
    cfg.output.jsonl_log = str(tmp_path / "log.jsonl")
    cfg.output.save_video = str(tmp_path / "out.mp4")
    cfg.output.event_snapshots_dir = str(tmp_path / "events")
    pipe = Pipeline(cfg)
    speeds, levels = [], []
    for res, vis in pipe.stream():
        speeds.append(res.meta["ego_speed"])
        levels.append(res.decision.level.label)
        assert vis.shape[0] == res.image.shape[0] and vis.shape[1] > res.image.shape[1]  # BEV panel
    # The lead car brakes around t = 5-7.5 s: the governor must slow the ego down
    assert min(speeds[180:400]) < 0.5 * max(speeds[:120])
    assert "warning" in levels or "critical" in levels
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert len(lines) == 420 and "decision" in json.loads(lines[0])
    assert (tmp_path / "out.mp4").stat().st_size > 10_000
    assert any((tmp_path / "events").glob("*.jpg"))


def test_tracker_disabled_still_ranges():
    cfg = load_config("urban")
    cfg.tracker.enabled = False
    cfg.source.max_frames = 30
    pipe = Pipeline(cfg)
    results = [r for r, _ in pipe.stream()]
    assert any(t.world is not None for r in results for t in r.tracks)


def test_frame_result_is_json_serialisable():
    cfg = load_config("warehouse")
    cfg.source.max_frames = 60
    pipe = Pipeline(cfg)
    for res, _ in pipe.stream():
        json.dumps(res.to_dict())
    summary = pipe.timer.summary()
    assert summary["total"]["p95"] >= summary["total"]["p50"] > 0
    assert np.isfinite(summary["detect"]["mean"])
