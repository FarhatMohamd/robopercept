"""Command-line interface: ``robopercept <command> [options]``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .config import Config, dump_config, load_config

log = logging.getLogger("robopercept")


def _load(args: argparse.Namespace) -> Config:
    overrides = list(getattr(args, "set", None) or [])
    cfg = load_config(getattr(args, "config", None), overrides)
    if getattr(args, "source", None):
        cfg.source.uri = args.source
    if getattr(args, "show", False):
        cfg.output.show = True
    if getattr(args, "save", None):
        cfg.output.save_video = args.save
    if getattr(args, "serve", False):
        cfg.server.enabled = True
    if getattr(args, "no_serve", False):
        cfg.server.enabled = False
    if getattr(args, "port", None):
        cfg.server.port = args.port
    if getattr(args, "frames", None):
        cfg.source.max_frames = args.frames
    return cfg


def _run_pipeline(cfg: Config, realtime: bool | None = None) -> dict:
    from .pipeline import Pipeline

    pipe = Pipeline(cfg)
    server = None
    on_result = None
    pause = None
    if cfg.server.enabled:
        from .server import DashboardServer

        server = DashboardServer(cfg.server.host, cfg.server.port, cfg.server.jpeg_quality).start()
        server.state.config = cfg.model_dump(mode="json")
        pause = server.state.paused

        def on_result(res, vis):  # noqa: ANN001 - simple callback
            server.state.update(res, vis, pipe.timer.summary(), pipe.timer.fps)

        print(f"Dashboard: http://localhost:{server.port}  (Ctrl+C to stop)")
    if realtime is None:
        realtime = cfg.server.enabled or cfg.output.show
    try:
        summary = pipe.run(on_result=on_result, realtime=realtime, pause=pause)
    except KeyboardInterrupt:
        summary = {"frames": pipe.timer.samples["total"].__len__(), "latency_ms": pipe.timer.summary()}
    finally:
        if server is not None:
            server.stop()
    return summary


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load(args)
    summary = _run_pipeline(cfg)
    total = summary["latency_ms"].get("total", {})
    print(f"Processed {summary['frames']} frames, latency mean {total.get('mean', 0):.1f} ms, "
          f"p95 {total.get('p95', 0):.1f} ms")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    cfg = load_config(args.scenario, list(args.set or []))
    cfg.server.enabled = not args.no_serve
    if args.port:
        cfg.server.port = args.port
    if args.show:
        cfg.output.show = True
    if args.save:
        cfg.output.save_video = args.save
    if args.frames:
        cfg.source.max_frames = args.frames
    _run_pipeline(cfg, realtime=not args.fast)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .eval import evaluate_synthetic

    if args.config is None:
        args.config = args.scenario or "urban"
    cfg = _load(args)
    if args.scenario:
        cfg.source.uri = f"synthetic:{args.scenario}"
    metrics = evaluate_synthetic(cfg, args.frames or 900)
    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        width = max(len(k) for k in metrics)
        for k, v in metrics.items():
            print(f"{k:<{width}}  {v:10.4f}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    cfg = _load(args)
    cfg.server.enabled = False
    cfg.output.show = False
    cfg.source.max_frames = args.frames or 300
    summary = _run_pipeline(cfg, realtime=False)
    rows = summary["latency_ms"]
    print(f"{'stage':<8} {'mean':>8} {'p50':>8} {'p95':>8} {'max':>8}   (ms, {summary['frames']} frames)")
    for stage, vals in rows.items():
        print(f"{stage:<8} {vals['mean']:8.2f} {vals['p50']:8.2f} {vals['p95']:8.2f} {vals['max']:8.2f}")
    if "total" in rows:
        print(f"throughput ~ {1000.0 / max(rows['total']['mean'], 1e-6):.0f} FPS (processing only)")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    print(dump_config(_load(args)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="robopercept", description="Real-time perception for autonomous robots and vehicles")
    p.add_argument("--version", action="version", version=f"robopercept {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("-c", "--config", help="YAML configuration file")
        sp.add_argument("--set", action="append", metavar="KEY=VALUE", help="override, e.g. detector.conf=0.4")
        sp.add_argument("--frames", type=int, help="stop after N frames")

    r = sub.add_parser("run", help="run the pipeline on a camera, video, stream or folder")
    common(r)
    r.add_argument("-s", "--source", help="0, video.mp4, rtsp://..., folder/ or synthetic:urban")
    r.add_argument("--show", action="store_true", help="open an OpenCV window")
    r.add_argument("--save", help="write the annotated video to this path")
    r.add_argument("--serve", action="store_true", help="start the web dashboard")
    r.add_argument("--no-serve", action="store_true", help="disable the web dashboard")
    r.add_argument("--port", type=int)
    r.set_defaults(func=cmd_run)

    d = sub.add_parser("demo", help="closed-loop simulator demo with the dashboard (no GPU needed)")
    d.add_argument("--scenario", choices=["urban", "warehouse"], default="urban")
    d.add_argument("--set", action="append", metavar="KEY=VALUE")
    d.add_argument("--port", type=int)
    d.add_argument("--show", action="store_true")
    d.add_argument("--save")
    d.add_argument("--frames", type=int)
    d.add_argument("--no-serve", action="store_true")
    d.add_argument("--fast", action="store_true", help="do not throttle to real time")
    d.set_defaults(func=cmd_demo)

    e = sub.add_parser("eval", help="score tracking and ranging against simulator ground truth")
    common(e)
    e.add_argument("--scenario", choices=["urban", "warehouse"])
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=cmd_eval)

    b = sub.add_parser("bench", help="per-stage latency benchmark")
    common(b)
    b.add_argument("-s", "--source")
    b.set_defaults(func=cmd_bench)

    c = sub.add_parser("config", help="print the resolved configuration")
    common(c)
    c.set_defaults(func=cmd_config)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
