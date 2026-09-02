"""
Single entry point for the pipeline.

    python -m egoannot <stage> [args]

Stages run in dependency order; each writes into artifacts/ and reads what the
previous one wrote, so a full build is:

    python -m egoannot quality  measure  --all
    python -m egoannot events   measure  --all
    python -m egoannot features fit      --all
    python -m egoannot features build    --all
    python -m egoannot segments render
    python -m egoannot spans    build
    python -m egoannot caption  run   --backend stub
    python -m egoannot score
"""
from __future__ import annotations

import argparse
import json
import sys

from . import config


def _episodes(args):
    """Episode paths from --all, explicit names, or the segment definitions."""
    if args.all:
        root = config.corpus()
        paths = sorted(root.glob("*.mcap")) + sorted((root / "ep").glob("*.mcap"))
        return paths
    if args.episodes:
        return [config.episode_path(e) for e in args.episodes]
    segs = json.load(open(config.SEGMENT_DEFS))
    names = sorted({s["source"].split("/")[-1].rsplit(".", 1)[0] for s in segs})
    return [config.episode_path(n) for n in names]


def _add_episode_args(p):
    p.add_argument("episodes", nargs="*", help="episode names (default: those "
                                               "referenced by data/segments.json)")
    p.add_argument("--all", action="store_true", help="every .mcap in the corpus")


def build_parser():
    p = argparse.ArgumentParser("egoannot", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    # -- quality -------------------------------------------------------
    q = sub.add_parser("quality", help="T1-T4 per-clip quality records")
    qs = q.add_subparsers(dest="cmd", required=True)
    for cmd, helptext in (("measure", "write clip records"),
                          ("calibrate", "derive blur/exposure floors from footage"),
                          ("selftest", "fault injection against T1")):
        sp = qs.add_parser(cmd, help=helptext)
        _add_episode_args(sp)
        if cmd == "measure":
            sp.add_argument("--out", default=None)
    qr = qs.add_parser("report", help="re-print a report from existing records")
    qr.add_argument("records", nargs="?", default=None)

    # -- events --------------------------------------------------------
    e = sub.add_parser("events", help="contact/release events and actionness")
    es = e.add_subparsers(dest="cmd", required=True)
    em = es.add_parser("measure", help="write events, spans and summaries")
    _add_episode_args(em)
    em.add_argument("--out", default=None)
    ec = es.add_parser("calibrate", help="pooled actionness percentiles")
    _add_episode_args(ec)

    # -- features ------------------------------------------------------
    f = sub.add_parser("features", help="pose-only per-frame features")
    fs = f.add_subparsers(dest="cmd", required=True)
    ff = fs.add_parser("fit", help="fit the global closure PCA")
    _add_episode_args(ff)
    fb = fs.add_parser("build", help="write per-episode feature archives")
    _add_episode_args(fb)
    fb.add_argument("--out", default=None)
    fa = fs.add_parser("auc", help="univariate discriminability vs gold")
    fa.add_argument("--features", default=None)

    # -- segments ------------------------------------------------------
    sg = sub.add_parser("segments", help="render review clips with pose overlays")
    sgs = sg.add_subparsers(dest="cmd", required=True)
    sgr = sgs.add_parser("render", help="cut mp4s + overlay json + manifest")
    sgr.add_argument("--only", nargs="*", default=None, help="segment ids")
    sgr.add_argument("--out", default=None)

    # -- spans ---------------------------------------------------------
    s = sub.add_parser("spans", help="cut annotation spans and attach pose facts")
    ss = s.add_subparsers(dest="cmd", required=True)
    sb = ss.add_parser("build", help="write spans.jsonl")
    sb.add_argument("--only", nargs="*", default=None, help="segment ids")
    sb.add_argument("--out", default=None)
    sb.add_argument("--signal", choices=["activity", "velocity"], default=None)
    sb.add_argument("--no-band", action="store_true",
                    help="skip duration-band enforcement (rule A1)")
    sb.add_argument("--no-quality-gate", action="store_true")
    sb.add_argument("--gate-tiers", nargs="*", default=None, choices=["T1", "T2"])

    # -- caption -------------------------------------------------------
    c = sub.add_parser("caption", help="caption spans with a VLM backend")
    cs = c.add_subparsers(dest="cmd", required=True)
    cr = cs.add_parser("run", help="produce captions")
    cr.add_argument("--spans", default=None)
    cr.add_argument("--backend", default="stub",
                    choices=["stub", "anthropic", "openai", "qwen-local"])
    cr.add_argument("--out", default=None)
    cr.add_argument("--limit", type=int, default=None)
    cp = cs.add_parser("prompt", help="print the system prompt for a domain pack")
    cp.add_argument("--pack", default="retail_shelf")

    # -- score ---------------------------------------------------------
    b = sub.add_parser("baseline",
                       help="vision-only control arm: the VLM segments too")
    bs = b.add_subparsers(dest="cmd", required=True)
    br = bs.add_parser("run", help="caption with no pose input")
    br.add_argument("--segments", nargs="*", default=None)
    br.add_argument("--backend", default="qwen-local",
                    choices=["stub", "anthropic", "openai", "qwen-local"])
    br.add_argument("--out", default=None)
    bc = bs.add_parser("compare", help="head-to-head against the pose-guided arm")
    bc.add_argument("--pose", default=None)
    bc.add_argument("--vision", default=None)
    bc.add_argument("--segments", nargs="*", default=None)

    sc = sub.add_parser("score", help="format, diversity and grounding metrics")
    sc.add_argument("captions", nargs="?", default=None)

    # -- gold / stereo / lint -----------------------------------------
    g = sub.add_parser("gold", help="score events against the human gold set")
    g.add_argument("--candidates", default=None)

    st = sub.add_parser("stereo", help="validate hand pose against stereo depth")
    _add_episode_args(st)
    st.add_argument("--every", type=float, default=2.0)
    st.add_argument("--scale", type=float, default=0.5)

    ln = sub.add_parser("lint", help="run the atomicity linter")
    ln.add_argument("labels", nargs="?", default=None,
                    help="a .jsonl of labels (default: run the self-test)")
    ln.add_argument("--pack", default=None)
    ln.add_argument("--band", default="measured", choices=["measured", "target"])

    d = sub.add_parser("demo", help="render an annotated demo clip")
    d.add_argument("--segments", nargs="+", required=True)
    d.add_argument("--out", default=None)
    d.add_argument("--gif", default=None)
    d.add_argument("--seconds", type=float, default=None,
                   help="cap each segment at this many seconds")
    d.add_argument("--gif-fps", type=int, default=8)
    d.add_argument("--gif-width", type=int, default=430)
    d.add_argument("--gif-colors", type=int, default=64)
    d.add_argument("--gif-dither", default="bayer:bayer_scale=4",
                   help="ffmpeg paletteuse dither; `none` compresses far "
                        "better on video content")

    sub.add_parser("paths", help="show resolved paths and exit")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    stage = args.stage

    if stage == "paths":
        print(f"repo      {config.ROOT}")
        try:
            print(f"corpus    {config.corpus()}")
        except FileNotFoundError as e:
            print(f"corpus    UNRESOLVED - {e}")
        print(f"data      {config.DATA}")
        print(f"artifacts {config.ARTIFACTS}")
        print(f"segments  {config.SEGMENTS_DIR}")
        for label, path in (("quality", config.QUALITY_RECORDS),
                            ("events", config.EVENTS_RECORDS),
                            ("features", config.FEATURES_DIR),
                            ("spans", config.SPANS),
                            ("captions", config.CAPTIONS)):
            exists = "ok " if path.exists() else "-- "
            print(f"  {exists} {label:<9s} {path}")
        return 0

    if stage == "quality":
        from .stages import quality
        if args.cmd == "measure":
            quality.measure(_episodes(args), args.out)
        elif args.cmd == "calibrate":
            quality.calibrate(_episodes(args))
        elif args.cmd == "selftest":
            return 0 if quality.selftest(_episodes(args)) else 1
        elif args.cmd == "report":
            quality.report(quality.load_records(args.records))
        return 0

    if stage == "events":
        from .stages import events
        if args.cmd == "measure":
            events.measure(_episodes(args), args.out)
        else:
            events.calibrate_actionness(_episodes(args))
        return 0

    if stage == "features":
        from .stages import features
        if args.cmd == "fit":
            features.fit(_episodes(args))
        elif args.cmd == "build":
            features.build(_episodes(args), args.out)
        else:
            features.auc(args.features)
        return 0

    if stage == "segments":
        from .tools import make_segments
        make_segments.render(args.only, args.out)
        return 0

    if stage == "spans":
        from .stages import spans
        cfg = dict(spans.CFG)
        if args.signal:
            cfg["signal"] = args.signal
        if args.no_band:
            cfg["enforce_band"] = False
        if args.no_quality_gate:
            cfg["quality_gate"] = False
        if args.gate_tiers is not None:
            cfg["quality_gate_tiers"] = tuple(args.gate_tiers)
        spans.build(out=args.out, only=args.only, cfg=cfg)
        return 0

    if stage == "caption":
        from .stages import caption
        if args.cmd == "prompt":
            print(caption.system_prompt(args.pack))
        else:
            caption.run(args.spans, args.backend, args.out, args.limit)
        return 0

    if stage == "baseline":
        from .stages import baseline
        if args.cmd == "run":
            baseline.run(args.segments, args.out, args.backend)
        else:
            baseline.compare(args.pose, args.vision, args.segments)
        return 0

    if stage == "score":
        from .stages import score
        score.score(args.captions)
        return 0

    if stage == "gold":
        from .evaluation import gold
        gold.evaluate(candidates_path=args.candidates)
        return 0

    if stage == "stereo":
        from .evaluation import stereo
        results = []
        for path in _episodes(args):
            try:
                r = stereo.validate(str(path), args.every, args.scale)
                if r:
                    results.append({k: v for k, v in r.items() if k != "rows"})
            except Exception as e:
                print(f"FAIL {path}: {type(e).__name__}: {e}")
        if results:
            print("=" * 78)
            print("%-22s %6s %10s %8s %7s %9s %9s" % (
                "episode", "n", "median_err", "MAE", "corr", "<5cm", "coverage"))
            for r in results:
                print("%-22s %6d %+10.3f %8.3f %+7.3f %8.0f%% %8.0f%%" % (
                    r["name"][:22], r["n"], r["median_err"], r["mae"], r["corr"],
                    100 * r["within_5cm"], 100 * r["coverage"]))
            out = config.artifact("events", "stereo_validation.json")
            json.dump(results, open(out, "w"), indent=1)
            print("wrote", out)
        return 0

    if stage == "demo":
        from .tools import render_demo
        render_demo.render(args.segments,
                           args.out or config.artifact("reports", "demo.mp4"),
                           args.seconds, args.gif,
                           gif_fps=args.gif_fps, gif_width=args.gif_width,
                           gif_colors=args.gif_colors,
                           gif_dither=args.gif_dither)
        return 0

    if stage == "lint":
        from .labels import atomicity
        if not args.labels:
            return 0 if atomicity.selftest() else 1
        if args.pack:
            atomicity.use_domain(args.pack)
        labels = [json.loads(x) for x in open(args.labels)]
        return 1 if atomicity.report(labels, args.band, show=10,
                                     name=args.labels) else 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
