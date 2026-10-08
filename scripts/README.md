# Script map

For daily use, start with `python ego.py --help` at the repository root.
All existing scripts and flags remain available for reproducible experiments.

| Task | Scripts |
|---|---|
| Current quality workflow | `process_quality_fast.py` |
| Current hand workflow | `benchmark_hand_tracking.py`, `download_owl_hand.py` |
| Human review | `build_quality_review.py`, `build_hand_review.py`, `summarize_quality_review.py` |
| Full RGB annotation / near-real-time candidate | `process_rgb_pipeline.py`, `process_rgb_realtime.py` |
| Public benchmark inputs / Qwen weights | `download_speed_data.py`, `download_speed_model.py` |
| Episode concurrency | `benchmark_episode_concurrency.py` |
| Quality performance | `benchmark_quality.py`, `benchmark_quality_measurements.py`, `compare_quality_benchmarks.py` |
| RGB runtime comparisons | `replay_speed.py`, `summarize_speed.py`, `compare_speed_outputs.py`, `compare_realtime_outputs.py` |
| Hand recovery experiments | `probe_hand_recovery.py`, `probe_hand_competing.py`, `compare_hand_recovery.py` |
| Other hand-detector comparisons | `download_hand_detector.py`, `download_grounded_hand.py` |
| Annotation references and audits | `benchmark_rgb_caption_reference.py`, `export_epic_reference.py`, `evaluate_annotation_audit.py` |
| Native HOT3D benchmarking | `benchmark_hot3d.py`, `finish_hot3d_benchmark.py`, `hot3d_contact_sheet.py` |
| Demo preparation | `annotate_rgb_demo.py`, `export_demo_pose.py`, `make_quality_controls.py` |

Model downloaders write files and contact their published model sources; read
the relevant setup/license document before using an alternative detector.
See [the docs index](../docs/README.md). New routine runs should go to `outputs/`;
existing `artifacts/` results remain in place.
