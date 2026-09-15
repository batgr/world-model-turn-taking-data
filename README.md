## Development

### Environment

Install the project and its dependencies with:

```bash
uv sync
```

### Structural data audit

Run phase A2 against the configured EgoCom and Ego4D data:

```bash
uv run python -m conv_wm.data.audits.run_structural
```

The JSON report is written to
`${paths.reports}/structural/structural_audit.json`. See
[`docs/data_pipeline/02_structural_audit.md`](docs/data_pipeline/02_structural_audit.md)
for the checks and deliberate exclusions.

### Temporal and media audit

Run the reproducible Phase B population audits in dependency order:

```bash
uv run python -m conv_wm.data.audits.run_media_metadata
uv run python -m conv_wm.data.audits.run_video_timeline
uv run python -m conv_wm.data.audits.run_audio_timeline
uv run python -m conv_wm.data.audits.run_av_sync
```

Reports are written below `${paths.reports}/temporal`. See
[`docs/data_pipeline/03_temporal_audit.md`](docs/data_pipeline/03_temporal_audit.md)
for B1--B3 and the B6 temporal contract, and
[`docs/data_pipeline/04_synchronization.md`](docs/data_pipeline/04_synchronization.md)
for B4 and the deferred B5 scope.
